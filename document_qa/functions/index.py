from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import jieba
except Exception:  # pragma: no cover - optional dependency fallback
    jieba = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency fallback
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency fallback
    SentenceTransformer = None


IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DATE_RE = re.compile(
    r"("
    r"\d{8}|"
    r"20\d{2}年(?:0?[1-9]|1[0-2])月(?:0?[1-9]|[12]\d|3[01])日?|"
    r"20\d{2}年(?:0?[1-9]|1[0-2])月|"
    r"20\d{2}年|"
    r"20\d{2}[./-](?:0?[1-9]|1[0-2])[./-](?:0?[1-9]|[12]\d|3[01])|"
    r"20\d{2}[./-](?:0?[1-9]|1[0-2])"
    r")"
)
DOC_TYPE_WORDS = ("报告", "通知", "请示", "函", "纪要", "方案", "预案", "要点", "意见", "办法", "规则")
DEFAULT_EXCLUDED_TOP_LEVEL_DIRS = {"其他"}
DOCUMENT_KB_EXTRACTOR_VERSION = 2
TERM_STOPWORDS = {
    "关于", "情况", "工作", "汇报", "报告", "通知", "文件", "批办", "有关",
    "进行", "开展", "推进", "建设", "年度", "落实", "印发", "一个", "我们",
    "以及", "通过", "进一步", "相关", "如下", "其中", "目前", "同时", "不断",
    "全市", "全省", "全区", "有没有", "是否", "哪些", "什么", "请问", "帮我",
    "寻找", "查找", "材料", "资料",
}
GENERIC_TERM_SUFFIXES = ("计划", "方案", "报告", "通知", "请示", "情况", "工作", "办法", "意见", "公告", "清单", "要点")
SYNTHESIS_INTENT_HINTS = {
    "requirements": ("要求", "规定", "政策", "措施", "办法", "要点", "规范", "标准", "落实"),
    "status": ("情况", "运行", "现状", "成效", "进展", "数据", "态势"),
    "problems": ("问题", "困难", "不足", "风险", "隐患", "原因"),
    "actions": ("措施", "任务", "安排", "部署", "整治", "推进", "开展"),
    "comparison": ("比较", "差异", "变化", "对比", "不同", "相同"),
    "timeline": ("时间", "年份", "年度", "历年", "阶段", "进度"),
    "materials": ("相关文档", "相关材料", "找", "查找", "寻找", "哪些文档"),
}
SYNTHESIS_STRUCTURE_PATTERNS = (
    ("目标任务", ("目标", "任务", "总体要求", "指导思想", "主要目标")),
    ("重点内容", ("重点", "要点", "主要内容", "重点工作", "工作重点")),
    ("措施要求", ("措施", "要求", "规定", "办法", "规范", "标准", "制度")),
    ("职责分工", ("职责", "分工", "责任", "主体责任", "牵头", "配合")),
    ("流程条件", ("流程", "程序", "申报", "条件", "材料", "步骤", "办理")),
    ("监督管理", ("监管", "监督", "检查", "执法", "考核", "评审", "验收")),
    ("问题风险", ("问题", "风险", "隐患", "困难", "不足", "原因")),
    ("保障建议", ("保障", "建议", "下一步", "工作建议", "改进", "落实")),
)


@dataclass
class DocumentPaths:
    repo_root: Path
    corpus_root: Path
    index_path: Path


@dataclass
class EmbeddingConfig:
    enabled: bool
    provider: str
    api_key: str
    api_url: str
    model: str
    local_model_path: str
    batch_size: int
    max_chars: int
    min_similarity: float


def resolve_paths(domain_dir: Path) -> DocumentPaths:
    repo_root = domain_dir.resolve().parents[1]
    corpus_root = Path(os.getenv("DOCUMENT_QA_ROOT", repo_root / "documents_mineru")).resolve()
    index_path = Path(
        os.getenv("DOCUMENT_QA_INDEX", domain_dir / ".document_qa" / "document_index.sqlite")
    ).resolve()
    return DocumentPaths(repo_root=repo_root, corpus_root=corpus_root, index_path=index_path)


def resolve_embedding_config() -> EmbeddingConfig:
    provider = os.getenv("DOCUMENT_QA_EMBEDDING_PROVIDER", "openai").strip().lower()
    default_model = "BAAI/bge-small-zh-v1.5" if provider == "local" else "text-embedding-3-small"
    return EmbeddingConfig(
        enabled=coerce_bool(os.getenv("DOCUMENT_QA_EMBEDDINGS", "")),
        provider=provider,
        api_key=os.getenv("DOCUMENT_QA_EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY", "sk-placeholder"),
        api_url=os.getenv("DOCUMENT_QA_EMBEDDING_API_URL") or os.getenv("LLM_API_URL", "http://localhost:8090/v1"),
        model=os.getenv("DOCUMENT_QA_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL", default_model),
        local_model_path=os.getenv("DOCUMENT_QA_EMBEDDING_LOCAL_PATH", ""),
        batch_size=max(1, min(int(os.getenv("DOCUMENT_QA_EMBEDDING_BATCH_SIZE", "32")), 128)),
        max_chars=max(300, min(int(os.getenv("DOCUMENT_QA_EMBEDDING_MAX_CHARS", "1800")), 6000)),
        min_similarity=max(0.0, min(float(os.getenv("DOCUMENT_QA_EMBEDDING_MIN_SIMILARITY", "0.55")), 1.0)),
    )


def cached_hf_snapshot(model_id: str) -> Path | None:
    if not model_id or "/" not in model_id:
        return None
    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [
        path for path in snapshots.iterdir()
        if path.is_dir() and (path / "config.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


class DocumentIndex:
    def __init__(self, paths: DocumentPaths):
        self.paths = paths
        self.embedding_config = resolve_embedding_config()

    def ensure(self) -> dict[str, Any]:
        if not self.paths.index_path.exists():
            raise RuntimeError(
                f"文档索引不存在: {self.paths.index_path}。"
                "全量自动重建已禁用，请先恢复索引备份，或离线执行受控重建。"
            )
        try:
            with self._connect() as conn:
                indexed = conn.execute("select count(*) from documents").fetchone()[0]
            files = len(self._markdown_files())
            if indexed == 0 and files:
                raise RuntimeError(
                    f"文档索引为空: {self.paths.index_path}。"
                    "全量自动重建已禁用，请先恢复索引备份，或离线执行受控重建。"
                )
            return {"status": "ready", "documents": indexed, "index_path": str(self.paths.index_path)}
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower():
                raise RuntimeError(
                    f"文档索引当前被其他任务写入锁定: {self.paths.index_path}。"
                    "请稍后重试，或等待 KB/embedding 批处理完成。"
                ) from exc
            raise RuntimeError(
                f"文档索引损坏或不可读: {self.paths.index_path}。"
                "全量自动重建已禁用，请先恢复索引备份，或离线执行受控重建。"
            ) from exc
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(
                f"文档索引损坏或不可读: {self.paths.index_path}。"
                "全量自动重建已禁用，请先恢复索引备份，或离线执行受控重建。"
            ) from exc

    def rebuild(self, force: bool = True) -> dict[str, Any]:
        raise RuntimeError(
            "全量重建文档索引已禁用。请使用 sync_document_index 做增量同步；"
            "如确需全量重建，请使用离线脚本以临时库构建并原子替换主索引。"
        )
        # 保留旧实现供后续改造成离线原子重建；运行时入口禁止调用。
        self.paths.index_path.parent.mkdir(parents=True, exist_ok=True)
        if force and self.paths.index_path.exists():
            self.paths.index_path.unlink()

        md_files = self._markdown_files()
        parsed_documents = [self._parse_document(path) for path in md_files]
        total_documents = len(parsed_documents)
        doc_term_weights = {
            doc["document_id"]: collect_document_term_weights(doc, chunks)
            for doc, chunks in parsed_documents
        }
        term_doc_freq: dict[str, int] = {}
        for weights in doc_term_weights.values():
            for term in weights:
                term_doc_freq[term] = term_doc_freq.get(term, 0) + 1

        with self._connect() as conn:
            self._init_schema(conn)
            for term, doc_freq in term_doc_freq.items():
                conn.execute(
                    "insert or replace into term_stats (term, doc_freq, idf) values (?, ?, ?)",
                    (term, doc_freq, compute_idf(total_documents, doc_freq)),
                )
            documents = 0
            chunks = 0
            for doc, doc_chunks in parsed_documents:
                conn.execute(
                    """
                    insert or replace into documents
                    (document_id, path, title, category, agency, doc_type, date_text, file_mtime, char_count)
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc["document_id"], doc["path"], doc["title"], doc["category"],
                        doc["agency"], doc["doc_type"], doc["date_text"], doc["file_mtime"],
                        doc["char_count"],
                    ),
                )
                for chunk in doc_chunks:
                    conn.execute(
                        """
                        insert or replace into chunks
                        (chunk_id, document_id, ordinal, heading, content, image_refs, char_start)
                        values (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk["chunk_id"], chunk["document_id"], chunk["ordinal"],
                            chunk["heading"], chunk["content"], chunk["image_refs"],
                            chunk["char_start"],
                        ),
                    )
                    conn.execute(
                        """
                        insert into chunks_fts
                        (chunk_id, document_id, title, category, heading, content_tokens, content)
                        values (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk["chunk_id"], chunk["document_id"], doc["title"],
                            doc["category"], chunk["heading"], tokenize_for_search(chunk["content"]),
                            chunk["content"],
                        ),
                    )
                documents += 1
                chunks += len(doc_chunks)
                for term, term_weight in top_weighted_terms(
                    doc_term_weights.get(doc["document_id"], {}),
                    limit=30,
                ):
                    doc_freq = term_doc_freq.get(term, 0)
                    idf = compute_idf(total_documents, doc_freq)
                    conn.execute(
                        """
                        insert into document_terms
                        (document_id, term, term_weight, doc_freq, idf, score)
                        values (?, ?, ?, ?, ?, ?)
                        """,
                        (doc["document_id"], term, term_weight, doc_freq, idf, term_weight * idf),
                    )
        result = {
            "status": "rebuilt",
            "documents": documents,
            "chunks": chunks,
            "corpus_root": str(self.paths.corpus_root),
            "index_path": str(self.paths.index_path),
        }
        if self.embedding_config.enabled:
            result["embeddings"] = {
                "status": "not_rebuilt",
                "message": "文档索引已重建；如需语义检索，请显式调用 rebuild_document_embeddings。",
            }
        return result

    def sync(self) -> dict[str, Any]:
        """Incrementally synchronize the SQLite index with the markdown corpus."""
        if not self.paths.index_path.exists():
            raise RuntimeError(
                f"文档索引不存在: {self.paths.index_path}。"
                "增量同步需要已有索引；请先恢复索引备份，或离线执行受控全量重建。"
            )

        md_files = self._markdown_files()
        current_by_path = {
            path.relative_to(self.paths.repo_root).as_posix(): path
            for path in md_files
        }

        with self._connect() as conn:
            self._init_schema(conn)
            existing_rows = conn.execute(
                "select document_id, path, file_mtime from documents"
            ).fetchall()
            existing_by_path = {row["path"]: row for row in existing_rows}

            deleted_paths = sorted(set(existing_by_path) - set(current_by_path))
            upsert_paths = []
            unchanged = 0
            for rel_path, path in sorted(current_by_path.items()):
                row = existing_by_path.get(rel_path)
                if not row:
                    upsert_paths.append((rel_path, path, "added"))
                    continue
                if abs(float(row["file_mtime"] or 0.0) - path.stat().st_mtime) > 1e-6:
                    upsert_paths.append((rel_path, path, "updated"))
                else:
                    unchanged += 1

            deleted = 0
            for rel_path in deleted_paths:
                self._delete_document(conn, existing_by_path[rel_path]["document_id"])
                deleted += 1

            added = 0
            updated = 0
            chunks_upserted = 0
            for _rel_path, path, action in upsert_paths:
                doc, chunks = self._parse_document(path)
                self._delete_document(conn, doc["document_id"])
                self._insert_parsed_document(conn, doc, chunks)
                chunks_upserted += len(chunks)
                if action == "added":
                    added += 1
                else:
                    updated += 1

            if deleted or upsert_paths:
                self._refresh_term_statistics(conn)

            documents = conn.execute("select count(*) from documents").fetchone()[0]
            chunks = conn.execute("select count(*) from chunks").fetchone()[0]

        result = {
            "status": "synced",
            "documents": documents,
            "chunks": chunks,
            "added": added,
            "updated": updated,
            "deleted": deleted,
            "unchanged": unchanged,
            "chunks_upserted": chunks_upserted,
            "corpus_root": str(self.paths.corpus_root),
            "index_path": str(self.paths.index_path),
        }
        if self.embedding_config.enabled:
            result["embeddings"] = {
                "status": "not_synced",
                "message": "如需同步语义向量，请调用 rebuild_document_embeddings(force=false)，它只会补缺失或文本变化的 chunk。",
            }
        return result

    def rebuild_embeddings(self, force: bool = False,
                           ensure_index: bool = True) -> dict[str, Any]:
        if ensure_index:
            self.ensure()
        config = self.embedding_config
        if not config.enabled:
            return {
                "status": "disabled",
                "message": "设置 DOCUMENT_QA_EMBEDDINGS=true 后再重建语义向量索引。",
            }
        if config.provider not in {"openai", "local"}:
            return {
                "status": "unavailable",
                "provider": config.provider,
                "message": "Unsupported embedding provider. Use openai or local.",
            }
        if config.provider == "openai" and OpenAI is None:
            return {"status": "unavailable", "message": "openai package is not available"}
        if config.provider == "local" and SentenceTransformer is None:
            return {"status": "unavailable", "message": "sentence-transformers package is not available"}

        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                """
                select c.chunk_id, c.heading, c.content, d.title, d.path, d.category, d.agency, d.doc_type
                from chunks c join documents d using (document_id)
                order by d.path, c.ordinal
                """
            ).fetchall()
            existing = {}
            if not force:
                existing_rows = conn.execute(
                    "select chunk_id, text_hash from chunk_embeddings where model = ?",
                    (config.model,),
                ).fetchall()
                existing = {row["chunk_id"]: row["text_hash"] for row in existing_rows}

        inputs = []
        for row in rows:
            text = embedding_text(dict(row), config.max_chars)
            text_hash = stable_id(text)
            if not force and existing.get(row["chunk_id"]) == text_hash:
                continue
            inputs.append((row["chunk_id"], text, text_hash))

        embedded = 0
        dimensions = 0
        with self._connect() as conn:
            self._init_schema(conn)
            if force:
                conn.execute("delete from chunk_embeddings where model = ?", (config.model,))
            for batch in batched(inputs, config.batch_size):
                vectors = self._embed_texts([item[1] for item in batch])
                for (chunk_id, _text, text_hash), vector in zip(batch, vectors, strict=True):
                    dimensions = len(vector)
                    conn.execute(
                        """
                        insert or replace into chunk_embeddings
                        (chunk_id, model, dim, embedding, text_hash)
                        values (?, ?, ?, ?, ?)
                        """,
                        (chunk_id, config.model, dimensions, pack_vector(vector), text_hash),
                    )
                    embedded += 1

        return {
            "status": "rebuilt" if force else "updated",
            "provider": config.provider,
            "model": config.model,
            "local_model_path": self._local_embedding_model_path() if config.provider == "local" else "",
            "chunks_considered": len(rows),
            "chunks_embedded": embedded,
            "dimensions": dimensions,
        }

    def search(self, query: str, *, category: str = "", agency: str = "",
               doc_type: str = "", limit: int = 8,
               rerank: bool = False, semantic: bool = False,
               debug: bool = False) -> dict[str, Any]:
        self.ensure()
        clean_query = (query or "").strip()
        limit = max(1, min(int(limit or 8), 30))
        rerank = coerce_bool(rerank)
        semantic = coerce_bool(semantic)
        debug = coerce_bool(debug)
        filters, params = self._sql_filters(category=category, agency=agency, doc_type=doc_type)

        recall_limit = max(limit * 8, 30) if rerank and clean_query else limit
        ranked_candidates = self._recall_chunks(
            clean_query,
            filters,
            params,
            limit=recall_limit,
            semantic=semantic,
        )
        if rerank and clean_query:
            profile = self._build_query_profile(clean_query)
            recall_hits = ranked_candidates[:recall_limit]
            hits = self._rerank_profiled_hits(recall_hits, profile, limit=limit, debug=debug)
            result = {
                "query": clean_query,
                "filters": compact_dict({"category": category, "agency": agency, "doc_type": doc_type}),
                "rerank": "document_level_query_profile",
                "semantic": semantic,
                "query_terms": profile[:12],
                "count": len(hits),
                "hits": hits,
            }
            if debug:
                result["debug"] = {
                    "recall_count": len(recall_hits),
                    "rerank": "document_level_query_profile",
                }
            return result
        hits = self._strip_internal_fields(ranked_candidates[:limit])
        return {
            "query": clean_query,
            "filters": compact_dict({"category": category, "agency": agency, "doc_type": doc_type}),
            "semantic": semantic,
            "count": len(hits),
            "hits": hits,
        }

    def list_documents(self, *, category: str = "", agency: str = "",
                       doc_type: str = "", title_like: str = "",
                       limit: int = 20) -> list[dict[str, Any]]:
        self.ensure()
        filters = []
        params: list[Any] = []
        for field, value in (("category", category), ("agency", agency), ("doc_type", doc_type)):
            if value:
                filters.append(f"{field} like ?")
                params.append(f"%{value}%")
        if title_like:
            filters.append("title like ?")
            params.append(f"%{title_like}%")
        limit = max(1, min(int(limit or 20), 100))
        sql = f"""
            select document_id, title, path, category, agency, doc_type, date_text, char_count
            from documents
            where {' and '.join(filters) if filters else '1=1'}
            order by category, title
            limit ?
        """
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, [*params, limit])]

    def read_document(self, *, path: str = "", document_id: str = "",
                      chunk_id: str = "", heading: str = "",
                      max_chars: int = 6000) -> dict[str, Any]:
        self.ensure()
        max_chars = max(1000, min(int(max_chars or 6000), 20000))
        with self._connect() as conn:
            if chunk_id:
                chunk = conn.execute(
                    "select document_id, ordinal from chunks where chunk_id = ?",
                    (chunk_id,),
                ).fetchone()
                if not chunk:
                    return {"error": f"未找到 chunk_id: {chunk_id}"}
                doc = self._get_doc(conn, chunk["document_id"])
                rows = conn.execute(
                    """
                    select chunk_id, heading, content, image_refs, ordinal
                    from chunks
                    where document_id = ? and ordinal between ? and ?
                    order by ordinal
                    """,
                    (chunk["document_id"], chunk["ordinal"] - 1, chunk["ordinal"] + 1),
                ).fetchall()
                return self._format_read_result(doc, rows, max_chars, focus_chunk_id=chunk_id)

            doc = None
            if document_id:
                doc = self._get_doc(conn, document_id)
            elif path:
                doc = conn.execute(
                    "select * from documents where path = ? or path like ? limit 1",
                    (path, f"%{path}%"),
                ).fetchone()
            if not doc:
                return {"error": "请提供有效的 path、document_id 或 chunk_id"}

            filters = "document_id = ?"
            params: list[Any] = [doc["document_id"]]
            if heading:
                filters += " and (heading like ? or content like ?)"
                params.extend([f"%{heading}%", f"%{heading}%"])
            rows = conn.execute(
                f"""
                select chunk_id, heading, content, image_refs, ordinal
                from chunks
                where {filters}
                order by ordinal
                """,
                params,
            ).fetchall()
            return self._format_read_result(doc, rows, max_chars)

    def find_related_documents(self, *, seed_path: str = "", seed_title: str = "",
                               query_hint: str = "", limit: int = 10,
                               debug: bool = False) -> dict[str, Any]:
        limit = max(1, min(int(limit or 10), 30))
        debug = coerce_bool(debug)
        seed = self._find_seed_document(seed_path=seed_path, seed_title=seed_title)
        if not seed:
            return {
                "error": "未找到 seed 文档。seed_path 或 seed_title 必须指向被排除目录中的 Markdown 文档。",
                "allowed_seed_top_level_dirs": sorted(DEFAULT_EXCLUDED_TOP_LEVEL_DIRS),
            }

        text = seed["path"].read_text(encoding="utf-8", errors="ignore")
        profile = self._build_seed_profile(seed["title"], text, query_hint=query_hint)
        terms = [item["term"] for item in profile[:12]]
        search_query = " ".join(terms[:10])
        recall_limit = max(limit * 8, 30)
        recall_hits = self._recall_chunks(search_query, [], [], limit=recall_limit)
        hits = self._rerank_profiled_hits(recall_hits[:recall_limit], profile, limit=limit, debug=debug)
        result = {
            "seed": {
                "title": seed["title"],
                "path": seed["relative_path"],
                "top_level_dir": seed["top_level_dir"],
                "usage": "seed_only",
                "note": "seed 文档只用于生成检索线索，不能作为最终回答出处。",
            },
            "query_hint": query_hint,
            "query_terms": profile[:12],
            "generated_query": search_query,
            "count": len(hits),
            "hits": hits,
        }
        if debug:
            result["debug"] = {
                "recall_count": len(recall_hits),
                "rerank": "document_level_seed_profile",
            }
        return result

    def build_document_kb(self, *, force: bool = False,
                          include_soft: bool = False,
                          limit: int = 0) -> dict[str, Any]:
        self.ensure()
        force = coerce_bool(force)
        include_soft = coerce_bool(include_soft)
        limit = max(0, int(limit or 0))

        with self._connect() as conn:
            self._init_schema(conn)
            docs = self._load_relation_documents(conn)
            doc_terms = self._load_document_terms(conn)
            title_keys = {
                doc["document_id"]: normalize_series_title(doc.get("title", ""))
                for doc in docs
            }
            signatures = {
                doc["document_id"]: document_kb_signature(doc)
                for doc in docs
            }
            changed_ids: set[str] | None = None
            removed_state = 0

            if force:
                conn.execute("delete from document_relations")
                conn.execute("delete from document_metadata")
                conn.execute("delete from document_kb_state")
            else:
                state = self._load_document_kb_state(conn)
                doc_ids = set(signatures)
                removed_ids = sorted(set(state) - doc_ids)
                if removed_ids:
                    self._delete_document_kb_for_ids(conn, removed_ids, delete_incoming=True)
                    removed_state = len(removed_ids)
                changed_ids = {
                    document_id
                    for document_id, signature in signatures.items()
                    if (
                        document_id not in state
                        or state[document_id]["signature"] != signature
                        or bool(state[document_id]["include_soft"]) != include_soft
                    )
                }
                if not changed_ids and not removed_state:
                    stats = self._document_relation_stats(conn)
                    return {
                        "status": "unchanged",
                        "mode": "incremental",
                        "documents": len(docs),
                        "documents_changed": 0,
                        "documents_pending": 0,
                        "state_removed": 0,
                        "relations": 0,
                        "include_soft": include_soft,
                        "stats": stats,
                    }
                total_changed = len(changed_ids)
                if limit and len(changed_ids) > limit:
                    changed_ids = set(sorted(changed_ids)[:limit])
                self._delete_document_kb_for_ids(conn, sorted(changed_ids), delete_incoming=False)
            if force:
                total_changed = len(docs)

            docs_to_refresh = docs if force else [
                doc for doc in docs if doc["document_id"] in (changed_ids or set())
            ]
            self._upsert_document_metadata(conn, docs_to_refresh)

            relations = self._build_document_relations(
                conn,
                docs,
                doc_terms,
                doc_term_index=self._build_doc_term_index(doc_terms),
                title_keys=title_keys,
                include_soft=include_soft,
                focus_ids=None if force else changed_ids,
            )

            inserted = 0
            for relation in relations.values():
                conn.execute(
                    """
                    insert or replace into document_relations
                    (relation_id, source_document_id, target_document_id, relation_type,
                     confidence, evidence, source_chunk_id, target_chunk_id, method, metadata)
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relation["relation_id"],
                        relation["source_document_id"],
                        relation["target_document_id"],
                        relation["relation_type"],
                        relation["confidence"],
                        relation["evidence"],
                        relation.get("source_chunk_id", ""),
                        relation.get("target_chunk_id", ""),
                        relation["method"],
                        json.dumps(relation.get("metadata", {}), ensure_ascii=False),
                    ),
                )
                inserted += 1

            self._upsert_document_kb_state(
                conn,
                docs if force else docs_to_refresh,
                signatures,
                include_soft=include_soft,
            )
            stats = self._document_relation_stats(conn)

        return {
            "status": "rebuilt" if force else "updated",
            "mode": "full" if force else "incremental",
            "documents": len(docs),
            "documents_changed": len(docs) if force else len(changed_ids or set()),
            "documents_pending": 0 if force else max(0, total_changed - len(changed_ids or set())),
            "state_removed": removed_state,
            "relations": inserted,
            "include_soft": include_soft,
            "stats": stats,
        }

    def find_document_relations(self, *, path: str = "", document_id: str = "",
                                title_like: str = "", relation_type: str = "",
                                direction: str = "both", limit: int = 20) -> dict[str, Any]:
        self.ensure()
        limit = max(1, min(int(limit or 20), 100))
        direction = (direction or "both").strip().lower()
        if direction not in {"source", "target", "both"}:
            direction = "both"
        with self._connect() as conn:
            self._init_schema(conn)
            focus = self._resolve_relation_document(
                conn,
                path=path,
                document_id=document_id,
                title_like=title_like,
            )
            if not focus:
                return {"error": "请提供有效的 path、document_id 或 title_like 来定位文档"}

            relation_filters = []
            params: list[Any] = []
            if direction in {"source", "both"}:
                relation_filters.append("r.source_document_id = ?")
                params.append(focus["document_id"])
            if direction in {"target", "both"}:
                relation_filters.append("r.target_document_id = ?")
                params.append(focus["document_id"])
            where = f"({' or '.join(relation_filters)})"
            if relation_type:
                where += " and r.relation_type = ?"
                params.append(relation_type)

            rows = conn.execute(
                f"""
                select r.*, sd.title as source_title, sd.path as source_path,
                       sd.category as source_category, td.title as target_title,
                       td.path as target_path, td.category as target_category
                from document_relations r
                join documents sd on sd.document_id = r.source_document_id
                join documents td on td.document_id = r.target_document_id
                where {where}
                order by r.confidence desc, r.relation_type, td.title, sd.title
                limit ?
                """,
                [*params, limit],
            ).fetchall()
        return {
            "focus_document": dict(focus),
            "direction": direction,
            "relation_type": relation_type,
            "count": len(rows),
            "relations": [format_relation_row(row, focus["document_id"]) for row in rows],
        }

    def expand_document_context(self, *, path: str = "", document_id: str = "",
                                title_like: str = "", relation_types: str = "",
                                limit: int = 20) -> dict[str, Any]:
        wanted = {
            item.strip()
            for item in (relation_types or "").split(",")
            if item.strip()
        }
        result = self.find_document_relations(
            path=path,
            document_id=document_id,
            title_like=title_like,
            direction="both",
            limit=limit,
        )
        if "error" in result:
            return result
        grouped: dict[str, list[dict[str, Any]]] = {}
        recommended: dict[str, dict[str, Any]] = {}
        for relation in result["relations"]:
            if wanted and relation["relation_type"] not in wanted:
                continue
            grouped.setdefault(relation["relation_type"], []).append(relation)
            other = relation["other_document"]
            current = recommended.get(other["document_id"])
            if not current or relation["confidence"] > current["confidence"]:
                recommended[other["document_id"]] = {
                    **other,
                    "relation_type": relation["relation_type"],
                    "confidence": relation["confidence"],
                    "evidence": relation["evidence"],
                    "method": relation["method"],
                }
        return {
            "focus_document": result["focus_document"],
            "relation_types": sorted(grouped),
            "groups": grouped,
            "recommended_documents": sorted(
                recommended.values(),
                key=lambda item: (-float(item["confidence"]), item["title"]),
            )[:max(1, min(int(limit or 20), 100))],
        }

    def prepare_answer_context(self, *, query: str, limit_docs: int = 6,
                               max_chars_per_doc: int = 4000,
                               include_relations: bool = True,
                               debug: bool = False) -> dict[str, Any]:
        self.ensure()
        clean_query = (query or "").strip()
        if not clean_query:
            return {"error": "query 不能为空"}
        limit_docs = max(1, min(int(limit_docs or 6), 12))
        max_chars_per_doc = max(1200, min(int(max_chars_per_doc or 4000), 12000))
        include_relations = coerce_bool(include_relations)
        debug = coerce_bool(debug)
        expected_years = expected_years_from_query(clean_query)
        search_limit = min(max(limit_docs * 5, 18), 30)
        relation_seed_limit = min(max(limit_docs * 2, 8), search_limit)

        search_result = self.search(
            clean_query,
            limit=search_limit,
            rerank=True,
            debug=debug,
        )
        with self._connect() as conn:
            self._init_schema(conn)
            doc_scores = self._rank_context_documents(conn, search_result.get("hits", []), clean_query)
            relation_suggestions = []
            if include_relations:
                relation_suggestions = self._relation_context_suggestions(
                    conn,
                    [item["document_id"] for item in doc_scores[:relation_seed_limit]],
                    clean_query,
                    limit=max(6, limit_docs * 2),
                )
                known = {item["document_id"] for item in doc_scores}
                for suggestion in relation_suggestions:
                    doc_id = suggestion["document"]["document_id"]
                    if doc_id not in known:
                        doc_scores.append({
                            **suggestion["document"],
                            "context_score": max(0.0, float(suggestion["confidence"]) * 10.0),
                            "selection_reason": f"KB关系扩展：{suggestion['relation_type']} - {suggestion['evidence']}",
                            "search_hit_count": 0,
                            "best_hit": {},
                        })
                        known.add(doc_id)

            all_metadata = self._load_document_metadata(conn, [item["document_id"] for item in doc_scores])
            for item in doc_scores:
                temporal = document_temporal_metadata({
                    **item,
                    "metadata": all_metadata.get(item["document_id"], {}),
                })
                item["_temporal"] = temporal
                item["_out_of_time_range"] = (
                    bool(expected_years)
                    and bool(temporal.get("best_year"))
                    and temporal["best_year"] not in expected_years
                )

            eligible_doc_scores = [item for item in doc_scores if not item.get("_out_of_time_range")]
            if len(eligible_doc_scores) < min(limit_docs, 2):
                eligible_doc_scores = doc_scores

            selected = sorted(
                eligible_doc_scores,
                key=lambda item: (
                    bool(item.get("_out_of_time_range")),
                    -float(item.get("context_score", 0.0)),
                    int(item.get("char_count", 0)) < 200,
                    item.get("title", ""),
                ),
            )[:limit_docs]
            metadata = {
                item["document_id"]: all_metadata.get(item["document_id"], {})
                for item in selected
            }
            selected_ids = {item["document_id"] for item in selected}
            relation_suggestions_used = []
            if include_relations:
                for suggestion in relation_suggestions:
                    doc = suggestion.get("document", {})
                    doc_id = doc.get("document_id", "")
                    if doc_id not in selected_ids:
                        continue
                    temporal = document_temporal_metadata({
                        **doc,
                        "metadata": all_metadata.get(doc_id, {}),
                    })
                    if (
                        expected_years
                        and temporal.get("best_year")
                        and temporal["best_year"] not in expected_years
                    ):
                        continue
                    relation_suggestions_used.append({
                        **suggestion,
                        "temporal": temporal,
                    })
                    if len(relation_suggestions_used) >= limit_docs:
                        break

        evidence_documents = []
        for item in selected:
            read_result = self.read_document(
                document_id=item["document_id"],
                max_chars=max_chars_per_doc,
            )
            doc_meta = metadata.get(item["document_id"], {})
            quality = document_quality(item, read_result)
            temporal = document_temporal_metadata(
                {
                    **item,
                    "read_content": read_result.get("content", ""),
                    "metadata": doc_meta,
                }
            )
            evidence_documents.append({
                "document": {
                    "document_id": item["document_id"],
                    "title": item.get("title", ""),
                    "path": item.get("path", ""),
                    "category": item.get("category", ""),
                    "agency": item.get("agency", ""),
                    "doc_type": item.get("doc_type", ""),
                    "char_count": item.get("char_count", 0),
                },
                "selection": {
                    "context_score": round(float(item.get("context_score", 0.0)), 3),
                    "reason": item.get("selection_reason", ""),
                    "search_hit_count": item.get("search_hit_count", 0),
                    "best_hit": item.get("best_hit", {}),
                },
                "temporal": temporal,
                "quality": quality,
                "included_chunk_ids": read_result.get("included_chunk_ids", []),
                "content": read_result.get("content", ""),
                "truncated": read_result.get("truncated", False),
            })

        temporal_coverage = build_temporal_coverage(clean_query, evidence_documents)
        answer_guidance = [
            "优先基于 documents[].content 综合回答，并引用 title/path 或 included_chunk_ids。",
            "文档问答要先按问题意图归纳主题/要求/事项维度，再把多篇文档证据合并到相应维度下，不要按命中文档逐篇堆摘要。",
            "优先参考 synthesis_outline 的通用归纳维度；它来自问题意图、文档标题/小节结构和跨文档证据分布，不是业务专用模板。",
            "如果 synthesis_outline.recommended_answer_sections 非空，优先用这些 section 作为回答的一级小节，再在每节下合并多篇证据。",
            "同一事实或要求被多篇文档支持时，应合并表述并列出代表性出处；文档之间存在差异或适用范围不同时要说明。",
            "如果 temporal_coverage 显示时间范围覆盖不足，必须在答案中说明证据限制。",
            "不要声称覆盖 temporal_coverage.found_years 之外的年份；missing_years 只能表述为当前证据缺口。",
            "quality.status 为 low/empty 的文档只能作为线索，不宜作为关键事实依据。",
            "KB 关系用于补充上下文，不替代 read_document 证据。",
        ]
        strict_constraints = build_strict_answer_constraints(temporal_coverage)
        return {
            "query": clean_query,
            "strict_answer_constraints": strict_constraints,
            "workflow": [
                "search_documents(rerank=true)",
                "deduplicate_and_rank_documents",
                "expand_with_document_kb" if include_relations else "skip_kb_expansion",
                "read_core_documents",
                "summarize_with_temporal_and_quality_metadata",
            ],
            "search": {
                "count": search_result.get("count", 0),
                "candidate_limit": search_limit,
                "relation_seed_limit": relation_seed_limit if include_relations else 0,
                "query_terms": search_result.get("query_terms", []),
            },
            "temporal_coverage": temporal_coverage,
            "answer_guidance": answer_guidance,
            "synthesis_outline": build_synthesis_outline(clean_query, evidence_documents),
            "evidence_outline": build_evidence_outline(evidence_documents),
            "quality_notes": build_quality_notes(evidence_documents),
            "kb_relations_used": relation_suggestions_used if include_relations else [],
            "documents": evidence_documents,
        }

    def query_rows(self, object_type: str, filters: dict[str, Any] | None = None,
                   limit: int | None = None, order_by: str | None = None,
                   offset: int | None = None) -> list[dict[str, Any]]:
        self.ensure()
        limit = limit or 50
        if object_type == "Document":
            return self._query_documents(filters, limit, order_by, offset)
        if object_type == "DocumentChunk":
            return self._query_chunks(filters, limit, order_by, offset)
        if object_type == "DocumentRelation":
            return self._query_relations(filters, limit, order_by, offset)
        return []

    def query_by_id(self, object_type: str, id_value: Any) -> dict[str, Any] | None:
        id_field = {
            "Document": "document_id",
            "DocumentChunk": "chunk_id",
            "DocumentRelation": "relation_id",
        }.get(object_type, "document_id")
        rows = self.query_rows(object_type, {id_field: id_value}, limit=1)
        return rows[0] if rows else None

    def count(self, object_type: str, filters: dict[str, Any] | None = None) -> int:
        return len(self.query_rows(object_type, filters, limit=100000))

    def search_text(self, keyword: str, limit: int = 20) -> list[dict[str, Any]]:
        result = self.search(keyword, limit=limit)
        return [{**hit, "_object_type": "DocumentChunk"} for hit in result["hits"]]

    def _query_documents(self, filters: dict[str, Any] | None, limit: int,
                         order_by: str | None, offset: int | None) -> list[dict[str, Any]]:
        where, params = build_filter_sql(filters)
        sql = f"""
            select document_id, title, path, category, agency, doc_type, date_text, char_count
            from documents
            where {where}
            order by {safe_order_by(order_by, 'title')}
            limit ? offset ?
        """
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, [*params, limit, offset or 0])]

    def _query_chunks(self, filters: dict[str, Any] | None, limit: int,
                      order_by: str | None, offset: int | None) -> list[dict[str, Any]]:
        where, params = build_filter_sql(filters, table_alias="c")
        sql = f"""
            select c.chunk_id, c.document_id, d.title, d.path, d.category,
                   c.heading, c.content, c.image_refs
            from chunks c join documents d using (document_id)
            where {where}
            order by {safe_order_by(order_by, 'c.ordinal')}
            limit ? offset ?
        """
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, [*params, limit, offset or 0])]

    def _query_relations(self, filters: dict[str, Any] | None, limit: int,
                         order_by: str | None, offset: int | None) -> list[dict[str, Any]]:
        where, params = build_filter_sql(filters, table_alias="r")
        sql = f"""
            select r.relation_id, r.source_document_id, sd.title as source_title,
                   sd.path as source_path, r.target_document_id, td.title as target_title,
                   td.path as target_path, r.relation_type, r.confidence, r.evidence,
                   r.source_chunk_id, r.target_chunk_id, r.method, r.metadata
            from document_relations r
            join documents sd on sd.document_id = r.source_document_id
            join documents td on td.document_id = r.target_document_id
            where {where}
            order by {safe_order_by(order_by, 'r.confidence desc')}
            limit ? offset ?
        """
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, [*params, limit, offset or 0])]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.paths.index_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 30000")
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            create table if not exists documents (
                document_id text primary key,
                path text unique not null,
                title text,
                category text,
                agency text,
                doc_type text,
                date_text text,
                file_mtime real,
                char_count integer
            );
            create table if not exists chunks (
                chunk_id text primary key,
                document_id text not null,
                ordinal integer not null,
                heading text,
                content text,
                image_refs text,
                char_start integer,
                foreign key(document_id) references documents(document_id)
            );
            create virtual table if not exists chunks_fts using fts5(
                chunk_id unindexed,
                document_id unindexed,
                title,
                category,
                heading,
                content_tokens,
                content
            );
            create table if not exists term_stats (
                term text primary key,
                doc_freq integer not null,
                idf real not null
            );
            create table if not exists document_terms (
                document_id text not null,
                term text not null,
                term_weight real not null,
                doc_freq integer not null,
                idf real not null,
                score real not null,
                primary key (document_id, term),
                foreign key(document_id) references documents(document_id)
            );
            create table if not exists chunk_embeddings (
                chunk_id text not null,
                model text not null,
                dim integer not null,
                embedding blob not null,
                text_hash text not null,
                primary key (chunk_id, model),
                foreign key(chunk_id) references chunks(chunk_id)
            );
            create table if not exists document_relations (
                relation_id text primary key,
                source_document_id text not null,
                target_document_id text not null,
                relation_type text not null,
                confidence real not null,
                evidence text,
                source_chunk_id text,
                target_chunk_id text,
                method text not null,
                metadata text,
                foreign key(source_document_id) references documents(document_id),
                foreign key(target_document_id) references documents(document_id)
            );
            create table if not exists document_metadata (
                document_id text not null,
                key text not null,
                value text not null,
                confidence real not null,
                evidence text,
                method text not null,
                primary key (document_id, key, value, method),
                foreign key(document_id) references documents(document_id)
            );
            create table if not exists document_kb_state (
                document_id text primary key,
                signature text not null,
                include_soft integer not null,
                refreshed_at real not null,
                foreign key(document_id) references documents(document_id)
            );
            create index if not exists idx_chunks_document on chunks(document_id, ordinal);
            create index if not exists idx_documents_category on documents(category);
            create index if not exists idx_document_terms_term on document_terms(term);
            create index if not exists idx_chunk_embeddings_model on chunk_embeddings(model);
            create index if not exists idx_document_relations_source on document_relations(source_document_id);
            create index if not exists idx_document_relations_target on document_relations(target_document_id);
            create index if not exists idx_document_relations_type on document_relations(relation_type);
            create index if not exists idx_document_metadata_key on document_metadata(key, value);
            create index if not exists idx_document_kb_state_signature on document_kb_state(signature);
            """
        )

    def _delete_document(self, conn: sqlite3.Connection, document_id: str) -> None:
        chunk_ids = [
            row["chunk_id"]
            for row in conn.execute(
                "select chunk_id from chunks where document_id = ?",
                (document_id,),
            ).fetchall()
        ]
        if chunk_ids:
            conn.executemany(
                "delete from chunk_embeddings where chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )
            conn.executemany(
                "delete from chunks_fts where chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )
        conn.execute("delete from chunks where document_id = ?", (document_id,))
        conn.execute("delete from document_terms where document_id = ?", (document_id,))
        conn.execute("delete from document_metadata where document_id = ?", (document_id,))
        conn.execute(
            """
            delete from document_relations
            where source_document_id = ? or target_document_id = ?
            """,
            (document_id, document_id),
        )
        conn.execute("delete from document_kb_state where document_id = ?", (document_id,))
        conn.execute("delete from documents where document_id = ?", (document_id,))

    def _insert_parsed_document(self, conn: sqlite3.Connection,
                                doc: dict[str, Any],
                                chunks: list[dict[str, Any]]) -> None:
        conn.execute(
            """
            insert or replace into documents
            (document_id, path, title, category, agency, doc_type, date_text, file_mtime, char_count)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc["document_id"], doc["path"], doc["title"], doc["category"],
                doc["agency"], doc["doc_type"], doc["date_text"], doc["file_mtime"],
                doc["char_count"],
            ),
        )
        for chunk in chunks:
            conn.execute(
                """
                insert or replace into chunks
                (chunk_id, document_id, ordinal, heading, content, image_refs, char_start)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["chunk_id"], chunk["document_id"], chunk["ordinal"],
                    chunk["heading"], chunk["content"], chunk["image_refs"],
                    chunk["char_start"],
                ),
            )
            conn.execute(
                """
                insert into chunks_fts
                (chunk_id, document_id, title, category, heading, content_tokens, content)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["chunk_id"], chunk["document_id"], doc["title"],
                    doc["category"], chunk["heading"], tokenize_for_search(chunk["content"]),
                    chunk["content"],
                ),
            )
        weights = collect_document_term_weights(doc, chunks)
        for term, term_weight in top_weighted_terms(weights, limit=30):
            conn.execute(
                """
                insert or replace into document_terms
                (document_id, term, term_weight, doc_freq, idf, score)
                values (?, ?, ?, 0, 0.0, ?)
                """,
                (doc["document_id"], term, term_weight, term_weight),
            )

    def _refresh_term_statistics(self, conn: sqlite3.Connection) -> None:
        total_documents = conn.execute("select count(*) from documents").fetchone()[0]
        conn.execute("delete from term_stats")
        rows = conn.execute(
            """
            select term, count(distinct document_id) as doc_freq
            from document_terms
            group by term
            """
        ).fetchall()
        for row in rows:
            idf = compute_idf(total_documents, int(row["doc_freq"]))
            conn.execute(
                "insert or replace into term_stats (term, doc_freq, idf) values (?, ?, ?)",
                (row["term"], row["doc_freq"], idf),
            )
            conn.execute(
                """
                update document_terms
                set doc_freq = ?, idf = ?, score = term_weight * ?
                where term = ?
                """,
                (row["doc_freq"], idf, idf, row["term"]),
            )

    def _load_relation_documents(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            select d.document_id, d.path, d.title, d.category, d.agency, d.doc_type,
                   d.date_text, d.char_count,
                   min(c.chunk_id) as first_chunk_id,
                   group_concat(coalesce(c.heading, ''), '\n') as headings,
                   group_concat(substr(coalesce(c.content, ''), 1, 900), '\n') as content_sample
            from documents d
            left join chunks c on c.document_id = d.document_id
            group by d.document_id
            order by d.path
            """
        ).fetchall()
        docs = []
        for row in rows:
            doc = dict(row)
            title_relation_text = "\n".join(
                str(doc.get(key, "") or "")
                for key in ("title", "headings", "path")
            )
            relation_text = "\n".join(
                str(doc.get(key, "") or "")
                for key in ("title", "headings", "content_sample", "path")
            )
            doc["quoted_terms"] = extract_quoted_terms(title_relation_text)
            doc["relation_text"] = relation_text
            doc["doc_numbers"] = extract_doc_numbers(title_relation_text)
            doc["cited_doc_numbers"] = extract_doc_numbers(relation_text)
            docs.append(doc)
        return docs

    def _load_document_terms(self, conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
        rows = conn.execute(
            """
            select document_id, term, score
            from document_terms
            order by document_id, score desc
            """
        ).fetchall()
        by_doc: dict[str, dict[str, float]] = {}
        for row in rows:
            terms = by_doc.setdefault(row["document_id"], {})
            if len(terms) < 24:
                terms[row["term"]] = float(row["score"])
        return by_doc

    def _build_doc_term_index(self, doc_terms: dict[str, dict[str, float]]) -> dict[str, set[str]]:
        index: dict[str, set[str]] = {}
        for document_id, terms in doc_terms.items():
            for term in terms:
                if len(term) >= 3 and classify_term(term) not in {"form", "short"}:
                    index.setdefault(term, set()).add(document_id)
        return index

    def _load_document_kb_state(self, conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
        rows = conn.execute(
            """
            select document_id, signature, include_soft, refreshed_at
            from document_kb_state
            """
        ).fetchall()
        return {
            row["document_id"]: {
                "signature": row["signature"],
                "include_soft": bool(row["include_soft"]),
                "refreshed_at": float(row["refreshed_at"] or 0.0),
            }
            for row in rows
        }

    def _delete_document_kb_for_ids(self, conn: sqlite3.Connection,
                                    document_ids: list[str],
                                    *,
                                    delete_incoming: bool = False) -> None:
        if not document_ids:
            return
        conn.executemany(
            "delete from document_metadata where document_id = ?",
            [(document_id,) for document_id in document_ids],
        )
        if delete_incoming:
            conn.executemany(
                """
                delete from document_relations
                where source_document_id = ? or target_document_id = ?
                """,
                [(document_id, document_id) for document_id in document_ids],
            )
        else:
            conn.executemany(
                "delete from document_relations where source_document_id = ?",
                [(document_id,) for document_id in document_ids],
            )
        conn.executemany(
            "delete from document_kb_state where document_id = ?",
            [(document_id,) for document_id in document_ids],
        )

    def _upsert_document_kb_state(self, conn: sqlite3.Connection,
                                  docs: list[dict[str, Any]],
                                  signatures: dict[str, str],
                                  *,
                                  include_soft: bool) -> None:
        refreshed_at = time.time()
        conn.executemany(
            """
            insert or replace into document_kb_state
            (document_id, signature, include_soft, refreshed_at)
            values (?, ?, ?, ?)
            """,
            [
                (
                    doc["document_id"],
                    signatures[doc["document_id"]],
                    1 if include_soft else 0,
                    refreshed_at,
                )
                for doc in docs
                if doc["document_id"] in signatures
            ],
        )

    def _document_relation_stats(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        return [
            dict(row) for row in conn.execute(
                """
                select relation_type, method, count(*) as count,
                       round(avg(confidence), 3) as avg_confidence
                from document_relations
                group by relation_type, method
                order by relation_type, method
                """
            )
        ]

    def _build_document_relations(self, conn: sqlite3.Connection,
                                  docs: list[dict[str, Any]],
                                  doc_terms: dict[str, dict[str, float]],
                                  doc_term_index: dict[str, set[str]],
                                  title_keys: dict[str, str],
                                  *,
                                  include_soft: bool,
                                  focus_ids: set[str] | None = None,
                                  ) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        relations: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        title_index = build_title_index(docs)
        doc_number_index = build_doc_number_index(docs)
        quote_to_docs: dict[str, list[dict[str, Any]]] = {}
        series_to_docs: dict[str, list[dict[str, Any]]] = {}
        focus_ids = set(focus_ids or [])

        for doc in docs:
            for quote in doc["quoted_terms"]:
                quote_key = normalize_relation_text(quote)
                if len(quote_key) >= 6:
                    quote_to_docs.setdefault(quote_key, []).append(doc)
            series_key = normalize_series_title(doc["title"])
            if len(series_key) >= 8:
                series_to_docs.setdefault(series_key, []).append(doc)

        docs_to_scan = docs if not focus_ids else [
            doc for doc in docs if doc["document_id"] in focus_ids
        ]

        for doc in docs_to_scan:
            relation_role = classify_relation_role(doc["title"])
            for quote in doc["quoted_terms"]:
                for target in match_title_targets(quote, title_index):
                    if target["document_id"] == doc["document_id"]:
                        continue
                    target_role = classify_relation_role(target["title"])
                    if relation_role and target_role == relation_role:
                        continue
                    relation_type = relation_type_for_role(relation_role)
                    if not relation_role:
                        relation_type = classify_enhanced_relation_type(
                            doc,
                            quote,
                            default=relation_type,
                        )
                    add_relation(
                        relations,
                        doc,
                        target,
                        relation_type,
                        confidence_for_enhanced_relation(relation_type, relation_role),
                        f"标题或正文引用《{quote}》",
                        source_chunk_id=doc.get("first_chunk_id", ""),
                        method="rule",
                    )

            for doc_number in doc.get("cited_doc_numbers", []):
                for target in doc_number_index.get(doc_number, []):
                    if target["document_id"] == doc["document_id"]:
                        continue
                    relation_type = classify_enhanced_relation_type(
                        doc,
                        doc_number,
                        default="cites_by_doc_no",
                    )
                    add_relation(
                        relations,
                        doc,
                        target,
                        relation_type,
                        confidence_for_enhanced_relation(relation_type, ""),
                        f"标题或正文引用文号 {doc_number}",
                        source_chunk_id=doc.get("first_chunk_id", ""),
                        target_chunk_id=target.get("first_chunk_id", ""),
                        method="doc_number",
                        metadata={"doc_number": doc_number},
                    )

            if relation_role in {"feedback", "request", "approval", "attachment"}:
                for target in match_profile_targets(
                    doc,
                    docs,
                    doc_terms,
                    doc_term_index=doc_term_index,
                    title_keys=title_keys,
                    limit=4,
                ):
                    if target["document_id"] == doc["document_id"]:
                        continue
                    target_role = classify_relation_role(target["title"])
                    if target_role == relation_role:
                        continue
                    add_relation(
                        relations,
                        doc,
                        target,
                        relation_type_for_role(relation_role),
                        confidence_for_role(relation_role) - 0.08,
                        f"文种/标题显示为{relation_role_label(relation_role)}，且核心词重合",
                        source_chunk_id=doc.get("first_chunk_id", ""),
                        method="lexical_profile",
                    )

        for quote_key, members in quote_to_docs.items():
            if 2 <= len(members) <= 10 and self._relation_group_touches_focus(members, focus_ids):
                for source, target in directed_pairs(members):
                    add_relation(
                        relations,
                        source,
                        target,
                        "shared_reference",
                        min(0.84, 0.62 + len(quote_key) / 80),
                        f"共同引用《{pick_original_quote(source, quote_key)}》",
                        source_chunk_id=source.get("first_chunk_id", ""),
                        method="rule",
                    )

        for _series_key, members in series_to_docs.items():
            if 2 <= len(members) <= 8 and self._relation_group_touches_focus(members, focus_ids):
                for source, target in directed_pairs(members):
                    relation_type = "same_matter" if exact_relation_title_key(source["title"]) == exact_relation_title_key(target["title"]) else "same_series"
                    add_relation(
                        relations,
                        source,
                        target,
                        relation_type,
                        0.78 if relation_type == "same_matter" else 0.68,
                        "标题规范化后属于同一事项或同一系列",
                        source_chunk_id=source.get("first_chunk_id", ""),
                        method="rule",
                    )

        if include_soft:
            self._add_soft_term_relations(relations, docs, doc_terms, focus_ids=focus_ids or None)
            self._add_soft_embedding_relations(conn, relations, docs, focus_ids=focus_ids or None)

        return relations

    def _relation_group_touches_focus(self, docs: list[dict[str, Any]],
                                      focus_ids: set[str]) -> bool:
        if not focus_ids:
            return True
        return any(doc["document_id"] in focus_ids for doc in docs)

    def _add_soft_term_relations(self, relations: dict[tuple[str, str, str, str], dict[str, Any]],
                                 docs: list[dict[str, Any]],
                                 doc_terms: dict[str, dict[str, float]],
                                 focus_ids: set[str] | None = None) -> None:
        by_id = {doc["document_id"]: doc for doc in docs}
        sources = docs if not focus_ids else [
            doc for doc in docs if doc["document_id"] in focus_ids
        ]
        for source in sources:
            scored = []
            source_terms = doc_terms.get(source["document_id"], {})
            if not source_terms:
                continue
            for target in docs:
                if target["document_id"] == source["document_id"]:
                    continue
                target_terms = doc_terms.get(target["document_id"], {})
                score, overlap = weighted_term_overlap(source_terms, target_terms)
                if score >= 0.22 and overlap:
                    scored.append((score, overlap, target["document_id"]))
            for score, overlap, target_id in sorted(scored, reverse=True)[:4]:
                target = by_id[target_id]
                add_relation(
                    relations,
                    source,
                    target,
                    "topically_related",
                    min(0.76, 0.5 + score),
                    "核心短语重合：" + "、".join(overlap[:6]),
                    source_chunk_id=source.get("first_chunk_id", ""),
                    method="lexical_profile",
                    metadata={"overlap_terms": overlap[:8], "overlap_score": round(score, 4)},
                )

    def _add_soft_embedding_relations(self, conn: sqlite3.Connection,
                                      relations: dict[tuple[str, str, str, str], dict[str, Any]],
                                      docs: list[dict[str, Any]],
                                      focus_ids: set[str] | None = None) -> None:
        if not self.embedding_config.enabled:
            return
        rows = conn.execute(
            """
            select c.document_id, e.embedding, e.dim
            from chunk_embeddings e
            join chunks c on c.chunk_id = e.chunk_id
            where e.model = ?
            """,
            (self.embedding_config.model,),
        ).fetchall()
        if not rows:
            return
        doc_vectors: dict[str, list[list[float]]] = {}
        for row in rows:
            doc_vectors.setdefault(row["document_id"], []).append(
                unpack_vector(row["embedding"], row["dim"])
            )
        averaged = {
            doc_id: normalize_vector([
                sum(vector[i] for vector in vectors) / len(vectors)
                for i in range(len(vectors[0]))
            ])
            for doc_id, vectors in doc_vectors.items()
            if vectors
        }
        by_id = {doc["document_id"]: doc for doc in docs}
        threshold = max(0.64, self.embedding_config.min_similarity + 0.04)
        source_ids = set(averaged) if not focus_ids else set(focus_ids) & set(averaged)
        for source_id in source_ids:
            source_vector = averaged[source_id]
            source = by_id.get(source_id)
            if not source:
                continue
            scored = []
            for target_id, target_vector in averaged.items():
                if target_id == source_id or len(target_vector) != len(source_vector):
                    continue
                similarity = dot_product(source_vector, target_vector)
                if similarity >= threshold:
                    scored.append((similarity, target_id))
            for similarity, target_id in sorted(scored, reverse=True)[:3]:
                target = by_id.get(target_id)
                if not target:
                    continue
                add_relation(
                    relations,
                    source,
                    target,
                    "semantically_related",
                    min(0.82, float(similarity)),
                    "文档级 embedding 相似",
                    source_chunk_id=source.get("first_chunk_id", ""),
                    method="embedding",
                    metadata={"semantic_similarity": round(float(similarity), 4), "model": self.embedding_config.model},
                )

    def _resolve_relation_document(self, conn: sqlite3.Connection, *, path: str = "",
                                   document_id: str = "",
                                   title_like: str = "") -> sqlite3.Row | None:
        if document_id:
            return conn.execute(
                "select document_id, title, path, category, agency, doc_type, date_text, char_count from documents where document_id = ?",
                (document_id,),
            ).fetchone()
        if path:
            return conn.execute(
                """
                select document_id, title, path, category, agency, doc_type, date_text, char_count
                from documents where path = ? or path like ? order by length(path) limit 1
                """,
                (path, f"%{path}%"),
            ).fetchone()
        if title_like:
            return conn.execute(
                """
                select document_id, title, path, category, agency, doc_type, date_text, char_count
                from documents where title like ? or path like ? order by length(title), title limit 1
                """,
                (f"%{title_like}%", f"%{title_like}%"),
            ).fetchone()
        return None

    def _upsert_document_metadata(self, conn: sqlite3.Connection,
                                  docs: list[dict[str, Any]]) -> None:
        for doc in docs:
            metadata_items = infer_document_metadata(doc)
            for item in metadata_items:
                conn.execute(
                    """
                    insert or replace into document_metadata
                    (document_id, key, value, confidence, evidence, method)
                    values (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc["document_id"],
                        item["key"],
                        item["value"],
                        item["confidence"],
                        item.get("evidence", ""),
                        item["method"],
                    ),
                )

    def _load_document_metadata(self, conn: sqlite3.Connection,
                                document_ids: list[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
        if not document_ids:
            return {}
        placeholders = ",".join("?" for _ in document_ids)
        rows = conn.execute(
            f"""
            select document_id, key, value, confidence, evidence, method
            from document_metadata
            where document_id in ({placeholders})
            order by document_id, key, confidence desc
            """,
            document_ids,
        ).fetchall()
        by_doc: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for row in rows:
            by_key = by_doc.setdefault(row["document_id"], {})
            by_key.setdefault(row["key"], []).append({
                "value": row["value"],
                "confidence": float(row["confidence"]),
                "evidence": row["evidence"],
                "method": row["method"],
            })
        return by_doc

    def _rank_context_documents(self, conn: sqlite3.Connection,
                                hits: list[dict[str, Any]],
                                query: str) -> list[dict[str, Any]]:
        by_doc: dict[str, dict[str, Any]] = {}
        for hit in hits:
            doc_id = hit.get("document_id", "")
            if not doc_id:
                continue
            current = by_doc.get(doc_id)
            hit_score = float(hit.get("related_score", hit.get("score", 0.0)) or 0.0)
            title_bonus = 8.0 if any(token in str(hit.get("title", "")) for token in tokenize_terms(query, 8)) else 0.0
            semantic_bonus = float(hit.get("semantic_score", 0.0) or 0.0) * 8.0
            score = hit_score + title_bonus + semantic_bonus
            if not current:
                by_doc[doc_id] = {
                    "document_id": doc_id,
                    "title": hit.get("title", ""),
                    "path": hit.get("path", ""),
                    "category": hit.get("category", ""),
                    "agency": hit.get("agency", ""),
                    "doc_type": hit.get("doc_type", ""),
                    "context_score": score,
                    "selection_reason": "检索命中文档片段",
                    "search_hit_count": 1,
                    "best_hit": context_best_hit(hit),
                    "_best_score": score,
                }
            else:
                current["context_score"] += score * 0.25
                current["search_hit_count"] += 1
                if score > float(current.get("_best_score", 0.0)):
                    current["best_hit"] = context_best_hit(hit)
                    current["_best_score"] = score

        if not by_doc:
            return []
        rows = conn.execute(
            f"""
            select document_id, title, path, category, agency, doc_type, date_text, char_count
            from documents
            where document_id in ({','.join('?' for _ in by_doc)})
            """,
            list(by_doc),
        ).fetchall()
        for row in rows:
            item = by_doc[row["document_id"]]
            item.update(dict(row))
            item.pop("_best_score", None)
        return list(by_doc.values())

    def _relation_context_suggestions(self, conn: sqlite3.Connection,
                                      document_ids: list[str],
                                      query: str,
                                      limit: int) -> list[dict[str, Any]]:
        if not document_ids:
            return []
        placeholders = ",".join("?" for _ in document_ids)
        rows = conn.execute(
            f"""
            select r.*, sd.title as source_title, sd.path as source_path,
                   sd.category as source_category, td.title as target_title,
                   td.path as target_path, td.category as target_category,
                   sd.agency as source_agency, sd.doc_type as source_doc_type,
                   sd.char_count as source_char_count,
                   td.agency as target_agency, td.doc_type as target_doc_type,
                   td.char_count as target_char_count
            from document_relations r
            join documents sd on sd.document_id = r.source_document_id
            join documents td on td.document_id = r.target_document_id
            where r.source_document_id in ({placeholders})
               or r.target_document_id in ({placeholders})
            order by r.confidence desc
            limit ?
            """,
            [*document_ids, *document_ids, max(limit * 4, 12)],
        ).fetchall()
        seen = set(document_ids)
        suggestions = []
        query_terms = set(tokenize_terms(query, 20))
        for row in rows:
            source_in = row["source_document_id"] in seen
            other_prefix = "target" if source_in else "source"
            other_doc_id = row[f"{other_prefix}_document_id"]
            if other_doc_id in seen:
                continue
            title = row[f"{other_prefix}_title"] or ""
            evidence = row["evidence"] or ""
            relevance = 0.0
            for term in query_terms:
                if term and (term in title or term in evidence):
                    relevance += 1.0
            if row["relation_type"] in {"semantically_related", "topically_related"}:
                relevance += 0.5
            if relevance <= 0 and float(row["confidence"]) < 0.8:
                continue
            suggestions.append({
                "relation_id": row["relation_id"],
                "relation_type": row["relation_type"],
                "confidence": round(float(row["confidence"]), 3),
                "evidence": evidence,
                "method": row["method"],
                "document": {
                    "document_id": other_doc_id,
                    "title": title,
                    "path": row[f"{other_prefix}_path"],
                    "category": row[f"{other_prefix}_category"],
                    "agency": row[f"{other_prefix}_agency"],
                    "doc_type": row[f"{other_prefix}_doc_type"],
                    "char_count": row[f"{other_prefix}_char_count"],
                },
            })
            seen.add(other_doc_id)
            if len(suggestions) >= limit:
                break
        return suggestions

    def _parse_document(self, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel_path = path.relative_to(self.paths.repo_root).as_posix()
        rel_to_corpus = path.relative_to(self.paths.corpus_root)
        parts = rel_to_corpus.parts
        category = "/".join(parts[:-3]) if len(parts) >= 3 and parts[-2] == "auto" else "/".join(parts[:-1])
        heading_titles = [m.group(2).strip() for line in text.splitlines() if (m := HEADING_RE.match(line))]
        title = choose_title(path.stem, heading_titles)
        agency = infer_agency(parts, text)
        doc_type = infer_doc_type(parts, title)
        date_text = infer_date_text(rel_path, text)
        document_id = stable_id(rel_path)
        doc = {
            "document_id": document_id,
            "path": rel_path,
            "title": title,
            "category": category,
            "agency": agency,
            "doc_type": doc_type,
            "date_text": date_text,
            "file_mtime": path.stat().st_mtime,
            "char_count": len(text),
        }
        chunks = split_markdown_chunks(text)
        parsed_chunks = []
        for ordinal, chunk in enumerate(chunks):
            chunk_id = f"{document_id}-{ordinal:04d}"
            parsed_chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "ordinal": ordinal,
                "heading": chunk["heading"],
                "content": chunk["content"],
                "image_refs": ",".join(chunk["image_refs"]),
                "char_start": chunk["char_start"],
            })
        if not parsed_chunks:
            parsed_chunks.append({
                "chunk_id": f"{document_id}-0000",
                "document_id": document_id,
                "ordinal": 0,
                "heading": title,
                "content": text[:2000],
                "image_refs": ",".join(IMAGE_RE.findall(text)),
                "char_start": 0,
            })
        return doc, parsed_chunks

    def _markdown_files(self) -> list[Path]:
        if not self.paths.corpus_root.exists():
            return []
        files = []
        for path in sorted(self.paths.corpus_root.glob("**/*.md")):
            rel = path.relative_to(self.paths.corpus_root)
            if rel.parts and rel.parts[0] in DEFAULT_EXCLUDED_TOP_LEVEL_DIRS:
                continue
            files.append(path)
        return files

    def _find_seed_document(self, *, seed_path: str = "",
                            seed_title: str = "") -> dict[str, Any] | None:
        if not self.paths.corpus_root.exists():
            return None
        candidates = []
        for path in sorted(self.paths.corpus_root.glob("**/*.md")):
            rel_to_corpus = path.relative_to(self.paths.corpus_root)
            if not rel_to_corpus.parts or rel_to_corpus.parts[0] not in DEFAULT_EXCLUDED_TOP_LEVEL_DIRS:
                continue
            rel_path = path.relative_to(self.paths.repo_root).as_posix()
            title = path.stem
            candidates.append({
                "path": path,
                "relative_path": rel_path,
                "title": title,
                "top_level_dir": rel_to_corpus.parts[0],
            })

        clean_path = (seed_path or "").strip()
        clean_title = (seed_title or "").strip()
        if clean_path:
            for candidate in candidates:
                if candidate["relative_path"] == clean_path or clean_path in candidate["relative_path"]:
                    return candidate
        if clean_title:
            for candidate in candidates:
                if clean_title in candidate["title"] or clean_title in candidate["relative_path"]:
                    return candidate
        return candidates[0] if len(candidates) == 1 and not clean_path and not clean_title else None

    def _build_seed_profile(self, title: str, text: str,
                            query_hint: str = "") -> list[dict[str, Any]]:
        weights = collect_seed_term_weights(title, text, query_hint=query_hint)
        return self._build_term_profile(weights)

    def _build_query_profile(self, query: str) -> list[dict[str, Any]]:
        weights = collect_query_term_weights(query)
        return self._build_term_profile(weights)

    def _build_term_profile(self, weights: dict[str, float]) -> list[dict[str, Any]]:
        terms = list(weights.keys())
        if not terms:
            return []
        with self._connect() as conn:
            idfs = self._load_idfs(conn, terms)
            total_docs = conn.execute("select count(*) from documents").fetchone()[0]
        default_idf = compute_idf(total_docs, 0)
        profile = []
        for term, structure_weight in weights.items():
            idf = idfs.get(term, default_idf)
            kind = classify_term(term)
            score = structure_weight * idf * length_bonus(term) * kind_score_multiplier(kind)
            profile.append({
                "term": term,
                "kind": kind,
                "structure_weight": round(structure_weight, 3),
                "idf": round(idf, 3),
                "score": round(score, 3),
            })
        return sorted(profile, key=lambda item: (-item["score"], -len(item["term"]), item["term"]))

    def _load_idfs(self, conn: sqlite3.Connection, terms: list[str]) -> dict[str, float]:
        if not terms:
            return {}
        placeholders = ",".join("?" for _ in terms)
        rows = conn.execute(
            f"select term, idf from term_stats where term in ({placeholders})",
            terms,
        ).fetchall()
        return {row["term"]: float(row["idf"]) for row in rows}

    def _recall_chunks(self, query: str, filters: list[str],
                       params: list[Any], limit: int,
                       semantic: bool = False) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        clean_query = (query or "").strip()
        limit = max(1, min(int(limit or 8), 240))
        where_sql = " and ".join(filters) if filters else "1=1"

        with self._connect() as conn:
            if clean_query:
                like_sql = f"""
                    select c.*, d.title, d.path, d.category, d.agency, d.doc_type
                    from chunks c join documents d using (document_id)
                    where ({where_sql})
                      and (c.content like ? or c.heading like ? or d.title like ? or d.path like ?)
                    limit ?
                """
                like_value = f"%{clean_query}%"
                for row in conn.execute(
                    like_sql,
                    [*params, like_value, like_value, like_value, like_value, limit * 4],
                ):
                    self._add_candidate(candidates, dict(row), 20.0, clean_query)

                match = build_fts_query(clean_query)
                if match:
                    fts_sql = f"""
                        select c.*, d.title, d.path, d.category, d.agency, d.doc_type,
                               bm25(chunks_fts) as rank
                        from chunks_fts
                        join chunks c on c.chunk_id = chunks_fts.chunk_id
                        join documents d on d.document_id = c.document_id
                        where chunks_fts match ?
                          {'and ' + where_sql if filters else ''}
                        order by rank
                        limit ?
                    """
                    for row in conn.execute(fts_sql, [match, *params, limit * 4]):
                        score = 12.0 + max(0.0, 8.0 - abs(float(row["rank"])))
                        self._add_candidate(candidates, dict(row), score, clean_query)

                for token in tokenize_query(clean_query):
                    token_like = f"%{token}%"
                    token_sql = f"""
                        select c.*, d.title, d.path, d.category, d.agency, d.doc_type
                        from chunks c join documents d using (document_id)
                        where ({where_sql})
                          and (c.content like ? or d.title like ?)
                        limit ?
                    """
                    for row in conn.execute(token_sql, [*params, token_like, token_like, limit * 2]):
                        self._add_candidate(candidates, dict(row), 4.0, clean_query)

                if semantic:
                    for row in self._semantic_recall_chunks(clean_query, filters, params, limit):
                        self._add_semantic_candidate(candidates, row, clean_query)
            else:
                sql = f"""
                    select c.*, d.title, d.path, d.category, d.agency, d.doc_type
                    from chunks c join documents d using (document_id)
                    where {where_sql}
                    order by d.path, c.ordinal
                    limit ?
                """
                for row in conn.execute(sql, [*params, limit]):
                    self._add_candidate(candidates, dict(row), 1.0, clean_query)

        return sorted(candidates.values(), key=lambda r: (-r["score"], r["path"], r["ordinal"]))

    def _semantic_recall_chunks(self, query: str, filters: list[str],
                                params: list[Any], limit: int) -> list[dict[str, Any]]:
        config = self.embedding_config
        if not query or not config.enabled or config.provider not in {"openai", "local"}:
            return []
        try:
            with self._connect() as conn:
                embedded_count = conn.execute(
                    "select count(*) from chunk_embeddings where model = ?",
                    (config.model,),
                ).fetchone()[0]
                if embedded_count == 0:
                    return []
                where_sql = " and ".join(filters) if filters else "1=1"
                rows = conn.execute(
                    f"""
                    select c.*, d.title, d.path, d.category, d.agency, d.doc_type,
                           e.embedding, e.dim
                    from chunk_embeddings e
                    join chunks c on c.chunk_id = e.chunk_id
                    join documents d on d.document_id = c.document_id
                    where e.model = ? and {where_sql}
                    """,
                    [config.model, *params],
                ).fetchall()
        except sqlite3.DatabaseError:
            return []

        if not rows:
            return []

        try:
            query_vector = self._embed_query(query)
        except Exception:
            return []

        scored = []
        for row in rows:
            vector = unpack_vector(row["embedding"], row["dim"])
            if len(vector) != len(query_vector):
                continue
            similarity = dot_product(query_vector, vector)
            if similarity < config.min_similarity:
                continue
            item = dict(row)
            item.pop("embedding", None)
            item["semantic_score"] = round(float(similarity), 4)
            scored.append(item)

        return sorted(
            scored,
            key=lambda item: (-float(item.get("semantic_score", 0.0)), item.get("path", ""), item.get("ordinal", 0)),
        )[:max(1, min(limit, 240))]

    def _embed_query(self, query: str) -> list[float]:
        return self._embed_texts([query])[0]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        config = self.embedding_config
        if config.provider == "openai":
            client = OpenAI(api_key=config.api_key, base_url=config.api_url)
            response = client.embeddings.create(model=config.model, input=texts)
            return [normalize_vector(item.embedding) for item in response.data]
        if config.provider == "local":
            model = self._local_embedding_model()
            vectors = model.encode(
                texts,
                batch_size=config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [[float(value) for value in row] for row in vectors.tolist()]
        raise ValueError(f"Unsupported embedding provider: {config.provider}")

    def _local_embedding_model(self):
        if not hasattr(self, "_sentence_transformer"):
            path = self._local_embedding_model_path()
            if not path:
                raise RuntimeError(
                    "未找到本地 embedding 模型缓存。请设置 DOCUMENT_QA_EMBEDDING_LOCAL_PATH，"
                    "或先从 hf-mirror/modelscope 下载模型。"
                )
            self._sentence_transformer = SentenceTransformer(
                path,
                device=os.getenv("DOCUMENT_QA_EMBEDDING_DEVICE", "cpu"),
                local_files_only=True,
            )
        return self._sentence_transformer

    def _local_embedding_model_path(self) -> str:
        config = self.embedding_config
        if config.local_model_path:
            path = Path(config.local_model_path).expanduser()
            return str(path) if path.exists() else ""
        cached = cached_hf_snapshot(config.model)
        return str(cached) if cached else ""

    def _rerank_profiled_hits(self, recall_hits: list[dict[str, Any]],
                              profile: list[dict[str, Any]],
                              limit: int,
                              debug: bool = False) -> list[dict[str, Any]]:
        hits = self._rerank_related_hits(recall_hits, profile, limit=limit, debug=debug)
        return self._strip_internal_fields(hits)

    def _rerank_related_hits(self, hits: list[dict[str, Any]],
                             seed_profile: list[dict[str, Any]],
                             limit: int,
                             debug: bool = False) -> list[dict[str, Any]]:
        if not hits:
            return []
        seed_terms = seed_profile[:20]
        by_doc: dict[str, dict[str, Any]] = {}
        for hit in hits:
            doc_id = hit.get("document_id", "")
            if not doc_id:
                continue
            searchable_title = str(hit.get("title", "") or "")
            searchable_heading = str(hit.get("heading", "") or "")
            searchable_snippet = str(hit.get("snippet", "") or "")
            searchable_content = str(hit.get("content", "") or "")
            matched = []
            phrase_score = 0.0
            matched_importance = 0.0
            anchor_importance = 0.0
            contributions = []
            for item in seed_terms:
                term = item["term"]
                kind = item.get("kind", classify_term(term))
                importance = math.sqrt(max(float(item.get("score", 1.0)), 1.0))
                kind_multiplier = kind_match_multiplier(kind)
                location = ""
                location_multiplier = 0.0
                if term in searchable_title:
                    location = "title"
                    location_multiplier = 2.0
                elif term in searchable_heading:
                    location = "heading"
                    location_multiplier = 1.2
                elif term in searchable_snippet:
                    location = "snippet"
                    location_multiplier = 0.6
                elif term in searchable_content:
                    location = "content"
                    location_multiplier = 0.35
                if location:
                    contribution = importance * kind_multiplier * location_multiplier
                    phrase_score += contribution
                    matched_importance += importance * kind_multiplier
                    if is_anchor_kind(kind):
                        anchor_importance += importance * kind_multiplier
                    matched.append(term)
                    if debug:
                        contributions.append({
                            "term": term,
                            "kind": kind,
                            "location": location,
                            "importance": round(importance, 3),
                            "kind_multiplier": kind_multiplier,
                            "location_multiplier": location_multiplier,
                            "contribution": round(contribution, 3),
                        })

            semantic_score = max(0.0, float(hit.get("semantic_score", 0.0) or 0.0))
            if semantic_score:
                semantic_component = semantic_score * 6.0
                semantic_importance = semantic_score * 15.0
                phrase_score += semantic_component
                matched_importance += semantic_importance
                anchor_importance += semantic_score * 6.0
                if not matched:
                    matched.append("semantic_similarity")
                if debug:
                    contributions.append({
                        "term": "semantic_similarity",
                        "kind": "semantic",
                        "location": "embedding",
                        "importance": round(semantic_importance, 3),
                        "kind_multiplier": 1.0,
                        "location_multiplier": 1.0,
                        "contribution": round(semantic_component, 3),
                    })

            if not matched:
                continue

            base_component = rerank_base_component(float(hit.get("score", 0.0)))
            related_score = base_component + phrase_score
            doc = by_doc.get(doc_id)
            if not doc:
                row = dict(hit)
                row["related_score"] = round(related_score, 3)
                row["matched_seed_terms"] = sorted(set(matched), key=lambda t: (-len(t), t))
                row["_matched_importance"] = matched_importance
                row["_anchor_importance"] = anchor_importance
                row["_best_chunk_score"] = related_score
                row["matched_chunk_count"] = 1
                if debug:
                    row["scoring_debug"] = {
                        "base_search_score": hit.get("score", 0.0),
                        "base_component": round(base_component, 3),
                        "phrase_component": round(phrase_score, 3),
                        "matched_importance": round(matched_importance, 3),
                        "anchor_importance": round(anchor_importance, 3),
                        "contributions": contributions,
                    }
                by_doc[doc_id] = row
                continue

            doc["related_score"] += related_score * 0.25
            doc["_matched_importance"] += matched_importance * 0.5
            doc["_anchor_importance"] += anchor_importance * 0.5
            doc["matched_chunk_count"] += 1
            doc["matched_seed_terms"] = sorted(
                set(doc.get("matched_seed_terms", [])) | set(matched),
                key=lambda t: (-len(t), t),
            )
            if related_score > float(doc.get("_best_chunk_score", doc.get("related_score", 0.0))):
                for key in ("chunk_id", "heading", "image_refs", "snippet", "source", "score"):
                    doc[key] = hit.get(key)
                doc["_best_chunk_score"] = related_score
                if debug:
                    doc["scoring_debug"] = {
                        "base_search_score": hit.get("score", 0.0),
                        "base_component": round(base_component, 3),
                        "phrase_component": round(phrase_score, 3),
                        "matched_importance": round(matched_importance, 3),
                        "anchor_importance": round(anchor_importance, 3),
                        "contributions": contributions,
                    }

        filtered = [
            row for row in by_doc.values()
            if float(row.get("_matched_importance", 0.0)) >= 8.0
            and float(row.get("_anchor_importance", 0.0)) >= 3.0
        ]
        ranked = sorted(
            filtered,
            key=lambda row: (-float(row.get("related_score", 0.0)), row.get("path", "")),
        )[:max(1, min(limit, 30))]
        for row in ranked:
            row["related_score"] = round(float(row.get("related_score", 0.0)), 3)
            row.pop("_best_chunk_score", None)
            row.pop("_matched_importance", None)
            row.pop("_anchor_importance", None)
        return ranked

    def _strip_internal_fields(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned = []
        for hit in hits:
            row = dict(hit)
            row.pop("content", None)
            row.pop("ordinal", None)
            row.pop("char_start", None)
            cleaned.append(row)
        return cleaned

    def _add_candidate(self, candidates: dict[str, dict[str, Any]],
                       row: dict[str, Any], score: float, query: str) -> None:
        chunk_id = row["chunk_id"]
        searchable = " ".join(
            str(row.get(key, "") or "")
            for key in ("title", "path", "category", "heading", "content")
        )
        tokens = tokenize_query(query)
        matched_tokens = [token for token in tokens if token in searchable]
        exact_match = bool(query and query in searchable)
        if query and len(tokens) >= 2 and not exact_match and len(matched_tokens) < min(2, len(tokens)):
            return
        score += len(matched_tokens) * 3
        if exact_match:
            score += 20
        existing = candidates.get(chunk_id)
        if existing:
            existing["score"] += score
            return
        content = row.get("content", "") or ""
        row["score"] = round(score, 3)
        row["snippet"] = make_snippet(content, query)
        row["source"] = {
            "title": row.get("title", ""),
            "path": row.get("path", ""),
            "heading": row.get("heading", ""),
            "chunk_id": chunk_id,
        }
        candidates[chunk_id] = row

    def _add_semantic_candidate(self, candidates: dict[str, dict[str, Any]],
                                row: dict[str, Any], query: str) -> None:
        chunk_id = row["chunk_id"]
        semantic_score = max(0.0, float(row.get("semantic_score", 0.0) or 0.0))
        score = round(semantic_score * 18.0, 3)
        existing = candidates.get(chunk_id)
        if existing:
            existing["score"] += score
            existing["semantic_score"] = max(
                float(existing.get("semantic_score", 0.0) or 0.0),
                semantic_score,
            )
            existing["recall_sources"] = sorted(set(existing.get("recall_sources", ["lexical"])) | {"semantic"})
            return

        content = row.get("content", "") or ""
        row["score"] = score
        row["snippet"] = make_snippet(content, query)
        row["source"] = {
            "title": row.get("title", ""),
            "path": row.get("path", ""),
            "heading": row.get("heading", ""),
            "chunk_id": chunk_id,
        }
        row["recall_sources"] = ["semantic"]
        candidates[chunk_id] = row

    def _sql_filters(self, *, category: str = "", agency: str = "",
                     doc_type: str = "") -> tuple[list[str], list[Any]]:
        filters = []
        params: list[Any] = []
        for field, value in (("category", category), ("agency", agency), ("doc_type", doc_type)):
            if value:
                filters.append(f"d.{field} like ?")
                params.append(f"%{value}%")
        return filters, params

    def _get_doc(self, conn: sqlite3.Connection, document_id: str) -> sqlite3.Row | None:
        return conn.execute("select * from documents where document_id = ?", (document_id,)).fetchone()

    def _format_read_result(self, doc: sqlite3.Row | None, rows: list[sqlite3.Row],
                            max_chars: int, focus_chunk_id: str = "") -> dict[str, Any]:
        if not doc:
            return {"error": "未找到文档"}
        doc_dict = dict(doc)
        blocks = []
        used = 0
        included = []
        for row in rows:
            label = f"### {row['heading']}" if row["heading"] else f"### chunk {row['ordinal']}"
            block = f"{label}\n{row['content'].strip()}"
            if used + len(block) > max_chars:
                remain = max_chars - used
                if remain > 200:
                    block = block[:remain] + "\n...[truncated]"
                    blocks.append(block)
                    included.append(row["chunk_id"])
                break
            blocks.append(block)
            included.append(row["chunk_id"])
            used += len(block)
        content = "\n\n".join(blocks)
        with self._connect() as conn:
            self._init_schema(conn)
            metadata = self._load_document_metadata(conn, [doc["document_id"]]).get(doc["document_id"], {})
        temporal = document_temporal_metadata({
            **doc_dict,
            "metadata": metadata,
            "read_content": content,
        })
        quality = document_quality(doc_dict, {
            "content": content,
            "truncated": len(content) >= max_chars,
        })
        return {
            "document": {
                "document_id": doc["document_id"],
                "title": doc["title"],
                "path": doc["path"],
                "category": doc["category"],
                "agency": doc["agency"],
                "doc_type": doc["doc_type"],
                "char_count": doc["char_count"],
            },
            "temporal": temporal,
            "quality": quality,
            "evidence_warning": "read_document 只说明本篇文档证据；多文档综合的时间覆盖范围应以 prepare_answer_context.temporal_coverage 为准。",
            "focus_chunk_id": focus_chunk_id,
            "included_chunk_ids": included,
            "content": content,
            "truncated": len(content) >= max_chars,
        }


class DocumentResolver:
    def __init__(self, index: DocumentIndex):
        self.index = index

    def query(self, object_type: str, filters: dict[str, Any] | None = None,
              limit: int | None = None, order_by: str | None = None,
              offset: int | None = None) -> list[dict[str, Any]]:
        return self.index.query_rows(object_type, filters, limit, order_by, offset)

    def count(self, object_type: str, filters: dict[str, Any] | None = None) -> int:
        return self.index.count(object_type, filters)

    def query_by_id(self, object_type: str, id_value: Any) -> dict[str, Any] | None:
        return self.index.query_by_id(object_type, id_value)

    def search_text(self, keyword: str, object_types: list[str] | None = None,
                    limit: int = 20) -> list[dict[str, Any]]:
        return self.index.search_text(keyword, limit)


def split_markdown_chunks(text: str, max_chars: int = 1400) -> list[dict[str, Any]]:
    chunks = []
    heading = ""
    buffer: list[str] = []
    start = 0
    cursor = 0

    def flush() -> None:
        nonlocal buffer, start
        content = "\n".join(buffer).strip()
        if content:
            chunks.append({
                "heading": heading,
                "content": content,
                "image_refs": IMAGE_RE.findall(content),
                "char_start": start,
            })
        buffer = []
        start = cursor

    for line in text.splitlines():
        line_len = len(line) + 1
        match = HEADING_RE.match(line)
        if match and buffer:
            flush()
        if match:
            heading = match.group(2).strip()
        if not buffer:
            start = cursor
        buffer.append(line)
        if sum(len(item) + 1 for item in buffer) >= max_chars and line.strip() == "":
            flush()
        cursor += line_len
    flush()

    expanded = []
    for chunk in chunks:
        content = chunk["content"]
        if len(content) <= max_chars * 1.4:
            expanded.append(chunk)
            continue
        paragraphs = re.split(r"\n\s*\n", content)
        buf: list[str] = []
        for para in paragraphs:
            if sum(len(item) + 2 for item in buf) + len(para) > max_chars and buf:
                expanded.append({**chunk, "content": "\n\n".join(buf)})
                buf = []
            buf.append(para)
        if buf:
            expanded.append({**chunk, "content": "\n\n".join(buf)})
    return expanded


def choose_title(stem: str, headings: list[str]) -> str:
    for h in headings:
        if any(word in h for word in DOC_TYPE_WORDS) and len(h) > 4:
            return h
    return headings[0] if headings else stem


def infer_agency(parts: tuple[str, ...], text: str) -> str:
    if len(parts) >= 2 and parts[0] == "通知抬头文件":
        return parts[1]
    first_heading = next((m.group(2).strip() for line in text.splitlines() if (m := HEADING_RE.match(line))), "")
    return first_heading.removesuffix("文件").strip()


def infer_doc_type(parts: tuple[str, ...], title: str) -> str:
    for part in parts:
        if part in DOC_TYPE_WORDS:
            return part
    for word in DOC_TYPE_WORDS:
        if word in title:
            return word
    return ""


def infer_date_text(path: str, text: str) -> str:
    for source in (path, text[:500]):
        match = DATE_RE.search(source)
        if match:
            return match.group(1)
    return ""


def infer_document_metadata(doc: dict[str, Any]) -> list[dict[str, Any]]:
    structural_text = "\n".join(
        str(doc.get(key, "") or "")
        for key in ("title", "date_text", "path", "headings")
    )
    content_text = str(doc.get("content_sample", "") or "")
    items: list[dict[str, Any]] = []
    for candidate in extract_date_candidates(structural_text):
        if candidate["kind"] == "year":
            items.append({
                "key": "year",
                "value": candidate["value"],
                "confidence": candidate["confidence"],
                "evidence": candidate["evidence"],
                "method": candidate["method"],
            })
        else:
            items.append({
                "key": "date",
                "value": candidate["value"],
                "confidence": candidate["confidence"],
                "evidence": candidate["evidence"],
                "method": candidate["method"],
            })
            items.append({
                "key": "year",
                "value": candidate["value"][:4],
                "confidence": max(0.0, candidate["confidence"] - 0.05),
                "evidence": candidate["evidence"],
                "method": candidate["method"],
            })

    structural_years = {item["value"] for item in items if item["key"] == "year"}
    for candidate in extract_date_candidates(content_text):
        confidence = max(0.0, candidate["confidence"] - 0.28)
        method = f"content_{candidate['method']}"
        if candidate["kind"] == "year":
            if candidate["value"] in structural_years:
                continue
            items.append({
                "key": "year",
                "value": candidate["value"],
                "confidence": confidence,
                "evidence": candidate["evidence"],
                "method": method,
            })
        else:
            year = candidate["value"][:4]
            if year not in structural_years:
                items.append({
                    "key": "date",
                    "value": candidate["value"],
                    "confidence": confidence,
                    "evidence": candidate["evidence"],
                    "method": method,
                })
                items.append({
                    "key": "year",
                    "value": year,
                    "confidence": max(0.0, confidence - 0.05),
                    "evidence": candidate["evidence"],
                    "method": method,
                })

    for doc_number in extract_doc_numbers(structural_text):
        items.append({
            "key": "doc_number",
            "value": doc_number,
            "confidence": 0.88,
            "evidence": doc_number,
            "method": "structural_doc_number",
        })

    char_count = int(doc.get("char_count", 0) or 0)
    if char_count < 80:
        quality, confidence = "empty", 0.95
    elif char_count < 500:
        quality, confidence = "low", 0.8
    else:
        quality, confidence = "usable", 0.7
    items.append({
        "key": "extraction_quality",
        "value": quality,
        "confidence": confidence,
        "evidence": f"char_count={char_count}",
        "method": "char_count",
    })
    return dedupe_metadata_items(items)


def extract_date_candidates(text: str) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    patterns = [
        (r"(20\d{2})年(\d{1,2})月(\d{1,2})日?", 0.92, "full_date"),
        (r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})", 0.9, "full_date"),
        (r"(20\d{2})年(\d{1,2})月", 0.78, "year_month"),
        (r"(20\d{2})[./-](\d{1,2})(?![./-]\d)", 0.72, "year_month"),
        (r"[〔\[（(](20\d{2})[〕\]）)]", 0.84, "doc_number_year"),
        (r"(20\d{2})\s*年", 0.76, "plain_year"),
    ]
    for pattern, confidence, method in patterns:
        for match in re.finditer(pattern, text or ""):
            if method == "full_date":
                month = int(match.group(2))
                day = int(match.group(3))
                if not (1 <= month <= 12 and 1 <= day <= 31):
                    continue
                value = f"{match.group(1)}-{month:02d}-{day:02d}"
                kind = "date"
            elif method == "year_month":
                month = int(match.group(2))
                if not (1 <= month <= 12):
                    continue
                value = f"{match.group(1)}-{month:02d}"
                kind = "date"
            else:
                value = match.group(1)
                kind = "year"
            key = (kind, value, method)
            if key in seen:
                continue
            seen.add(key)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            candidates.append({
                "kind": kind,
                "value": value,
                "confidence": confidence,
                "evidence": re.sub(r"\s+", " ", text[start:end]).strip(),
                "method": method,
            })
    return sorted(candidates, key=lambda item: (-item["confidence"], item["value"]))[:12]


def dedupe_metadata_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in items:
        key = (item["key"], item["value"], item["method"])
        existing = result.get(key)
        if not existing or item["confidence"] > existing["confidence"]:
            result[key] = item
    return list(result.values())


def context_best_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": hit.get("chunk_id", ""),
        "heading": hit.get("heading", ""),
        "snippet": hit.get("snippet", ""),
        "score": hit.get("score", 0),
        "related_score": hit.get("related_score", 0),
        "semantic_score": hit.get("semantic_score", 0),
        "recall_sources": hit.get("recall_sources", []),
    }


def document_quality(item: dict[str, Any], read_result: dict[str, Any]) -> dict[str, Any]:
    char_count = int(item.get("char_count", 0) or 0)
    content = read_result.get("content", "") or ""
    if char_count < 80 or len(content.strip()) < 80:
        status = "empty"
        note = "正文极短，可能是 MinerU 抽取失败或原文为空。"
    elif char_count < 500 or len(content.strip()) < 500:
        status = "low"
        note = "正文较短，只适合作为弱证据或线索。"
    elif read_result.get("truncated"):
        status = "partial"
        note = "已读取核心上下文，但返回内容被截断。"
    else:
        status = "usable"
        note = "可作为回答证据。"
    return {
        "status": status,
        "char_count": char_count,
        "read_chars": len(content),
        "note": note,
    }


def document_temporal_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = doc.get("metadata", {}) or {}
    years = list(metadata.get("year", []))
    dates = list(metadata.get("date", []))
    if not years and not dates:
        inferred = infer_document_metadata({
            "title": doc.get("title", ""),
            "path": doc.get("path", ""),
            "date_text": doc.get("date_text", ""),
            "content_sample": str(doc.get("read_content", ""))[:1200],
            "char_count": doc.get("char_count", 0),
        })
        for item in inferred:
            if item["key"] == "year":
                years.append({
                    "value": item["value"],
                    "confidence": item["confidence"],
                    "evidence": item.get("evidence", ""),
                    "method": item["method"],
                })
            elif item["key"] == "date":
                dates.append({
                    "value": item["value"],
                    "confidence": item["confidence"],
                    "evidence": item.get("evidence", ""),
                    "method": item["method"],
                })
    return {
        "best_year": best_metadata_value(years),
        "best_date": best_metadata_value(dates),
        "years": years[:5],
        "dates": dates[:5],
    }


def best_metadata_value(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    return str(sorted(items, key=lambda item: (-float(item.get("confidence", 0)), item.get("value", "")))[0]["value"])


def build_temporal_coverage(query: str,
                            evidence_documents: list[dict[str, Any]]) -> dict[str, Any]:
    expected_years = expected_years_from_query(query)
    found_years = sorted({
        doc.get("temporal", {}).get("best_year", "")
        for doc in evidence_documents
        if doc.get("temporal", {}).get("best_year")
    })
    missing_years = [year for year in expected_years if year not in found_years]
    return {
        "expected_years": expected_years,
        "found_years": found_years,
        "missing_years": missing_years,
        "coverage_note": temporal_coverage_note(expected_years, found_years, missing_years),
    }


def expected_years_from_query(query: str) -> list[str]:
    query_years = sorted(set(re.findall(r"20\d{2}", query or "")))
    if len(query_years) >= 2:
        start, end = int(query_years[0]), int(query_years[-1])
        if 1900 <= start <= end <= 2100 and end - start <= 30:
            return [str(year) for year in range(start, end + 1)]
    if len(query_years) == 1:
        return query_years
    return []


def temporal_coverage_note(expected: list[str], found: list[str], missing: list[str]) -> str:
    if not expected:
        return "用户问题未指定明确年份范围，按证据文档自身时间属性组织即可。"
    if not found:
        return "未能从候选文档中可靠识别年份，回答时需说明时间证据不足。"
    if missing:
        return f"证据只覆盖 {', '.join(found)}；缺少 {', '.join(missing)} 的可靠文档。"
    return f"证据覆盖用户指定年份：{', '.join(found)}。"


def build_strict_answer_constraints(temporal_coverage: dict[str, Any]) -> dict[str, Any]:
    found = temporal_coverage.get("found_years", []) or []
    missing = temporal_coverage.get("missing_years", []) or []
    constraints = {
        "must_state_temporal_coverage": bool(temporal_coverage.get("expected_years")),
        "evidence_years_only": found,
        "missing_years_are_not_covered": missing,
        "do_not_claim_coverage_for": missing,
    }
    if found:
        constraints["required_limitation_sentence"] = (
            f"当前证据只覆盖 {', '.join(found)}；"
            f"缺少 {', '.join(missing)} 的可靠文档。"
            if missing
            else f"当前证据覆盖 {', '.join(found)}。"
        )
    elif temporal_coverage.get("expected_years"):
        constraints["required_limitation_sentence"] = "当前证据未能可靠识别年份，不能按指定年份范围作完整汇总。"
    else:
        constraints["required_limitation_sentence"] = ""
    return constraints


def build_evidence_outline(evidence_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outline = []
    for index, doc in enumerate(evidence_documents, 1):
        document = doc.get("document", {})
        temporal = doc.get("temporal", {})
        quality = doc.get("quality", {})
        outline.append({
            "rank": index,
            "document_id": document.get("document_id", ""),
            "title": document.get("title", ""),
            "path": document.get("path", ""),
            "best_year": temporal.get("best_year", ""),
            "quality_status": quality.get("status", ""),
            "included_chunk_ids": doc.get("included_chunk_ids", [])[:3],
            "selection_reason": doc.get("selection", {}).get("reason", ""),
        })
    return outline


def build_synthesis_outline(query: str,
                            evidence_documents: list[dict[str, Any]]) -> dict[str, Any]:
    intents = infer_synthesis_intents(query)
    dimension_map: dict[str, dict[str, Any]] = {}

    for index, doc in enumerate(evidence_documents, 1):
        document = doc.get("document", {})
        if doc.get("quality", {}).get("status") in {"empty", "low"}:
            continue
        doc_ref = {
            "rank": index,
            "document_id": document.get("document_id", ""),
            "title": document.get("title", ""),
            "path": document.get("path", ""),
            "best_year": doc.get("temporal", {}).get("best_year", ""),
            "included_chunk_ids": doc.get("included_chunk_ids", [])[:3],
        }
        for candidate in synthesis_dimension_candidates(query, doc):
            item = dimension_map.setdefault(candidate["dimension"], {
                "dimension": candidate["dimension"],
                "reason_signals": set(),
                "supporting_documents": {},
                "evidence_terms": {},
                "score": 0.0,
            })
            item["score"] += candidate["score"]
            item["reason_signals"].update(candidate["signals"])
            item["supporting_documents"][doc_ref["document_id"]] = doc_ref
            for term in candidate["terms"]:
                item["evidence_terms"][term] = item["evidence_terms"].get(term, 0) + 1

    dimensions = []
    for item in dimension_map.values():
        supporting = list(item["supporting_documents"].values())
        support_count = len(supporting)
        terms = sorted(
            item["evidence_terms"],
            key=lambda term: (-item["evidence_terms"][term], -len(term), term),
        )[:8]
        score = float(item["score"]) + support_count * 2.5
        if support_count >= 2:
            score += 3.0
        dimensions.append({
            "dimension": item["dimension"],
            "reason": synthesis_reason(item["reason_signals"], support_count),
            "support_count": support_count,
            "evidence_terms": terms,
            "supporting_documents": supporting[:5],
            "_score": round(score, 3),
        })

    ranked = sorted(
        dimensions,
        key=lambda item: (-item["_score"], -item["support_count"], item["dimension"]),
    )
    for item in ranked:
        item.pop("_score", None)
    recommended_sections = [
        {
            "section": item["dimension"],
            "support_count": item["support_count"],
            "evidence_terms": item["evidence_terms"][:5],
        }
        for item in ranked[:6]
    ]

    return {
        "intent": intents,
        "recommended_answer_sections": recommended_sections,
        "generation_basis": [
            "用户问题中的通用意图词",
            "证据文档标题和小节结构",
            "多篇文档共同出现的结构性短语",
            "文档抽取质量和时间覆盖边界",
        ],
        "dimensions": ranked[:6],
        "usage": "回答时优先使用 recommended_answer_sections 作为一级归纳框架，再在各节中合并 documents 证据；dimension 只是通用文档结构建议，不是业务专用模板。",
    }


def infer_synthesis_intents(query: str) -> list[str]:
    found = []
    for intent, hints in SYNTHESIS_INTENT_HINTS.items():
        if any(hint in query for hint in hints):
            found.append(intent)
    if not found and any(word in query for word in ("汇总", "总结", "梳理", "归纳")):
        found.append("summary")
    return found or ["answer_question"]


def synthesis_dimension_candidates(query: str, doc: dict[str, Any]) -> list[dict[str, Any]]:
    document = doc.get("document", {})
    content = doc.get("content", "") or ""
    chunks = split_read_content_blocks(content)
    text_scope = "\n".join([
        document.get("title", ""),
        "\n".join(block["heading"] for block in chunks[:10]),
        content[:1800],
    ])
    candidates: list[dict[str, Any]] = []

    for dimension, hints in SYNTHESIS_STRUCTURE_PATTERNS:
        matched = [hint for hint in hints if hint and hint in text_scope]
        if not matched:
            continue
        intent_bonus = 1.5 if any(hint in query for hint in matched) else 0.0
        heading_bonus = 1.0 if any(any(hint in block["heading"] for hint in matched) for block in chunks) else 0.0
        candidates.append({
            "dimension": dimension,
            "signals": matched[:4],
            "terms": matched[:4],
            "score": len(matched) + intent_bonus + heading_bonus,
        })

    for phrase, count in repeated_document_phrases(document.get("title", ""), chunks).items():
        if count < 2:
            continue
        candidates.append({
            "dimension": phrase,
            "signals": ["标题/小节重复短语"],
            "terms": [phrase],
            "score": min(4.0, 1.0 + count * 0.8),
        })

    if not candidates:
        fallback_terms = tokenize_terms(document.get("title", ""), limit=3)
        if fallback_terms:
            candidates.append({
                "dimension": "核心事项",
                "signals": ["标题主题"],
                "terms": fallback_terms,
                "score": 1.0,
            })
    return candidates


def split_read_content_blocks(content: str) -> list[dict[str, str]]:
    blocks = []
    current_heading = ""
    current_lines: list[str] = []
    for line in (content or "").splitlines():
        if line.startswith("### "):
            if current_heading or current_lines:
                blocks.append({
                    "heading": current_heading,
                    "text": "\n".join(current_lines).strip(),
                })
            current_heading = line[4:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_heading or current_lines:
        blocks.append({
            "heading": current_heading,
            "text": "\n".join(current_lines).strip(),
        })
    return blocks


def repeated_document_phrases(title: str,
                              chunks: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    sources = [title, *(chunk["heading"] for chunk in chunks[:12])]
    for source in sources:
        for term in tokenize_terms(source, limit=12):
            if classify_term(term) in {"form", "short", "numeric_or_code"}:
                continue
            if len(term) < 3 or len(term) > 12:
                continue
            counts[term] = counts.get(term, 0) + 1
    return counts


def synthesis_reason(signals: set[str], support_count: int) -> str:
    signal_text = "、".join(sorted(signals)[:5]) or "证据文档结构"
    if support_count >= 2:
        return f"由 {support_count} 篇证据文档共同支持；结构信号：{signal_text}"
    return f"由单篇核心证据支持；结构信号：{signal_text}"


def build_quality_notes(evidence_documents: list[dict[str, Any]]) -> list[str]:
    notes = []
    weak = [
        doc["document"]["title"]
        for doc in evidence_documents
        if doc.get("quality", {}).get("status") in {"empty", "low"}
    ]
    if weak:
        notes.append("以下文档抽取质量较低，只能作为线索：" + "；".join(weak[:5]))
    truncated = [
        doc["document"]["title"]
        for doc in evidence_documents
        if doc.get("truncated")
    ]
    if truncated:
        notes.append("以下文档读取内容被截断，如需精确引用应继续 read_document：" + "；".join(truncated[:5]))
    return notes


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def document_kb_signature(doc: dict[str, Any]) -> str:
    payload = {
        "kb_extractor_version": DOCUMENT_KB_EXTRACTOR_VERSION,
        "title": doc.get("title", ""),
        "category": doc.get("category", ""),
        "agency": doc.get("agency", ""),
        "doc_type": doc.get("doc_type", ""),
        "date_text": doc.get("date_text", ""),
        "char_count": doc.get("char_count", 0),
        "headings": doc.get("headings", ""),
        "content_sample": doc.get("content_sample", ""),
        "quoted_terms": doc.get("quoted_terms", []),
        "doc_numbers": doc.get("doc_numbers", []),
        "cited_doc_numbers": doc.get("cited_doc_numbers", []),
        "relation_text": doc.get("relation_text", ""),
    }
    return stable_id(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def embedding_text(row: dict[str, Any], max_chars: int) -> str:
    header = "\n".join(
        part for part in [
            f"标题：{row.get('title', '')}",
            f"类别：{row.get('category', '')}",
            f"机关：{row.get('agency', '')}",
            f"类型：{row.get('doc_type', '')}",
            f"小节：{row.get('heading', '')}",
        ]
        if part and not part.endswith("：")
    )
    content = re.sub(r"\s+", " ", str(row.get("content", "") or "")).strip()
    return f"{header}\n正文：{content}"[:max_chars]


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(item) / norm for item in vector]


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(data: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}f", data))


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def batched(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def tokenize_query(text: str) -> list[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    if jieba:
        tokens = [tok.strip() for tok in jieba.lcut(text) if tok.strip()]
    else:
        tokens = text.split()
    result = []
    for token in tokens:
        if len(token) >= 2 or token.isdigit():
            result.append(token)
    return list(dict.fromkeys(result))[:12]


def tokenize_terms(text: str, limit: int | None = None) -> list[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    if jieba:
        raw_tokens = [tok.strip() for tok in jieba.lcut(text) if tok.strip()]
    else:
        raw_tokens = text.split()
    result = []
    seen = set()
    for token in raw_tokens:
        if not is_useful_term(token) or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if limit and len(result) >= limit:
            break
    return result


def is_useful_term(term: str) -> bool:
    if not term or term in TERM_STOPWORDS:
        return False
    if term.isdigit() and len(term) < 4:
        return False
    if len(term) < 2:
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", term))


def collect_document_term_weights(doc: dict[str, Any],
                                  chunks: list[dict[str, Any]]) -> dict[str, float]:
    weights: dict[str, float] = {}

    add_structured_terms(weights, doc.get("title", ""), source_weight=8.0)
    add_quoted_terms(weights, doc.get("title", ""), source_weight=12.0)
    add_structured_terms(weights, doc.get("agency", ""), source_weight=3.0)
    add_structured_terms(weights, doc.get("category", ""), source_weight=2.0)

    seen_headings = set()
    body_parts = []
    for chunk in chunks:
        heading = chunk.get("heading", "")
        if heading and heading not in seen_headings:
            add_structured_terms(weights, heading, source_weight=4.0)
            add_quoted_terms(weights, heading, source_weight=6.0)
            seen_headings.add(heading)
        body_parts.append(chunk.get("content", ""))
    add_frequency_terms(weights, "\n".join(body_parts), source_weight=0.5, limit=80)
    return weights


def collect_seed_term_weights(title: str, text: str,
                              query_hint: str = "") -> dict[str, float]:
    weights: dict[str, float] = {}
    add_structured_terms(weights, title, source_weight=10.0)
    add_quoted_terms(weights, title, source_weight=18.0)
    add_structured_terms(weights, query_hint, source_weight=12.0)

    heading_text = "\n".join(
        m.group(2).strip()
        for line in text.splitlines()
        if (m := HEADING_RE.match(line))
    )
    add_structured_terms(weights, heading_text, source_weight=5.0)
    add_quoted_terms(weights, heading_text, source_weight=8.0)
    add_frequency_terms(weights, text[:5000], source_weight=0.8, limit=80)
    return weights


def collect_query_term_weights(query: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    tokens = tokenize_terms(query, limit=40)
    for token in tokens:
        weights[token] = weights.get(token, 0.0) + 10.0 * length_bonus(token) * term_quality(token)
    for phrase in query_phrases(tokens, limit=12):
        weights[phrase] = weights.get(phrase, 0.0) + 9.0 * length_bonus(phrase) * term_quality(phrase)
    add_quoted_terms(weights, query, source_weight=14.0)
    return weights


def query_phrases(tokens: list[str], limit: int) -> list[str]:
    phrases = []
    seen = set()
    useful_tokens = [token for token in tokens if classify_term(token) != "form"]
    for size in (2, 3):
        for idx in range(0, max(0, len(useful_tokens) - size + 1)):
            phrase = "".join(useful_tokens[idx:idx + size])
            if phrase in seen or not is_useful_term(phrase) or len(phrase) > 16:
                continue
            seen.add(phrase)
            phrases.append(phrase)
            if len(phrases) >= limit:
                return phrases
    return phrases


def add_structured_terms(weights: dict[str, float], text: str,
                         source_weight: float) -> None:
    if not text:
        return
    clean = re.sub(r"\s+", "", text)
    if (
        is_useful_term(clean)
        and len(clean) <= 24
        and not clean.startswith("关于")
        and "《" not in clean
    ):
        weights[clean] = weights.get(clean, 0.0) + source_weight * length_bonus(clean) * term_quality(clean)
    for token in tokenize_terms(text, limit=60):
        weights[token] = weights.get(token, 0.0) + source_weight * length_bonus(token) * term_quality(token)


def add_quoted_terms(weights: dict[str, float], text: str,
                     source_weight: float) -> None:
    for term in re.findall(r"[《“\"]([^》”\"]{2,80})[》”\"]", text or ""):
        clean = re.sub(r"\s+", "", term)
        if is_useful_term(clean) and len(clean) <= 60:
            weights[clean] = weights.get(clean, 0.0) + source_weight * length_bonus(clean) * term_quality(clean)
        for token in tokenize_terms(term, limit=20):
            weights[token] = weights.get(token, 0.0) + (source_weight * 0.5) * length_bonus(token) * term_quality(token)


def add_frequency_terms(weights: dict[str, float], text: str,
                        source_weight: float, limit: int) -> None:
    counts: dict[str, int] = {}
    for token in tokenize_terms(text):
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(
        counts.items(),
        key=lambda item: (-(1 + math.log(item[1])) * length_bonus(item[0]), item[0]),
    )[:limit]
    for term, count in ranked:
        weights[term] = weights.get(term, 0.0) + source_weight * (1 + math.log(count)) * length_bonus(term) * term_quality(term)


def compute_idf(total_docs: int, doc_freq: int) -> float:
    return math.log((total_docs + 1) / (doc_freq + 1)) + 1.0


def length_bonus(term: str) -> float:
    return min(2.5, 1.0 + len(term) / 12)


def term_quality(term: str) -> float:
    if re.fullmatch(r"[一二三四五六七八九十0-9]+年", term):
        return 0.2
    if len(term) <= 2:
        return 0.45
    if len(term) <= 4 and any(term.endswith(suffix) for suffix in GENERIC_TERM_SUFFIXES):
        return 0.35
    return 1.0


def classify_term(term: str) -> str:
    if re.fullmatch(r"[0-9A-Za-z.+-]+", term):
        return "numeric_or_code"
    if re.fullmatch(r"[一二三四五六七八九十0-9]+年", term):
        return "form"
    if len(term) <= 2:
        return "short"
    if len(term) <= 6 and any(term.endswith(suffix) for suffix in GENERIC_TERM_SUFFIXES):
        return "form"
    if len(term) >= 10:
        return "quoted_theme"
    return "topic"


def kind_score_multiplier(kind: str) -> float:
    return {
        "quoted_theme": 1.0,
        "topic": 1.0,
        "numeric_or_code": 0.75,
        "short": 0.55,
        "form": 0.35,
    }.get(kind, 1.0)


def kind_match_multiplier(kind: str) -> float:
    return {
        "quoted_theme": 1.2,
        "topic": 1.0,
        "numeric_or_code": 0.8,
        "short": 0.55,
        "form": 0.25,
    }.get(kind, 1.0)


def is_anchor_kind(kind: str) -> bool:
    return kind in {"quoted_theme", "topic", "numeric_or_code"}


def rerank_base_component(score: float) -> float:
    return min(8.0, math.log1p(max(score, 0.0)) * 1.5)


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def top_weighted_terms(weights: dict[str, float],
                       limit: int) -> list[tuple[str, float]]:
    return sorted(
        weights.items(),
        key=lambda item: (-item[1] * length_bonus(item[0]), -len(item[0]), item[0]),
    )[:limit]


def extract_seed_terms(title: str, text: str, *, query_hint: str = "",
                       limit: int = 16) -> list[str]:
    stopwords = {
        "关于", "情况", "工作", "汇报", "报告", "通知", "文件", "批办", "有关",
        "进行", "开展", "推进", "建设", "年度", "落实", "印发", "一个",
    }
    candidates: dict[str, float] = {}

    def add_terms(source: str, weight: float) -> None:
        for token in tokenize_query(source):
            if token in stopwords or token.isdigit():
                continue
            if len(token) < 2:
                continue
            candidates[token] = candidates.get(token, 0.0) + weight + min(len(token), 8) * 0.2

    add_terms(title, 8.0)
    add_terms(query_hint, 10.0)
    add_terms(text[:3000], 1.0)

    quoted_terms = re.findall(r"[《“\"]([^》”\"]{2,40})[》”\"]", title)
    for term in quoted_terms:
        clean = re.sub(r"\s+", "", term)
        if clean:
            candidates[clean] = candidates.get(clean, 0.0) + 20.0
            add_terms(clean, 4.0)

    sorted_terms = sorted(candidates.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    result = []
    for term, _score in sorted_terms:
        result.append(term)
        if len(result) >= limit:
            break
    return result


def extract_quoted_terms(text: str) -> list[str]:
    terms = []
    seen = set()
    for term in re.findall(r"[《“\"]([^》”\"]{2,100})[》”\"]", text or ""):
        clean = re.sub(r"\s+", "", term)
        if not clean or clean in seen:
            continue
        if clean.endswith(("（征求意见稿）", "(征求意见稿)")):
            clean = re.sub(r"[（(]征求意见稿[）)]$", "", clean)
        if is_useful_term(clean):
            seen.add(clean)
            terms.append(clean)
    return terms[:30]


def extract_doc_numbers(text: str) -> list[str]:
    numbers = []
    seen = set()
    patterns = [
        r"[\u4e00-\u9fffA-Za-z]{1,12}[〔\[]20\d{2}[〕\]][\u4e00-\u9fffA-Za-z]{0,8}\d{1,5}号?",
        r"[\u4e00-\u9fffA-Za-z]{1,12}\(20\d{2}\)[\u4e00-\u9fffA-Za-z]{0,8}\d{1,5}号?",
        r"[\u4e00-\u9fffA-Za-z]{1,12}发〔20\d{2}〕\d{1,5}号?",
        r"[\u4e00-\u9fffA-Za-z]{1,12}函〔20\d{2}〕\d{1,5}号?",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text or ""):
            normalized = normalize_doc_number(match.group(0))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            numbers.append(normalized)
            if len(numbers) >= 20:
                return numbers
    return numbers


def normalize_doc_number(text: str) -> str:
    clean = re.sub(r"\s+", "", text or "")
    clean = clean.replace("［", "[").replace("］", "]")
    clean = clean.replace("[", "〔").replace("]", "〕")
    clean = clean.replace("（", "(").replace("）", ")")
    clean = re.sub(r"\((20\d{2})\)", r"〔\1〕", clean)
    clean = re.sub(r"号$", "", clean)
    clean = re.sub(r"^[和及与、，,；;：:。关于依据根据按照遵照落实执行修订废止替代取代转发印发]+", "", clean)
    if not re.search(r"〔20\d{2}〕", clean):
        return ""
    if not re.search(r"\d+$", clean):
        return ""
    if len(clean) < 7 or len(clean) > 40:
        return ""
    return clean


def normalize_relation_text(text: str) -> str:
    clean = re.sub(r"\s+", "", text or "")
    clean = re.sub(r"[\(\)（）\[\]【】〔〕“”\"'、，。；;:：!！?？]", "", clean)
    clean = re.sub(r"^(关于|转发|印发|下发|征求|反馈)", "", clean)
    clean = re.sub(r"(的通知|的请示|的函|的报告|的意见|的批复|的公告|的方案|的计划|征求意见稿)$", "", clean)
    clean = re.sub(r"(20\d{2}|二〇二[一二三四五六七八九十]|202[0-9])年度?", "", clean)
    clean = re.sub(r"\d+$", "", clean)
    return clean


def exact_relation_title_key(title: str) -> str:
    return normalize_relation_text(title).replace("副本", "").replace("附件", "")


def normalize_series_title(title: str) -> str:
    clean = exact_relation_title_key(title)
    clean = re.sub(r"(第[一二三四五六七八九十0-9]+届|第[一二三四五六七八九十0-9]+次)", "", clean)
    clean = re.sub(r"[一二三四五六七八九十0-9]+月", "", clean)
    clean = re.sub(r"[一二三四五六七八九十0-9]+日", "", clean)
    clean = re.sub(r"(一|二|三|四|五|六|七|八|九|十|[0-9]+)$", "", clean)
    return clean


def build_title_index(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = []
    for doc in docs:
        title = str(doc.get("title", "") or "")
        path_title = Path(str(doc.get("path", "") or "")).stem
        keys = {
            normalize_relation_text(title),
            exact_relation_title_key(title),
            normalize_relation_text(path_title),
            exact_relation_title_key(path_title),
        }
        index.append({**doc, "_title_keys": {key for key in keys if len(key) >= 4}})
    return index


def build_doc_number_index(docs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        numbers = list(dict.fromkeys(doc.get("doc_numbers", [])))
        for number in numbers[:6]:
            index.setdefault(number, []).append(doc)
    return {
        number: members
        for number, members in index.items()
        if 1 <= len(members) <= 5
    }


def match_title_targets(quote: str, title_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quote_key = normalize_relation_text(quote)
    if len(quote_key) < 8:
        return []
    matches = []
    for doc in title_index:
        keys = doc.get("_title_keys", set())
        title_key = exact_relation_title_key(str(doc.get("title", "") or ""))
        if quote_key in keys or any(quote_key in key or key in quote_key for key in keys if len(key) >= 8):
            matches.append(doc)
        elif len(quote_key) >= 10 and quote_key in title_key:
            matches.append(doc)
    return matches[:12]


def classify_relation_role(title: str) -> str:
    clean = title or ""
    if re.search(r"反馈意见|反馈.*意见|回复|回函|答复", clean):
        return "feedback"
    if re.search(r"批复|批办单|批办|批示|审批", clean):
        return "approval"
    if re.search(r"请示|申请", clean):
        return "request"
    if re.search(r"附件|附表|附录", clean):
        return "attachment"
    return ""


def relation_type_for_role(role: str) -> str:
    return {
        "feedback": "feedback_to",
        "approval": "approval_for",
        "request": "request_for",
        "attachment": "attachment_of",
    }.get(role, "cites")


def confidence_for_role(role: str) -> float:
    return {
        "feedback": 0.88,
        "approval": 0.86,
        "request": 0.82,
        "attachment": 0.78,
    }.get(role, 0.74)


def relation_role_label(role: str) -> str:
    return {
        "feedback": "反馈/回复",
        "approval": "批办/批复",
        "request": "请示/申请",
        "attachment": "附件",
    }.get(role, "引用")


def classify_enhanced_relation_type(source: dict[str, Any],
                                    evidence_text: str = "",
                                    *,
                                    default: str = "cites") -> str:
    text = "\n".join([
        str(source.get("title", "") or ""),
        relation_evidence_context(str(source.get("relation_text", "") or ""), evidence_text),
    ])
    if re.search(r"废止|停止执行|宣布失效|予以废止|不再执行", text):
        return "abolishes"
    if re.search(r"修订|修改|修正|调整|补充|更新", text):
        return "revises"
    if re.search(r"替代|取代|代替|替换|原.*同时废止", text):
        return "replaces"
    if re.search(r"征求意见|征询意见|征求.*建议|公开征求", text):
        return "solicits_opinion_on"
    if re.search(r"依据|根据|按照|遵照|依照", text):
        return "based_on"
    if re.search(r"贯彻|落实|实施|执行", text):
        return "implements"
    return default


def relation_evidence_context(text: str, needle: str, window: int = 180) -> str:
    if not needle:
        return text[:window]
    normalized_needle = normalize_doc_number(needle) or needle
    candidates = [needle, normalized_needle]
    for candidate in candidates:
        if not candidate:
            continue
        pos = text.find(candidate)
        if pos >= 0:
            start = max(0, pos - window)
            end = min(len(text), pos + len(candidate) + window)
            return text[start:end]
    compact_text = re.sub(r"\s+", "", text)
    compact_needle = re.sub(r"\s+", "", normalized_needle)
    pos = compact_text.find(compact_needle)
    if pos >= 0:
        start = max(0, pos - window)
        end = min(len(compact_text), pos + len(compact_needle) + window)
        return compact_text[start:end]
    return text[:window]


def confidence_for_enhanced_relation(relation_type: str, role: str) -> float:
    if role:
        return confidence_for_role(role)
    return {
        "based_on": 0.86,
        "abolishes": 0.88,
        "revises": 0.84,
        "replaces": 0.86,
        "solicits_opinion_on": 0.82,
        "cites_by_doc_no": 0.82,
        "implements": 0.84,
    }.get(relation_type, 0.74)


def match_profile_targets(source: dict[str, Any], docs: list[dict[str, Any]],
                          doc_terms: dict[str, dict[str, float]],
                          *,
                          doc_term_index: dict[str, set[str]] | None = None,
                          title_keys: dict[str, str] | None = None,
                          limit: int = 4) -> list[dict[str, Any]]:
    source_id = source["document_id"]
    source_terms = doc_terms.get(source_id, {})
    if not source_terms:
        return []
    by_id = {doc["document_id"]: doc for doc in docs}
    title_keys = title_keys or {}
    source_title_key = title_keys.get(source_id) or normalize_series_title(source.get("title", ""))
    candidate_ids: set[str] = set()
    if doc_term_index:
        useful_terms = [
            term for term in sorted(
                source_terms,
                key=lambda term: (-source_terms[term], -len(term), term),
            )
            if len(term) >= 3 and classify_term(term) not in {"form", "short"}
        ][:12]
        for term in useful_terms:
            candidate_ids.update(doc_term_index.get(term, set()))
    if source_title_key:
        for doc in docs:
            target_title_key = title_keys.get(doc["document_id"], "")
            if target_title_key and (
                source_title_key in target_title_key or target_title_key in source_title_key
            ):
                candidate_ids.add(doc["document_id"])
    if not candidate_ids:
        return []

    scored = []
    for target_id in candidate_ids:
        if target_id == source_id:
            continue
        target = by_id.get(target_id)
        if not target:
            continue
        score, overlap = weighted_term_overlap(source_terms, doc_terms.get(target["document_id"], {}))
        target_title_key = title_keys.get(target["document_id"], "")
        if source_title_key and target_title_key and (
            source_title_key in target_title_key or target_title_key in source_title_key
        ):
            score += 0.2
        if score >= 0.18 and overlap:
            scored.append((score, target))
    return [target for _score, target in sorted(scored, key=lambda item: (-item[0], item[1]["title"]))[:limit]]


def weighted_term_overlap(left: dict[str, float], right: dict[str, float]) -> tuple[float, list[str]]:
    if not left or not right:
        return 0.0, []
    common = set(left) & set(right)
    useful_common = [
        term for term in common
        if classify_term(term) not in {"form", "short"} and len(term) >= 3
    ]
    if not useful_common:
        return 0.0, []
    common_weight = sum(math.sqrt(left[term] * right[term]) for term in useful_common)
    left_weight = sum(left.values()) or 1.0
    right_weight = sum(right.values()) or 1.0
    score = common_weight / math.sqrt(left_weight * right_weight)
    overlap = sorted(useful_common, key=lambda term: (-(left[term] + right[term]), -len(term), term))
    return score, overlap


def directed_pairs(items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = []
    for source in items:
        for target in items:
            if source["document_id"] != target["document_id"]:
                pairs.append((source, target))
    return pairs


def pick_original_quote(doc: dict[str, Any], quote_key: str) -> str:
    for quote in doc.get("quoted_terms", []):
        if normalize_relation_text(quote) == quote_key:
            return quote
    return quote_key


def add_relation(relations: dict[tuple[str, str, str, str], dict[str, Any]],
                 source: dict[str, Any], target: dict[str, Any], relation_type: str,
                 confidence: float, evidence: str, *, source_chunk_id: str = "",
                 target_chunk_id: str = "", method: str = "rule",
                 metadata: dict[str, Any] | None = None) -> None:
    if source["document_id"] == target["document_id"]:
        return
    confidence = round(max(0.0, min(float(confidence), 1.0)), 3)
    key = (source["document_id"], target["document_id"], relation_type, method)
    relation_id = stable_id("|".join(key))
    candidate = {
        "relation_id": relation_id,
        "source_document_id": source["document_id"],
        "target_document_id": target["document_id"],
        "relation_type": relation_type,
        "confidence": confidence,
        "evidence": evidence[:500],
        "source_chunk_id": source_chunk_id,
        "target_chunk_id": target_chunk_id,
        "method": method,
        "metadata": metadata or {},
    }
    existing = relations.get(key)
    if not existing or confidence > float(existing["confidence"]):
        relations[key] = candidate


def format_relation_row(row: sqlite3.Row, focus_document_id: str = "") -> dict[str, Any]:
    is_outgoing = row["source_document_id"] == focus_document_id
    other_prefix = "target" if is_outgoing else "source"
    metadata = {}
    if row["metadata"]:
        try:
            metadata = json.loads(row["metadata"])
        except json.JSONDecodeError:
            metadata = {}
    return {
        "relation_id": row["relation_id"],
        "relation_type": row["relation_type"],
        "direction": "outgoing" if is_outgoing else "incoming",
        "confidence": round(float(row["confidence"]), 3),
        "evidence": row["evidence"],
        "method": row["method"],
        "source_document": {
            "document_id": row["source_document_id"],
            "title": row["source_title"],
            "path": row["source_path"],
            "category": row["source_category"],
        },
        "target_document": {
            "document_id": row["target_document_id"],
            "title": row["target_title"],
            "path": row["target_path"],
            "category": row["target_category"],
        },
        "other_document": {
            "document_id": row[f"{other_prefix}_document_id"],
            "title": row[f"{other_prefix}_title"],
            "path": row[f"{other_prefix}_path"],
            "category": row[f"{other_prefix}_category"],
        },
        "source_chunk_id": row["source_chunk_id"],
        "target_chunk_id": row["target_chunk_id"],
        "metadata": metadata,
    }


def tokenize_for_search(text: str) -> str:
    tokens = tokenize_query(text)
    if tokens:
        return " ".join(tokens)
    return " ".join(text[i:i + 2] for i in range(max(0, len(text) - 1)))


def build_fts_query(text: str) -> str:
    tokens = [escape_fts_token(token) for token in tokenize_query(text)]
    tokens = [token for token in tokens if token]
    return " OR ".join(tokens[:8])


def escape_fts_token(token: str) -> str:
    token = re.sub(r'["\']', "", token)
    token = re.sub(r"[^\w\u4e00-\u9fff]+", "", token)
    return f'"{token}"' if token else ""


def make_snippet(content: str, query: str, width: int = 260) -> str:
    clean = re.sub(r"\s+", " ", content).strip()
    if not clean:
        return ""
    positions = [clean.find(query)] if query else [-1]
    positions.extend(clean.find(token) for token in tokenize_query(query))
    positions = [pos for pos in positions if pos >= 0]
    start = max(0, (min(positions) if positions else 0) - 60)
    snippet = clean[start:start + width]
    if start:
        snippet = "..." + snippet
    if start + width < len(clean):
        snippet += "..."
    return snippet


def compact_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value}


def build_filter_sql(filters: dict[str, Any] | None,
                     table_alias: str = "") -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    prefix = f"{table_alias}." if table_alias else ""
    for key, value in (filters or {}).items():
        field, op = key.split("__", 1) if "__" in key else (key, "eq")
        field_expr = f"{prefix}{field}"
        if op == "like":
            clauses.append(f"{field_expr} like ?")
            params.append(f"%{value}%")
        else:
            clauses.append(f"{field_expr} = ?")
            params.append(value)
    return (" and ".join(clauses) if clauses else "1=1", params)


def safe_order_by(order_by: str | None, default: str) -> str:
    if not order_by:
        return default
    reverse = order_by.startswith("-")
    field = order_by.lstrip("-")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", field):
        return default
    return f"{field} desc" if reverse else field
