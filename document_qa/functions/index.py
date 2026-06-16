from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency fallback
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency fallback
    SentenceTransformer = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency fallback
    np = None

from .graph_analysis import (
    DEFAULT_RELATION_GROUP_TYPES,
    analyze_relation_graph,
    build_relation_groups,
    find_relation_paths,
)
from .embeddings import (
    batched,
    cached_hf_snapshot,
    coerce_bool,
    dot_product,
    embedding_text,
    load_faiss,
    normalize_vector,
    pack_vector,
    resolve_embedding_config,
    unpack_vector,
)
from .retrieval import (
    add_candidate,
    add_semantic_candidate,
    rerank_profiled_hits,
    strip_internal_fields,
)
from .relations import (
    add_relation,
    build_doc_number_index,
    build_title_index,
    classify_enhanced_relation_type,
    classify_relation_role,
    confidence_for_enhanced_relation,
    confidence_for_role,
    directed_pairs,
    exact_relation_title_key,
    format_relation_row,
    match_profile_targets,
    match_title_targets,
    normalize_relation_text,
    normalize_series_title,
    parse_relation_types,
    pick_original_quote,
    relation_role_label,
    relation_type_for_role,
)
from .search_utils import (
    anchored_like_terms,
    build_filter_sql,
    build_fts_query,
    compact_dict,
    make_snippet,
    safe_order_by,
    summarize_recall_sources,
    tokenize_for_search,
)
from .storage import (
    connect as _connect,
    delete_document,
    delete_document_kb_for_ids,
    document_relation_stats,
    init_schema,
    insert_parsed_document,
    load_document_kb_state,
    refresh_term_statistics,
    upsert_document_kb_state,
)
from .text_processing import (
    DEFAULT_EXCLUDED_TOP_LEVEL_DIRS,
    DOCUMENT_KB_EXTRACTOR_VERSION,
    HEADING_RE,
    IMAGE_RE,
    build_evidence_outline,
    build_quality_notes,
    build_strict_answer_constraints,
    build_synthesis_outline,
    build_temporal_coverage,
    choose_title,
    classify_term,
    collect_document_term_weights,
    collect_query_term_weights,
    collect_seed_term_weights,
    compute_idf,
    context_best_hit,
    document_kb_signature,
    document_quality,
    document_temporal_metadata,
    extract_doc_numbers,
    extract_quoted_terms,
    extract_seed_terms,
    infer_agency,
    infer_date_text,
    infer_doc_type,
    infer_document_metadata,
    is_anchor_kind,
    kind_match_multiplier,
    kind_score_multiplier,
    length_bonus,
    rerank_base_component,
    split_markdown_chunks,
    stable_id,
    top_weighted_terms,
    tokenize_query,
)


@dataclass
class DocumentPaths:
    repo_root: Path
    corpus_root: Path
    index_path: Path

    @property
    def derived_dir(self) -> Path:
        return self.index_path.parent

    @property
    def faiss_index_path(self) -> Path:
        return self.derived_dir / "chunk_embeddings.faiss"

    @property
    def faiss_meta_path(self) -> Path:
        return self.derived_dir / "chunk_embeddings.faiss.json"

def resolve_paths(domain_dir: Path) -> DocumentPaths:
    repo_root = domain_dir.resolve().parents[1]
    corpus_root = Path(os.getenv("DOCUMENT_QA_ROOT", repo_root / "documents_mineru")).resolve()
    index_path = Path(
        os.getenv("DOCUMENT_QA_INDEX", domain_dir / ".document_qa" / "document_index.sqlite")
    ).resolve()
    return DocumentPaths(repo_root=repo_root, corpus_root=corpus_root, index_path=index_path)





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
            with _connect(self.paths.index_path) as conn:
                indexed = conn.execute("select count(*) from documents").fetchone()[0]
            if indexed == 0:
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

        with _connect(self.paths.index_path) as conn:
            init_schema(conn)
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

        with _connect(self.paths.index_path) as conn:
            init_schema(conn)
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
                delete_document(conn, existing_by_path[rel_path]["document_id"])
                deleted += 1

            added = 0
            updated = 0
            chunks_upserted = 0
            for _rel_path, path, action in upsert_paths:
                doc, chunks = self._parse_document(path)
                delete_document(conn, doc["document_id"])
                insert_parsed_document(conn, doc, chunks)
                chunks_upserted += len(chunks)
                if action == "added":
                    added += 1
                else:
                    updated += 1

            if deleted or upsert_paths:
                refresh_term_statistics(conn)

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

        with _connect(self.paths.index_path) as conn:
            init_schema(conn)
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
        with _connect(self.paths.index_path) as conn:
            init_schema(conn)
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
            "faiss": self.rebuild_vector_index(ensure_index=False),
        }

    def rebuild_vector_index(self, ensure_index: bool = True) -> dict[str, Any]:
        if ensure_index:
            self.ensure()
        if np is None:
            return {"status": "unavailable", "message": "numpy package is not available"}
        faiss = load_faiss()
        if faiss is None:
            return {"status": "unavailable", "message": "faiss-cpu package is not available"}

        config = self.embedding_config
        with _connect(self.paths.index_path) as conn:
            init_schema(conn)
            rows = conn.execute(
                """
                select e.chunk_id, e.dim, e.embedding
                from chunk_embeddings e
                join chunks c on c.chunk_id = e.chunk_id
                where e.model = ?
                order by c.document_id, c.ordinal
                """,
                (config.model,),
            ).fetchall()
        if not rows:
            return {"status": "empty", "model": config.model, "vectors": 0}

        dim = int(rows[0]["dim"])
        vectors = np.empty((len(rows), dim), dtype="float32")
        chunk_ids = []
        for idx, row in enumerate(rows):
            if int(row["dim"]) != dim:
                continue
            vectors[idx] = np.frombuffer(row["embedding"], dtype="<f4", count=dim)
            chunk_ids.append(row["chunk_id"])
        if len(chunk_ids) != len(rows):
            vectors = vectors[:len(chunk_ids)]

        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self.paths.derived_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.paths.faiss_index_path))
        self.paths.faiss_meta_path.write_text(
            json.dumps(
                {
                    "model": config.model,
                    "dim": dim,
                    "metric": "inner_product",
                    "normalized": True,
                    "count": len(chunk_ids),
                    "chunk_ids": chunk_ids,
                    "built_at": time.time(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "status": "rebuilt",
            "model": config.model,
            "vectors": len(chunk_ids),
            "dimensions": dim,
            "index_path": str(self.paths.faiss_index_path),
            "metadata_path": str(self.paths.faiss_meta_path),
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
            hits = rerank_profiled_hits(recall_hits, profile, limit=limit, debug=debug)
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
                    "recall_sources": summarize_recall_sources(recall_hits),
                }
            return result
        hits = strip_internal_fields(ranked_candidates[:limit])
        result = {
            "query": clean_query,
            "filters": compact_dict({"category": category, "agency": agency, "doc_type": doc_type}),
            "semantic": semantic,
            "count": len(hits),
            "hits": hits,
        }
        if debug:
            result["debug"] = {"recall_sources": summarize_recall_sources(hits)}
        return result

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
        with _connect(self.paths.index_path) as conn:
            return [dict(row) for row in conn.execute(sql, [*params, limit])]

    def read_document(self, *, path: str = "", document_id: str = "",
                      chunk_id: str = "", heading: str = "",
                      max_chars: int = 6000) -> dict[str, Any]:
        self.ensure()
        max_chars = max(1000, min(int(max_chars or 6000), 20000))
        with _connect(self.paths.index_path) as conn:
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
                    """
                    select *
                    from documents
                    where path = ? or path like ?
                    order by case when path = ? then 0 else 1 end, length(path), path
                    limit 1
                    """,
                    (path, f"%{path}%", path),
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
        hits = rerank_profiled_hits(recall_hits[:recall_limit], profile, limit=limit, debug=debug)
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

        with _connect(self.paths.index_path) as conn:
            init_schema(conn)
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
                state = load_document_kb_state(conn)
                doc_ids = set(signatures)
                removed_ids = sorted(set(state) - doc_ids)
                if removed_ids:
                    delete_document_kb_for_ids(conn, removed_ids, delete_incoming=True)
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
                    stats = document_relation_stats(conn)
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
                delete_document_kb_for_ids(conn, sorted(changed_ids), delete_incoming=False)
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

            upsert_document_kb_state(
                conn,
                docs if force else docs_to_refresh,
                signatures,
                include_soft=include_soft,
            )
            stats = document_relation_stats(conn)

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
        with _connect(self.paths.index_path) as conn:
            init_schema(conn)
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

    def find_document_relation_groups(self, *, query: str = "",
                                      relation_types: str = "",
                                      title_like: str = "",
                                      limit: int = 5,
                                      min_relations: int = 2,
                                      max_documents_per_group: int = 8) -> dict[str, Any]:
        self.ensure()
        limit = max(1, min(int(limit or 5), 20))
        min_relations = max(1, min(int(min_relations or 2), 20))
        max_documents_per_group = max(2, min(int(max_documents_per_group or 8), 20))
        wanted_types = parse_relation_types(relation_types)
        query_terms = tokenize_terms(query, 12)
        title_filter = (title_like or "").strip()

        with _connect(self.paths.index_path) as conn:
            init_schema(conn)
            rows = self._load_relation_group_rows(
                conn,
                wanted_types=wanted_types,
                query_terms=query_terms,
                title_like=title_filter,
            )

        components = build_relation_groups(
            rows,
            min_relations=min_relations,
            max_documents_per_group=max_documents_per_group,
        )
        return {
            "query": query,
            "title_like": title_filter,
            "relation_types": wanted_types,
            "min_relations": min_relations,
            "count": len(components[:limit]),
            "groups": components[:limit],
            "usage_note": "关系组用于发现候选关联公文；关键事实、正文细节和最终结论仍需 read_document 核对原文。",
        }

    def find_document_relation_paths(self, *, start_document_id: str = "",
                                     start_path: str = "",
                                     start_title_like: str = "",
                                     end_document_id: str = "",
                                     end_path: str = "",
                                     end_title_like: str = "",
                                     relation_types: str = "",
                                     direction: str = "both",
                                     max_depth: int = 3,
                                     limit: int = 10) -> dict[str, Any]:
        self.ensure()
        max_depth = max(1, min(int(max_depth or 3), 5))
        limit = max(1, min(int(limit or 10), 50))
        direction = (direction or "both").strip().lower()
        if direction not in {"outgoing", "incoming", "both"}:
            direction = "both"
        wanted_types = parse_relation_types(relation_types)

        with _connect(self.paths.index_path) as conn:
            init_schema(conn)
            start = self._resolve_relation_document(
                conn,
                path=start_path,
                document_id=start_document_id,
                title_like=start_title_like,
            )
            if not start:
                return {"error": "请提供有效的 start_document_id、start_path 或 start_title_like 来定位起点文档"}
            end = None
            if end_document_id or end_path or end_title_like:
                end = self._resolve_relation_document(
                    conn,
                    path=end_path,
                    document_id=end_document_id,
                    title_like=end_title_like,
                )
                if not end:
                    return {"error": "未找到终点文档，请检查 end_document_id、end_path 或 end_title_like"}
            rows = self._load_path_relation_rows(conn, wanted_types=wanted_types)

        paths = find_relation_paths(
            [dict(row) for row in rows],
            start_id=start["document_id"],
            end_id=end["document_id"] if end else "",
            direction=direction,
            max_depth=max_depth,
            limit=limit,
        )
        return {
            "start_document": dict(start),
            "end_document": dict(end) if end else None,
            "direction": direction,
            "relation_types": wanted_types,
            "max_depth": max_depth,
            "count": len(paths),
            "paths": paths,
            "usage_note": "多跳路径是关系线索，不等于事实结论；回答具体事实前应读取路径上的关键文档原文。",
        }

    def analyze_document_graph(self, *, query: str = "",
                               relation_types: str = "",
                               top_n: int = 10,
                               min_confidence: float = 0.0,
                               max_edges: int = 20000) -> dict[str, Any]:
        self.ensure()
        wanted_types = parse_relation_types(relation_types)
        query_terms = tokenize_terms(query, 12)
        top_n = max(1, min(int(top_n or 10), 50))
        min_confidence = max(0.0, min(float(min_confidence or 0.0), 1.0))
        max_edges = max(100, min(int(max_edges or 20000), 100000))

        with _connect(self.paths.index_path) as conn:
            rows = self._load_graph_relation_rows(
                conn,
                wanted_types=wanted_types,
                query_terms=query_terms,
                min_confidence=min_confidence,
                max_edges=max_edges,
            )
        return analyze_relation_graph(rows, top_n=top_n)

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
        with _connect(self.paths.index_path) as conn:
            init_schema(conn)
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
                            "context_score": max(0.0, float(suggestion["confidence"]) * 4.0),
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
                best_year = temporal.get("best_year")
                item["_matches_expected_year"] = (
                    bool(expected_years)
                    and bool(best_year)
                    and best_year in expected_years
                )
                item["_out_of_time_range"] = (
                    bool(expected_years)
                    and bool(best_year)
                    and best_year not in expected_years
                )

            if expected_years:
                eligible_doc_scores = [item for item in doc_scores if item.get("_matches_expected_year")]
            else:
                eligible_doc_scores = [item for item in doc_scores if not item.get("_out_of_time_range")]
            if len(eligible_doc_scores) < min(limit_docs, 2):
                eligible_doc_scores = doc_scores

            selected = sorted(
                eligible_doc_scores,
                key=lambda item: (
                    expected_years and not bool(item.get("_matches_expected_year")),
                    not bool(item.get("search_hit_count", 0)),
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
        with _connect(self.paths.index_path) as conn:
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
        with _connect(self.paths.index_path) as conn:
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
        with _connect(self.paths.index_path) as conn:
            return [dict(row) for row in conn.execute(sql, [*params, limit, offset or 0])]

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
                from documents
                where path = ? or path like ?
                order by case when path = ? then 0 else 1 end, length(path), path
                limit 1
                """,
                (path, f"%{path}%", path),
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

    def _load_relation_group_rows(self, conn: sqlite3.Connection,
                                  *,
                                  wanted_types: list[str],
                                  query_terms: list[str],
                                  title_like: str) -> list[dict[str, Any]]:
        where = []
        params: list[Any] = []
        if wanted_types:
            placeholders = ",".join("?" for _ in wanted_types)
            where.append(f"r.relation_type in ({placeholders})")
            params.extend(wanted_types)
        else:
            placeholders = ",".join("?" for _ in DEFAULT_RELATION_GROUP_TYPES)
            where.append(f"r.relation_type in ({placeholders})")
            params.extend(DEFAULT_RELATION_GROUP_TYPES)
        if title_like:
            like = f"%{title_like}%"
            where.append(
                "(sd.title like ? or td.title like ? or sd.path like ? or td.path like ? or r.evidence like ?)"
            )
            params.extend([like, like, like, like, like])
        for term in query_terms:
            like = f"%{term}%"
            where.append(
                "(sd.title like ? or td.title like ? or sd.category like ? or td.category like ? or r.evidence like ?)"
            )
            params.extend([like, like, like, like, like])

        sql = f"""
            select r.relation_id, r.source_document_id, r.target_document_id,
                   r.relation_type, r.confidence, r.evidence, r.method, r.metadata,
                   sd.title as source_title, sd.path as source_path,
                   sd.category as source_category, sd.agency as source_agency,
                   sd.doc_type as source_doc_type, sd.date_text as source_date_text,
                   sd.char_count as source_char_count,
                   td.title as target_title, td.path as target_path,
                   td.category as target_category, td.agency as target_agency,
                   td.doc_type as target_doc_type, td.date_text as target_date_text,
                   td.char_count as target_char_count
            from document_relations r
            join documents sd on sd.document_id = r.source_document_id
            join documents td on td.document_id = r.target_document_id
            where {' and '.join(where)}
            order by r.confidence desc, r.relation_type
            limit 5000
        """
        return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def _load_path_relation_rows(self, conn: sqlite3.Connection,
                                 *,
                                 wanted_types: list[str]) -> list[sqlite3.Row]:
        where = []
        params: list[Any] = []
        if wanted_types:
            placeholders = ",".join("?" for _ in wanted_types)
            where.append(f"r.relation_type in ({placeholders})")
            params.extend(wanted_types)
        else:
            where.append("r.relation_type not in ('topically_related', 'semantically_related')")
        sql = f"""
            select r.relation_id, r.source_document_id, r.target_document_id,
                   r.relation_type, r.confidence, r.evidence, r.method, r.metadata,
                   sd.title as source_title, sd.path as source_path,
                   sd.category as source_category, sd.agency as source_agency,
                   sd.doc_type as source_doc_type, sd.date_text as source_date_text,
                   sd.char_count as source_char_count,
                   td.title as target_title, td.path as target_path,
                   td.category as target_category, td.agency as target_agency,
                   td.doc_type as target_doc_type, td.date_text as target_date_text,
                   td.char_count as target_char_count
            from document_relations r
            join documents sd on sd.document_id = r.source_document_id
            join documents td on td.document_id = r.target_document_id
            where {' and '.join(where)}
            order by r.confidence desc
        """
        return conn.execute(sql, params).fetchall()

    def _load_graph_relation_rows(self, conn: sqlite3.Connection,
                                  *,
                                  wanted_types: list[str],
                                  query_terms: list[str],
                                  min_confidence: float,
                                  max_edges: int) -> list[dict[str, Any]]:
        where = ["r.confidence >= ?"]
        params: list[Any] = [min_confidence]
        if wanted_types:
            placeholders = ",".join("?" for _ in wanted_types)
            where.append(f"r.relation_type in ({placeholders})")
            params.extend(wanted_types)
        for term in query_terms:
            like = f"%{term}%"
            where.append(
                "(sd.title like ? or td.title like ? or sd.category like ? or td.category like ? or r.evidence like ?)"
            )
            params.extend([like, like, like, like, like])
        sql = f"""
            select r.relation_id, r.source_document_id, r.target_document_id,
                   r.relation_type, r.confidence, r.evidence, r.method,
                   sd.title as source_title, sd.path as source_path,
                   sd.category as source_category, sd.agency as source_agency,
                   sd.doc_type as source_doc_type, sd.date_text as source_date_text,
                   sd.char_count as source_char_count,
                   td.title as target_title, td.path as target_path,
                   td.category as target_category, td.agency as target_agency,
                   td.doc_type as target_doc_type, td.date_text as target_date_text,
                   td.char_count as target_char_count
            from document_relations r
            join documents sd on sd.document_id = r.source_document_id
            join documents td on td.document_id = r.target_document_id
            where {' and '.join(where)}
            order by r.confidence desc
            limit ?
        """
        return [dict(row) for row in conn.execute(sql, [*params, max_edges]).fetchall()]

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
        with _connect(self.paths.index_path) as conn:
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

        with _connect(self.paths.index_path) as conn:
            if clean_query:
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
                        add_candidate(candidates, dict(row), score, clean_query)

                title_terms = [clean_query, *tokenize_query(clean_query)]
                for term in title_terms[:8]:
                    if not term:
                        continue
                    title_like = f"%{term}%"
                    title_sql = f"""
                        select c.*, d.title, d.path, d.category, d.agency, d.doc_type
                        from documents d
                        join chunks c on c.document_id = d.document_id
                        where ({where_sql})
                          and (d.title like ? or d.path like ?)
                        order by d.path, c.ordinal
                        limit ?
                    """
                    for row in conn.execute(title_sql, [*params, title_like, title_like, limit * 2]):
                        add_candidate(candidates, dict(row), 8.0, clean_query)

                for row in self._anchored_like_recall(conn, clean_query, where_sql, params, limit):
                    add_candidate(candidates, row, 16.0, clean_query)

                if semantic:
                    for row in self._semantic_recall_chunks(clean_query, filters, params, limit):
                        add_semantic_candidate(candidates, row, clean_query)
            else:
                sql = f"""
                    select c.*, d.title, d.path, d.category, d.agency, d.doc_type
                    from chunks c join documents d using (document_id)
                    where {where_sql}
                    order by d.path, c.ordinal
                    limit ?
                """
                for row in conn.execute(sql, [*params, limit]):
                    add_candidate(candidates, dict(row), 1.0, clean_query)

        return sorted(candidates.values(), key=lambda r: (-r["score"], r["path"], r["ordinal"]))

    def _anchored_like_recall(self, conn: sqlite3.Connection,
                              query: str,
                              where_sql: str,
                              params: list[Any],
                              limit: int) -> list[dict[str, Any]]:
        anchors = anchored_like_terms(query)
        if len(anchors) < 2:
            return []
        clauses = []
        like_params: list[Any] = []
        for term in anchors[:4]:
            clauses.append("(c.content like ? or c.heading like ? or d.title like ? or d.path like ?)")
            like = f"%{term}%"
            like_params.extend([like, like, like, like])
        sql = f"""
            select c.*, d.title, d.path, d.category, d.agency, d.doc_type
            from chunks c join documents d using (document_id)
            where ({where_sql}) and {' and '.join(clauses)}
            order by d.path, c.ordinal
            limit ?
        """
        return [dict(row) for row in conn.execute(sql, [*params, *like_params, max(limit * 2, 20)]).fetchall()]

    def _semantic_recall_chunks(self, query: str, filters: list[str],
                                params: list[Any], limit: int) -> list[dict[str, Any]]:
        config = self.embedding_config
        if not query or not config.enabled or config.provider not in {"openai", "local"}:
            return []
        faiss_hits = self._semantic_recall_chunks_faiss(query, filters, params, limit)
        if faiss_hits:
            return faiss_hits
        try:
            with _connect(self.paths.index_path) as conn:
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

    def _semantic_recall_chunks_faiss(self, query: str, filters: list[str],
                                      params: list[Any], limit: int) -> list[dict[str, Any]]:
        if np is None:
            return []
        faiss = load_faiss()
        if faiss is None:
            return []
        if not self.paths.faiss_index_path.exists() or not self.paths.faiss_meta_path.exists():
            return []
        try:
            meta = json.loads(self.paths.faiss_meta_path.read_text(encoding="utf-8"))
            if meta.get("model") != self.embedding_config.model:
                return []
            chunk_ids = meta.get("chunk_ids") or []
            index = faiss.read_index(str(self.paths.faiss_index_path))
            query_vector = np.asarray([self._embed_query(query)], dtype="float32")
            recall_k = min(max(limit * 12, 80), len(chunk_ids))
            scores, indexes = index.search(query_vector, recall_k)
        except Exception:
            return []

        candidates = []
        for score, vector_index in zip(scores[0].tolist(), indexes[0].tolist(), strict=True):
            if vector_index < 0 or vector_index >= len(chunk_ids):
                continue
            if float(score) < self.embedding_config.min_similarity:
                continue
            candidates.append((chunk_ids[vector_index], float(score)))
        if not candidates:
            return []

        score_by_chunk = {chunk_id: score for chunk_id, score in candidates}
        placeholders = ",".join("?" for _chunk_id, _score in candidates)
        where_sql = " and ".join(filters) if filters else "1=1"
        with _connect(self.paths.index_path) as conn:
            rows = conn.execute(
                f"""
                select c.*, d.title, d.path, d.category, d.agency, d.doc_type
                from chunks c join documents d using (document_id)
                where c.chunk_id in ({placeholders}) and {where_sql}
                """,
                [*[chunk_id for chunk_id, _score in candidates], *params],
            ).fetchall()
        hits = []
        for row in rows:
            item = dict(row)
            item["semantic_score"] = round(score_by_chunk.get(row["chunk_id"], 0.0), 4)
            hits.append(item)
        return sorted(
            hits,
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
        with _connect(self.paths.index_path) as conn:
            init_schema(conn)
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

















































































