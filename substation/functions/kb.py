from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    HEADING_RE, IMAGE_RE, KB_SCHEMA_VERSION, PRIMARY_KNOWLEDGE_BOOST,
    PRIMARY_KNOWLEDGE_ROLE, HIGH_VALUE_EVIDENCE_BOOST, TOC_PENALTY,
)
from .evidence import _evidence_role, _is_toc_chunk
from .utils import _compact, _query_terms, _stable_id, _strip_frontmatter

@dataclass
class SubstationKb:
    domain_dir: Path

    def __post_init__(self) -> None:
        self.docs_root = self.domain_dir / "docs_md"
        self.md_root = self.docs_root / "md"
        self.kb_dir = self.domain_dir / ".substation_kb"
        self.db_path = self.kb_dir / "substation_event_kb.sqlite"
        self._loaded = False
        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._needs_sync():
            self.sync()
        self._load()
        self._loaded = True

    def sync(self) -> dict[str, Any]:
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        docs, chunks = self._scan()
        with sqlite3.connect(self.db_path) as conn:
            self._init_schema(conn)
            conn.execute("delete from documents")
            conn.execute("delete from chunks")
            conn.execute("delete from meta")
            conn.executemany(
                """
                insert into documents
                (document_id, title, category, source_path, md_path, doc_role, char_count, file_mtime)
                values
                (:document_id, :title, :category, :source_path, :md_path, :doc_role, :char_count, :file_mtime)
                """,
                docs,
            )
            conn.executemany(
                """
                insert into chunks
                (chunk_id, document_id, title, category, md_path, heading, content, evidence_role, ordinal, image_refs)
                values
                (:chunk_id, :document_id, :title, :category, :md_path, :heading, :content, :evidence_role, :ordinal, :image_refs)
                """,
                chunks,
            )
            conn.execute("insert into meta(key, value) values('schema_version', ?)", (KB_SCHEMA_VERSION,))
            conn.execute("insert into meta(key, value) values('source_mtime', ?)", (str(self._source_mtime()),))
        self._loaded = False
        self.documents = []
        self.chunks = []
        return {
            "status": "ok",
            "db_path": str(self.db_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            create table if not exists meta (
              key text primary key,
              value text not null
            );
            create table if not exists documents (
              document_id text primary key,
              title text,
              category text,
              source_path text,
              md_path text,
              doc_role text,
              char_count integer,
              file_mtime real
            );
            create table if not exists chunks (
              chunk_id text primary key,
              document_id text,
              title text,
              category text,
              md_path text,
              heading text,
              content text,
              evidence_role text,
              ordinal integer,
              image_refs text
            );
            """
        )

    def _load(self) -> None:
        with self._connect() as conn:
            self.documents = [dict(r) for r in conn.execute("select * from documents order by document_id")]
            self.chunks = [dict(r) for r in conn.execute("select * from chunks order by document_id, ordinal")]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _source_mtime(self) -> float:
        paths = list(self.md_root.rglob("*.md")) if self.md_root.exists() else []
        for path in [self.docs_root / "manifest.csv", self.docs_root / "status.csv"]:
            if path.exists():
                paths.append(path)
        return max((p.stat().st_mtime for p in paths), default=0.0)

    def _needs_sync(self) -> bool:
        if not self.db_path.exists():
            return True
        try:
            with self._connect() as conn:
                self._init_schema(conn)
                count = conn.execute("select count(*) from documents").fetchone()[0]
                source_row = conn.execute("select value from meta where key='source_mtime'").fetchone()
                schema_row = conn.execute("select value from meta where key='schema_version'").fetchone()
            return (
                count == 0
                or source_row is None
                or schema_row is None
                or schema_row["value"] != KB_SCHEMA_VERSION
                or float(source_row["value"]) < self._source_mtime()
            )
        except sqlite3.DatabaseError:
            return True

    def _read_manifest(self) -> dict[str, dict[str, str]]:
        path = self.docs_root / "manifest.csv"
        if not path.exists():
            return {}
        rows: dict[str, dict[str, str]] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows[row.get("md_path", "")] = row
        return rows

    def _scan(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        manifest = self._read_manifest()
        docs: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        for md_path in sorted(self.md_root.rglob("*.md")):
            rel_to_md = md_path.relative_to(self.md_root).as_posix()
            rel_to_domain = md_path.relative_to(self.domain_dir).as_posix()
            meta_row = manifest.get(f"md/{rel_to_md}", {})
            raw = md_path.read_text(encoding="utf-8", errors="ignore")
            frontmatter, body = _strip_frontmatter(raw)
            original_name = meta_row.get("original_name") or frontmatter.get("original_name") or ""
            title = Path(original_name).stem if original_name else self._first_title(body, md_path.stem)
            category = meta_row.get("category") or frontmatter.get("category") or md_path.parent.name
            doc = {
                "document_id": md_path.stem,
                "title": title,
                "category": category,
                "source_path": meta_row.get("source_path") or frontmatter.get("source_path", ""),
                "md_path": rel_to_domain,
                "doc_role": self._doc_role(title, category),
                "char_count": len(body),
                "file_mtime": md_path.stat().st_mtime,
            }
            docs.append(doc)
            chunks.extend(self._split_chunks(doc, body))
        return docs, chunks

    def _first_title(self, text: str, fallback: str) -> str:
        for line in text.splitlines():
            m = HEADING_RE.match(line)
            if m:
                return m.group(2).strip()
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("![]("):
                return line[:80]
        return fallback

    def _doc_role(self, title: str, category: str) -> str:
        text = f"{title} {category}"
        if any(k in text for k in ["设备异常", "故障应急", "规程之五"]):
            return "emergency_procedure"
        if any(k in text for k in ["运行方式", "运行规定", "规程之三"]):
            return "operation_rule"
        if any(k in text for k in ["典型操作票", "规程之四"]):
            return "operation_ticket"
        if any(k in text for k in ["设计规范", "说明书", "控制保护", "极控制系统", "直流控制", "系统总体设计"]):
            return "control_protection_spec"
        if "设备概况" in text:
            return "equipment_overview"
        if "定值" in text:
            return "setting_sheet"
        return "document"

    def _split_chunks(self, doc: dict[str, Any], text: str) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        heading = doc["title"]
        buf: list[str] = []
        ordinal = 0

        def flush() -> None:
            nonlocal buf, ordinal
            content = "\n".join(buf).strip()
            buf = []
            if not content:
                return
            for piece in self._chunk_text(content):
                ordinal += 1
                chunk_id = f"{doc['document_id']}_c{ordinal:04d}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": doc["document_id"],
                        "title": doc["title"],
                        "category": doc["category"],
                        "md_path": doc["md_path"],
                        "heading": heading,
                        "content": piece,
                        "evidence_role": _evidence_role(f"{heading}\n{piece}"),
                        "ordinal": ordinal,
                        "image_refs": ",".join(IMAGE_RE.findall(piece)),
                    }
                )

        for line in text.splitlines():
            m = HEADING_RE.match(line)
            if m:
                flush()
                heading = m.group(2).strip()
            buf.append(line)
        flush()
        return chunks

    def _chunk_text(self, text: str, max_chars: int = 1800) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        paras = re.split(r"\n\s*\n", text)
        out: list[str] = []
        buf = ""
        for para in paras:
            if len(para) > max_chars:
                if buf:
                    out.append(buf)
                    buf = ""
                out.extend(para[i : i + max_chars] for i in range(0, len(para), max_chars))
                continue
            if buf and len(buf) + len(para) + 2 > max_chars:
                out.append(buf)
                buf = para
            else:
                buf = f"{buf}\n\n{para}".strip()
        if buf:
            out.append(buf)
        return out

    def query(self, object_type: str, filters: dict[str, Any] | None = None,
              limit: int | None = None, order_by: str | None = None,
              offset: int | None = None) -> list[dict[str, Any]]:
        self.ensure_loaded()
        table = {"SubstationDocument": "documents", "EvidenceChunk": "chunks"}.get(object_type)
        if not table:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in (filters or {}).items():
            if key.endswith("__like"):
                clauses.append(f"{key[:-6]} like ?")
                params.append(f"%{value}%")
            else:
                clauses.append(f"{key} = ?")
                params.append(value)
        sql = f"select * from {table}"
        if clauses:
            sql += " where " + " and ".join(clauses)
        if order_by:
            reverse = order_by.startswith("-")
            field = order_by[1:] if reverse else order_by
            sql += f" order by {field} {'desc' if reverse else 'asc'}"
        elif table == "chunks":
            sql += " order by document_id, ordinal"
        else:
            sql += " order by document_id"
        if limit:
            sql += " limit ?"
            params.append(limit)
            if offset:
                sql += " offset ?"
                params.append(offset)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def count(self, object_type: str, filters: dict[str, Any] | None = None) -> int:
        return len(self.query(object_type, filters))

    def query_by_id(self, object_type: str, id_value: Any) -> dict[str, Any] | None:
        id_field = {"SubstationDocument": "document_id", "EvidenceChunk": "chunk_id"}.get(object_type)
        if not id_field:
            return None
        rows = self.query(object_type, {id_field: id_value}, limit=1)
        return rows[0] if rows else None

    def search_text(self, keyword: str, object_types: list[str] | None = None, limit: int = 20) -> list[dict[str, Any]]:
        return self.search(keyword, limit=limit).get("chunks", [])

    def search(self, query: str, category: str = "", doc_role: str = "",
               evidence_role: str = "", limit: int = 8,
               prefer_primary: bool = False) -> dict[str, Any]:
        self.ensure_loaded()
        terms = _query_terms(query)
        clauses = []
        params: list[Any] = []
        for term in terms[:16]:
            clauses.append("(c.title like ? or c.heading like ? or c.content like ?)")
            like = f"%{term}%"
            params.extend([like, like, like])
        if clauses:
            where = ["(" + " or ".join(clauses) + ")"]
        else:
            where = ["1=1"]
        if category:
            where.append("c.category like ?")
            params.append(f"%{category}%")
        if evidence_role:
            where.append("c.evidence_role = ?")
            params.append(evidence_role)
        if doc_role:
            where.append("d.doc_role = ?")
            params.append(doc_role)
        sql = """
            select c.*, d.doc_role, d.source_path
            from chunks c join documents d on d.document_id = c.document_id
            where """ + " and ".join(where) + " limit 500"
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(sql, params)]
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            hay = "\n".join([row.get("title", ""), row.get("heading", ""), row.get("content", "")])
            score = self._score(hay, terms)
            if row.get("doc_role") == PRIMARY_KNOWLEDGE_ROLE:
                row["knowledge_priority"] = "primary_fifth_volume"
                if prefer_primary:
                    score += PRIMARY_KNOWLEDGE_BOOST
            else:
                row["knowledge_priority"] = "supporting_document"
            if row.get("evidence_role") in {"handling_rule", "severe_signal", "abnormal_signal"}:
                score += HIGH_VALUE_EVIDENCE_BOOST
            if _is_toc_chunk(row):
                score -= TOC_PENALTY
                row["is_toc"] = True
            else:
                row["is_toc"] = False
            if score <= 0:
                continue
            row["score"] = round(score, 3)
            row["snippet"] = self._snippet(row.get("content", ""), terms)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return {
            "query": query,
            "terms": terms,
            "chunks": [row for _, row in scored[:limit]],
            "usage_note": "这些片段是研判依据候选；事件性质仍需结合用户给出的时序、复归情况和后果判断。",
        }

    def read(self, document_id: str = "", path: str = "", chunk_id: str = "",
             heading: str = "", max_chars: int = 6000) -> dict[str, Any]:
        self.ensure_loaded()
        focus_doc = None
        focus_chunk = None
        if chunk_id:
            focus_chunk = self.query_by_id("EvidenceChunk", chunk_id)
            if focus_chunk:
                focus_doc = self.query_by_id("SubstationDocument", focus_chunk["document_id"])
        elif document_id:
            focus_doc = self.query_by_id("SubstationDocument", document_id)
        elif path:
            focus_doc = next((d for d in self.documents if d["md_path"] == path or path.endswith(d["md_path"])), None)
        if not focus_doc:
            return {"error": "未找到文档", "document_id": document_id, "path": path, "chunk_id": chunk_id}
        chunks = [c for c in self.chunks if c["document_id"] == focus_doc["document_id"]]
        if heading:
            chunks = [c for c in chunks if heading in c.get("heading", "") or heading in c.get("content", "")]
        elif focus_chunk:
            base = int(focus_chunk["ordinal"])
            chunks = [c for c in chunks if abs(int(c["ordinal"]) - base) <= 2]
        parts = []
        included = []
        for chunk in chunks:
            block = f"## {chunk['heading']}\n{chunk['content']}"
            if len("\n\n".join(parts + [block])) > max_chars:
                break
            parts.append(block)
            included.append(chunk["chunk_id"])
        return {
            "document": focus_doc,
            "included_chunk_ids": included,
            "content": "\n\n".join(parts),
            "truncated": len(included) < len(chunks),
        }

    def _score(self, text: str, terms: list[str]) -> float:
        lower = text.lower()
        score = 0.0
        for term in terms:
            count = lower.count(term.lower())
            if count:
                score += min(count, 8) * (2.5 if len(term) >= 4 else 1.0)
        return score

    def _snippet(self, content: str, terms: list[str], window: int = 360) -> str:
        lower = content.lower()
        hit = -1
        for term in terms:
            hit = lower.find(term.lower())
            if hit >= 0:
                break
        if hit < 0:
            return _compact(content[:window])
        start = max(hit - window // 2, 0)
        end = min(start + window, len(content))
        return _compact(content[start:end])

class SubstationResolver:
    def __init__(self, kb: SubstationKb):
        self.kb = kb

    def query(self, object_type: str, filters: dict[str, Any] | None = None,
              limit: int | None = None, order_by: str | None = None,
              offset: int | None = None) -> list[dict[str, Any]]:
        return self.kb.query(object_type, filters, limit, order_by, offset)

    def count(self, object_type: str, filters: dict[str, Any] | None = None) -> int:
        return self.kb.count(object_type, filters)

    def query_by_id(self, object_type: str, id_value: Any) -> dict[str, Any] | None:
        return self.kb.query_by_id(object_type, id_value)

    def search_text(self, keyword: str, object_types: list[str] | None = None, limit: int = 20) -> list[dict[str, Any]]:
        return self.kb.search_text(keyword, object_types, limit)
