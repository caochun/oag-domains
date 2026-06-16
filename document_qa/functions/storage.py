from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from .search_utils import tokenize_for_search
from .text_processing import collect_document_term_weights, compute_idf, top_weighted_terms


def connect(index_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(index_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma busy_timeout = 30000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
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


def delete_document(conn: sqlite3.Connection, document_id: str) -> None:
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


def insert_parsed_document(conn: sqlite3.Connection,
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


def refresh_term_statistics(conn: sqlite3.Connection) -> None:
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


def load_document_kb_state(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
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


def delete_document_kb_for_ids(conn: sqlite3.Connection,
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


def upsert_document_kb_state(conn: sqlite3.Connection,
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


def document_relation_stats(conn: sqlite3.Connection) -> list[dict[str, Any]]:
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
