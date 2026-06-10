from __future__ import annotations

import json
from pathlib import Path

from oag.ontology.registry import FunctionRegistry
from oag.ontology.repository import ObjectRepository
from oag.ontology.schema import Ontology

from .index import DocumentIndex, DocumentResolver, resolve_paths


def _json_result(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def register(registry: FunctionRegistry, store: ObjectRepository, ontology: Ontology):
    domain_dir = Path(__file__).resolve().parent.parent
    index = DocumentIndex(resolve_paths(domain_dir))

    registry.register_resolver("document_index", DocumentResolver(index))

    fn_map = {
        "search_documents": lambda **kw: index.search(
            kw.get("query", ""),
            category=kw.get("category", "") or "",
            agency=kw.get("agency", "") or "",
            doc_type=kw.get("doc_type", "") or "",
            limit=kw.get("limit", 8) or 8,
            rerank=kw.get("rerank", False),
            debug=kw.get("debug", False),
        ),
        "read_document": lambda **kw: index.read_document(
            path=kw.get("path", "") or "",
            document_id=kw.get("document_id", "") or "",
            chunk_id=kw.get("chunk_id", "") or "",
            heading=kw.get("heading", "") or "",
            max_chars=kw.get("max_chars", 6000) or 6000,
        ),
        "list_documents": lambda **kw: index.list_documents(
            category=kw.get("category", "") or "",
            agency=kw.get("agency", "") or "",
            doc_type=kw.get("doc_type", "") or "",
            title_like=kw.get("title_like", "") or "",
            limit=kw.get("limit", 20) or 20,
        ),
        "find_related_documents": lambda **kw: index.find_related_documents(
            seed_path=kw.get("seed_path", "") or "",
            seed_title=kw.get("seed_title", "") or "",
            query_hint=kw.get("query_hint", "") or "",
            limit=kw.get("limit", 10) or 10,
            debug=kw.get("debug", False),
        ),
        "build_document_kb": lambda **kw: index.build_document_kb(
            force=kw.get("force", True),
            include_soft=kw.get("include_soft", True),
        ),
        "find_document_relations": lambda **kw: index.find_document_relations(
            path=kw.get("path", "") or "",
            document_id=kw.get("document_id", "") or "",
            title_like=kw.get("title_like", "") or "",
            relation_type=kw.get("relation_type", "") or "",
            direction=kw.get("direction", "both") or "both",
            limit=kw.get("limit", 20) or 20,
        ),
        "expand_document_context": lambda **kw: index.expand_document_context(
            path=kw.get("path", "") or "",
            document_id=kw.get("document_id", "") or "",
            title_like=kw.get("title_like", "") or "",
            relation_types=kw.get("relation_types", "") or "",
            limit=kw.get("limit", 20) or 20,
        ),
        "prepare_answer_context": lambda **kw: index.prepare_answer_context(
            query=kw.get("query", "") or "",
            limit_docs=kw.get("limit_docs", 6) or 6,
            max_chars_per_doc=kw.get("max_chars_per_doc", 4000) or 4000,
            include_relations=kw.get("include_relations", True),
            debug=kw.get("debug", False),
        ),
        "rebuild_document_index": lambda **kw: index.rebuild(
            force=kw.get("force", True),
        ),
        "rebuild_document_embeddings": lambda **kw: index.rebuild_embeddings(
            force=kw.get("force", False),
        ),
    }

    for name, fn in fn_map.items():
        func_def = ontology.functions.get(name)
        if func_def:
            registry.register(name, lambda _fn=fn, **kw: _json_result(_fn(**kw)), func_def)
