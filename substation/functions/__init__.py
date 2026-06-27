from __future__ import annotations

from pathlib import Path

from oag.ontology.registry import FunctionRegistry
from oag.ontology.repository import ObjectRepository
from oag.ontology.schema import Ontology

from .answer_context import (
    prepare_substation_answer_context,
    read_substation_evidence,
    search_substation_evidence,
)
from .diagnosis import (
    assess_substation_event_chain,
    build_substation_case_context,
    format_substation_conclusion,
    judge_substation_case,
)
from .kb import SubstationKb, SubstationResolver
from .parser import parse_substation_events
from .runtime import RuntimeMemoryAdapter, _upsert_runtime_rows
from .utils import _json_result

def register(registry: FunctionRegistry, repository: ObjectRepository, ontology: Ontology):
    domain_dir = Path(__file__).resolve().parent.parent
    kb = SubstationKb(domain_dir)
    registry.register_resolver("substation_event_kb", SubstationResolver(kb))
    registry.register_adapter("runtime_memory", RuntimeMemoryAdapter.factory(domain_dir))

    fn_map = {
        "search_substation_evidence": lambda **kw: search_substation_evidence(
            query=kw.get("query", "") or "",
            category=kw.get("category", "") or "",
            doc_role=kw.get("doc_role", "") or "",
            evidence_role=kw.get("evidence_role", "") or "",
            limit=kw.get("limit", 8) or 8,
            kb=kb,
        ),
        "prepare_substation_answer_context": lambda **kw: prepare_substation_answer_context(
            query=kw.get("query", "") or "",
            category=kw.get("category", "") or "",
            limit_evidence=kw.get("limit_evidence", 5) or 5,
            kb=kb,
        ),
        "read_substation_evidence": lambda **kw: read_substation_evidence(
            document_id=kw.get("document_id", "") or "",
            path=kw.get("path", "") or "",
            chunk_id=kw.get("chunk_id", "") or "",
            heading=kw.get("heading", "") or "",
            max_chars=kw.get("max_chars", 6000) or 6000,
            kb=kb,
        ),
        "parse_substation_events": lambda **kw: parse_substation_events(
            event_text=kw.get("event_text", "") or "",
            station_name=kw.get("station_name", "") or "",
        ),
        "build_substation_case_context": lambda **kw: build_substation_case_context(
            event_text=kw.get("event_text", "") or "",
            events_json=kw.get("events_json", "") or "",
            evidence_query=kw.get("evidence_query", "") or "",
            limit_evidence=kw.get("limit_evidence", 8) or 8,
            kb=kb,
            repository=repository,
        ),
        "judge_substation_case": lambda **kw: judge_substation_case(
            case_context=kw.get("case_context", "") or "",
            kb=kb,
            repository=repository,
        ),
        "format_substation_conclusion": lambda **kw: format_substation_conclusion(
            judgment=kw.get("judgment", "") or "",
            repository=repository,
        ),
        "assess_substation_event_chain": lambda **kw: assess_substation_event_chain(
            event_text=kw.get("event_text", "") or "",
            events_json=kw.get("events_json", "") or "",
            evidence_query=kw.get("evidence_query", "") or "",
            limit_evidence=kw.get("limit_evidence", 8) or 8,
            kb=kb,
            repository=repository,
        ),
        "sync_substation_kb": lambda **kw: kb.sync(),
    }

    for name, fn in fn_map.items():
        func_def = ontology.functions.get(name)
        if func_def:
            def _wrap(_fn=fn, _name=name, **kw):
                result = _fn(**kw)
                if _name == "parse_substation_events" and isinstance(result, dict):
                    parsed = result.get("events", [])
                    chain = result.get("event_chain")
                    events_to_store = []
                    for row in parsed:
                        stored = dict(row)
                        if isinstance(stored.get("subsystems"), list):
                            stored["subsystems"] = ",".join(stored["subsystems"])
                        events_to_store.append(stored)
                    _upsert_runtime_rows(repository, "StationEvent", events_to_store)
                    if chain:
                        stored_chain = dict(chain)
                        if isinstance(stored_chain.get("involved_subsystems"), list):
                            stored_chain["involved_subsystems"] = ",".join(stored_chain["involved_subsystems"])
                        _upsert_runtime_rows(repository, "EventChain", [stored_chain])
                elif _name == "format_substation_conclusion" and isinstance(result, dict):
                    row = result.get("event_conclusion")
                    if isinstance(row, dict):
                        _upsert_runtime_rows(repository, "EventConclusion", [row])
                return _json_result(result)

            registry.register(name, _wrap, func_def)


__all__ = [
    "assess_substation_event_chain",
    "build_substation_case_context",
    "format_substation_conclusion",
    "judge_substation_case",
    "parse_substation_events",
    "prepare_substation_answer_context",
    "read_substation_evidence",
    "register",
    "search_substation_evidence",
]
