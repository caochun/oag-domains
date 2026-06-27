from __future__ import annotations

import re
from typing import Any

from .constants import (
    ABNORMAL_TERMS,
    HIGH_VALUE_EVIDENCE_BOOST,
    NORMAL_TERMS,
    PRIMARY_EVIDENCE_LIMIT,
    PRIMARY_KNOWLEDGE_ROLE,
    SEVERE_TERMS,
    STATION_EVIDENCE_BOOST,
    TOC_HEADINGS,
)
from .utils import _compact, _truncate_text

def _evidence_role(text: str) -> str:
    if any(k in text for k in ["处理预案", "处理原则", "事故处理", "异常处理", "故障应急"]):
        return "handling_rule"
    if any(k in text for k in ["保护动作", "闭锁", "跳闸", "低电压", "故障穿越"]):
        return "severe_signal"
    if any(k in text for k in ["异常", "告警", "轻微故障", "自监视"]):
        return "abnormal_signal"
    if any(k in text for k in ["解锁", "功率控制", "滤波器", "分接", "运行方式", "操作"]):
        return "normal_operation"
    if any(k in text for k in ["录波", "SER", "事件记录"]):
        return "event_record"
    if IMAGE_RE.search(text):
        return "figure_context"
    return "definition"

def _is_toc_chunk(row: dict[str, Any]) -> bool:
    heading = _compact(row.get("heading", "")).lower()
    content = _compact(row.get("content", ""))
    if heading in TOC_HEADINGS:
        return True
    if "目 录" in heading or heading == "目录":
        return True
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if lines and len(lines) <= 30:
        numbered = sum(1 for line in lines if re.search(r"\.{2,}\s*\d+$|第\s*\d+\s*页|^\d+(\.\d+){1,}", line))
        if numbered >= max(3, len(lines) // 3):
            return True
    return False

def _evidence_ref(row: dict[str, Any]) -> dict[str, Any]:
    ref = {
        "chunk_id": row.get("chunk_id", ""),
        "document_id": row.get("document_id", ""),
        "title": row.get("title", ""),
        "heading": row.get("heading", ""),
        "doc_role": row.get("doc_role", ""),
        "evidence_role": row.get("evidence_role", ""),
        "knowledge_priority": row.get("knowledge_priority", ""),
        "md_path": row.get("md_path", ""),
        "score": row.get("score", 0),
    }
    if row.get("scenario_match"):
        ref["scenario_match"] = row.get("scenario_match")
    if row.get("matched_keywords"):
        ref["matched_keywords"] = row.get("matched_keywords")
    return ref

def _select_evidence_refs(evidence: dict[str, Any], limit: int = PRIMARY_EVIDENCE_LIMIT,
                          features: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    chunks = evidence.get("chunks", []) if isinstance(evidence, dict) else []
    if not isinstance(chunks, list):
        return []
    features = features or {}

    def scenario_hit(row: dict[str, Any]) -> int:
        text = " ".join(str(row.get(key, "")) for key in ("title", "heading", "content"))
        heading = str(row.get("heading", ""))
        if heading in {"前言", "目录", "目 录"}:
            return 0
        if features.get("has_transformer_tap_change") and any(k in text for k in ["分接头", "分接开关", "档位"]):
            return 1
        if features.get("has_valve_cooling_abnormal") and any(k in text for k in ["阀冷", "内水冷", "主循环泵"]):
            return 1
        if features.get("has_dc_line_fault_ride_through") and any(k in text for k in ["直流线路", "低电压", "故障穿越", "再启动"]):
            return 1
        return 0

    def rank(row: dict[str, Any]) -> tuple[int, int, int, int, int, float]:
        primary = 1 if row.get("doc_role") == PRIMARY_KNOWLEDGE_ROLE else 0
        same_station = 1 if row.get("station_priority") == "same_station" else 0
        useful_role = 1 if row.get("evidence_role") in {"handling_rule", "severe_signal", "abnormal_signal"} else 0
        body = 0 if _is_toc_chunk(row) else 1
        return (primary, same_station, scenario_hit(row), body, useful_role, float(row.get("score") or 0))

    sorted_chunks = sorted((c for c in chunks if isinstance(c, dict)), key=rank, reverse=True)
    seen_refs: set[tuple[str, str]] = set()
    body_refs = []
    scenario_refs = []
    for row in sorted_chunks:
        if _is_toc_chunk(row):
            continue
        hit = scenario_hit(row)
        if hit:
            if features.get("has_transformer_tap_change"):
                row["scenario_match"] = "transformer_tap_change"
                row["matched_keywords"] = [k for k in ["分接头", "分接开关", "档位"] if k in " ".join(str(row.get(key, "")) for key in ("title", "heading", "content"))]
            elif features.get("has_valve_cooling_abnormal"):
                row["scenario_match"] = "valve_cooling_abnormal"
            elif features.get("has_dc_line_fault_ride_through"):
                row["scenario_match"] = "dc_line_fault_ride_through"
        ref = _evidence_ref(row)
        key = (ref.get("title", ""), ref.get("heading", ""))
        if key in seen_refs:
            continue
        seen_refs.add(key)
        body_refs.append(ref)
        if hit:
            scenario_refs.append(ref)
    if features.get("has_transformer_tap_change") and scenario_refs:
        return scenario_refs[:limit]
    if body_refs:
        return body_refs[:limit]
    return [_evidence_ref(row) for row in sorted_chunks[:limit]]

def _evidence_summary(refs: list[dict[str, Any]]) -> str:
    if not refs:
        return "未检索到可直接支撑结论的正文证据。"
    parts = []
    for ref in refs[:PRIMARY_EVIDENCE_LIMIT]:
        role = "第五分册" if ref.get("doc_role") == PRIMARY_KNOWLEDGE_ROLE else "补充文档"
        title = ref.get("title") or ref.get("document_id") or "未命名文档"
        heading = ref.get("heading") or "未命名章节"
        parts.append(f"{role}《{title}》“{heading}”")
    return "；".join(parts)

def _compact_chunk(row: dict[str, Any], max_chars: int = 520) -> dict[str, Any]:
    text = row.get("snippet") or row.get("content") or ""
    return {
        "chunk_id": row.get("chunk_id", ""),
        "document_id": row.get("document_id", ""),
        "title": row.get("title", ""),
        "category": row.get("category", ""),
        "md_path": row.get("md_path", ""),
        "heading": row.get("heading", ""),
        "doc_role": row.get("doc_role", ""),
        "evidence_role": row.get("evidence_role", ""),
        "knowledge_priority": row.get("knowledge_priority", ""),
        "score": row.get("score", 0),
        "snippet": _truncate_text(text, max_chars),
    }

def _compact_search_result(result: dict[str, Any], max_chars: int = 520) -> dict[str, Any]:
    chunks = result.get("chunks", []) if isinstance(result, dict) else []
    return {
        "query": result.get("query", ""),
        "terms": result.get("terms", []),
        "chunks": [_compact_chunk(chunk, max_chars=max_chars) for chunk in chunks if isinstance(chunk, dict)],
        "usage_note": "这些是候选片段。若用户要答案本身，优先使用 prepare_substation_answer_context；必要时再用 read_substation_evidence 核对原文。",
    }

def _boost_station_evidence(chunk: dict[str, Any], station_category: str) -> None:
    if not station_category:
        return
    category = str(chunk.get("category", ""))
    title = str(chunk.get("title", ""))
    path = str(chunk.get("md_path", ""))
    if station_category and (station_category in category or station_category in title or f"/{station_category}/" in path):
        chunk["score"] = round(float(chunk.get("score") or 0) + STATION_EVIDENCE_BOOST, 3)
        chunk["station_priority"] = "same_station"

def _merge_evidence_chunk(merged: dict[str, dict[str, Any]], chunk: dict[str, Any]) -> None:
    chunk_id = chunk.get("chunk_id")
    if not chunk_id:
        return
    existing = merged.get(chunk_id)
    if not existing or float(chunk.get("score") or 0) > float(existing.get("score") or 0):
        merged[chunk_id] = chunk
