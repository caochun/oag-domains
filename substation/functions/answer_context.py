from __future__ import annotations

from typing import Any

from .constants import PRIMARY_KNOWLEDGE_ROLE
from .evidence import _boost_station_evidence, _compact_search_result, _is_toc_chunk, _merge_evidence_chunk
from .kb import SubstationKb
from .parser import _weak_station_name
from .utils import _compact, _station_category, _truncate_text

def _question_intent(query: str) -> str:
    text = query or ""
    if any(k in text for k in ["是否异常", "是不是异常", "风险", "意味着什么", "说明什么", "判断", "研判"]):
        return "diagnosis"
    if any(k in text for k in ["是什么意思", "什么含义", "是什么", "解释", "概念", "A/B套", "A套", "B套", "双套"]):
        return "concept"
    if any(k in text for k in ["怎么处理", "如何处理", "处置", "预案", "应急"]):
        return "handling"
    return "document_answer"

def _scenario_queries(query: str, intent: str) -> list[dict[str, str]]:
    text = query or ""
    queries: list[dict[str, str]] = []
    if any(k in text for k in ["A/B套", "A套", "B套", "双套", "冗余", "值班", "备用", "active", "standby"]):
        queries.extend([
            {"query": "控制保护系统 双套 冗余 值班 备用 active standby 系统切换", "doc_role": "control_protection_spec"},
            {"query": "直流控制 主设备 冗余 两套系统 值班系统 备用系统 切换逻辑", "doc_role": "control_protection_spec"},
        ])
    if any(k in text for k in ["分接", "档位", "站用变", "降压变", "指令下发", "时间差"]):
        queries.extend([
            {"query": "分接开关 档位 不一致 控制系统 切换 信号回路", "doc_role": "emergency_procedure"},
            {"query": "换流变 分接开关 档位不一致 无法调节 备用控制系统", "doc_role": "emergency_procedure"},
            {"query": "分接开关 控制系统 冗余 指令 同步 A套 B套", "doc_role": "control_protection_spec"},
        ])
    if intent in {"diagnosis", "handling"}:
        queries.append({"query": f"{query} 设备异常 故障应急 处理 复归 跳闸 闭锁", "doc_role": "emergency_procedure"})
    queries.append({"query": query, "doc_role": ""})

    seen: set[tuple[str, str]] = set()
    unique = []
    for item in queries:
        key = (_compact(item["query"]), item.get("doc_role", ""))
        if key[0] and key not in seen:
            seen.add(key)
            unique.append({"query": key[0], "doc_role": key[1]})
    return unique[:6]

def _rank_answer_chunk(row: dict[str, Any], query: str, intent: str) -> tuple[float, float]:
    text = " ".join(str(row.get(key, "")) for key in ("title", "heading", "content", "snippet"))
    score = float(row.get("score") or 0)
    if _is_toc_chunk(row):
        score -= 200
    if intent == "concept":
        if row.get("doc_role") == "control_protection_spec":
            score += 80
        if any(k in text for k in ["冗余", "双重化", "两套系统", "值班", "备用", "Active", "Standby", "active", "standby"]):
            score += 50
        if row.get("doc_role") == PRIMARY_KNOWLEDGE_ROLE and not any(k in text for k in ["控制保护主机", "冗余", "双套"]):
            score -= 60
    elif intent in {"diagnosis", "handling"}:
        if row.get("doc_role") == PRIMARY_KNOWLEDGE_ROLE:
            score += 80
        if any(k in text for k in ["处理", "处理原则", "处理预案", "异常", "不一致", "无法调节"]):
            score += 35
    if any(k in query for k in ["分接", "档位", "站用变", "降压变"]):
        score += 60 if any(k in text for k in ["分接", "分接开关", "档位"]) else -50
    if any(k in query for k in ["A/B", "A套", "B套", "双套"]):
        if any(k in text for k in ["A套", "B套", "双套", "两套系统", "冗余", "值班", "备用", "Active", "Standby"]):
            score += 45
    return (score, float(row.get("score") or 0))

def _answer_context_constraints(intent: str) -> list[str]:
    base = [
        "回答必须区分文档明示内容、用户事件事实和推断。",
        "不得把检索到的异常处置场景反推为用户事件已经发生。",
        "引用 evidence[].source 中的文档标题和章节作为出处。",
    ]
    if intent == "concept":
        base.append("概念解释优先采用控制保护设计/说明书证据；第五分册仅作为异常处置补充。")
    if intent in {"diagnosis", "handling"}:
        base.append("异常性或处置结论优先采用第五分册；控保设计文档只解释机制。")
    return base

def _selection_reason(row: dict[str, Any], intent: str) -> str:
    if intent == "concept" and row.get("doc_role") == "control_protection_spec":
        return "控制保护设计/说明书用于解释机制和术语。"
    if row.get("doc_role") == PRIMARY_KNOWLEDGE_ROLE:
        return "第五分册用于异常研判和处置依据。"
    return "补充专业文档用于解释背景机制。"

def _answer_context_quality_notes(query: str, evidence: list[dict[str, Any]]) -> list[str]:
    notes = []
    if not evidence:
        return ["未检索到可用文档证据，回答应说明证据不足。"]
    if any(k in query for k in ["A/B", "A套", "B套", "双套"]) and not any(
        e["source"].get("doc_role") == "control_protection_spec" for e in evidence
    ):
        notes.append("未命中控制保护设计文档，A/B套概念解释证据可能不足。")
    if any(k in query for k in ["是否异常", "意味着什么", "异常", "处置", "处理"]) and not any(
        e["source"].get("doc_role") == PRIMARY_KNOWLEDGE_ROLE for e in evidence
    ):
        notes.append("未命中第五分册，异常性或处置判断证据不足。")
    return notes

def prepare_substation_answer_context(query: str, category: str = "", limit_evidence: int = 5,
                                      kb: SubstationKb | None = None) -> dict[str, Any]:
    if kb is None:
        return {"error": "kb not available"}
    intent = _question_intent(query)
    station_category = category or _station_category(_weak_station_name(query))
    merged: dict[str, dict[str, Any]] = {}
    searches = _scenario_queries(query, intent)
    for item in searches:
        result = kb.search(
            item["query"],
            category=station_category,
            doc_role=item.get("doc_role", ""),
            limit=max(5, limit_evidence),
            prefer_primary=intent in {"diagnosis", "handling"},
        )
        for chunk in result.get("chunks", []):
            _boost_station_evidence(chunk, station_category)
            _merge_evidence_chunk(merged, chunk)

    ranked = sorted(merged.values(), key=lambda row: _rank_answer_chunk(row, query, intent), reverse=True)
    evidence = []
    seen: set[tuple[str, str]] = set()

    def to_answer_evidence(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": {
                "chunk_id": row.get("chunk_id", ""),
                "document_id": row.get("document_id", ""),
                "title": row.get("title", ""),
                "heading": row.get("heading", ""),
                "doc_role": row.get("doc_role", ""),
                "md_path": row.get("md_path", ""),
                "score": row.get("score", 0),
            },
            "excerpt": _truncate_text(row.get("snippet") or row.get("content", ""), 700),
            "selection_reason": _selection_reason(row, intent),
        }

    for row in ranked:
        if _is_toc_chunk(row):
            continue
        key = (row.get("title", "") or row.get("document_id", ""), row.get("heading", ""))
        if key in seen:
            continue
        seen.add(key)
        evidence.append(to_answer_evidence(row))
        if len(evidence) >= limit_evidence:
            break

    if any(k in query for k in ["A/B", "A套", "B套", "双套"]) and not any(
        item["source"].get("doc_role") == "control_protection_spec" for item in evidence
    ):
        control_row = next(
            (
                row for row in ranked
                if row.get("doc_role") == "control_protection_spec"
                and not _is_toc_chunk(row)
                and (row.get("title", "") or row.get("document_id", ""), row.get("heading", "")) not in seen
            ),
            None,
        )
        if control_row:
            control_evidence = to_answer_evidence(control_row)
            if len(evidence) < limit_evidence:
                evidence.append(control_evidence)
            else:
                replace_idx = next(
                    (
                        idx for idx in range(len(evidence) - 1, -1, -1)
                        if evidence[idx]["source"].get("doc_role") != PRIMARY_KNOWLEDGE_ROLE
                    ),
                    len(evidence) - 1,
                )
                evidence[replace_idx] = control_evidence

    return {
        "query": query,
        "intent": intent,
        "evidence": evidence,
        "answer_guidance": _answer_context_constraints(intent),
        "quality_notes": _answer_context_quality_notes(query, evidence),
        "search_plan": searches,
        "compact": True,
    }

def search_substation_evidence(query: str, category: str = "", doc_role: str = "",
                               evidence_role: str = "", limit: int = 8,
                               kb: SubstationKb | None = None) -> dict[str, Any]:
    if kb is None:
        return {"error": "kb not available"}
    safe_limit = max(1, min(int(limit or 8), 8))
    result = kb.search(
        query,
        category=category,
        doc_role=doc_role,
        evidence_role=evidence_role,
        limit=safe_limit,
        prefer_primary=doc_role == PRIMARY_KNOWLEDGE_ROLE,
    )
    return _compact_search_result(result)

def read_substation_evidence(document_id: str = "", path: str = "", chunk_id: str = "",
                             heading: str = "", max_chars: int = 6000,
                             kb: SubstationKb | None = None) -> dict[str, Any]:
    if kb is None:
        return {"error": "kb not available"}
    return kb.read(document_id=document_id, path=path, chunk_id=chunk_id, heading=heading, max_chars=max_chars)
