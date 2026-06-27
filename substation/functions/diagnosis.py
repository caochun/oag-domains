from __future__ import annotations

import json
import re
from typing import Any

from oag.ontology.repository import ObjectRepository

from .constants import EVIDENCE_QUERY_TERMS, PRIMARY_KNOWLEDGE_ROLE
from .evidence import _boost_station_evidence, _evidence_summary, _merge_evidence_chunk, _select_evidence_refs
from .features import (
    _chain_features, _chain_text, _coerce_event_input, _event_query_from_raw,
    _observed_facts, _select_events, _timeline,
)
from .kb import SubstationKb
from .runtime import _upsert_runtime_rows
from .utils import _compact, _make_conclusion_id, _station_category, _truncate_text

def build_substation_case_context(event_text: str = "", events_json: str = "",
                                  evidence_query: str = "", limit_evidence: int = 8,
                                  kb: SubstationKb | None = None,
                                  repository: ObjectRepository | None = None) -> dict[str, Any]:
    parsed = _coerce_event_input(event_text, events_json)
    events = parsed.get("events", [])
    if not events:
        return {
            "error": "未能解析事件链",
            "parsed": parsed,
            "case_context": None,
        }
    _upsert_runtime_rows(repository, "StationEvent", [
        {**event, "subsystems": ",".join(event["subsystems"]) if isinstance(event.get("subsystems"), list) else event.get("subsystems", "")}
        for event in events
    ])
    features = _chain_features(events)
    effective_evidence_query = evidence_query or _event_query_from_raw(events)
    evidence = _collect_evidence(
        kb,
        features,
        effective_evidence_query,
        limit_evidence,
        station_category=_station_category((parsed.get("event_chain") or {}).get("station_name", "")),
    ) if kb else {}
    case_context = {
        "event_chain": parsed.get("event_chain"),
        "timeline": _timeline(events),
        "chain_features": features,
        "main_events": _select_events(events, "main"),
        "companion_events": _select_events(events, "companion"),
        "abnormal_signals": _select_events(events, "abnormal"),
        "evidence": evidence,
        "answer_guidance": [
            "先给结构化结论，再用简短文字解释。",
            "区分用户给出的事件事实、文档证据、智能体推断和仍需核查的数据。",
            "event_nature 判断是否符合运行/保护逻辑，risk_level 表示潜在风险强度，两者不要混同。",
        ],
    }
    chain = parsed.get("event_chain")
    if isinstance(chain, dict):
        stored_chain = dict(chain)
        if isinstance(stored_chain.get("involved_subsystems"), list):
            stored_chain["involved_subsystems"] = ",".join(stored_chain["involved_subsystems"])
        _upsert_runtime_rows(repository, "EventChain", [stored_chain])
    return {
        "parsed": parsed,
        "case_context": case_context,
    }

def judge_substation_case(case_context: str = "", kb: SubstationKb | None = None,
                          repository: ObjectRepository | None = None) -> dict[str, Any]:
    if isinstance(case_context, dict):
        data = case_context
    else:
        case_context = str(case_context or "")
        if case_context and not case_context.lstrip().startswith(("{", "[")):
            maybe_json = re.search(r"(\{.*\})", case_context, re.S)
            if maybe_json:
                case_context = maybe_json.group(1)
        try:
            data = json.loads(case_context) if case_context else {}
        except json.JSONDecodeError:
            return {"error": "case_context 不是有效 JSON"}
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return {"error": "case_context 不是有效 JSON"}
    if isinstance(data, dict) and isinstance(data.get("case_context"), str):
        try:
            data["case_context"] = json.loads(data["case_context"])
        except json.JSONDecodeError:
            pass
    context = data.get("case_context") if isinstance(data, dict) else None
    if not context and isinstance(data, dict) and {"event_chain", "timeline", "chain_features"} & set(data):
        context = data
        data = {"case_context": context, "parsed": {"events": data.get("events", [])}}
    if not context:
        return {"error": "缺少 case_context"}
    events = []
    parsed = data.get("parsed") if isinstance(data, dict) else None
    if isinstance(parsed, dict):
        events = parsed.get("events", [])
    if not events:
        return {
            "error": "case_context 缺少事件",
            "case_context": context,
        }
    features = context.get("chain_features", {})
    evidence_refs = _select_evidence_refs(context.get("evidence", {}), features=features)
    evidence_summary = _evidence_summary(evidence_refs)
    observed_facts = _observed_facts(events, features)
    abnormal_triggers = _abnormal_triggers(features, evidence_refs)
    trigger_match = _match_triggers(events, features, abnormal_triggers)
    conclusion = _make_conclusion(
        events,
        features,
        evidence_refs,
        evidence_summary,
        observed_facts,
        abnormal_triggers,
        trigger_match,
    )
    conclusion_id = _make_conclusion_id(
        json.dumps(conclusion, ensure_ascii=False)
        + json.dumps(features, ensure_ascii=False)
        + json.dumps(evidence_refs, ensure_ascii=False)
    )
    return {
        "case_context": context,
        "conclusion_id": conclusion_id,
        "judgment": {
            "conclusion_id": conclusion_id,
            "event_description": conclusion["event_description"],
            "event_type": conclusion["event_type"],
            "event_nature": conclusion["event_nature"],
            "risk_level": conclusion["risk_level"],
            "reason": conclusion["reason"],
            "observed_facts": observed_facts,
            "abnormal_triggers": abnormal_triggers,
            "trigger_match": trigger_match,
            "evidence_refs": evidence_refs,
            "evidence_summary": evidence_summary,
        },
    }

def format_substation_conclusion(judgment: str = "", repository: ObjectRepository | None = None) -> dict[str, Any]:
    try:
        data = json.loads(judgment) if judgment else {}
    except json.JSONDecodeError:
        return {"error": "judgment 不是有效 JSON"}
    judgment_data = data.get("judgment") if isinstance(data, dict) else None
    if not judgment_data:
        return {"error": "缺少 judgment"}
    _upsert_runtime_rows(repository, "EventConclusion", [judgment_data])
    return {
        "structured_conclusion": judgment_data,
        "event_conclusion": judgment_data,
    }

def assess_substation_event_chain(event_text: str = "", events_json: str = "",
                                  evidence_query: str = "", limit_evidence: int = 8,
                                  kb: SubstationKb | None = None,
                                  repository: ObjectRepository | None = None) -> dict[str, Any]:
    context_result = build_substation_case_context(
        event_text=event_text,
        events_json=events_json,
        evidence_query=evidence_query,
        limit_evidence=limit_evidence,
        kb=kb,
        repository=repository,
    )
    if "error" in context_result:
        return context_result
    judgment_result = judge_substation_case(
        case_context=json.dumps(context_result, ensure_ascii=False),
        kb=kb,
        repository=repository,
    )
    if "error" in judgment_result:
        return judgment_result
    formatted = format_substation_conclusion(
        judgment=json.dumps(judgment_result, ensure_ascii=False),
        repository=repository,
    )
    if "error" in formatted:
        return formatted
    return _compact_assessment_result(context_result, judgment_result, formatted)

def _compact_assessment_result(context_result: dict[str, Any],
                               judgment_result: dict[str, Any],
                               formatted: dict[str, Any]) -> dict[str, Any]:
    parsed = context_result.get("parsed") if isinstance(context_result, dict) else {}
    context = context_result.get("case_context") if isinstance(context_result, dict) else {}
    events = parsed.get("events", []) if isinstance(parsed, dict) else []
    chain = parsed.get("event_chain") if isinstance(parsed, dict) else None
    features = context.get("chain_features", {}) if isinstance(context, dict) else {}
    conclusion = formatted.get("structured_conclusion", {})
    result = {
        "structured_conclusion": conclusion,
        "event_summary": _compact_event_summary(events, chain),
        "feature_summary": _compact_feature_summary(features),
        "parse_note": parsed.get("parse_note", "") if isinstance(parsed, dict) else "",
        "conclusion_id": judgment_result.get("conclusion_id", ""),
        "result_note": "已省略完整事件原文、case_context 和全文证据片段，避免长事件链占用模型上下文；最终回答应以 structured_conclusion、event_summary 和 evidence_refs 为准。",
    }
    result["answer_md"] = _render_assessment_answer(
        conclusion,
        result["event_summary"],
        result.get("feature_summary", {}),
    )
    return {key: value for key, value in result.items() if value not in ("", None, [], {})}

def _compact_event_summary(events: list[dict[str, Any]], chain: dict[str, Any] | None,
                           max_events: int = 8, max_chars: int = 180) -> dict[str, Any]:
    chain = chain or {}
    items = []
    for event in events[:max_events]:
        text = event.get("summary") or event.get("raw_text") or ""
        items.append({
            "sequence": event.get("sequence"),
            "start_time": event.get("start_time", ""),
            "end_time": event.get("end_time", ""),
            "station_name": event.get("station_name", ""),
            "device_name": event.get("device_name", ""),
            "event_type": event.get("event_type", ""),
            "alarm_level": event.get("alarm_level", 99),
            "summary": _truncate_text(text, max_chars),
            "parse_quality": event.get("parse_quality", ""),
        })
    return {
        "event_count": len(events),
        "window_start": chain.get("window_start", ""),
        "window_end": chain.get("window_end", ""),
        "station_name": chain.get("station_name", ""),
        "involved_subsystems": chain.get("involved_subsystems", []),
        "sample_events": items,
        "omitted_event_count": max(0, len(events) - len(items)),
    }

def _compact_feature_summary(features: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "has_dc_line_fault_ride_through",
        "has_transformer_tap_change",
        "has_protection_action",
        "has_abnormal_signal",
        "has_severe_signal",
        "has_negative_outcome",
        "has_normal_completion",
        "has_recorder",
        "has_operation_sequence",
        "has_control_response_difference",
        "has_tap_position_inconsistency",
        "has_tap_software_jump",
        "min_alarm_level",
        "max_alarm_level",
        "keywords",
    ]
    return {key: features.get(key) for key in keys if key in features}

def _render_assessment_answer(conclusion: dict[str, Any],
                              event_summary: dict[str, Any],
                              feature_summary: dict[str, Any]) -> str:
    lines = [
        "## 事件链研判结论",
        "",
        f"- 事件性质：{conclusion.get('event_nature', '证据不足')}",
        f"- 风险等级：{conclusion.get('risk_level', '未知')}",
        f"- 事件类型：{conclusion.get('event_type', '未归类')}",
        f"- 概括：{conclusion.get('event_description', '')}",
        "",
        "### 事件事实",
        f"- 时间窗口：{event_summary.get('window_start', '')} 至 {event_summary.get('window_end', '')}",
        f"- 事件数量：{event_summary.get('event_count', 0)}",
    ]
    omitted = event_summary.get("omitted_event_count", 0)
    sample_events = event_summary.get("sample_events", []) or []
    if sample_events:
        lines.append("- 主要片段：")
        for item in sample_events[:5]:
            lines.append(
                f"  - {item.get('sequence')}. {item.get('event_type') or '事件'}：{item.get('summary', '')}"
            )
        hidden_count = max(0, int(event_summary.get("event_count", 0) or 0) - min(5, len(sample_events)))
        if hidden_count:
            lines.append(f"  - 另有 {hidden_count} 条事件已纳入工具侧特征提取，未在此展开。")

    matches = conclusion.get("trigger_match", []) or []
    matched = [m for m in matches if m.get("status") == "matched"]
    related = [m for m in matches if m.get("status") == "related_but_not_matched"]
    not_matched = [m for m in matches if m.get("status") == "not_matched"]

    lines.extend(["", "### 触发条件匹配"])
    if matched:
        lines.append("- 已命中：")
        for item in matched[:4]:
            lines.append(f"  - {item.get('trigger_id')}: {item.get('basis')}")
    if related:
        lines.append("- 相关但未命中：")
        for item in related[:4]:
            lines.append(f"  - {item.get('trigger_id')}: {item.get('basis')}")
    if not_matched:
        lines.append("- 明确未命中：")
        for item in not_matched[:4]:
            lines.append(f"  - {item.get('trigger_id')}: {item.get('basis')}")
    if not matched and not related and not not_matched:
        lines.append("- 当前证据不足以匹配具体触发条件。")

    reason = conclusion.get("reason", "")
    if reason:
        lines.extend(["", "### 判断依据", f"- {reason}"])

    evidence_summary = conclusion.get("evidence_summary", "")
    if evidence_summary:
        lines.extend(["", "### 文档依据", f"- {evidence_summary}"])

    checks = _recommended_checks(conclusion, feature_summary, matches)
    if checks:
        lines.extend(["", "### 建议核查"])
        lines.extend(f"- {item}" for item in checks)

    lines.extend([
        "",
        "### 结构化结论",
        "```json",
        json.dumps(
            {
                "event_description": conclusion.get("event_description", ""),
                "event_type": conclusion.get("event_type", ""),
                "event_nature": conclusion.get("event_nature", ""),
                "risk_level": conclusion.get("risk_level", ""),
                "reason": conclusion.get("reason", ""),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
    ])
    return "\n".join(line for line in lines if line is not None)

def _recommended_checks(conclusion: dict[str, Any],
                        feature_summary: dict[str, Any],
                        matches: list[dict[str, Any]]) -> list[str]:
    checks: list[str] = []
    if feature_summary.get("has_recorder"):
        checks.append("调阅同一时间窗录波，确认是否存在电压、电流或保护出口扰动。")
    if feature_summary.get("has_control_response_difference"):
        checks.append("核查A/B套主机通信链路、时钟同步、CPU/内存负荷和主备状态一致性。")
    if feature_summary.get("has_tap_software_jump") or any(
        m.get("trigger_id", "").startswith("tap_") and m.get("status") in {"matched", "related_but_not_matched"}
        for m in matches
    ):
        checks.append("核对分接开关机械档位、OWS/软件档位和信号回路，确认无持续不一致。")
    if conclusion.get("event_nature") == "需核查" and not checks:
        checks.append("结合实时量、告警复归状态和现场设备状态做一次闭环核查。")
    return checks[:4]

def _evidence_ref_ids(evidence_refs: list[dict[str, Any]], *keywords: str) -> list[str]:
    ids = []
    for ref in evidence_refs:
        text = " ".join(str(ref.get(key, "")) for key in ("title", "heading", "doc_role", "evidence_role", "scenario_match"))
        matched_keywords = ref.get("matched_keywords") if isinstance(ref.get("matched_keywords"), list) else []
        if not keywords or any(keyword in text for keyword in keywords) or any(keyword in matched_keywords for keyword in keywords):
            chunk_id = str(ref.get("chunk_id") or "")
            if chunk_id:
                ids.append(chunk_id)
    return ids[:3]

def _abnormal_triggers(features: dict[str, Any], evidence_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    triggers: list[dict[str, Any]] = []
    if features.get("has_transformer_tap_change"):
        tap_refs = _evidence_ref_ids(evidence_refs, "分接", "档位")
        triggers.extend([
            {
                "trigger_id": "tap_inconsistency_alarm",
                "trigger_category": "tap_position",
                "compatible_fact_categories": ["tap_position"],
                "description": "换流变分接头不一致、三相或同阀组档位不同步等告警。",
                "source": "第五分册设备异常与故障应急处理",
                "evidence_ref_ids": tap_refs,
            },
            {
                "trigger_id": "tap_adjustment_failed_or_unfinished",
                "trigger_category": "operation_failure",
                "compatible_fact_categories": ["negative_outcome"],
                "description": "分接开关无法调节、拒动、操作失败、未完成或告警不复归。",
                "source": "第五分册设备异常与故障应急处理",
                "evidence_ref_ids": tap_refs,
            },
            {
                "trigger_id": "tap_mechanism_abnormal",
                "trigger_category": "tap_mechanism",
                "compatible_fact_categories": ["tap_mechanism"],
                "description": "分接开关传动机构异常、变形、脱扣、滑档或机构箱异常。",
                "source": "第五分册设备异常与故障应急处理",
                "evidence_ref_ids": tap_refs,
            },
            {
                "trigger_id": "tap_position_difference_over_2",
                "trigger_category": "tap_position",
                "compatible_fact_categories": ["tap_position"],
                "description": "分接头档位差超过规程允许范围，或相间档位差持续存在。",
                "source": "第五分册设备异常与故障应急处理",
                "evidence_ref_ids": tap_refs,
            },
            {
                "trigger_id": "tap_motor_power_trip",
                "trigger_category": "power_supply",
                "compatible_fact_categories": ["power_supply", "negative_outcome"],
                "description": "分接开关电机电源小开关跳开等电源异常。",
                "source": "第五分册设备异常与故障应急处理",
                "evidence_ref_ids": tap_refs,
            },
        ])
    triggers.extend([
        {
            "trigger_id": "protection_action",
            "trigger_category": "protection_action",
            "compatible_fact_categories": ["protection_action"],
            "description": "保护动作、保护出口或与故障穿越相关的控制保护响应。",
            "source": "事件链通用异常/风险规则",
            "evidence_ref_ids": _evidence_ref_ids(evidence_refs, "保护", "故障穿越", "低电压"),
        },
        {
            "trigger_id": "trip_lockout_or_negative_outcome",
            "trigger_category": "negative_outcome",
            "compatible_fact_categories": ["negative_outcome"],
            "description": "跳闸、极闭锁、停运、失败、持续告警或未复归等后果性线索。",
            "source": "事件链通用异常/风险规则",
            "evidence_ref_ids": _evidence_ref_ids(evidence_refs, "闭锁", "跳闸", "停运", "告警"),
        },
    ])
    return triggers

def _match_triggers(events: list[dict[str, Any]], features: dict[str, Any],
                    triggers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = _chain_text(events)
    matches = []
    has_control_related_tap = (
        features.get("has_transformer_tap_change")
        and features.get("has_control_response_difference")
    )
    for trigger in triggers:
        trigger_id = trigger.get("trigger_id", "")
        status = "unknown"
        basis = "当前事件摘要不足以确认该触发条件。"
        fact_ids: list[str] = []
        if trigger_id == "tap_inconsistency_alarm":
            if features.get("has_tap_position_inconsistency"):
                status = "matched"
                basis = "事件摘要出现分接头/档位不一致、不同步或档位差等同类事实。"
                fact_ids = ["transformer_tap_change", "tap_position_inconsistency"]
            elif has_control_related_tap:
                status = "related_but_not_matched"
                basis = "事件中A/B套指令或响应差异与分接调节相关，但它属于控制响应差异，不等同于分接头/档位不一致。"
                fact_ids = ["transformer_tap_change", "control_response_difference"]
            elif features.get("has_transformer_tap_change") and features.get("has_normal_completion"):
                status = "not_matched"
                basis = "事件仅描述分接开关操作和档位调整完成，未见不一致或档位差告警。"
                fact_ids = ["transformer_tap_change", "normal_completion"]
        elif trigger_id == "tap_adjustment_failed_or_unfinished":
            if re.search(r"无法调节|拒动|操作失败|失败|未完成|未复归|持续告警", text):
                status = "matched"
                basis = "事件摘要出现无法调节、失败、未完成、未复归或持续告警线索。"
                fact_ids = ["negative_outcome"]
            elif features.get("has_transformer_tap_change") and features.get("has_normal_completion"):
                status = "not_matched"
                basis = "事件摘要明确出现操作完成或完成类线索，未见失败或不复归。"
                fact_ids = ["transformer_tap_change", "normal_completion"]
        elif trigger_id == "tap_mechanism_abnormal":
            if re.search(r"传动机构|机构异常|机构箱异常|变形|脱扣|滑档", text):
                status = "matched"
                basis = "事件摘要出现机构异常、变形、脱扣或滑档相关线索。"
                fact_ids = ["transformer_tap_change", "abnormal_signal"]
            elif features.get("has_transformer_tap_change"):
                status = "not_matched"
                basis = "事件摘要未见传动机构异常、变形、脱扣或滑档描述。"
                fact_ids = ["transformer_tap_change"]
        elif trigger_id == "tap_position_difference_over_2":
            if re.search(r"档位差.{0,6}超过|超过\s*2\s*档|相差\s*[3-9]\s*档", text):
                status = "matched"
                basis = "事件摘要出现档位差超过阈值的线索。"
                fact_ids = ["transformer_tap_change", "abnormal_signal"]
            elif has_control_related_tap:
                status = "related_but_not_matched"
                basis = "事件中A/B套指令时间差与分接操作相关，但未出现相间档位差、超过2档或持续不同步等同类事实。"
                fact_ids = ["transformer_tap_change", "control_response_difference"]
            elif features.get("has_transformer_tap_change") and features.get("has_normal_completion"):
                status = "not_matched"
                basis = "事件摘要只给出一次档位调整完成，未见档位差超过阈值或持续不同步。"
                fact_ids = ["transformer_tap_change", "normal_completion"]
        elif trigger_id == "tap_motor_power_trip":
            if re.search(r"电机电源.{0,12}(跳|跳开|跳闸)|小开关.{0,8}(跳|跳开|跳闸)", text):
                status = "matched"
                basis = "事件摘要出现分接开关电机电源或小开关跳开线索。"
                fact_ids = ["transformer_tap_change", "negative_outcome"]
            elif features.get("has_transformer_tap_change"):
                status = "not_matched"
                basis = "事件摘要未见电机电源小开关跳开或电源异常描述。"
                fact_ids = ["transformer_tap_change"]
        elif trigger_id == "protection_action":
            status = "matched" if features.get("has_protection_action") else "not_matched"
            basis = "事件摘要中出现保护动作相关表述。" if status == "matched" else "事件摘要未见保护动作、保护出口或故障穿越表述。"
            fact_ids = ["protection_action"]
        elif trigger_id == "trip_lockout_or_negative_outcome":
            status = "matched" if features.get("has_negative_outcome") else "not_matched"
            basis = "事件摘要中存在跳闸、闭锁、失败、持续或未复归等后果线索。" if status == "matched" else "事件摘要未见跳闸、闭锁、失败、持续或未复归等后果线索。"
            fact_ids = ["negative_outcome"]
        matches.append({
            "trigger_id": trigger_id,
            "status": status,
            "basis": basis,
            "matched_fact_ids": fact_ids,
        })
    return matches

def _make_conclusion(events: list[dict[str, Any]], features: dict[str, Any],
                     evidence_refs: list[dict[str, Any]],
                     evidence_summary: str,
                     observed_facts: list[dict[str, Any]],
                     abnormal_triggers: list[dict[str, Any]],
                     trigger_match: list[dict[str, Any]]) -> dict[str, str]:
    event_type = _synthesize_event_type(features)
    description = _synthesize_description(events, features, event_type)

    if features["has_negative_outcome"]:
        event_nature = "异常"
    elif features["has_dc_line_fault_ride_through"] and features["has_normal_completion"]:
        event_nature = "正常"
    elif features["has_protection_action"] and features["has_normal_completion"] and not features["has_negative_outcome"]:
        event_nature = "正常"
    elif features["has_abnormal_signal"]:
        event_nature = "需核查"
    elif features["has_operation_sequence"] or features["has_normal_completion"]:
        event_nature = "正常"
    else:
        event_nature = "证据不足"

    if features["has_dc_line_fault_ride_through"] or features["has_protection_action"]:
        risk_level = "严重"
    elif features["has_severe_signal"] or features["min_alarm_level"] <= 2:
        risk_level = "较高"
    elif features["has_abnormal_signal"]:
        risk_level = "一般"
    else:
        risk_level = "低"

    reason = _synthesize_reason(
        features,
        event_nature,
        risk_level,
        evidence_refs,
        evidence_summary,
        trigger_match,
    )
    return {
        "event_description": description,
        "event_type": event_type,
        "event_nature": event_nature,
        "risk_level": risk_level,
        "reason": reason,
        "observed_facts": observed_facts,
        "abnormal_triggers": abnormal_triggers,
        "trigger_match": trigger_match,
        "evidence_summary": evidence_summary,
    }

def _synthesize_event_type(features: dict[str, Any]) -> str:
    systems = set(features["subsystems"])
    if features["has_dc_line_fault_ride_through"]:
        return "直流线路故障穿越及控保系统切换" if features["has_control_switch"] else "直流线路保护动作及故障穿越"
    if features.get("has_valve_cooling_abnormal"):
        return "阀冷系统异常及伴随事件"
    if features.get("has_transformer_tap_change"):
        return "换流变分接开关档位调整"
    if "ac_filter" in systems and "station_power" in systems:
        return "换流器解锁及交流滤波器/站用电联动操作"
    if "dc_control" in systems and features["has_operation_sequence"]:
        return "直流控制操作及伴随事件"
    if features["has_recorder"]:
        return "故障录波启动及伴随告警"
    if features["has_abnormal_signal"]:
        return "二次设备异常告警"
    return "站内连续运行事件"

def _synthesize_description(events: list[dict[str, Any]], features: dict[str, Any], event_type: str) -> str:
    station = events[0].get("station_name") or "该站"
    text = " ".join(f"{e.get('event_type','')} {e.get('summary','')} {e.get('raw_text','')}" for e in events)
    phrases: list[str] = []
    if "低电压保护" in text:
        phrases.append("直流线路低电压保护动作并复归")
    if "再启动" in text:
        phrases.append("故障穿越再启动命令短时出现")
    if features["has_control_switch"]:
        phrases.append("控保系统发生A/B套切换或差异响应")
    if "解锁" in text:
        phrases.append("换流器解锁过程完成")
    if "滤波器" in text:
        phrases.append("交流滤波器及无功设备投退")
    if "分接" in text:
        phrases.append("换流变/变压器分接头调节")
    if features["has_recorder"]:
        phrases.append("多个录波启动信号短时出现后消失")
    if features["has_abnormal_signal"] and "自监视" in text:
        phrases.append("测控自监视异常及轻微故障告警出现后复归")
    if not phrases:
        phrases.append(f"发生{event_type}")
    return f"{station}" + "，".join(phrases) + "。"

def _synthesize_reason(features: dict[str, Any], event_nature: str, risk_level: str,
                       evidence_refs: list[dict[str, Any]], evidence_summary: str,
                       trigger_match: list[dict[str, Any]]) -> str:
    if evidence_refs:
        prefix = f"文档依据：{evidence_summary}。事件事实显示"
    else:
        prefix = "未检索到可直接支撑结论的正文证据，以下仅按用户给出的事件事实作保守研判："
    if features["has_dc_line_fault_ride_through"]:
        if event_nature == "正常":
            return f"{prefix}保护动作、故障穿越/再启动和控保响应按时序发生，且文本中可见复归、完成或未见停运跳闸等后果；性质可判为正常响应，但因涉及直流线路保护和故障穿越，风险等级为严重。"
        return f"{prefix}事件涉及直流线路保护、故障穿越或再启动，且存在未复归、失败、闭锁、跳闸等不利后果线索，应按严重风险异常处理。"
    if event_nature == "异常":
        return f"{prefix}事件链中存在未复归、失败、持续、闭锁、跳闸或停运等后果性线索，已超出普通操作伴随信号。"
    if event_nature == "需核查":
        return f"{prefix}事件链包含自监视异常、轻微故障、A/B套响应差异或短时告警等异常信号，但同时存在复归/消失/完成线索，暂不宜直接判为严重异常，需要核查录波、保护出口和实时量。"
    if event_nature == "正常":
        if features.get("has_transformer_tap_change"):
            unmatched = [m for m in trigger_match if str(m.get("trigger_id", "")).startswith("tap_") and m.get("status") == "not_matched"]
            matched = [m for m in trigger_match if str(m.get("trigger_id", "")).startswith("tap_") and m.get("status") == "matched"]
            if unmatched and not matched:
                return f"{prefix}本次只见分接开关/分接头档位调整并完成；第五分册中用于识别分接开关异常的触发条件，如分接头不一致、调节失败/未完成、机构异常、档位差超限或电机电源跳开，在事件摘要中均未命中，因此按正常操作研判，风险等级评为{risk_level}。"
        return f"{prefix}事件主要表现为操作、控制或保护逻辑中的短时信号，且有复归、消失或完成线索；未见失败、持续告警、跳闸或闭锁后果，风险等级评为{risk_level}。"
    return f"{prefix}当前摘要缺少保护出口、实时量、录波和告警复归状态，无法可靠判断性质。"

def _collect_evidence(kb: SubstationKb | None, features: dict[str, Any], evidence_query: str,
                      limit: int, station_category: str = "") -> dict[str, Any]:
    if not kb:
        return {}
    queries = []
    if evidence_query:
        queries.append(evidence_query)
    for system in features.get("subsystems", []):
        query = EVIDENCE_QUERY_TERMS.get(system)
        if query:
            queries.append(query)
    if features.get("has_dc_line_fault_ride_through"):
        queries.insert(0, EVIDENCE_QUERY_TERMS["dc_line"])
    if features.get("has_abnormal_signal"):
        queries.append("设备异常 故障应急 轻微故障 自监视 告警 处理")
    merged: dict[str, dict[str, Any]] = {}

    primary_queries = list(dict.fromkeys([
        "设备异常 故障应急 处理预案 保护动作 闭锁 跳闸 告警 复归",
        *queries,
    ]))
    if station_category:
        for query in primary_queries[:6]:
            result = kb.search(
                query,
                category=station_category,
                doc_role=PRIMARY_KNOWLEDGE_ROLE,
                limit=max(4, limit // 2),
                prefer_primary=True,
            )
            for chunk in result.get("chunks", []):
                _boost_station_evidence(chunk, station_category)
                _merge_evidence_chunk(merged, chunk)

    for query in primary_queries[:6]:
        result = kb.search(query, doc_role=PRIMARY_KNOWLEDGE_ROLE, limit=max(4, limit // 2), prefer_primary=True)
        for chunk in result.get("chunks", []):
            _boost_station_evidence(chunk, station_category)
            _merge_evidence_chunk(merged, chunk)

    for query in queries[:6]:
        result = kb.search(query, limit=max(3, limit // 2), prefer_primary=True)
        for chunk in result.get("chunks", []):
            _boost_station_evidence(chunk, station_category)
            _merge_evidence_chunk(merged, chunk)
    chunks = sorted(merged.values(), key=lambda row: row.get("score", 0), reverse=True)[:limit]
    return {
        "primary_knowledge_source": "第五分册（设备异常与故障应急处理）",
        "queries": queries[:6],
        "primary_queries": primary_queries[:6],
        "chunks": chunks,
        "usage_note": "诊断类问题优先采用第五分册（设备异常与故障应急处理）作为处置和异常性判断依据；运行方式、控保规范和设备概况作为补充解释，不覆盖用户提供的实时事件事实。",
    }
