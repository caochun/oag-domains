from __future__ import annotations

import json
import re
from typing import Any

from .constants import ABNORMAL_TERMS, NORMAL_TERMS, SEVERE_TERMS
from .parser import parse_substation_events
from .utils import _compact

def _coerce_event_input(event_text: str, events_json: str) -> dict[str, Any]:
    if events_json:
        try:
            data = json.loads(events_json)
            if isinstance(data, dict) and "events" in data:
                return data
            if isinstance(data, list):
                return {"events": data, "event_chain": None}
        except json.JSONDecodeError:
            pass
    return parse_substation_events(event_text or "")

def _chain_features(events: list[dict[str, Any]]) -> dict[str, Any]:
    text = " ".join(
        f"{e.get('station_name','')} {e.get('device_name','')} {e.get('event_type','')} {e.get('summary','')} {e.get('raw_text','')}"
        for e in events
    )
    systems = sorted({s for e in events for s in e.get("subsystems", [])})
    flags = [e.get("flags", {}) for e in events]
    return {
        "subsystems": systems,
        "has_dc_line_fault_ride_through": any(k in text for k in ["低电压保护", "故障穿越", "再启动"]),
        "has_transformer_tap_change": any(k in text for k in ["分接开关", "分接头", "档位", "降至", "升至"]) and any(k in text for k in ["换流变", "变压器档位调整"]),
        "has_protection_action": any(f.get("protection") for f in flags),
        "has_control_switch": any(f.get("control_switch") for f in flags),
        "has_abnormal_signal": any(f.get("abnormal_signal") for f in flags),
        "has_severe_signal": any(f.get("severe_signal") for f in flags),
        "has_valve_cooling_abnormal": any(f.get("valve_cooling_abnormal") for f in flags),
        "has_negative_outcome": any(f.get("negative_outcome") for f in flags),
        "has_normal_completion": any(f.get("normal_completion") for f in flags),
        "has_recorder": any(f.get("recorder") for f in flags),
        "has_operation_sequence": any(f.get("operation") for f in flags),
        "has_control_response_difference": bool(re.search(r"(A/B|A、B|A套|B套).{0,24}(差异|时间差|响应不一致|不同步)|指令下发存在异常", text)),
        "has_tap_position_inconsistency": bool(re.search(r"分接(?:头|开关)?.{0,24}(不一致|不同步|档位差)|(?:机械|就地|OWS|软件).{0,12}档位.{0,12}不一致", text)),
        "has_tap_software_jump": "档位BCD码发生跳变" in text or bool(re.search(r"软件.{0,6}档位.{0,8}跳变", text)),
        "min_alarm_level": min(int(e.get("alarm_level") or 99) for e in events),
        "max_alarm_level": max(int(e.get("alarm_level") or 0) for e in events),
        "keywords": _feature_keywords(text),
    }

def _event_query_from_raw(events: list[dict[str, Any]], limit: int = 12) -> str:
    text = _chain_text(events)
    terms: list[str] = []
    priority_terms = [
        "分接开关", "分接头", "换流变", "档位", "阀冷", "主循环泵", "内水冷",
        "低电压保护", "故障穿越", "再启动", "闭锁", "跳闸", "保护动作",
        "自监视", "轻微故障", "录波", "站用变", "交流滤波器", "解锁",
    ]
    for term in priority_terms:
        if term in text and term not in terms:
            terms.append(term)
    for event in events:
        for key in ("station_name", "device_name", "event_type"):
            value = _compact(str(event.get(key, "")))
            if value and value not in terms:
                terms.append(value)
    return " ".join(terms[:limit])

def _feature_keywords(text: str) -> list[str]:
    keys = []
    for term in SEVERE_TERMS + ABNORMAL_TERMS + NORMAL_TERMS:
        if term in text and term not in keys:
            keys.append(term)
    return keys[:20]

def _chain_text(events: list[dict[str, Any]]) -> str:
    return " ".join(
        f"{e.get('station_name','')} {e.get('device_name','')} {e.get('event_type','')} {e.get('summary','')} {e.get('raw_text','')}"
        for e in events
    )

def _tap_change_text(text: str) -> str:
    match = re.search(r"分接(?:头|开关)?.{0,20}?由\s*([0-9]+)\s*档\s*(升至|降至|调至)\s*([0-9]+)\s*档", text)
    if match:
        return f"分接头由{match.group(1)}档{match.group(2)}{match.group(3)}档"
    match = re.search(r"([0-9]+)\s*档\s*(升至|降至|调至)\s*([0-9]+)\s*档", text)
    if match:
        return f"档位由{match.group(1)}档{match.group(2)}{match.group(3)}档"
    return ""

def _observed_facts(events: list[dict[str, Any]], features: dict[str, Any]) -> list[dict[str, Any]]:
    text = _chain_text(events)
    facts: list[dict[str, Any]] = []
    if events:
        facts.append({
            "fact_id": "event_count",
            "label": "事件数量",
            "value": len(events),
            "basis": f"用户输入共解析出{len(events)}条事件。",
            "source": "user_event_text",
        })
        station = events[0].get("station_name", "")
        if station:
            facts.append({
                "fact_id": "station",
                "label": "站点",
                "value": station,
                "basis": f"事件链站点为{station}。",
                "source": "user_event_text",
            })
    if features.get("has_transformer_tap_change"):
        facts.append({
            "fact_id": "transformer_tap_change",
            "fact_category": "tap_operation",
            "label": "分接开关/分接头调节",
            "value": _tap_change_text(text) or "存在分接开关或分接头档位调整",
            "basis": "事件摘要包含分接开关操作、分接头或档位调整描述。",
            "source": "user_event_text",
        })
    if features.get("has_control_response_difference"):
        facts.append({
            "fact_id": "control_response_difference",
            "fact_category": "control_response",
            "label": "A/B套响应或指令下发差异",
            "value": True,
            "basis": "事件摘要出现A/B套时间差、响应不一致或指令下发存在异常等线索。",
            "source": "user_event_text",
        })
    if features.get("has_tap_position_inconsistency"):
        facts.append({
            "fact_id": "tap_position_inconsistency",
            "fact_category": "tap_position",
            "label": "分接头/档位不一致",
            "value": True,
            "basis": "事件摘要明确出现分接头不一致、不同步或档位差线索。",
            "source": "user_event_text",
        })
    if features.get("has_tap_software_jump"):
        facts.append({
            "fact_id": "tap_software_position_jump",
            "fact_category": "tap_signal",
            "label": "档位软件/BCD码跳变",
            "value": True,
            "basis": "事件摘要出现档位BCD码或软件档位跳变线索。",
            "source": "user_event_text",
        })
    if features.get("has_normal_completion"):
        facts.append({
            "fact_id": "normal_completion",
            "fact_category": "completion",
            "label": "操作完成/信号复归",
            "value": True,
            "basis": "事件摘要包含完成、复归或消失等结果线索。",
            "source": "user_event_text",
        })
    facts.extend([
        {
            "fact_id": "protection_action",
            "fact_category": "protection_action",
            "label": "保护动作",
            "value": bool(features.get("has_protection_action")),
            "basis": "事件摘要中" + ("出现保护动作相关表述。" if features.get("has_protection_action") else "未见保护动作相关表述。"),
            "source": "user_event_text",
        },
        {
            "fact_id": "abnormal_signal",
            "fact_category": "abnormal_signal",
            "label": "异常/轻微故障信号",
            "value": bool(features.get("has_abnormal_signal")),
            "basis": "事件摘要中" + ("出现异常、轻微故障或差异响应线索。" if features.get("has_abnormal_signal") else "未见异常、轻微故障或差异响应线索。"),
            "source": "user_event_text",
        },
        {
            "fact_id": "negative_outcome",
            "fact_category": "negative_outcome",
            "label": "失败、跳闸、闭锁、持续告警等后果",
            "value": bool(features.get("has_negative_outcome")),
            "basis": "事件摘要中" + ("存在失败、跳闸、闭锁、持续或未复归等后果性线索。" if features.get("has_negative_outcome") else "未见失败、跳闸、闭锁、持续或未复归等后果性线索。"),
            "source": "user_event_text",
        },
    ])
    return facts

def _timeline(events: list[dict[str, Any]]) -> list[str]:
    return [
        f"{e.get('sequence', i)}. {e.get('start_time') or '时间未抽取'} 至 {e.get('end_time') or '时间未抽取'}，{e.get('device_name') or '设备未抽取'}，{e.get('event_type') or '事件类型未抽取'}，告警等级{e.get('alarm_level')}：{e.get('summary') or e.get('raw_text')}"
        for i, e in enumerate(events, 1)
    ]

def _select_events(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    selected = []
    for event in events:
        flags = event.get("flags", {})
        if kind == "abnormal" and (flags.get("abnormal_signal") or flags.get("severe_signal") or flags.get("negative_outcome")):
            selected.append(event)
        elif kind == "companion" and (flags.get("recorder") or flags.get("normal_completion")) and not flags.get("severe_signal"):
            selected.append(event)
        elif kind == "main" and (flags.get("operation") or flags.get("protection") or flags.get("severe_signal")):
            selected.append(event)
    return selected
