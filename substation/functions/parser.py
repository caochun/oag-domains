from __future__ import annotations

import re
from typing import Any

from .constants import (
    ABNORMAL_TERMS, EVENT_RE, EVENT_START_PREFIX_RE, FIELD_RE,
    NEGATIVE_OUTCOME_TERMS, NORMAL_TERMS, SEVERE_TERMS, TIME_RE,
)
from .utils import _compact, _stable_id, _subsystems

def parse_substation_events(event_text: str, station_name: str = "") -> dict[str, Any]:
    text = _normalize_event_text(event_text or "")
    events = _parse_events_strict(text)
    parse_note = ""
    if not events:
        events = _parse_events_loose(text, station_name=station_name)
        if events:
            parse_note = "输入未完全符合标准模板，已按时间戳和字段标签进行宽松解析；缺失字段保留为空或原文。"
    events.sort(key=lambda row: (row["start_time"] or "", row["sequence"]))
    if not events:
        if not text.strip():
            return {"events": [], "event_chain": None, "parse_note": "输入为空，未能识别事件内容。"}
        events = [_raw_event(text, 1, station_name=station_name)]
        parse_note = "输入没有可识别的固定事件格式，已作为一条原始事件文本保留；结构化字段仅作弱抽取。"
    chain = {
        "chain_id": f"chain_{_stable_id((events[0].get('start_time') or '') + (events[-1].get('end_time') or '') + text[:80])}",
        "station_name": station_name or events[0]["station_name"],
        "window_start": events[0]["start_time"],
        "window_end": max((e.get("end_time") or e.get("start_time") or "") for e in events),
        "event_count": len(events),
        "involved_subsystems": sorted({s for event in events for s in event["subsystems"]}),
    }
    result = {"events": events, "event_chain": chain}
    if parse_note:
        result["parse_note"] = parse_note
    return result

def _normalize_event_text(text: str) -> str:
    return (
        str(text or "")
        .replace("\u3000", " ")
        .replace("：", ":")
        .replace("，", ",")
        .replace("；", ";")
        .replace("（", "(")
        .replace("）", ")")
    )

def _parse_events_strict(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    strict_text = text.replace(":", "：", 1) if False else text
    # EVENT_RE is kept for already-normalized historical inputs; loose parsing covers
    # punctuation and missing-index variants.
    for match in EVENT_RE.finditer(strict_text.replace(":", "：").replace(",", "，").replace(";", "；")):
        raw = match.group(0)
        summary = _compact(match.group("summary"))
        device = _compact(match.group("device"))
        text = f"{device} {match.group('event_type')} {summary}"
        idx = int(match.group("idx"))
        flags = _event_flags(text)
        events.append(
            {
                "event_id": f"ev_{idx:03d}_{_stable_id(raw, 8)}",
                "sequence": idx,
                "start_time": match.group("start"),
                "end_time": match.group("end"),
                "station_name": _compact(match.group("station")),
                "device_name": device,
                "event_type": _compact(match.group("event_type")),
                "summary": summary,
                "event_nature_text": _compact(match.group("nature")),
                "alarm_level": int(match.group("level")),
                "subsystems": _subsystems(text),
                "flags": flags,
            }
        )
    return events

def _parse_events_loose(text: str, station_name: str = "") -> list[dict[str, Any]]:
    segments = _event_segments(text)
    events = []
    for i, segment in enumerate(segments, 1):
        event = _parse_event_segment(segment, i, station_name=station_name)
        if event:
            events.append(event)
    return events

def _event_segments(text: str) -> list[str]:
    matches = list(TIME_RE.finditer(text or ""))
    if not matches:
        return []
    starts = []
    for match in matches:
        start = match.start()
        if _looks_like_event_start(text, start):
            starts.append(start)
    if not starts:
        starts = [matches[0].start()]
    starts = sorted(set(starts))
    segments = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        segment = text[start:end].strip(" \n\r\t,;")
        if segment:
            segments.append(segment)
    return segments

def _looks_like_event_start(text: str, start: int) -> bool:
    before = text[max(0, start - 32):start]
    stripped = before.rstrip()
    if not stripped:
        return True
    if stripped.endswith(("至", "到", "-", "~", "—")):
        return False
    prev = text[start - 1] if start > 0 else ""
    if prev and (prev.isspace() or prev in "。；;,，、.)）:："):
        return True
    return bool(EVENT_START_PREFIX_RE.search(before))

def _parse_event_segment(segment: str, sequence: int, station_name: str = "") -> dict[str, Any] | None:
    raw = segment.strip()
    times = TIME_RE.findall(raw)
    if not times:
        return None
    fields = _extract_event_fields(raw)
    start_time = times[0] if times else ""
    end_time = times[1] if len(times) > 1 else start_time
    station = _compact(fields.get("站点", "")) or station_name or _weak_station_name(raw)
    device = _compact(fields.get("设备", ""))
    event_type = _compact(fields.get("事件类型", ""))
    summary = _compact(fields.get("二级摘要", "") or fields.get("摘要", ""))
    if not summary:
        summary = _fallback_summary(raw)
    nature = _compact(fields.get("事件性质", ""))
    alarm_level = _extract_alarm_level(fields.get("告警等级", "") or raw)
    feature_text = f"{station} {device} {event_type} {summary} {raw}"
    flags = _event_flags(feature_text)
    return {
        "event_id": f"ev_{sequence:03d}_{_stable_id(raw, 8)}",
        "sequence": sequence,
        "start_time": start_time,
        "end_time": end_time,
        "station_name": station,
        "device_name": device,
        "event_type": event_type,
        "summary": summary,
        "event_nature_text": nature,
        "alarm_level": alarm_level,
        "subsystems": _subsystems(feature_text),
        "flags": flags,
        "raw_text": raw,
        "parse_quality": _parse_quality(fields, times),
    }

def _raw_event(text: str, sequence: int, station_name: str = "") -> dict[str, Any]:
    raw = _compact(text)
    times = TIME_RE.findall(raw)
    station = station_name or _weak_station_name(raw)
    feature_text = f"{station} {raw}"
    return {
        "event_id": f"ev_{sequence:03d}_{_stable_id(raw, 8)}",
        "sequence": sequence,
        "start_time": times[0] if times else "",
        "end_time": times[1] if len(times) > 1 else (times[0] if times else ""),
        "station_name": station,
        "device_name": "",
        "event_type": "",
        "summary": raw,
        "event_nature_text": "",
        "alarm_level": _extract_alarm_level(raw),
        "subsystems": _subsystems(feature_text),
        "flags": _event_flags(feature_text),
        "raw_text": raw,
        "parse_quality": "raw",
    }

def _extract_event_fields(segment: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in FIELD_RE.finditer(segment):
        label = match.group(1)
        value = _compact(match.group(2).strip(" ,;"))
        fields[label] = value
    return fields

def _weak_station_name(text: str) -> str:
    match = re.search(r"([\u4e00-\u9fff]{1,8}(?:换流站|站))", text or "")
    return _compact(match.group(1)) if match else ""

def _extract_alarm_level(text: str) -> int:
    match = re.search(r"(?:告警等级\s*[:：]?\s*)?([1-5])\s*(?:级)?", str(text or ""))
    if match:
        return int(match.group(1))
    return 99

def _fallback_summary(segment: str) -> str:
    text = TIME_RE.sub("", segment)
    text = re.sub(r"^\s*\d+[.、)]\s*", "", text)
    text = re.sub(r"\b至\b", " ", text)
    text = re.sub(r"(站点|设备|事件类型|事件性质|告警等级)\s*:\s*[^,;]*[,;]?", "", text)
    text = re.sub(r"二级摘要\s*:\s*", "", text)
    return _compact(text) or _compact(segment)

def _parse_quality(fields: dict[str, str], times: list[str]) -> str:
    required = ["站点", "设备", "事件类型"]
    if len(times) >= 2 and all(fields.get(key) for key in required) and (fields.get("二级摘要") or fields.get("摘要")):
        return "high"
    if times and (fields or len(_compact(" ".join(fields.values()))) > 0):
        return "medium"
    return "low"

def _event_flags(text: str) -> dict[str, bool]:
    abnormal_signal = any(term in text for term in ABNORMAL_TERMS)
    severe_signal = any(term in text for term in SEVERE_TERMS) or _has_true_lock_signal(text)
    return {
        "normal_completion": any(term in text for term in NORMAL_TERMS),
        "abnormal_signal": abnormal_signal,
        "severe_signal": severe_signal,
        "negative_outcome": any(term in text for term in NEGATIVE_OUTCOME_TERMS)
        and _has_negative_outcome_context(text),
        "recorder": "录波" in text,
        "protection": any(term in text for term in ["保护动作", "低电压保护", "保护出口"]),
        "control_switch": any(term in text for term in ["A切至B", "A套", "B套", "切换"]),
        "operation": any(term in text for term in ["解锁", "合闸", "投入", "投退", "分接", "功率升降"]),
        "valve_cooling_abnormal": "阀冷" in text and abnormal_signal and not any(k in text for k in ["阀冷控制系统", "闭锁顺序/阀冷控制系统"]),
    }

def _has_true_lock_signal(text: str) -> bool:
    if "闭锁" not in text:
        return False
    benign_context = ["闭锁顺序", "闭锁逻辑", "闭锁状态相继出现"]
    if any(term in text for term in benign_context) and not any(term in text for term in ["极闭锁", "双极闭锁", "闭锁动作", "闭锁告警"]):
        return False
    return True

def _has_negative_outcome_context(text: str) -> bool:
    if re.search(r"(未见|无|未发生|未出现).{0,8}(停运|跳闸|闭锁|失败|不成功|持续)", text):
        return False
    if re.search(r"(停运|跳闸|闭锁|失败|不成功|持续).{0,8}(未见|无|未发生|未出现)", text):
        return False
    if any(term in text for term in ["消失", "复归", "完成", "执行完毕"]):
        return False
    if "停运" in text and any(term in text for term in ["操作", "投退", "泵停运"]):
        return False
    if "闭锁顺序" in text and not any(term in text for term in ["极闭锁", "双极闭锁", "闭锁动作"]):
        return False
    return True
