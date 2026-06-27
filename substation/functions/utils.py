from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .constants import (
    ABNORMAL_TERMS,
    EVENT_CONCLUSION_PREFIX,
    NORMAL_TERMS,
    SEVERE_TERMS,
    SYSTEM_ALIASES,
)

def _json_result(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)

def _stable_id(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]

def _make_conclusion_id(text: str) -> str:
    return f"{EVENT_CONCLUSION_PREFIX}{_stable_id(text, 12)}"

def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def _strip_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, text[end + 5 :]

def _query_terms(query: str) -> list[str]:
    text = query or ""
    terms: list[str] = []
    for aliases in SYSTEM_ALIASES.values():
        if any(alias in text for alias in aliases):
            terms.extend(aliases)
    terms.extend([term for term in NORMAL_TERMS + ABNORMAL_TERMS + SEVERE_TERMS if term in text])
    terms.extend(re.findall(r"[A-Za-z0-9_.#/-]{2,}|[\u4e00-\u9fff]{2,}", text))
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        term = term.strip()
        if term and term not in seen:
            seen.add(term)
            result.append(term)
    return result or ([text.strip()] if text.strip() else [])

def _subsystems(text: str) -> list[str]:
    hits = []
    for system, aliases in SYSTEM_ALIASES.items():
        if any(alias in text for alias in aliases):
            hits.append(system)
    return hits or ["unknown"]

def _station_category(station_name: str) -> str:
    station = _compact(station_name)
    station = station.removesuffix("换流站").removesuffix("站")
    return station

def _truncate_text(text: str, max_chars: int) -> str:
    text = _compact(str(text or ""))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
