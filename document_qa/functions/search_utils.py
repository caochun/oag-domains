from __future__ import annotations

import re
from typing import Any

from .text_processing import tokenize_query

def summarize_recall_sources(hits: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hit in hits:
        sources = hit.get("recall_sources") or []
        if not sources:
            sources = ["unknown"]
        for source in sources:
            counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def anchored_like_terms(query: str) -> list[str]:
    tokens = tokenize_query(query)
    anchors = []
    if any(term in query for term in ("批示", "批办", "交办")):
        anchors.append("批示" if "批示" in query else ("批办" if "批办" in query else "交办"))
    for token in tokens:
        if token in {"查询", "一下", "所有", "文件", "材料", "相关"}:
            continue
        if re.fullmatch(r"20\d{2}", token):
            continue
        if token in anchors:
            continue
        if len(token) >= 3 or token.endswith(("局", "委", "办")):
            anchors.append(token)
    if "批示" in anchors:
        anchors = ["批示" if item == "批示" else item for item in anchors]
    return anchors[:4]


def tokenize_for_search(text: str) -> str:
    tokens = tokenize_query(text)
    if tokens:
        return " ".join(tokens)
    return " ".join(text[i:i + 2] for i in range(max(0, len(text) - 1)))


def build_fts_query(text: str) -> str:
    tokens = [escape_fts_token(token) for token in tokenize_query(text)]
    tokens = [token for token in tokens if token]
    return " OR ".join(tokens[:8])


def escape_fts_token(token: str) -> str:
    token = re.sub(r'["\']', "", token)
    token = re.sub(r"[^\w\u4e00-\u9fff]+", "", token)
    return f'"{token}"' if token else ""


def make_snippet(content: str, query: str, width: int = 260) -> str:
    clean = re.sub(r"\s+", " ", content).strip()
    if not clean:
        return ""
    positions = [clean.find(query)] if query else [-1]
    positions.extend(clean.find(token) for token in tokenize_query(query))
    positions = [pos for pos in positions if pos >= 0]
    start = max(0, (min(positions) if positions else 0) - 60)
    snippet = clean[start:start + width]
    if start:
        snippet = "..." + snippet
    if start + width < len(clean):
        snippet += "..."
    return snippet


def compact_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value}


def build_filter_sql(filters: dict[str, Any] | None,
                     table_alias: str = "") -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    prefix = f"{table_alias}." if table_alias else ""
    for key, value in (filters or {}).items():
        field, op = key.split("__", 1) if "__" in key else (key, "eq")
        field_expr = f"{prefix}{field}"
        if op == "like":
            clauses.append(f"{field_expr} like ?")
            params.append(f"%{value}%")
        else:
            clauses.append(f"{field_expr} = ?")
            params.append(value)
    return (" and ".join(clauses) if clauses else "1=1", params)


def safe_order_by(order_by: str | None, default: str) -> str:
    if not order_by:
        return default
    reverse = order_by.startswith("-")
    field = order_by.lstrip("-")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", field):
        return default
    return f"{field} desc" if reverse else field
