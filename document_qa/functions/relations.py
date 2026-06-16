from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from .text_processing import classify_term, normalize_doc_number, stable_id

def normalize_relation_text(text: str) -> str:
    clean = re.sub(r"\s+", "", text or "")
    clean = re.sub(r"[\(\)（）\[\]【】〔〕“”\"'、，。；;:：!！?？]", "", clean)
    clean = re.sub(r"^(关于|转发|印发|下发|征求|反馈)", "", clean)
    clean = re.sub(r"(的通知|的请示|的函|的报告|的意见|的批复|的公告|的方案|的计划|征求意见稿)$", "", clean)
    clean = re.sub(r"(20\d{2}|二〇二[一二三四五六七八九十]|202[0-9])年度?", "", clean)
    clean = re.sub(r"\d+$", "", clean)
    return clean


def exact_relation_title_key(title: str) -> str:
    return normalize_relation_text(title).replace("副本", "").replace("附件", "")


def normalize_series_title(title: str) -> str:
    clean = exact_relation_title_key(title)
    clean = re.sub(r"(第[一二三四五六七八九十0-9]+届|第[一二三四五六七八九十0-9]+次)", "", clean)
    clean = re.sub(r"[一二三四五六七八九十0-9]+月", "", clean)
    clean = re.sub(r"[一二三四五六七八九十0-9]+日", "", clean)
    clean = re.sub(r"(一|二|三|四|五|六|七|八|九|十|[0-9]+)$", "", clean)
    return clean


def build_title_index(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = []
    for doc in docs:
        title = str(doc.get("title", "") or "")
        path_title = Path(str(doc.get("path", "") or "")).stem
        keys = {
            normalize_relation_text(title),
            exact_relation_title_key(title),
            normalize_relation_text(path_title),
            exact_relation_title_key(path_title),
        }
        index.append({**doc, "_title_keys": {key for key in keys if len(key) >= 4}})
    return index


def build_doc_number_index(docs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        numbers = list(dict.fromkeys(doc.get("doc_numbers", [])))
        for number in numbers[:6]:
            index.setdefault(number, []).append(doc)
    return {
        number: members
        for number, members in index.items()
        if 1 <= len(members) <= 5
    }


def match_title_targets(quote: str, title_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quote_key = normalize_relation_text(quote)
    if len(quote_key) < 8:
        return []
    matches = []
    for doc in title_index:
        keys = doc.get("_title_keys", set())
        title_key = exact_relation_title_key(str(doc.get("title", "") or ""))
        if quote_key in keys or any(quote_key in key or key in quote_key for key in keys if len(key) >= 8):
            matches.append(doc)
        elif len(quote_key) >= 10 and quote_key in title_key:
            matches.append(doc)
    return matches[:12]


def classify_relation_role(title: str) -> str:
    clean = title or ""
    if re.search(r"反馈意见|反馈.*意见|回复|回函|答复", clean):
        return "feedback"
    if re.search(r"批复|批办单|批办|批示|审批", clean):
        return "approval"
    if re.search(r"请示|申请", clean):
        return "request"
    if re.search(r"附件|附表|附录", clean):
        return "attachment"
    return ""


def relation_type_for_role(role: str) -> str:
    return {
        "feedback": "feedback_to",
        "approval": "approval_for",
        "request": "request_for",
        "attachment": "attachment_of",
    }.get(role, "cites")


def confidence_for_role(role: str) -> float:
    return {
        "feedback": 0.88,
        "approval": 0.86,
        "request": 0.82,
        "attachment": 0.78,
    }.get(role, 0.74)


def relation_role_label(role: str) -> str:
    return {
        "feedback": "反馈/回复",
        "approval": "批办/批复",
        "request": "请示/申请",
        "attachment": "附件",
    }.get(role, "引用")


def classify_enhanced_relation_type(source: dict[str, Any],
                                    evidence_text: str = "",
                                    *,
                                    default: str = "cites") -> str:
    text = "\n".join([
        str(source.get("title", "") or ""),
        relation_evidence_context(str(source.get("relation_text", "") or ""), evidence_text),
    ])
    if re.search(r"废止|停止执行|宣布失效|予以废止|不再执行", text):
        return "abolishes"
    if re.search(r"修订|修改|修正|调整|补充|更新", text):
        return "revises"
    if re.search(r"替代|取代|代替|替换|原.*同时废止", text):
        return "replaces"
    if re.search(r"征求意见|征询意见|征求.*建议|公开征求", text):
        return "solicits_opinion_on"
    if re.search(r"依据|根据|按照|遵照|依照", text):
        return "based_on"
    if re.search(r"贯彻|落实|实施|执行", text):
        return "implements"
    return default


def relation_evidence_context(text: str, needle: str, window: int = 180) -> str:
    if not needle:
        return text[:window]
    normalized_needle = normalize_doc_number(needle) or needle
    candidates = [needle, normalized_needle]
    for candidate in candidates:
        if not candidate:
            continue
        pos = text.find(candidate)
        if pos >= 0:
            start = max(0, pos - window)
            end = min(len(text), pos + len(candidate) + window)
            return text[start:end]
    compact_text = re.sub(r"\s+", "", text)
    compact_needle = re.sub(r"\s+", "", normalized_needle)
    pos = compact_text.find(compact_needle)
    if pos >= 0:
        start = max(0, pos - window)
        end = min(len(compact_text), pos + len(compact_needle) + window)
        return compact_text[start:end]
    return text[:window]


def confidence_for_enhanced_relation(relation_type: str, role: str) -> float:
    if role:
        return confidence_for_role(role)
    return {
        "based_on": 0.86,
        "abolishes": 0.88,
        "revises": 0.84,
        "replaces": 0.86,
        "solicits_opinion_on": 0.82,
        "cites_by_doc_no": 0.82,
        "implements": 0.84,
    }.get(relation_type, 0.74)


def match_profile_targets(source: dict[str, Any], docs: list[dict[str, Any]],
                          doc_terms: dict[str, dict[str, float]],
                          *,
                          doc_term_index: dict[str, set[str]] | None = None,
                          title_keys: dict[str, str] | None = None,
                          limit: int = 4) -> list[dict[str, Any]]:
    source_id = source["document_id"]
    source_terms = doc_terms.get(source_id, {})
    if not source_terms:
        return []
    by_id = {doc["document_id"]: doc for doc in docs}
    title_keys = title_keys or {}
    source_title_key = title_keys.get(source_id) or normalize_series_title(source.get("title", ""))
    candidate_ids: set[str] = set()
    if doc_term_index:
        useful_terms = [
            term for term in sorted(
                source_terms,
                key=lambda term: (-source_terms[term], -len(term), term),
            )
            if len(term) >= 3 and classify_term(term) not in {"form", "short"}
        ][:12]
        for term in useful_terms:
            candidate_ids.update(doc_term_index.get(term, set()))
    if source_title_key:
        for doc in docs:
            target_title_key = title_keys.get(doc["document_id"], "")
            if target_title_key and (
                source_title_key in target_title_key or target_title_key in source_title_key
            ):
                candidate_ids.add(doc["document_id"])
    if not candidate_ids:
        return []

    scored = []
    for target_id in candidate_ids:
        if target_id == source_id:
            continue
        target = by_id.get(target_id)
        if not target:
            continue
        score, overlap = weighted_term_overlap(source_terms, doc_terms.get(target["document_id"], {}))
        target_title_key = title_keys.get(target["document_id"], "")
        if source_title_key and target_title_key and (
            source_title_key in target_title_key or target_title_key in source_title_key
        ):
            score += 0.2
        if score >= 0.18 and overlap:
            scored.append((score, target))
    return [target for _score, target in sorted(scored, key=lambda item: (-item[0], item[1]["title"]))[:limit]]


def weighted_term_overlap(left: dict[str, float], right: dict[str, float]) -> tuple[float, list[str]]:
    if not left or not right:
        return 0.0, []
    common = set(left) & set(right)
    useful_common = [
        term for term in common
        if classify_term(term) not in {"form", "short"} and len(term) >= 3
    ]
    if not useful_common:
        return 0.0, []
    common_weight = sum(math.sqrt(left[term] * right[term]) for term in useful_common)
    left_weight = sum(left.values()) or 1.0
    right_weight = sum(right.values()) or 1.0
    score = common_weight / math.sqrt(left_weight * right_weight)
    overlap = sorted(useful_common, key=lambda term: (-(left[term] + right[term]), -len(term), term))
    return score, overlap


def directed_pairs(items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = []
    for source in items:
        for target in items:
            if source["document_id"] != target["document_id"]:
                pairs.append((source, target))
    return pairs


def pick_original_quote(doc: dict[str, Any], quote_key: str) -> str:
    for quote in doc.get("quoted_terms", []):
        if normalize_relation_text(quote) == quote_key:
            return quote
    return quote_key


def add_relation(relations: dict[tuple[str, str, str, str], dict[str, Any]],
                 source: dict[str, Any], target: dict[str, Any], relation_type: str,
                 confidence: float, evidence: str, *, source_chunk_id: str = "",
                 target_chunk_id: str = "", method: str = "rule",
                 metadata: dict[str, Any] | None = None) -> None:
    if source["document_id"] == target["document_id"]:
        return
    confidence = round(max(0.0, min(float(confidence), 1.0)), 3)
    key = (source["document_id"], target["document_id"], relation_type, method)
    relation_id = stable_id("|".join(key))
    candidate = {
        "relation_id": relation_id,
        "source_document_id": source["document_id"],
        "target_document_id": target["document_id"],
        "relation_type": relation_type,
        "confidence": confidence,
        "evidence": evidence[:500],
        "source_chunk_id": source_chunk_id,
        "target_chunk_id": target_chunk_id,
        "method": method,
        "metadata": metadata or {},
    }
    existing = relations.get(key)
    if not existing or confidence > float(existing["confidence"]):
        relations[key] = candidate


def format_relation_row(row: sqlite3.Row, focus_document_id: str = "") -> dict[str, Any]:
    is_outgoing = row["source_document_id"] == focus_document_id
    other_prefix = "target" if is_outgoing else "source"
    metadata = {}
    if row["metadata"]:
        try:
            metadata = json.loads(row["metadata"])
        except json.JSONDecodeError:
            metadata = {}
    return {
        "relation_id": row["relation_id"],
        "relation_type": row["relation_type"],
        "direction": "outgoing" if is_outgoing else "incoming",
        "confidence": round(float(row["confidence"]), 3),
        "evidence": row["evidence"],
        "method": row["method"],
        "source_document": {
            "document_id": row["source_document_id"],
            "title": row["source_title"],
            "path": row["source_path"],
            "category": row["source_category"],
        },
        "target_document": {
            "document_id": row["target_document_id"],
            "title": row["target_title"],
            "path": row["target_path"],
            "category": row["target_category"],
        },
        "other_document": {
            "document_id": row[f"{other_prefix}_document_id"],
            "title": row[f"{other_prefix}_title"],
            "path": row[f"{other_prefix}_path"],
            "category": row[f"{other_prefix}_category"],
        },
        "source_chunk_id": row["source_chunk_id"],
        "target_chunk_id": row["target_chunk_id"],
        "metadata": metadata,
    }


def parse_relation_types(value: str) -> list[str]:
    return [
        item.strip()
        for item in (value or "").split(",")
        if item.strip()
    ]
