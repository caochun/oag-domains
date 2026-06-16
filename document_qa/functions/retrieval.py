from __future__ import annotations

import math
from typing import Any

from .search_utils import make_snippet
from .text_processing import (
    classify_term,
    is_anchor_kind,
    kind_match_multiplier,
    rerank_base_component,
    tokenize_query,
)


def rerank_profiled_hits(recall_hits: list[dict[str, Any]],
                         profile: list[dict[str, Any]],
                         limit: int,
                         debug: bool = False) -> list[dict[str, Any]]:
    hits = rerank_related_hits(recall_hits, profile, limit=limit, debug=debug)
    return strip_internal_fields(hits)


def rerank_related_hits(hits: list[dict[str, Any]],
                        seed_profile: list[dict[str, Any]],
                        limit: int,
                        debug: bool = False) -> list[dict[str, Any]]:
    if not hits:
        return []
    seed_terms = seed_profile[:20]
    by_doc: dict[str, dict[str, Any]] = {}
    for hit in hits:
        doc_id = hit.get("document_id", "")
        if not doc_id:
            continue
        searchable_title = str(hit.get("title", "") or "")
        searchable_heading = str(hit.get("heading", "") or "")
        searchable_snippet = str(hit.get("snippet", "") or "")
        searchable_content = str(hit.get("content", "") or "")
        matched = []
        phrase_score = 0.0
        matched_importance = 0.0
        anchor_importance = 0.0
        contributions = []
        for item in seed_terms:
            term = item["term"]
            kind = item.get("kind", classify_term(term))
            importance = math.sqrt(max(float(item.get("score", 1.0)), 1.0))
            kind_multiplier = kind_match_multiplier(kind)
            location = ""
            location_multiplier = 0.0
            if term in searchable_title:
                location = "title"
                location_multiplier = 2.0
            elif term in searchable_heading:
                location = "heading"
                location_multiplier = 1.2
            elif term in searchable_snippet:
                location = "snippet"
                location_multiplier = 0.6
            elif term in searchable_content:
                location = "content"
                location_multiplier = 0.35
            if location:
                contribution = importance * kind_multiplier * location_multiplier
                phrase_score += contribution
                matched_importance += importance * kind_multiplier
                if is_anchor_kind(kind):
                    anchor_importance += importance * kind_multiplier
                matched.append(term)
                if debug:
                    contributions.append({
                        "term": term,
                        "kind": kind,
                        "location": location,
                        "importance": round(importance, 3),
                        "kind_multiplier": kind_multiplier,
                        "location_multiplier": location_multiplier,
                        "contribution": round(contribution, 3),
                    })

        semantic_score = max(0.0, float(hit.get("semantic_score", 0.0) or 0.0))
        if semantic_score:
            semantic_component = semantic_score * 6.0
            semantic_importance = semantic_score * 15.0
            phrase_score += semantic_component
            matched_importance += semantic_importance
            anchor_importance += semantic_score * 6.0
            if not matched:
                matched.append("semantic_similarity")
            if debug:
                contributions.append({
                    "term": "semantic_similarity",
                    "kind": "semantic",
                    "location": "embedding",
                    "importance": round(semantic_importance, 3),
                    "kind_multiplier": 1.0,
                    "location_multiplier": 1.0,
                    "contribution": round(semantic_component, 3),
                })

        if not matched:
            continue

        base_component = rerank_base_component(float(hit.get("score", 0.0)))
        related_score = base_component + phrase_score
        doc = by_doc.get(doc_id)
        if not doc:
            row = dict(hit)
            row["related_score"] = round(related_score, 3)
            row["matched_seed_terms"] = sorted(set(matched), key=lambda t: (-len(t), t))
            row["_matched_importance"] = matched_importance
            row["_anchor_importance"] = anchor_importance
            row["_best_chunk_score"] = related_score
            row["matched_chunk_count"] = 1
            if debug:
                row["scoring_debug"] = {
                    "base_search_score": hit.get("score", 0.0),
                    "base_component": round(base_component, 3),
                    "phrase_component": round(phrase_score, 3),
                    "matched_importance": round(matched_importance, 3),
                    "anchor_importance": round(anchor_importance, 3),
                    "contributions": contributions,
                }
            by_doc[doc_id] = row
            continue

        doc["related_score"] += related_score * 0.25
        doc["_matched_importance"] += matched_importance * 0.5
        doc["_anchor_importance"] += anchor_importance * 0.5
        doc["matched_chunk_count"] += 1
        doc["matched_seed_terms"] = sorted(
            set(doc.get("matched_seed_terms", [])) | set(matched),
            key=lambda t: (-len(t), t),
        )
        if related_score > float(doc.get("_best_chunk_score", doc.get("related_score", 0.0))):
            for key in ("chunk_id", "heading", "image_refs", "snippet", "source", "score"):
                doc[key] = hit.get(key)
            doc["_best_chunk_score"] = related_score
            if debug:
                doc["scoring_debug"] = {
                    "base_search_score": hit.get("score", 0.0),
                    "base_component": round(base_component, 3),
                    "phrase_component": round(phrase_score, 3),
                    "matched_importance": round(matched_importance, 3),
                    "anchor_importance": round(anchor_importance, 3),
                    "contributions": contributions,
                }

    filtered = [
        row for row in by_doc.values()
        if float(row.get("_matched_importance", 0.0)) >= 8.0
        and float(row.get("_anchor_importance", 0.0)) >= 3.0
    ]
    ranked = sorted(
        filtered,
        key=lambda row: (-float(row.get("related_score", 0.0)), row.get("path", "")),
    )[:max(1, min(limit, 30))]
    for row in ranked:
        row["related_score"] = round(float(row.get("related_score", 0.0)), 3)
        row.pop("_best_chunk_score", None)
        row.pop("_matched_importance", None)
        row.pop("_anchor_importance", None)
    return ranked


def strip_internal_fields(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for hit in hits:
        row = dict(hit)
        row.pop("content", None)
        row.pop("ordinal", None)
        row.pop("char_start", None)
        cleaned.append(row)
    return cleaned


def add_candidate(candidates: dict[str, dict[str, Any]],
                  row: dict[str, Any], score: float, query: str) -> None:
    chunk_id = row["chunk_id"]
    searchable = " ".join(
        str(row.get(key, "") or "")
        for key in ("title", "path", "category", "heading", "content")
    )
    tokens = tokenize_query(query)
    matched_tokens = [token for token in tokens if token in searchable]
    exact_match = bool(query and query in searchable)
    if query and len(tokens) >= 2 and not exact_match and len(matched_tokens) < min(2, len(tokens)):
        return
    score += len(matched_tokens) * 3
    if exact_match:
        score += 20
    existing = candidates.get(chunk_id)
    if existing:
        existing["score"] += score
        if row.get("semantic_score"):
            existing["semantic_score"] = max(
                float(existing.get("semantic_score", 0.0) or 0.0),
                float(row.get("semantic_score", 0.0) or 0.0),
            )
        existing["recall_sources"] = sorted(set(existing.get("recall_sources", ["lexical"])) | {"lexical"})
        return
    content = row.get("content", "") or ""
    row["score"] = round(score, 3)
    row["snippet"] = make_snippet(content, query)
    row["recall_sources"] = ["lexical"]
    row["source"] = {
        "title": row.get("title", ""),
        "path": row.get("path", ""),
        "heading": row.get("heading", ""),
        "chunk_id": chunk_id,
    }
    candidates[chunk_id] = row


def add_semantic_candidate(candidates: dict[str, dict[str, Any]],
                           row: dict[str, Any], query: str) -> None:
    chunk_id = row["chunk_id"]
    semantic_score = max(0.0, float(row.get("semantic_score", 0.0) or 0.0))
    score = round(semantic_score * 18.0, 3)
    existing = candidates.get(chunk_id)
    if existing:
        existing["score"] += score
        existing["semantic_score"] = max(
            float(existing.get("semantic_score", 0.0) or 0.0),
            semantic_score,
        )
        existing["recall_sources"] = sorted(set(existing.get("recall_sources", ["lexical"])) | {"semantic"})
        return

    content = row.get("content", "") or ""
    row["score"] = score
    row["snippet"] = make_snippet(content, query)
    row["source"] = {
        "title": row.get("title", ""),
        "path": row.get("path", ""),
        "heading": row.get("heading", ""),
        "chunk_id": chunk_id,
    }
    row["recall_sources"] = ["semantic"]
    candidates[chunk_id] = row
