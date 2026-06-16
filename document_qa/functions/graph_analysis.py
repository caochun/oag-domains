from __future__ import annotations

import hashlib
import re
from typing import Any

try:
    import jieba
except Exception:  # pragma: no cover - optional dependency fallback
    jieba = None

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency fallback
    nx = None


RELATION_GROUP_TYPE_WEIGHTS = {
    "approval_for": 1.45,
    "request_for": 1.35,
    "feedback_to": 1.3,
    "attachment_of": 1.25,
    "abolishes": 1.25,
    "replaces": 1.22,
    "revises": 1.18,
    "based_on": 1.16,
    "implements": 1.14,
    "solicits_opinion_on": 1.12,
    "cites_by_doc_no": 1.1,
    "cites": 0.95,
    "same_series": 0.82,
    "shared_reference": 0.68,
    "same_matter": 0.55,
    "topically_related": 0.35,
    "semantically_related": 0.3,
}

DEFAULT_RELATION_GROUP_TYPES = tuple(RELATION_GROUP_TYPE_WEIGHTS)
GRAPH_TERM_STOPWORDS = {
    "关于", "情况", "工作", "汇报", "报告", "通知", "文件", "批办", "有关",
    "进行", "开展", "推进", "建设", "年度", "落实", "印发", "一个", "我们",
    "以及", "通过", "进一步", "相关", "如下", "其中", "目前", "同时", "不断",
    "全市", "全省", "全区", "有没有", "是否", "哪些", "什么", "请问", "帮我",
    "寻找", "查找", "材料", "资料",
}


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


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


def tokenize_terms(text: str, limit: int | None = None) -> list[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text or "")
    tokens = [tok.strip() for tok in jieba.lcut(text) if tok.strip()] if jieba else text.split()
    result = []
    seen = set()
    for token in tokens:
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if limit and len(result) >= limit:
            break
    return result


def build_relation_groups(rows: list[dict[str, Any]],
                          *,
                          min_relations: int,
                          max_documents_per_group: int) -> list[dict[str, Any]]:
    if not rows:
        return []

    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    documents: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = row["source_document_id"]
        target_id = row["target_document_id"]
        union(source_id, target_id)
        documents[source_id] = relation_group_document(row, "source")
        documents[target_id] = relation_group_document(row, "target")

    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(find(row["source_document_id"]), []).append(row)

    groups = []
    for _component_id, component_rows in grouped_rows.items():
        doc_ids = sorted({
            row["source_document_id"] for row in component_rows
        } | {
            row["target_document_id"] for row in component_rows
        })
        if len(component_rows) < min_relations or len(doc_ids) < 2:
            continue
        ranked_docs = sorted(
            (relation_group_ranked_document(doc_id, documents[doc_id], component_rows) for doc_id in doc_ids),
            key=lambda item: (-item["relation_count"], item["title"]),
        )
        representative_docs = dedupe_relation_group_documents(ranked_docs)
        if len(representative_docs) < 2:
            continue
        top_doc_ids = {item["document_id"] for item in representative_docs[:max_documents_per_group]}
        filtered_rows = [
            row for row in component_rows
            if row["source_document_id"] in top_doc_ids and row["target_document_id"] in top_doc_ids
        ]
        if len(filtered_rows) < min_relations:
            continue
        relation_types = sorted({row["relation_type"] for row in filtered_rows})
        score = relation_group_score(filtered_rows, len(top_doc_ids), len(relation_types))
        groups.append({
            "group_id": stable_id("|".join(sorted(top_doc_ids)) + f"|{len(filtered_rows)}"),
            "score": round(score, 3),
            "document_count": len(top_doc_ids),
            "relation_count": len(filtered_rows),
            "relation_types": relation_types,
            "documents": representative_docs[:max_documents_per_group],
            "relations": [
                relation_group_edge(row)
                for row in sorted(
                    filtered_rows,
                    key=lambda item: (
                        -relation_group_edge_score(item),
                        item["relation_type"],
                        item["source_title"] or "",
                    ),
                )[:max(max_documents_per_group * 2, min_relations)]
            ],
            "why_recommended": relation_group_reason(filtered_rows, relation_types),
        })

    return sorted(
        groups,
        key=lambda item: (-item["score"], -item["relation_count"], item["documents"][0]["title"]),
    )


def relation_group_document(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "document_id": row[f"{prefix}_document_id"],
        "title": row[f"{prefix}_title"],
        "path": row[f"{prefix}_path"],
        "category": row[f"{prefix}_category"],
        "agency": row[f"{prefix}_agency"],
        "doc_type": row[f"{prefix}_doc_type"],
        "date_text": row[f"{prefix}_date_text"],
        "char_count": row[f"{prefix}_char_count"],
    }


def relation_group_ranked_document(document_id: str,
                                   document: dict[str, Any],
                                   rows: list[dict[str, Any]]) -> dict[str, Any]:
    relation_count = sum(
        1 for row in rows
        if row["source_document_id"] == document_id or row["target_document_id"] == document_id
    )
    return {**document, "relation_count": relation_count}


def dedupe_relation_group_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_title: dict[str, dict[str, Any]] = {}
    duplicate_counts: dict[str, int] = {}
    for document in documents:
        title_key = exact_relation_title_key(str(document.get("title", "") or "")) or document["document_id"]
        duplicate_counts[title_key] = duplicate_counts.get(title_key, 0) + 1
        current = by_title.get(title_key)
        if not current or document["relation_count"] > current["relation_count"]:
            by_title[title_key] = document
    deduped = []
    for title_key, document in by_title.items():
        duplicate_count = duplicate_counts.get(title_key, 1)
        item = dict(document)
        if duplicate_count > 1:
            item["same_title_duplicate_count"] = duplicate_count
        deduped.append(item)
    return sorted(deduped, key=lambda item: (-item["relation_count"], item["title"]))


def relation_group_edge(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "relation_id": row["relation_id"],
        "relation_type": row["relation_type"],
        "confidence": round(float(row["confidence"] or 0.0), 3),
        "evidence": row["evidence"],
        "method": row["method"],
        "source_document": relation_group_document(row, "source"),
        "target_document": relation_group_document(row, "target"),
    }


def relation_group_edge_score(row: dict[str, Any]) -> float:
    weight = RELATION_GROUP_TYPE_WEIGHTS.get(row["relation_type"], 0.7)
    return float(row["confidence"] or 0.0) * weight


def relation_group_score(rows: list[dict[str, Any]],
                         document_count: int,
                         relation_type_count: int) -> float:
    edge_score = sum(relation_group_edge_score(row) for row in rows)
    hard_relation_bonus = sum(
        0.25 for row in rows
        if row["relation_type"] not in {"same_matter", "shared_reference", "topically_related", "semantically_related"}
    )
    return edge_score + hard_relation_bonus + document_count * 0.12 + relation_type_count * 0.35


def relation_group_reason(rows: list[dict[str, Any]], relation_types: list[str]) -> str:
    hard_types = [
        relation_type for relation_type in relation_types
        if relation_type not in {"same_matter", "shared_reference", "topically_related", "semantically_related"}
    ]
    if hard_types:
        return f"包含 {len(rows)} 条关系，且有 {', '.join(hard_types[:5])} 等较强公文关系。"
    return f"包含 {len(rows)} 条主题或引用线索关系，适合作为候选材料组继续核对。"


def find_relation_paths(rows: list[dict[str, Any]],
                        *,
                        start_id: str,
                        end_id: str,
                        direction: str,
                        max_depth: int,
                        limit: int) -> list[dict[str, Any]]:
    adjacency: dict[str, list[tuple[str, dict[str, Any], str]]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = row["source_document_id"]
        target_id = row["target_document_id"]
        documents[source_id] = relation_group_document(row, "source")
        documents[target_id] = relation_group_document(row, "target")
        if direction in {"outgoing", "both"}:
            adjacency.setdefault(source_id, []).append((target_id, row, "outgoing"))
        if direction in {"incoming", "both"}:
            adjacency.setdefault(target_id, []).append((source_id, row, "incoming"))

    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: (-relation_group_edge_score(item[1]), item[1]["relation_type"]))

    queue: list[tuple[str, list[str], list[tuple[dict[str, Any], str]]]] = [(start_id, [start_id], [])]
    found = []
    seen_signatures: set[tuple[Any, ...]] = set()
    max_queue = 20000
    while queue and len(found) < limit * 8 and max_queue > 0:
        current_id, visited, edges = queue.pop(0)
        max_queue -= 1
        if edges and (not end_id or current_id == end_id):
            signature = relation_path_signature(visited, edges, documents)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                found.append(format_relation_path(visited, edges, documents))
            if end_id and len(found) >= limit:
                break
        if len(edges) >= max_depth:
            continue
        for next_id, row, edge_direction in adjacency.get(current_id, [])[:80]:
            if next_id in visited:
                continue
            queue.append((next_id, [*visited, next_id], [*edges, (row, edge_direction)]))

    return sorted(
        found,
        key=lambda item: (-item["score"], item["depth"], item["documents"][-1]["title"]),
    )[:limit]


def relation_path_signature(visited: list[str],
                            edges: list[tuple[dict[str, Any], str]],
                            documents: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    title_keys = tuple(
        exact_relation_title_key(str(documents.get(document_id, {}).get("title", "") or document_id))
        for document_id in visited
    )
    edge_keys = tuple((row["relation_type"], edge_direction) for row, edge_direction in edges)
    return title_keys + edge_keys


def format_relation_path(visited: list[str],
                         edges: list[tuple[dict[str, Any], str]],
                         documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    edge_items = []
    score = 0.0
    for row, edge_direction in edges:
        edge_score = relation_group_edge_score(row)
        score += edge_score
        edge_items.append({
            **relation_group_edge(row),
            "traversal_direction": edge_direction,
            "edge_score": round(edge_score, 3),
        })
    score += len({edge["relation_type"] for edge in edge_items}) * 0.2
    return {
        "depth": len(edges),
        "score": round(score, 3),
        "documents": [documents[document_id] for document_id in visited if document_id in documents],
        "relations": edge_items,
        "path_summary": " -> ".join(
            documents[document_id]["title"] for document_id in visited if document_id in documents
        ),
    }


def analyze_relation_graph(rows: list[dict[str, Any]], *, top_n: int) -> dict[str, Any]:
    if nx is None:
        return {"status": "unavailable", "message": "networkx package is not available"}
    graph = nx.Graph()
    directed = nx.DiGraph()
    documents: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = row["source_document_id"]
        target_id = row["target_document_id"]
        documents[source_id] = relation_group_document(row, "source")
        documents[target_id] = relation_group_document(row, "target")
        weight = relation_group_edge_score(row)
        for doc_id in (source_id, target_id):
            if doc_id not in graph:
                graph.add_node(doc_id)
                directed.add_node(doc_id)
        if graph.has_edge(source_id, target_id):
            graph[source_id][target_id]["weight"] += weight
            graph[source_id][target_id]["relation_count"] += 1
        else:
            graph.add_edge(source_id, target_id, weight=weight, relation_count=1)
        directed.add_edge(source_id, target_id, weight=weight, relation_type=row["relation_type"])

    if graph.number_of_nodes() == 0:
        return {"status": "empty", "nodes": 0, "edges": 0, "central_documents": [], "communities": []}

    weighted_degree = dict(graph.degree(weight="weight"))
    try:
        pagerank = nx.pagerank(directed, weight="weight", max_iter=100)
    except Exception:
        pagerank = {node: 0.0 for node in graph.nodes}

    central = []
    for node in graph.nodes:
        central.append({
            **documents.get(node, {"document_id": node, "title": node}),
            "weighted_degree": round(float(weighted_degree.get(node, 0.0)), 3),
            "degree": int(graph.degree(node)),
            "pagerank": round(float(pagerank.get(node, 0.0)), 6),
        })
    central = sorted(
        central,
        key=lambda item: (-item["weighted_degree"], -item["pagerank"], item["title"]),
    )[:top_n]

    communities = []
    try:
        detected = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    except Exception:
        detected = []
    for idx, community in enumerate(detected[:top_n], start=1):
        members = sorted(
            (
                {
                    **documents.get(node, {"document_id": node, "title": node}),
                    "weighted_degree": round(float(weighted_degree.get(node, 0.0)), 3),
                    "degree": int(graph.degree(node)),
                }
                for node in community
            ),
            key=lambda item: (-item["weighted_degree"], item["title"]),
        )
        if len(members) < 2:
            continue
        communities.append({
            "community_id": idx,
            "document_count": len(members),
            "top_documents": members[:min(top_n, 8)],
            "label_terms": community_label_terms([item["title"] for item in members]),
        })

    return {
        "status": "ok",
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "directed_edges": directed.number_of_edges(),
        "central_documents": central,
        "communities": communities,
        "usage_note": "图指标用于发现结构性线索；具体公文事实仍需 read_document 核对原文。",
    }


def community_label_terms(titles: list[str]) -> list[str]:
    scores: dict[str, int] = {}
    for title in titles:
        for term in tokenize_terms(title, 20):
            if len(term) < 2 or term in GRAPH_TERM_STOPWORDS:
                continue
            scores[term] = scores.get(term, 0) + 1
    return [
        term for term, score in sorted(scores.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
        if score >= 2
    ][:8]
