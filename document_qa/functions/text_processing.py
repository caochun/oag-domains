from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

try:
    import jieba
except Exception:  # pragma: no cover - optional dependency fallback
    jieba = None

IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DATE_RE = re.compile(
    r"("
    r"\d{8}|"
    r"20\d{2}年(?:0?[1-9]|1[0-2])月(?:0?[1-9]|[12]\d|3[01])日?|"
    r"20\d{2}年(?:0?[1-9]|1[0-2])月|"
    r"20\d{2}年|"
    r"20\d{2}[./-](?:0?[1-9]|1[0-2])[./-](?:0?[1-9]|[12]\d|3[01])|"
    r"20\d{2}[./-](?:0?[1-9]|1[0-2])"
    r")"
)
DOC_TYPE_WORDS = ("报告", "通知", "请示", "函", "纪要", "方案", "预案", "要点", "意见", "办法", "规则")
DEFAULT_EXCLUDED_TOP_LEVEL_DIRS = {"其他"}
DOCUMENT_KB_EXTRACTOR_VERSION = 2
TERM_STOPWORDS = {
    "关于", "情况", "工作", "汇报", "报告", "通知", "文件", "批办", "有关",
    "进行", "开展", "推进", "建设", "年度", "落实", "印发", "一个", "我们",
    "以及", "通过", "进一步", "相关", "如下", "其中", "目前", "同时", "不断",
    "全市", "全省", "全区", "有没有", "是否", "哪些", "什么", "请问", "帮我",
    "寻找", "查找", "材料", "资料",
}
GENERIC_TERM_SUFFIXES = ("计划", "方案", "报告", "通知", "请示", "情况", "工作", "办法", "意见", "公告", "清单", "要点")
SYNTHESIS_INTENT_HINTS = {
    "requirements": ("要求", "规定", "政策", "措施", "办法", "要点", "规范", "标准", "落实"),
    "status": ("情况", "运行", "现状", "成效", "进展", "数据", "态势"),
    "problems": ("问题", "困难", "不足", "风险", "隐患", "原因"),
    "actions": ("措施", "任务", "安排", "部署", "整治", "推进", "开展"),
    "comparison": ("比较", "差异", "变化", "对比", "不同", "相同"),
    "timeline": ("时间", "年份", "年度", "历年", "阶段", "进度"),
    "materials": ("相关文档", "相关材料", "找", "查找", "寻找", "哪些文档"),
}
SYNTHESIS_STRUCTURE_PATTERNS = (
    ("目标任务", ("目标", "任务", "总体要求", "指导思想", "主要目标")),
    ("重点内容", ("重点", "要点", "主要内容", "重点工作", "工作重点")),
    ("措施要求", ("措施", "要求", "规定", "办法", "规范", "标准", "制度")),
    ("职责分工", ("职责", "分工", "责任", "主体责任", "牵头", "配合")),
    ("流程条件", ("流程", "程序", "申报", "条件", "材料", "步骤", "办理")),
    ("监督管理", ("监管", "监督", "检查", "执法", "考核", "评审", "验收")),
    ("问题风险", ("问题", "风险", "隐患", "困难", "不足", "原因")),
    ("保障建议", ("保障", "建议", "下一步", "工作建议", "改进", "落实")),
)


def split_markdown_chunks(text: str, max_chars: int = 1400) -> list[dict[str, Any]]:
    chunks = []
    heading = ""
    buffer: list[str] = []
    start = 0
    cursor = 0

    def flush() -> None:
        nonlocal buffer, start
        content = "\n".join(buffer).strip()
        if content:
            chunks.append({
                "heading": heading,
                "content": content,
                "image_refs": IMAGE_RE.findall(content),
                "char_start": start,
            })
        buffer = []
        start = cursor

    for line in text.splitlines():
        line_len = len(line) + 1
        match = HEADING_RE.match(line)
        if match and buffer:
            flush()
        if match:
            heading = match.group(2).strip()
        if not buffer:
            start = cursor
        buffer.append(line)
        if sum(len(item) + 1 for item in buffer) >= max_chars and line.strip() == "":
            flush()
        cursor += line_len
    flush()

    expanded = []
    for chunk in chunks:
        content = chunk["content"]
        if len(content) <= max_chars * 1.4:
            expanded.append(chunk)
            continue
        paragraphs = re.split(r"\n\s*\n", content)
        buf: list[str] = []
        for para in paragraphs:
            if sum(len(item) + 2 for item in buf) + len(para) > max_chars and buf:
                expanded.append({**chunk, "content": "\n\n".join(buf)})
                buf = []
            buf.append(para)
        if buf:
            expanded.append({**chunk, "content": "\n\n".join(buf)})
    return expanded


def choose_title(stem: str, headings: list[str]) -> str:
    for h in headings:
        if any(word in h for word in DOC_TYPE_WORDS) and len(h) > 4:
            return h
    return headings[0] if headings else stem


def infer_agency(parts: tuple[str, ...], text: str) -> str:
    if len(parts) >= 2 and parts[0] == "通知抬头文件":
        return parts[1]
    first_heading = next((m.group(2).strip() for line in text.splitlines() if (m := HEADING_RE.match(line))), "")
    return first_heading.removesuffix("文件").strip()


def infer_doc_type(parts: tuple[str, ...], title: str) -> str:
    for part in parts:
        if part in DOC_TYPE_WORDS:
            return part
    for word in DOC_TYPE_WORDS:
        if word in title:
            return word
    return ""


def infer_date_text(path: str, text: str) -> str:
    for source in (path, text[:500]):
        match = DATE_RE.search(source)
        if match:
            return match.group(1)
    return ""


def infer_document_metadata(doc: dict[str, Any]) -> list[dict[str, Any]]:
    structural_text = "\n".join(
        str(doc.get(key, "") or "")
        for key in ("title", "date_text", "path", "headings")
    )
    content_text = str(doc.get("content_sample", "") or "")
    items: list[dict[str, Any]] = []
    for candidate in extract_date_candidates(structural_text):
        if candidate["kind"] == "year":
            items.append({
                "key": "year",
                "value": candidate["value"],
                "confidence": candidate["confidence"],
                "evidence": candidate["evidence"],
                "method": candidate["method"],
            })
        else:
            items.append({
                "key": "date",
                "value": candidate["value"],
                "confidence": candidate["confidence"],
                "evidence": candidate["evidence"],
                "method": candidate["method"],
            })
            items.append({
                "key": "year",
                "value": candidate["value"][:4],
                "confidence": max(0.0, candidate["confidence"] - 0.05),
                "evidence": candidate["evidence"],
                "method": candidate["method"],
            })

    structural_years = {item["value"] for item in items if item["key"] == "year"}
    for candidate in extract_date_candidates(content_text):
        confidence = max(0.0, candidate["confidence"] - 0.28)
        method = f"content_{candidate['method']}"
        if candidate["kind"] == "year":
            if candidate["value"] in structural_years:
                continue
            items.append({
                "key": "year",
                "value": candidate["value"],
                "confidence": confidence,
                "evidence": candidate["evidence"],
                "method": method,
            })
        else:
            year = candidate["value"][:4]
            if year not in structural_years:
                items.append({
                    "key": "date",
                    "value": candidate["value"],
                    "confidence": confidence,
                    "evidence": candidate["evidence"],
                    "method": method,
                })
                items.append({
                    "key": "year",
                    "value": year,
                    "confidence": max(0.0, confidence - 0.05),
                    "evidence": candidate["evidence"],
                    "method": method,
                })

    for doc_number in extract_doc_numbers(structural_text):
        items.append({
            "key": "doc_number",
            "value": doc_number,
            "confidence": 0.88,
            "evidence": doc_number,
            "method": "structural_doc_number",
        })

    char_count = int(doc.get("char_count", 0) or 0)
    if char_count < 80:
        quality, confidence = "empty", 0.95
    elif char_count < 500:
        quality, confidence = "low", 0.8
    else:
        quality, confidence = "usable", 0.7
    items.append({
        "key": "extraction_quality",
        "value": quality,
        "confidence": confidence,
        "evidence": f"char_count={char_count}",
        "method": "char_count",
    })
    return dedupe_metadata_items(items)


def extract_date_candidates(text: str) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    patterns = [
        (r"(20\d{2})年(\d{1,2})月(\d{1,2})日?", 0.92, "full_date"),
        (r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})", 0.9, "full_date"),
        (r"(20\d{2})年(\d{1,2})月", 0.78, "year_month"),
        (r"(20\d{2})[./-](\d{1,2})(?![./-]\d)", 0.72, "year_month"),
        (r"[〔\[（(](20\d{2})[〕\]）)]", 0.84, "doc_number_year"),
        (r"(20\d{2})\s*年", 0.76, "plain_year"),
    ]
    for pattern, confidence, method in patterns:
        for match in re.finditer(pattern, text or ""):
            if method == "full_date":
                month = int(match.group(2))
                day = int(match.group(3))
                if not (1 <= month <= 12 and 1 <= day <= 31):
                    continue
                value = f"{match.group(1)}-{month:02d}-{day:02d}"
                kind = "date"
            elif method == "year_month":
                month = int(match.group(2))
                if not (1 <= month <= 12):
                    continue
                value = f"{match.group(1)}-{month:02d}"
                kind = "date"
            else:
                value = match.group(1)
                kind = "year"
            key = (kind, value, method)
            if key in seen:
                continue
            seen.add(key)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            candidates.append({
                "kind": kind,
                "value": value,
                "confidence": confidence,
                "evidence": re.sub(r"\s+", " ", text[start:end]).strip(),
                "method": method,
            })
    return sorted(candidates, key=lambda item: (-item["confidence"], item["value"]))[:12]


def dedupe_metadata_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in items:
        key = (item["key"], item["value"], item["method"])
        existing = result.get(key)
        if not existing or item["confidence"] > existing["confidence"]:
            result[key] = item
    return list(result.values())


def context_best_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": hit.get("chunk_id", ""),
        "heading": hit.get("heading", ""),
        "snippet": hit.get("snippet", ""),
        "score": hit.get("score", 0),
        "related_score": hit.get("related_score", 0),
        "semantic_score": hit.get("semantic_score", 0),
        "recall_sources": hit.get("recall_sources", []),
    }


def document_quality(item: dict[str, Any], read_result: dict[str, Any]) -> dict[str, Any]:
    char_count = int(item.get("char_count", 0) or 0)
    content = read_result.get("content", "") or ""
    if char_count < 80 or len(content.strip()) < 80:
        status = "empty"
        note = "正文极短，可能是 MinerU 抽取失败或原文为空。"
    elif char_count < 500 or len(content.strip()) < 500:
        status = "low"
        note = "正文较短，只适合作为弱证据或线索。"
    elif read_result.get("truncated"):
        status = "partial"
        note = "已读取核心上下文，但返回内容被截断。"
    else:
        status = "usable"
        note = "可作为回答证据。"
    return {
        "status": status,
        "char_count": char_count,
        "read_chars": len(content),
        "note": note,
    }


def document_temporal_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = doc.get("metadata", {}) or {}
    years = list(metadata.get("year", []))
    dates = list(metadata.get("date", []))
    if not years and not dates:
        inferred = infer_document_metadata({
            "title": doc.get("title", ""),
            "path": doc.get("path", ""),
            "date_text": doc.get("date_text", ""),
            "content_sample": str(doc.get("read_content", ""))[:1200],
            "char_count": doc.get("char_count", 0),
        })
        for item in inferred:
            if item["key"] == "year":
                years.append({
                    "value": item["value"],
                    "confidence": item["confidence"],
                    "evidence": item.get("evidence", ""),
                    "method": item["method"],
                })
            elif item["key"] == "date":
                dates.append({
                    "value": item["value"],
                    "confidence": item["confidence"],
                    "evidence": item.get("evidence", ""),
                    "method": item["method"],
                })
    return {
        "best_year": best_metadata_value(years),
        "best_date": best_metadata_value(dates),
        "years": years[:5],
        "dates": dates[:5],
    }


def best_metadata_value(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    return str(sorted(items, key=lambda item: (-float(item.get("confidence", 0)), item.get("value", "")))[0]["value"])


def build_temporal_coverage(query: str,
                            evidence_documents: list[dict[str, Any]]) -> dict[str, Any]:
    expected_years = expected_years_from_query(query)
    found_years = sorted({
        doc.get("temporal", {}).get("best_year", "")
        for doc in evidence_documents
        if doc.get("temporal", {}).get("best_year")
    })
    missing_years = [year for year in expected_years if year not in found_years]
    return {
        "expected_years": expected_years,
        "found_years": found_years,
        "missing_years": missing_years,
        "coverage_note": temporal_coverage_note(expected_years, found_years, missing_years),
    }


def expected_years_from_query(query: str) -> list[str]:
    query_years = sorted(set(re.findall(r"20\d{2}", query or "")))
    if len(query_years) >= 2:
        start, end = int(query_years[0]), int(query_years[-1])
        if 1900 <= start <= end <= 2100 and end - start <= 30:
            return [str(year) for year in range(start, end + 1)]
    if len(query_years) == 1:
        return query_years
    return []


def temporal_coverage_note(expected: list[str], found: list[str], missing: list[str]) -> str:
    if not expected:
        return "用户问题未指定明确年份范围，按证据文档自身时间属性组织即可。"
    if not found:
        return "未能从候选文档中可靠识别年份，回答时需说明时间证据不足。"
    if missing:
        return f"证据只覆盖 {', '.join(found)}；缺少 {', '.join(missing)} 的可靠文档。"
    return f"证据覆盖用户指定年份：{', '.join(found)}。"


def build_strict_answer_constraints(temporal_coverage: dict[str, Any]) -> dict[str, Any]:
    found = temporal_coverage.get("found_years", []) or []
    missing = temporal_coverage.get("missing_years", []) or []
    constraints = {
        "must_state_temporal_coverage": bool(temporal_coverage.get("expected_years")),
        "evidence_years_only": found,
        "missing_years_are_not_covered": missing,
        "do_not_claim_coverage_for": missing,
    }
    if found:
        constraints["required_limitation_sentence"] = (
            f"当前证据只覆盖 {', '.join(found)}；"
            f"缺少 {', '.join(missing)} 的可靠文档。"
            if missing
            else f"当前证据覆盖 {', '.join(found)}。"
        )
    elif temporal_coverage.get("expected_years"):
        constraints["required_limitation_sentence"] = "当前证据未能可靠识别年份，不能按指定年份范围作完整汇总。"
    else:
        constraints["required_limitation_sentence"] = ""
    return constraints


def build_evidence_outline(evidence_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outline = []
    for index, doc in enumerate(evidence_documents, 1):
        document = doc.get("document", {})
        temporal = doc.get("temporal", {})
        quality = doc.get("quality", {})
        outline.append({
            "rank": index,
            "document_id": document.get("document_id", ""),
            "title": document.get("title", ""),
            "path": document.get("path", ""),
            "best_year": temporal.get("best_year", ""),
            "quality_status": quality.get("status", ""),
            "included_chunk_ids": doc.get("included_chunk_ids", [])[:3],
            "selection_reason": doc.get("selection", {}).get("reason", ""),
        })
    return outline


def build_synthesis_outline(query: str,
                            evidence_documents: list[dict[str, Any]]) -> dict[str, Any]:
    intents = infer_synthesis_intents(query)
    dimension_map: dict[str, dict[str, Any]] = {}

    for index, doc in enumerate(evidence_documents, 1):
        document = doc.get("document", {})
        if doc.get("quality", {}).get("status") in {"empty", "low"}:
            continue
        doc_ref = {
            "rank": index,
            "document_id": document.get("document_id", ""),
            "title": document.get("title", ""),
            "path": document.get("path", ""),
            "best_year": doc.get("temporal", {}).get("best_year", ""),
            "included_chunk_ids": doc.get("included_chunk_ids", [])[:3],
        }
        for candidate in synthesis_dimension_candidates(query, doc):
            item = dimension_map.setdefault(candidate["dimension"], {
                "dimension": candidate["dimension"],
                "reason_signals": set(),
                "supporting_documents": {},
                "evidence_terms": {},
                "score": 0.0,
            })
            item["score"] += candidate["score"]
            item["reason_signals"].update(candidate["signals"])
            item["supporting_documents"][doc_ref["document_id"]] = doc_ref
            for term in candidate["terms"]:
                item["evidence_terms"][term] = item["evidence_terms"].get(term, 0) + 1

    dimensions = []
    for item in dimension_map.values():
        supporting = list(item["supporting_documents"].values())
        support_count = len(supporting)
        terms = sorted(
            item["evidence_terms"],
            key=lambda term: (-item["evidence_terms"][term], -len(term), term),
        )[:8]
        score = float(item["score"]) + support_count * 2.5
        if support_count >= 2:
            score += 3.0
        dimensions.append({
            "dimension": item["dimension"],
            "reason": synthesis_reason(item["reason_signals"], support_count),
            "support_count": support_count,
            "evidence_terms": terms,
            "supporting_documents": supporting[:5],
            "_score": round(score, 3),
        })

    ranked = sorted(
        dimensions,
        key=lambda item: (-item["_score"], -item["support_count"], item["dimension"]),
    )
    for item in ranked:
        item.pop("_score", None)
    recommended_sections = [
        {
            "section": item["dimension"],
            "support_count": item["support_count"],
            "evidence_terms": item["evidence_terms"][:5],
        }
        for item in ranked[:6]
    ]

    return {
        "intent": intents,
        "recommended_answer_sections": recommended_sections,
        "generation_basis": [
            "用户问题中的通用意图词",
            "证据文档标题和小节结构",
            "多篇文档共同出现的结构性短语",
            "文档抽取质量和时间覆盖边界",
        ],
        "dimensions": ranked[:6],
        "usage": "回答时优先使用 recommended_answer_sections 作为一级归纳框架，再在各节中合并 documents 证据；dimension 只是通用文档结构建议，不是业务专用模板。",
    }


def infer_synthesis_intents(query: str) -> list[str]:
    found = []
    for intent, hints in SYNTHESIS_INTENT_HINTS.items():
        if any(hint in query for hint in hints):
            found.append(intent)
    if not found and any(word in query for word in ("汇总", "总结", "梳理", "归纳")):
        found.append("summary")
    return found or ["answer_question"]


def synthesis_dimension_candidates(query: str, doc: dict[str, Any]) -> list[dict[str, Any]]:
    document = doc.get("document", {})
    content = doc.get("content", "") or ""
    chunks = split_read_content_blocks(content)
    text_scope = "\n".join([
        document.get("title", ""),
        "\n".join(block["heading"] for block in chunks[:10]),
        content[:1800],
    ])
    candidates: list[dict[str, Any]] = []

    for dimension, hints in SYNTHESIS_STRUCTURE_PATTERNS:
        matched = [hint for hint in hints if hint and hint in text_scope]
        if not matched:
            continue
        intent_bonus = 1.5 if any(hint in query for hint in matched) else 0.0
        heading_bonus = 1.0 if any(any(hint in block["heading"] for hint in matched) for block in chunks) else 0.0
        candidates.append({
            "dimension": dimension,
            "signals": matched[:4],
            "terms": matched[:4],
            "score": len(matched) + intent_bonus + heading_bonus,
        })

    for phrase, count in repeated_document_phrases(document.get("title", ""), chunks).items():
        if count < 2:
            continue
        candidates.append({
            "dimension": phrase,
            "signals": ["标题/小节重复短语"],
            "terms": [phrase],
            "score": min(4.0, 1.0 + count * 0.8),
        })

    if not candidates:
        fallback_terms = tokenize_terms(document.get("title", ""), limit=3)
        if fallback_terms:
            candidates.append({
                "dimension": "核心事项",
                "signals": ["标题主题"],
                "terms": fallback_terms,
                "score": 1.0,
            })
    return candidates


def split_read_content_blocks(content: str) -> list[dict[str, str]]:
    blocks = []
    current_heading = ""
    current_lines: list[str] = []
    for line in (content or "").splitlines():
        if line.startswith("### "):
            if current_heading or current_lines:
                blocks.append({
                    "heading": current_heading,
                    "text": "\n".join(current_lines).strip(),
                })
            current_heading = line[4:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_heading or current_lines:
        blocks.append({
            "heading": current_heading,
            "text": "\n".join(current_lines).strip(),
        })
    return blocks


def repeated_document_phrases(title: str,
                              chunks: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    sources = [title, *(chunk["heading"] for chunk in chunks[:12])]
    for source in sources:
        for term in tokenize_terms(source, limit=12):
            if classify_term(term) in {"form", "short", "numeric_or_code"}:
                continue
            if len(term) < 3 or len(term) > 12:
                continue
            counts[term] = counts.get(term, 0) + 1
    return counts


def synthesis_reason(signals: set[str], support_count: int) -> str:
    signal_text = "、".join(sorted(signals)[:5]) or "证据文档结构"
    if support_count >= 2:
        return f"由 {support_count} 篇证据文档共同支持；结构信号：{signal_text}"
    return f"由单篇核心证据支持；结构信号：{signal_text}"


def build_quality_notes(evidence_documents: list[dict[str, Any]]) -> list[str]:
    notes = []
    weak = [
        doc["document"]["title"]
        for doc in evidence_documents
        if doc.get("quality", {}).get("status") in {"empty", "low"}
    ]
    if weak:
        notes.append("以下文档抽取质量较低，只能作为线索：" + "；".join(weak[:5]))
    truncated = [
        doc["document"]["title"]
        for doc in evidence_documents
        if doc.get("truncated")
    ]
    if truncated:
        notes.append("以下文档读取内容被截断，如需精确引用应继续 read_document：" + "；".join(truncated[:5]))
    return notes


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def document_kb_signature(doc: dict[str, Any]) -> str:
    payload = {
        "kb_extractor_version": DOCUMENT_KB_EXTRACTOR_VERSION,
        "title": doc.get("title", ""),
        "category": doc.get("category", ""),
        "agency": doc.get("agency", ""),
        "doc_type": doc.get("doc_type", ""),
        "date_text": doc.get("date_text", ""),
        "char_count": doc.get("char_count", 0),
        "headings": doc.get("headings", ""),
        "content_sample": doc.get("content_sample", ""),
        "quoted_terms": doc.get("quoted_terms", []),
        "doc_numbers": doc.get("doc_numbers", []),
        "cited_doc_numbers": doc.get("cited_doc_numbers", []),
        "relation_text": doc.get("relation_text", ""),
    }
    return stable_id(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def tokenize_query(text: str) -> list[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    if jieba:
        tokens = [tok.strip() for tok in jieba.lcut(text) if tok.strip()]
    else:
        tokens = text.split()
    result = []
    for token in tokens:
        if len(token) >= 2 or token.isdigit():
            result.append(token)
    return list(dict.fromkeys(result))[:12]


def tokenize_terms(text: str, limit: int | None = None) -> list[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    if jieba:
        raw_tokens = [tok.strip() for tok in jieba.lcut(text) if tok.strip()]
    else:
        raw_tokens = text.split()
    result = []
    seen = set()
    for token in raw_tokens:
        if not is_useful_term(token) or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if limit and len(result) >= limit:
            break
    return result


def is_useful_term(term: str) -> bool:
    if not term or term in TERM_STOPWORDS:
        return False
    if term.isdigit() and len(term) < 4:
        return False
    if len(term) < 2:
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", term))


def collect_document_term_weights(doc: dict[str, Any],
                                  chunks: list[dict[str, Any]]) -> dict[str, float]:
    weights: dict[str, float] = {}

    add_structured_terms(weights, doc.get("title", ""), source_weight=8.0)
    add_quoted_terms(weights, doc.get("title", ""), source_weight=12.0)
    add_structured_terms(weights, doc.get("agency", ""), source_weight=3.0)
    add_structured_terms(weights, doc.get("category", ""), source_weight=2.0)

    seen_headings = set()
    body_parts = []
    for chunk in chunks:
        heading = chunk.get("heading", "")
        if heading and heading not in seen_headings:
            add_structured_terms(weights, heading, source_weight=4.0)
            add_quoted_terms(weights, heading, source_weight=6.0)
            seen_headings.add(heading)
        body_parts.append(chunk.get("content", ""))
    add_frequency_terms(weights, "\n".join(body_parts), source_weight=0.5, limit=80)
    return weights


def collect_seed_term_weights(title: str, text: str,
                              query_hint: str = "") -> dict[str, float]:
    weights: dict[str, float] = {}
    add_structured_terms(weights, title, source_weight=10.0)
    add_quoted_terms(weights, title, source_weight=18.0)
    add_structured_terms(weights, query_hint, source_weight=12.0)

    heading_text = "\n".join(
        m.group(2).strip()
        for line in text.splitlines()
        if (m := HEADING_RE.match(line))
    )
    add_structured_terms(weights, heading_text, source_weight=5.0)
    add_quoted_terms(weights, heading_text, source_weight=8.0)
    add_frequency_terms(weights, text[:5000], source_weight=0.8, limit=80)
    return weights


def collect_query_term_weights(query: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    tokens = tokenize_terms(query, limit=40)
    for token in tokens:
        weights[token] = weights.get(token, 0.0) + 10.0 * length_bonus(token) * term_quality(token)
    for phrase in query_phrases(tokens, limit=12):
        weights[phrase] = weights.get(phrase, 0.0) + 9.0 * length_bonus(phrase) * term_quality(phrase)
    add_quoted_terms(weights, query, source_weight=14.0)
    return weights


def query_phrases(tokens: list[str], limit: int) -> list[str]:
    phrases = []
    seen = set()
    useful_tokens = [token for token in tokens if classify_term(token) != "form"]
    for size in (2, 3):
        for idx in range(0, max(0, len(useful_tokens) - size + 1)):
            phrase = "".join(useful_tokens[idx:idx + size])
            if phrase in seen or not is_useful_term(phrase) or len(phrase) > 16:
                continue
            seen.add(phrase)
            phrases.append(phrase)
            if len(phrases) >= limit:
                return phrases
    return phrases


def add_structured_terms(weights: dict[str, float], text: str,
                         source_weight: float) -> None:
    if not text:
        return
    clean = re.sub(r"\s+", "", text)
    if (
        is_useful_term(clean)
        and len(clean) <= 24
        and not clean.startswith("关于")
        and "《" not in clean
    ):
        weights[clean] = weights.get(clean, 0.0) + source_weight * length_bonus(clean) * term_quality(clean)
    for token in tokenize_terms(text, limit=60):
        weights[token] = weights.get(token, 0.0) + source_weight * length_bonus(token) * term_quality(token)


def add_quoted_terms(weights: dict[str, float], text: str,
                     source_weight: float) -> None:
    for term in re.findall(r"[《“\"]([^》”\"]{2,80})[》”\"]", text or ""):
        clean = re.sub(r"\s+", "", term)
        if is_useful_term(clean) and len(clean) <= 60:
            weights[clean] = weights.get(clean, 0.0) + source_weight * length_bonus(clean) * term_quality(clean)
        for token in tokenize_terms(term, limit=20):
            weights[token] = weights.get(token, 0.0) + (source_weight * 0.5) * length_bonus(token) * term_quality(token)


def add_frequency_terms(weights: dict[str, float], text: str,
                        source_weight: float, limit: int) -> None:
    counts: dict[str, int] = {}
    for token in tokenize_terms(text):
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(
        counts.items(),
        key=lambda item: (-(1 + math.log(item[1])) * length_bonus(item[0]), item[0]),
    )[:limit]
    for term, count in ranked:
        weights[term] = weights.get(term, 0.0) + source_weight * (1 + math.log(count)) * length_bonus(term) * term_quality(term)


def compute_idf(total_docs: int, doc_freq: int) -> float:
    return math.log((total_docs + 1) / (doc_freq + 1)) + 1.0


def length_bonus(term: str) -> float:
    return min(2.5, 1.0 + len(term) / 12)


def term_quality(term: str) -> float:
    if re.fullmatch(r"[一二三四五六七八九十0-9]+年", term):
        return 0.2
    if len(term) <= 2:
        return 0.45
    if len(term) <= 4 and any(term.endswith(suffix) for suffix in GENERIC_TERM_SUFFIXES):
        return 0.35
    return 1.0


def classify_term(term: str) -> str:
    if re.fullmatch(r"[0-9A-Za-z.+-]+", term):
        return "numeric_or_code"
    if re.fullmatch(r"[一二三四五六七八九十0-9]+年", term):
        return "form"
    if len(term) <= 2:
        return "short"
    if len(term) <= 6 and any(term.endswith(suffix) for suffix in GENERIC_TERM_SUFFIXES):
        return "form"
    if len(term) >= 10:
        return "quoted_theme"
    return "topic"


def kind_score_multiplier(kind: str) -> float:
    return {
        "quoted_theme": 1.0,
        "topic": 1.0,
        "numeric_or_code": 0.75,
        "short": 0.55,
        "form": 0.35,
    }.get(kind, 1.0)


def kind_match_multiplier(kind: str) -> float:
    return {
        "quoted_theme": 1.2,
        "topic": 1.0,
        "numeric_or_code": 0.8,
        "short": 0.55,
        "form": 0.25,
    }.get(kind, 1.0)


def is_anchor_kind(kind: str) -> bool:
    return kind in {"quoted_theme", "topic", "numeric_or_code"}


def rerank_base_component(score: float) -> float:
    return min(8.0, math.log1p(max(score, 0.0)) * 1.5)


def top_weighted_terms(weights: dict[str, float],
                       limit: int) -> list[tuple[str, float]]:
    return sorted(
        weights.items(),
        key=lambda item: (-item[1] * length_bonus(item[0]), -len(item[0]), item[0]),
    )[:limit]


def extract_seed_terms(title: str, text: str, *, query_hint: str = "",
                       limit: int = 16) -> list[str]:
    stopwords = {
        "关于", "情况", "工作", "汇报", "报告", "通知", "文件", "批办", "有关",
        "进行", "开展", "推进", "建设", "年度", "落实", "印发", "一个",
    }
    candidates: dict[str, float] = {}

    def add_terms(source: str, weight: float) -> None:
        for token in tokenize_query(source):
            if token in stopwords or token.isdigit():
                continue
            if len(token) < 2:
                continue
            candidates[token] = candidates.get(token, 0.0) + weight + min(len(token), 8) * 0.2

    add_terms(title, 8.0)
    add_terms(query_hint, 10.0)
    add_terms(text[:3000], 1.0)

    quoted_terms = re.findall(r"[《“\"]([^》”\"]{2,40})[》”\"]", title)
    for term in quoted_terms:
        clean = re.sub(r"\s+", "", term)
        if clean:
            candidates[clean] = candidates.get(clean, 0.0) + 20.0
            add_terms(clean, 4.0)

    sorted_terms = sorted(candidates.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    result = []
    for term, _score in sorted_terms:
        result.append(term)
        if len(result) >= limit:
            break
    return result


def extract_quoted_terms(text: str) -> list[str]:
    terms = []
    seen = set()
    for term in re.findall(r"[《“\"]([^》”\"]{2,100})[》”\"]", text or ""):
        clean = re.sub(r"\s+", "", term)
        if not clean or clean in seen:
            continue
        if clean.endswith(("（征求意见稿）", "(征求意见稿)")):
            clean = re.sub(r"[（(]征求意见稿[）)]$", "", clean)
        if is_useful_term(clean):
            seen.add(clean)
            terms.append(clean)
    return terms[:30]


def extract_doc_numbers(text: str) -> list[str]:
    numbers = []
    seen = set()
    patterns = [
        r"[\u4e00-\u9fffA-Za-z]{1,12}[〔\[]20\d{2}[〕\]][\u4e00-\u9fffA-Za-z]{0,8}\d{1,5}号?",
        r"[\u4e00-\u9fffA-Za-z]{1,12}\(20\d{2}\)[\u4e00-\u9fffA-Za-z]{0,8}\d{1,5}号?",
        r"[\u4e00-\u9fffA-Za-z]{1,12}发〔20\d{2}〕\d{1,5}号?",
        r"[\u4e00-\u9fffA-Za-z]{1,12}函〔20\d{2}〕\d{1,5}号?",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text or ""):
            normalized = normalize_doc_number(match.group(0))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            numbers.append(normalized)
            if len(numbers) >= 20:
                return numbers
    return numbers


def normalize_doc_number(text: str) -> str:
    clean = re.sub(r"\s+", "", text or "")
    clean = clean.replace("［", "[").replace("］", "]")
    clean = clean.replace("[", "〔").replace("]", "〕")
    clean = clean.replace("（", "(").replace("）", ")")
    clean = re.sub(r"\((20\d{2})\)", r"〔\1〕", clean)
    clean = re.sub(r"号$", "", clean)
    clean = re.sub(r"^[和及与、，,；;：:。关于依据根据按照遵照落实执行修订废止替代取代转发印发]+", "", clean)
    if not re.search(r"〔20\d{2}〕", clean):
        return ""
    if not re.search(r"\d+$", clean):
        return ""
    if len(clean) < 7 or len(clean) > 40:
        return ""
    return clean
