from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oag.ontology.registry import FunctionRegistry
from oag.ontology.repository import ObjectRepository
from oag.ontology.schema import Ontology


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
EVENT_RE = re.compile(
    r"(?P<idx>\d+)\.\s*"
    r"(?P<start>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*至\s*"
    r"(?P<end>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)，"
    r"站点：(?P<station>[^，]+)，设备：(?P<device>[^，]+)，"
    r"事件类型：(?P<event_type>[^，]+)，二级摘要：(?P<summary>.*?)(?:，事件性质：(?P<nature>.*?)，告警等级：(?P<level>\d+))",
    re.S,
)

DOMAIN_SYNONYMS = {
    "自动功率升降": ["自动功率升降", "自动功率控制", "功率升降", "目标功率", "功率变化速率", "斜坡发生器", "双极功率控制"],
    "阀冷": ["阀冷", "阀冷却", "内水冷", "冷却水", "开关阀", "高压泵", "加药泵", "工业水泵", "补水泵"],
    "录波": ["录波", "故障录波", "录波启动", "故障录波器启动", "事件记录", "SER", "顺序事件"],
    "直流站控": ["直流站控", "直流站控制", "双极功率协调控制", "功率调节器"],
    "VCE": ["VCE", "阀控制单元", "阀组控制", "阀组控制系统"],
}


def _json_result(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _stable_id(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _strip_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[end + 5 :]


def _text_terms(query: str) -> list[str]:
    query = query.strip()
    terms: list[str] = []
    for key, values in DOMAIN_SYNONYMS.items():
        if key in query or any(v in query for v in values):
            terms.extend(values)
    domain_phrases = [
        "主循环泵",
        "备用泵",
        "主泵",
        "开关阀",
        "到位信号",
        "流量低",
        "温度高",
        "水位低",
        "压力低",
        "泄漏",
        "异常",
        "处理预案",
        "故障录波",
        "保护动作",
        "自动功率",
        "功率升降",
        "双极功率",
        "目标功率",
        "变化速率",
    ]
    terms.extend([phrase for phrase in domain_phrases if phrase in query])
    terms.extend(re.findall(r"[A-Za-z0-9_.#-]{2,}|[\u4e00-\u9fff]{2,}", query))
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        t = term.strip()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return unique or ([query] if query else [])


def _matches_filters(row: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if key.endswith("__like"):
            col = key[:-6]
            if str(expected) not in str(row.get(col, "")):
                return False
        elif row.get(key) != expected:
            return False
    return True


@dataclass
class SubstationDocumentIndex:
    domain_dir: Path

    def __post_init__(self) -> None:
        self.docs_root = self.domain_dir / "docs_md"
        self.md_root = self.docs_root / "md"
        self.assets_root = self.docs_root / "assets"
        self.kb_dir = self.domain_dir / ".substation_kb"
        self.db_path = self.kb_dir / "substation_index.sqlite"
        self._loaded = False
        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.figures: list[dict[str, Any]] = []

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._index_needs_sync():
            self.sync()
        self._load_from_sqlite()
        self._loaded = True

    def sync(self) -> dict[str, Any]:
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        documents, chunks, figures = self._scan_files()
        with sqlite3.connect(self.db_path) as conn:
            self._init_schema(conn)
            conn.execute("delete from documents")
            conn.execute("delete from chunks")
            conn.execute("delete from chunks_fts")
            conn.execute("delete from figures")
            conn.execute("delete from event_patterns")
            conn.execute("delete from procedure_map")
            conn.executemany(
                """
                insert into documents
                (document_id, title, category, source_path, md_path, pages, doc_role, char_count, file_mtime)
                values (:document_id, :title, :category, :source_path, :md_path, :pages, :doc_role, :char_count, :file_mtime)
                """,
                documents,
            )
            conn.executemany(
                """
                insert into chunks
                (chunk_id, document_id, title, path, category, heading, content, image_refs, evidence_role, ordinal, char_start)
                values (:chunk_id, :document_id, :title, :path, :category, :heading, :content, :image_refs, :evidence_role, :ordinal, :char_start)
                """,
                chunks,
            )
            conn.executemany(
                """
                insert into chunks_fts (chunk_id, document_id, title, category, heading, content)
                values (:chunk_id, :document_id, :title, :category, :heading, :content)
                """,
                chunks,
            )
            conn.executemany(
                """
                insert into figures
                (figure_id, document_id, path, caption, width, height, figure_type, title, category)
                values (:figure_id, :document_id, :path, :caption, :width, :height, :figure_type, :title, :category)
                """,
                figures,
            )
            conn.executemany(
                """
                insert into event_patterns
                (pattern_id, name, trigger_terms, normal_indicators, abnormal_indicators, judgement_hint, required_checks, evidence_query)
                values (:pattern_id, :name, :trigger_terms, :normal_indicators, :abnormal_indicators, :judgement_hint, :required_checks, :evidence_query)
                """,
                self._seed_event_patterns(),
            )
            conn.executemany(
                """
                insert into procedure_map
                (procedure_id, phenomenon, system, query, document_id, heading, chunk_id, priority, notes)
                values (:procedure_id, :phenomenon, :system, :query, :document_id, :heading, :chunk_id, :priority, :notes)
                """,
                self._build_procedure_map(chunks),
            )
            conn.execute(
                "insert or replace into meta (key, value) values ('schema_version', '1')"
            )
            conn.execute(
                "insert or replace into meta (key, value) values ('source_mtime', ?)",
                (str(self._source_mtime()),),
            )
        self._loaded = False
        self.documents = []
        self.chunks = []
        self.figures = []
        return {
            "status": "ok",
            "db_path": str(self.db_path),
            "documents": len(documents),
            "chunks": len(chunks),
            "figures": len(figures),
            "event_patterns": len(self._seed_event_patterns()),
        }

    def _scan_files(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        manifest = self._read_manifest()
        documents: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        figures: list[dict[str, Any]] = []
        for md_path in sorted(self.md_root.rglob("*.md")):
            rel_md = md_path.relative_to(self.domain_dir).as_posix()
            doc_id = md_path.stem
            m = manifest.get(f"md/{md_path.relative_to(self.md_root).as_posix()}", {})
            raw_text = md_path.read_text(encoding="utf-8", errors="ignore")
            frontmatter, body = _strip_frontmatter(raw_text)
            original_name = m.get("original_name") or frontmatter.get("original_name") or ""
            title = Path(original_name).stem if original_name else (self._title(body) or doc_id)
            category = m.get("category") or frontmatter.get("category") or md_path.parent.name
            doc = {
                "document_id": doc_id,
                "title": title,
                "category": category,
                "source_path": m.get("source_path") or frontmatter.get("source_path", ""),
                "md_path": rel_md,
                "pages": int(m.get("pages") or frontmatter.get("pages") or 0),
                "doc_role": self._doc_role(title, category),
                "char_count": len(body),
                "file_mtime": md_path.stat().st_mtime,
            }
            documents.append(doc)
            chunks.extend(self._split_chunks(doc, body))
            figures.extend(self._figures_for_doc(doc, body))
        return documents, chunks, figures

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            create table if not exists meta (
              key text primary key,
              value text not null
            );
            create table if not exists documents (
              document_id text primary key,
              title text,
              category text,
              source_path text,
              md_path text,
              pages integer,
              doc_role text,
              char_count integer,
              file_mtime real
            );
            create table if not exists chunks (
              chunk_id text primary key,
              document_id text,
              title text,
              path text,
              category text,
              heading text,
              content text,
              image_refs text,
              evidence_role text,
              ordinal integer,
              char_start integer
            );
            create virtual table if not exists chunks_fts using fts5(
              chunk_id unindexed,
              document_id unindexed,
              title,
              category,
              heading,
              content
            );
            create table if not exists figures (
              figure_id text primary key,
              document_id text,
              path text,
              caption text,
              width integer,
              height integer,
              figure_type text,
              title text,
              category text
            );
            create table if not exists event_patterns (
              pattern_id text primary key,
              name text,
              trigger_terms text,
              normal_indicators text,
              abnormal_indicators text,
              judgement_hint text,
              required_checks text,
              evidence_query text
            );
            create table if not exists procedure_map (
              procedure_id text primary key,
              phenomenon text,
              system text,
              query text,
              document_id text,
              heading text,
              chunk_id text,
              priority integer,
              notes text
            );
            """
        )

    def _load_from_sqlite(self) -> None:
        with self._connect() as conn:
            self.documents = [dict(r) for r in conn.execute("select * from documents order by document_id")]
            self.chunks = [dict(r) for r in conn.execute("select * from chunks order by document_id, ordinal")]
            self.figures = [dict(r) for r in conn.execute("select * from figures order by document_id, figure_id")]

    def _source_mtime(self) -> float:
        mtimes = [p.stat().st_mtime for p in self.md_root.rglob("*.md")]
        for p in [self.docs_root / "manifest.csv", self.docs_root / "status.csv"]:
            if p.exists():
                mtimes.append(p.stat().st_mtime)
        return max(mtimes) if mtimes else 0.0

    def _index_needs_sync(self) -> bool:
        if not self.db_path.exists():
            return True
        try:
            with self._connect() as conn:
                self._init_schema(conn)
                indexed = conn.execute("select count(*) from documents").fetchone()[0]
                row = conn.execute("select value from meta where key='source_mtime'").fetchone()
            if indexed == 0 or row is None:
                return True
            return float(row["value"]) < self._source_mtime()
        except sqlite3.DatabaseError:
            return True

    def _seed_event_patterns(self) -> list[dict[str, Any]]:
        rows = [
            {
                "name": "自动功率升降正常完成",
                "trigger_terms": "自动功率升降,自动功率控制,目标功率,功率变化速率,双极功率升降完成",
                "normal_indicators": "升降完成,告警消失,无闭锁,无保护动作,无失败",
                "abnormal_indicators": "功率升降失败,暂停,闭锁,保护动作,跳闸,功率控制模式切换异常",
                "judgement_hint": "若功率升降均完成，且无保护出口、闭锁、失败或持续告警，通常倾向可能正常；需核查功率曲线和命令来源。",
                "required_checks": "目标功率;功率变化速率;实际功率曲线;控制位置;是否人工/调度/自动曲线下发;是否有停止升降命令",
                "evidence_query": "自动功率控制 目标功率 功率变化速率 双极功率控制 斜坡发生器",
            },
            {
                "name": "阀冷泵阀切换伴随事件",
                "trigger_terms": "阀冷,阀冷却,内水冷,开关阀,高压泵,加药泵,工业水泵,补水泵",
                "normal_indicators": "到位信号恢复,备用切换成功,告警消失,水温正常,流量正常,压力正常",
                "abnormal_indicators": "水温高,流量低,压力低,泄漏,主循环泵异常,未切至备用泵,告警持续",
                "judgement_hint": "阀冷设备启停或到位变化不能单独判故障；若伴随温度/流量/压力异常或备用未切换，应提高异常等级。",
                "required_checks": "阀冷水温;流量;压力;泵运行状态;阀门命令来源;备用切换;告警复归",
                "evidence_query": "阀冷 内水冷 主循环泵 异常 处理预案",
            },
            {
                "name": "录波启动短时复归",
                "trigger_terms": "故障录波,录波启动,阀测控机箱录波,SER,事件记录",
                "normal_indicators": "短时产生,相继消失,无保护出口,无闭锁,无跳闸",
                "abnormal_indicators": "保护动作,跳闸,闭锁,持续告警,录波伴随一次设备异常",
                "judgement_hint": "录波启动是重要线索，但不等同于设备故障；必须结合保护动作报告、录波文件和一次/二次设备检查。",
                "required_checks": "录波触发通道;保护动作报告;是否有出口;录波波形;事件记录;告警是否复归",
                "evidence_query": "故障录波 启动 事件记录 SER 保护动作",
            },
            {
                "name": "保护动作或闭锁异常",
                "trigger_terms": "保护动作,闭锁,跳闸,功率回降,极闭锁,双极闭锁",
                "normal_indicators": "无",
                "abnormal_indicators": "保护动作,闭锁,跳闸,功率回降,重启不成功,设备异常",
                "judgement_hint": "出现保护动作、闭锁、跳闸或功率回降失败时，通常应判为可疑或异常，并按运行规程第五分册查处理预案。",
                "required_checks": "保护动作信号;出口记录;故障录波;闭锁对象;一次设备状态;调度汇报记录",
                "evidence_query": "保护动作 闭锁 跳闸 功率回降 故障录波 处理预案",
            },
        ]
        return [{**row, "pattern_id": f"pat_{_stable_id(row['name'], 8)}"} for row in rows]

    def _build_procedure_map(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        signals = [
            ("自动功率升降", "dc_station_control", ["功率升降", "双极功率控制", "斜坡发生器"]),
            ("阀冷异常", "valve_cooling", ["阀冷", "内水冷", "主循环泵", "冷却水"]),
            ("故障录波", "protection_recorder", ["故障录波", "录波启动", "事件记录"]),
            ("保护动作闭锁", "protection", ["保护动作", "闭锁", "跳闸"]),
            ("SER事件记录", "ser", ["SER", "顺序事件", "事件记录"]),
        ]
        for chunk in chunks:
            if self._is_toc_chunk(chunk):
                continue
            hay = f"{chunk.get('heading','')} {chunk.get('content','')}"
            for phenomenon, system, terms in signals:
                if any(term in hay for term in terms):
                    priority = self._procedure_priority(chunk, hay, phenomenon)
                    rows.append(
                        {
                            "procedure_id": f"proc_{_stable_id(phenomenon + chunk['chunk_id'], 12)}",
                            "phenomenon": phenomenon,
                            "system": system,
                            "query": ",".join(terms),
                            "document_id": chunk["document_id"],
                            "heading": chunk["heading"],
                            "chunk_id": chunk["chunk_id"],
                            "priority": priority,
                            "notes": chunk.get("evidence_role", ""),
                        }
                    )
        dedup: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = f"{row['phenomenon']}:{row['chunk_id']}"
            if key not in dedup or row["priority"] > dedup[key]["priority"]:
                dedup[key] = row
        return list(dedup.values())

    def _is_toc_chunk(self, chunk: dict[str, Any]) -> bool:
        heading = (chunk.get("heading") or "").strip().replace(" ", "")
        content = chunk.get("content") or ""
        ordinal = int(chunk.get("ordinal") or 0)
        if heading in {"目录", "目錄"}:
            return True
        if ordinal <= 15 and heading in {"目录", "目錄", "目次"}:
            return True

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return False
        toc_like = 0
        for line in lines:
            # MinerU/OCR often turns dot leaders into repeated dots/spaces before page numbers.
            if re.search(r"\.{2,}\s*\d+\s*$", line):
                toc_like += 1
            elif re.match(r"^\d+(?:\.\d+){1,4}\s+\S.{0,80}\s+\d+\s*$", line):
                toc_like += 1
        return ordinal <= 30 and len(lines) >= 4 and toc_like / len(lines) >= 0.45

    def _procedure_priority(self, chunk: dict[str, Any], hay: str, phenomenon: str = "") -> int:
        heading = chunk.get("heading") or ""
        title = chunk.get("title") or ""
        priority = 3

        if re.match(r"^\d+(?:\.\d+)*\s+\S+", heading):
            priority += 2
        if "处理预案" in heading:
            priority += 6
        elif "处理预案" in hay:
            priority += 4
        if "处理" in heading:
            priority += 3
        if chunk.get("evidence_role") == "handling_rule":
            priority += 4
        elif chunk.get("evidence_role") == "abnormal_symptom":
            priority += 2
        if any(k in title for k in ["设备异常", "故障应急", "应急处理", "规程之五"]):
            priority += 5
        if phenomenon == "阀冷异常" and any(k in heading for k in ["阀内水冷", "内水冷", "主循环泵"]):
            priority += 4
        if phenomenon == "自动功率升降" and any(k in heading for k in ["双极功率控制", "功率控制", "功率升降"]):
            priority += 4
        if phenomenon in {"故障录波", "SER事件记录"} and any(k in heading for k in ["故障录波", "顺序事件", "SER"]):
            priority += 4
        return priority

    def _read_manifest(self) -> dict[str, dict[str, str]]:
        path = self.docs_root / "manifest.csv"
        if not path.exists():
            return {}
        rows: dict[str, dict[str, str]] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows[row.get("md_path", "")] = row
        return rows

    def _title(self, text: str) -> str:
        for line in text.splitlines():
            m = HEADING_RE.match(line)
            if m:
                return m.group(2).strip()
        for line in text.splitlines():
            if line.strip() and not line.startswith("![]("):
                return line.strip()[:80]
        return ""

    def _doc_role(self, title: str, category: str) -> str:
        if "定值单" in title or "定值单" in category:
            return "setting_sheet"
        if "图册" in title:
            return "diagram_atlas"
        if "典型操作票" in title:
            return "operation_ticket"
        if "异常" in title or "故障" in title or "应急" in title:
            return "emergency_procedure"
        if "设备概况" in title:
            return "equipment_overview"
        if "运行方式" in title or "运行规定" in title:
            return "operation_rule"
        if "设计规范" in title or "技术说明书" in title:
            return "control_protection_spec"
        return "document"

    def _split_chunks(self, doc: dict[str, Any], text: str) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        current_heading = doc["title"]
        current: list[str] = []
        ordinal = 0
        char_pos = 0

        def flush() -> None:
            nonlocal ordinal, current
            content = "\n".join(current).strip()
            if not content:
                current = []
                return
            pieces = self._chunk_long_text(content, 2200)
            for piece in pieces:
                ordinal += 1
                chunk_id = f"{doc['document_id']}_c{ordinal:04d}"
                image_refs = ",".join(IMAGE_RE.findall(piece))
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": doc["document_id"],
                        "title": doc["title"],
                        "path": doc["md_path"],
                        "category": doc["category"],
                        "heading": current_heading,
                        "content": piece,
                        "image_refs": image_refs,
                        "evidence_role": self._evidence_role(current_heading + "\n" + piece),
                        "ordinal": ordinal,
                        "char_start": char_pos,
                    }
                )
            current = []

        for line in text.splitlines():
            m = HEADING_RE.match(line)
            if m:
                flush()
                current_heading = m.group(2).strip()
                current.append(line)
            else:
                current.append(line)
            char_pos += len(line) + 1
        flush()
        return chunks

    def _chunk_long_text(self, text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        paragraphs = re.split(r"\n\s*\n", text)
        pieces: list[str] = []
        buf = ""
        for para in paragraphs:
            if buf and len(buf) + len(para) + 2 > max_chars:
                pieces.append(buf)
                buf = ""
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    pieces.append(para[i : i + max_chars])
            else:
                buf = f"{buf}\n\n{para}".strip()
        if buf:
            pieces.append(buf)
        return pieces

    def _evidence_role(self, text: str) -> str:
        if any(k in text for k in ["处理预案", "处理", "事故处理"]):
            return "handling_rule"
        if any(k in text for k in ["现象", "告警", "故障", "异常", "闭锁"]):
            return "abnormal_symptom"
        if any(k in text for k in ["操作任务", "操作步骤", "典型操作"]):
            return "operation_step"
        if any(k in text for k in ["图", "接线图", "原理图", "系统图", "布置图"]):
            return "diagram"
        if any(k in text for k in ["定值", "整定"]):
            return "setting"
        if any(k in text for k in ["功能", "条件", "运行方式"]):
            return "normal_behavior"
        return "definition"

    def _figures_for_doc(self, doc: dict[str, Any], text: str) -> list[dict[str, Any]]:
        figures: list[dict[str, Any]] = []
        for i, match in enumerate(IMAGE_RE.finditer(text), 1):
            img = match.group(1)
            figure_id = f"{doc['document_id']}_f{i:04d}"
            caption = self._nearby_caption(text, match.start())
            figures.append(
                {
                    "figure_id": figure_id,
                    "document_id": doc["document_id"],
                    "path": img,
                    "caption": caption,
                    "width": 0,
                    "height": 0,
                    "figure_type": self._figure_type(caption, doc),
                    "title": doc["title"],
                    "category": doc["category"],
                }
            )
        return figures

    def _nearby_caption(self, text: str, pos: int) -> str:
        prefix = text[:pos].splitlines()[-5:]
        suffix = text[pos:].splitlines()[:5]
        candidates = [line.strip("# \t") for line in prefix + suffix if line.strip() and not line.strip().startswith("![](")]
        return candidates[-1][:120] if candidates else ""

    def _figure_type(self, caption: str, doc: dict[str, Any]) -> str:
        text = caption + " " + doc["title"]
        if "定值" in text:
            return "setting_scan"
        if "保护" in text and "图" in text:
            return "protection_logic"
        if "控制" in text and "图" in text:
            return "control_logic"
        if "接线图" in text or "系统图" in text:
            return "wiring_diagram"
        if "布置" in text or "屏位" in text:
            return "layout"
        return "other"

    def query(self, object_type: str, filters: dict[str, Any] | None = None,
              limit: int | None = None, order_by: str | None = None,
              offset: int | None = None) -> list[dict[str, Any]]:
        self.ensure_loaded()
        table = {
            "HvdcDocument": "documents",
            "DocumentChunk": "chunks",
            "DocumentFigure": "figures",
        }.get(object_type)
        if table:
            return self._query_sqlite(table, filters, limit, order_by, offset)
        rows = {
            "HvdcDocument": self.documents,
            "DocumentChunk": self.chunks,
            "DocumentFigure": self.figures,
        }.get(object_type, [])
        filtered = [row for row in rows if _matches_filters(row, filters)]
        if order_by:
            reverse = order_by.startswith("-")
            key = order_by[1:] if reverse else order_by
            filtered.sort(key=lambda r: r.get(key) or "", reverse=reverse)
        start = offset or 0
        end = start + limit if limit else None
        return filtered[start:end]

    def count(self, object_type: str, filters: dict[str, Any] | None = None) -> int:
        self.ensure_loaded()
        table = {
            "HvdcDocument": "documents",
            "DocumentChunk": "chunks",
            "DocumentFigure": "figures",
        }.get(object_type)
        if table:
            return len(self._query_sqlite(table, filters, None, None, None))
        return len(self.query(object_type, filters))

    def _query_sqlite(self, table: str, filters: dict[str, Any] | None,
                      limit: int | None, order_by: str | None, offset: int | None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for key, expected in (filters or {}).items():
            if key.endswith("__like"):
                clauses.append(f"{key[:-6]} like ?")
                params.append(f"%{expected}%")
            else:
                clauses.append(f"{key} = ?")
                params.append(expected)
        sql = f"select * from {table}"
        if clauses:
            sql += " where " + " and ".join(clauses)
        if order_by:
            reverse = order_by.startswith("-")
            key = order_by[1:] if reverse else order_by
            sql += f" order by {key} {'desc' if reverse else 'asc'}"
        elif table == "chunks":
            sql += " order by document_id, ordinal"
        elif table == "figures":
            sql += " order by document_id, figure_id"
        else:
            sql += " order by document_id"
        if limit:
            sql += " limit ?"
            params.append(limit)
            if offset:
                sql += " offset ?"
                params.append(offset)
        elif offset:
            sql += " limit -1 offset ?"
            params.append(offset)
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, params)]

    def query_by_id(self, object_type: str, id_value: Any) -> dict[str, Any] | None:
        id_fields = {
            "HvdcDocument": "document_id",
            "DocumentChunk": "chunk_id",
            "DocumentFigure": "figure_id",
        }
        id_field = id_fields.get(object_type)
        if not id_field:
            return None
        rows = self.query(object_type, {id_field: id_value}, limit=1)
        return rows[0] if rows else None

    def search_text(self, keyword: str, object_types: list[str] | None = None, limit: int = 20) -> list[dict[str, Any]]:
        hits = self.search(keyword, limit=limit, include_figures=True)
        return hits.get("chunks", [])[:limit]

    def search(self, query: str, category: str = "", doc_role: str = "",
               limit: int = 8, include_figures: bool = True) -> dict[str, Any]:
        self.ensure_loaded()
        terms = _text_terms(query)
        db_chunks = self._search_chunks_sqlite(terms, category, doc_role, limit)
        if db_chunks:
            figures = self._search_figures_sqlite(terms, limit=limit) if include_figures else []
            return {
                "query": query,
                "terms": terms,
                "chunks": db_chunks,
                "figures": figures,
                "usage_note": "片段用于定位依据；形成结论前建议 read_substation_doc 读取章节上下文。",
            }
        scored: list[tuple[float, dict[str, Any]]] = []
        docs_by_id = {doc["document_id"]: doc for doc in self.documents}
        for chunk in self.chunks:
            doc = docs_by_id.get(chunk["document_id"], {})
            if category and category not in chunk.get("category", ""):
                continue
            if doc_role and doc.get("doc_role") != doc_role:
                continue
            hay = "\n".join([chunk.get("title", ""), chunk.get("heading", ""), chunk.get("content", "")])
            score = self._score(hay, terms)
            if score <= 0:
                continue
            item = dict(chunk)
            item["score"] = round(score, 3)
            item["document_role"] = doc.get("doc_role", "")
            item["source_path"] = doc.get("source_path", "")
            item["snippet"] = self._snippet(chunk["content"], terms)
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        chunks = [item for _, item in scored[:limit]]
        figures = self._search_figures(terms, limit=limit) if include_figures else []
        return {
            "query": query,
            "terms": terms,
            "chunks": chunks,
            "figures": figures,
            "usage_note": "片段用于定位依据；形成结论前建议 read_substation_doc 读取章节上下文。",
        }

    def _search_chunks_sqlite(self, terms: list[str], category: str, doc_role: str, limit: int) -> list[dict[str, Any]]:
        fts_terms = [t for t in terms if re.match(r"^[A-Za-z0-9_.#-]+$|^[\u4e00-\u9fff]{2,}$", t)]
        if not fts_terms:
            return []
        # FTS5 handles Chinese poorly without tokenizer config, so use LIKE OR terms for dependable recall.
        clauses = []
        params: list[Any] = []
        for term in fts_terms[:16]:
            clauses.append("(c.title like ? or c.heading like ? or c.content like ?)")
            like = f"%{term}%"
            params.extend([like, like, like])
        sql = """
            select c.*, d.doc_role as document_role, d.source_path as source_path
            from chunks c join documents d on d.document_id = c.document_id
            where
        """ + " or ".join(clauses)
        if category:
            sql += " and c.category like ?"
            params.append(f"%{category}%")
        if doc_role:
            sql += " and d.doc_role = ?"
            params.append(doc_role)
        sql += " limit 400"
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(sql, params)]
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            hay = "\n".join([row.get("title", ""), row.get("heading", ""), row.get("content", "")])
            score = self._score(hay, terms)
            if score <= 0:
                continue
            row["score"] = round(score, 3)
            row["snippet"] = self._snippet(row.get("content", ""), terms)
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in scored[:limit]]

    def _search_figures_sqlite(self, terms: list[str], limit: int) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        for term in terms[:12]:
            clauses.append("(title like ? or caption like ? or path like ? or figure_type like ?)")
            like = f"%{term}%"
            params.extend([like, like, like, like])
        if not clauses:
            return []
        sql = "select * from figures where " + " or ".join(clauses) + " limit 200"
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(sql, params)]
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            hay = " ".join([row.get("title", ""), row.get("caption", ""), row.get("path", ""), row.get("figure_type", "")])
            score = self._score(hay, terms)
            if score > 0:
                row["score"] = round(score, 3)
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in scored[:limit]]

    def _score(self, text: str, terms: list[str]) -> float:
        score = 0.0
        lowered = text.lower()
        for term in terms:
            if not term:
                continue
            count = lowered.count(term.lower())
            if count:
                score += min(count, 6) * (2.5 if len(term) >= 4 else 1.0)
        return score

    def _snippet(self, content: str, terms: list[str], window: int = 280) -> str:
        best = -1
        lower = content.lower()
        for term in terms:
            idx = lower.find(term.lower())
            if idx >= 0:
                best = idx
                break
        if best < 0:
            return content[:window]
        start = max(best - window // 2, 0)
        end = min(start + window, len(content))
        return content[start:end].replace("\n", " ")

    def _search_figures(self, terms: list[str], limit: int) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for fig in self.figures:
            hay = " ".join([fig.get("title", ""), fig.get("caption", ""), fig.get("path", ""), fig.get("figure_type", "")])
            score = self._score(hay, terms)
            if score > 0:
                item = dict(fig)
                item["score"] = round(score, 3)
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def read_document(self, document_id: str = "", path: str = "", chunk_id: str = "",
                      heading: str = "", max_chars: int = 6000) -> dict[str, Any]:
        self.ensure_loaded()
        focus_doc: dict[str, Any] | None = None
        focus_chunk: dict[str, Any] | None = None
        if chunk_id:
            focus_chunk = self.query_by_id("DocumentChunk", chunk_id)
            if focus_chunk:
                focus_doc = self.query_by_id("HvdcDocument", focus_chunk["document_id"])
        elif document_id:
            focus_doc = self.query_by_id("HvdcDocument", document_id)
        elif path:
            focus_doc = next((d for d in self.documents if d["md_path"] == path or path.endswith(d["md_path"])), None)
        if not focus_doc:
            return {"error": "未找到文档", "document_id": document_id, "path": path, "chunk_id": chunk_id}

        doc_chunks = [c for c in self.chunks if c["document_id"] == focus_doc["document_id"]]
        if heading:
            doc_chunks = [c for c in doc_chunks if heading in c.get("heading", "") or heading in c.get("content", "")]
        elif focus_chunk:
            ordinal = focus_chunk["ordinal"]
            doc_chunks = [c for c in doc_chunks if abs(c["ordinal"] - ordinal) <= 2]

        content_parts: list[str] = []
        included: list[str] = []
        for chunk in doc_chunks:
            block = f"## {chunk['heading']}\n{chunk['content']}"
            if len("\n\n".join(content_parts + [block])) > max_chars:
                break
            content_parts.append(block)
            included.append(chunk["chunk_id"])
        figures = [f for f in self.figures if f["document_id"] == focus_doc["document_id"]][:20]
        return {
            "document": focus_doc,
            "focus_chunk_id": chunk_id,
            "included_chunk_ids": included,
            "content": "\n\n".join(content_parts),
            "figures_preview": figures,
            "truncated": len(doc_chunks) > len(included),
        }

    def event_patterns_for_text(self, text: str) -> list[dict[str, Any]]:
        self.ensure_loaded()
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute("select * from event_patterns order by name")]
        matched: list[dict[str, Any]] = []
        for row in rows:
            terms = [t for t in row.get("trigger_terms", "").split(",") if t]
            score = sum(1 for term in terms if term in text)
            if score:
                item = dict(row)
                item["match_score"] = score
                item["matched_terms"] = [term for term in terms if term in text]
                matched.append(item)
        matched.sort(key=lambda r: r["match_score"], reverse=True)
        return matched

    def procedure_map_search(self, phenomenon: str, device_name: str = "", limit: int = 5) -> list[dict[str, Any]]:
        self.ensure_loaded()
        terms = _text_terms(" ".join([phenomenon, device_name]))
        clauses = []
        params: list[Any] = []
        for term in terms[:12]:
            clauses.append("(phenomenon like ? or system like ? or query like ? or heading like ?)")
            like = f"%{term}%"
            params.extend([like, like, like, like])
        if not clauses:
            return []
        sql = "select * from procedure_map where " + " or ".join(clauses) + " order by priority desc limit 300"
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(sql, params)]
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            chunk = self.query_by_id("DocumentChunk", row["chunk_id"]) or {}
            hay = " ".join(
                [
                    row.get("phenomenon", ""),
                    row.get("system", ""),
                    row.get("query", ""),
                    row.get("heading", ""),
                    chunk.get("title", ""),
                    chunk.get("heading", ""),
                    chunk.get("content", ""),
                ]
            )
            dynamic = float(row.get("priority") or 0) + self._score(hay, terms)
            if self._is_toc_chunk(chunk):
                dynamic -= 20
            if chunk.get("evidence_role") == "handling_rule":
                dynamic += 3
            row["score"] = round(dynamic, 3)
            scored.append((dynamic, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:limit]]


class SubstationResolver:
    def __init__(self, index: SubstationDocumentIndex):
        self.index = index

    def query(self, object_type: str, filters: dict[str, Any] | None = None,
              limit: int | None = None, order_by: str | None = None,
              offset: int | None = None) -> list[dict[str, Any]]:
        return self.index.query(object_type, filters, limit, order_by, offset)

    def count(self, object_type: str, filters: dict[str, Any] | None = None) -> int:
        return self.index.count(object_type, filters)

    def query_by_id(self, object_type: str, id_value: Any) -> dict[str, Any] | None:
        return self.index.query_by_id(object_type, id_value)

    def search_text(self, keyword: str, object_types: list[str] | None = None, limit: int = 20) -> list[dict[str, Any]]:
        return self.index.search_text(keyword, object_types, limit)


def _subsystem(device: str, summary: str = "") -> str:
    text = device + " " + summary
    if "录波" in text:
        return "protection_recorder"
    if "阀冷" in text or "内水冷" in text or "冷却" in text:
        return "valve_cooling"
    if "VCE" in text or "阀控制" in text:
        return "vce"
    if "阀组控制" in text:
        return "valve_group_control"
    if "直流站控" in text:
        return "dc_station_control"
    if "交流站控" in text:
        return "ac_station_control"
    if "站用电" in text:
        return "station_service_power"
    return "unknown"


def parse_event_chain(event_text: str, station_name: str = "") -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for match in EVENT_RE.finditer(event_text):
        summary = re.sub(r"\s+", " ", match.group("summary")).strip()
        device = match.group("device").strip()
        level = int(match.group("level"))
        event_id = f"ev_{int(match.group('idx')):03d}_{_stable_id(match.group(0), 8)}"
        events.append(
            {
                "event_id": event_id,
                "station_name": match.group("station").strip(),
                "device_name": device,
                "subsystem": _subsystem(device, summary),
                "event_type": match.group("event_type").strip(),
                "start_time": match.group("start"),
                "end_time": match.group("end"),
                "level": level,
                "summary": summary,
                "nature": (match.group("nature") or "").strip(),
            }
        )
    events.sort(key=lambda e: e["start_time"])
    if not events:
        return {"events": [], "event_chain": None, "parse_note": "未能按预期格式解析事件，请检查输入。"}
    chain = {
        "chain_id": f"chain_{_stable_id(events[0]['start_time'] + events[-1]['end_time'])}",
        "station_name": station_name or events[0]["station_name"],
        "window_start": events[0]["start_time"],
        "window_end": max(e["end_time"] for e in events),
        "primary_event_type": _primary_event_type(events),
        "involved_subsystems": ",".join(sorted({e["subsystem"] for e in events})),
        "event_count": len(events),
        "chain_summary": "",
    }
    return {"events": events, "event_chain": chain}


def _primary_event_type(events: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for event in events:
        counts[event["event_type"]] = counts.get(event["event_type"], 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _event_lines(events: list[dict[str, Any]]) -> list[str]:
    return [
        f"{i}. {e['start_time']} 至 {e['end_time']}，{e['device_name']}，{e['event_type']}，等级{e['level']}：{e['summary']}"
        for i, e in enumerate(events, 1)
    ]


def analyze_event_chain(events: str, evidence_query: str = "", require_citations: bool = True,
                        index: SubstationDocumentIndex | None = None) -> dict[str, Any]:
    parsed = _coerce_events(events)
    evs = parsed.get("events", [])
    if not evs:
        return {"error": "未能解析事件链", "parsed": parsed}
    main = _primary_event_type(evs)
    summaries = " ".join(e["summary"] + " " + e["event_type"] + " " + e["device_name"] for e in evs)
    patterns = index.event_patterns_for_text(summaries) if index is not None else []
    has_completed_power = "功率升降完成" in summaries and "自动功率升降" in summaries
    has_recorder_clear = "录波" in summaries and any(k in summaries for k in ["消失", "相继消失"])
    has_fault_keywords = any(k in summaries for k in ["闭锁", "跳闸", "保护动作", "失败", "不成功", "持续", "死机"])
    has_valve_cooling = any(e["subsystem"] == "valve_cooling" for e in evs)
    max_level = max(int(e.get("level") or 0) for e in evs)

    if has_fault_keywords or max_level <= 1:
        judgement = "可疑"
        confidence = 0.62
        reason = "事件链含保护动作/闭锁/失败/高等级等异常关键词或等级较高，需要结合录波和实时量复核。"
    elif has_completed_power and has_recorder_clear:
        judgement = "可能正常"
        confidence = 0.72
        reason = "两次自动功率升降均显示完成，录波/告警为短时产生后消失；当前文本未见保护出口、闭锁或功率升降失败。"
    elif has_completed_power:
        judgement = "可能正常"
        confidence = 0.66
        reason = "主导事件像自动功率控制过程且显示完成，但仍需核查功率曲线和相关伴随事件。"
    else:
        judgement = "证据不足"
        confidence = 0.45
        reason = "仅凭事件摘要无法判断主因和异常性，需要补充运行量、保护动作报告和规程依据。"

    required_checks = [
        "核查双极实际功率曲线、目标功率、功率变化速率和两次升降命令来源",
        "核查是否存在保护出口、闭锁、功率升降暂停/失败或人工干预记录",
        "核查故障录波启动原因、录波文件和保护动作报告，确认是否仅为扰动/操作伴随启动",
    ]
    if has_valve_cooling:
        required_checks.append("核查阀冷水温、流量、压力、泵/阀门命令来源、备用切换及告警是否复归")

    evidence = {}
    if require_citations and index is not None:
        query = evidence_query or "自动功率升降 直流站控 功率变化速率 目标功率 阀冷 故障录波 SER"
        evidence = index.search(query, limit=8, include_figures=True)

    return {
        "timeline": _event_lines(evs),
        "primary_event": main,
        "companion_events": [
            e for e in evs
            if e["event_type"] != main or e["subsystem"] in {"valve_cooling", "protection_recorder", "vce", "valve_group_control"}
        ],
        "interpretation": {
            "narrative": _narrative(evs),
            "main_cause_hypothesis": "以直流站控自动功率升降/双极功率控制过程为主线，阀冷操作和多系统录波启动更可能是同时间窗内的伴随事件或扰动响应。",
            "anomaly_judgement": judgement,
            "judgement_reason": reason,
            "confidence": confidence,
            "required_checks": required_checks,
            "matched_patterns": [
                {
                    "name": p["name"],
                    "matched_terms": p["matched_terms"],
                    "judgement_hint": p["judgement_hint"],
                    "required_checks": p["required_checks"],
                }
                for p in patterns[:4]
            ],
        },
        "evidence": evidence,
        "answer_contract": "回答时需区分：用户事件事实、文档依据、智能体推断、需补充核查项。",
    }


def _coerce_events(events: str) -> dict[str, Any]:
    if not events:
        return {"events": []}
    try:
        data = json.loads(events)
        if isinstance(data, dict) and "events" in data:
            return data
        if isinstance(data, list):
            return {"events": data}
    except json.JSONDecodeError:
        pass
    return parse_event_chain(events)


def _narrative(events: list[dict[str, Any]]) -> str:
    power = [e for e in events if "功率升降" in e["event_type"] or "功率" in e["summary"]]
    valve = [e for e in events if e["subsystem"] == "valve_cooling"]
    recorder = [e for e in events if "录波" in e["summary"]]
    parts: list[str] = []
    if power:
        parts.append(f"时间窗内出现 {len(power)} 次直流站控自动功率升降，摘要均指向设定目标功率/变化速率后执行升降并完成。")
    if valve:
        parts.append("功率升降期间夹杂阀冷系统阀门到位信号变化及泵停运/切换类事件。")
    if recorder:
        parts.append("随后多个站控、VCE 或阀组控制相关录波启动告警短时产生并消失。")
    return "".join(parts) or "事件链包含多个同站运行事件，需结合时间关系和文档依据进一步判别。"


def find_relevant_procedures(phenomenon: str, device_name: str = "", limit: int = 5,
                             index: SubstationDocumentIndex | None = None) -> dict[str, Any]:
    if index is None:
        return {"error": "index not available"}
    mapped = index.procedure_map_search(phenomenon, device_name, limit=limit)
    query = " ".join([phenomenon, device_name, "处理预案 运行规定 设计规范"])
    result = index.search(query, limit=limit, include_figures=True)
    mapped_chunks: list[dict[str, Any]] = []
    for hit in mapped:
        chunk = index.query_by_id("DocumentChunk", hit["chunk_id"])
        if chunk:
            chunk = dict(chunk)
            chunk["procedure_map"] = hit
            mapped_chunks.append(chunk)
    seen = {c["chunk_id"] for c in mapped_chunks}
    procedures = mapped_chunks + [c for c in result["chunks"] if c["chunk_id"] not in seen]
    return {
        "phenomenon": phenomenon,
        "device_name": device_name,
        "procedure_map_hits": mapped,
        "procedures": procedures[:limit],
        "figures": result["figures"],
        "usage_note": "这些是候选规程/预案依据；最终判断仍需结合用户事件中的持续时间、是否复归、保护出口和实时量。",
    }


def register(registry: FunctionRegistry, repository: ObjectRepository, ontology: Ontology):
    domain_dir = Path(__file__).resolve().parent.parent
    index = SubstationDocumentIndex(domain_dir)
    registry.register_resolver("substation_document_index", SubstationResolver(index))

    fn_map = {
        "search_substation_docs": lambda **kw: index.search(
            kw.get("query", "") or "",
            category=kw.get("category", "") or "",
            doc_role=kw.get("doc_role", "") or "",
            limit=kw.get("limit", 8) or 8,
            include_figures=kw.get("include_figures", True),
        ),
        "read_substation_doc": lambda **kw: index.read_document(
            document_id=kw.get("document_id", "") or "",
            path=kw.get("path", "") or "",
            chunk_id=kw.get("chunk_id", "") or "",
            heading=kw.get("heading", "") or "",
            max_chars=kw.get("max_chars", 6000) or 6000,
        ),
        "parse_event_chain": lambda **kw: parse_event_chain(
            kw.get("event_text", "") or "",
            station_name=kw.get("station_name", "") or "",
        ),
        "analyze_event_chain": lambda **kw: analyze_event_chain(
            kw.get("events", "") or "",
            evidence_query=kw.get("evidence_query", "") or "",
            require_citations=kw.get("require_citations", True),
            index=index,
        ),
        "find_relevant_procedures": lambda **kw: find_relevant_procedures(
            kw.get("phenomenon", "") or "",
            device_name=kw.get("device_name", "") or "",
            limit=kw.get("limit", 5) or 5,
            index=index,
        ),
        "sync_substation_index": lambda **kw: index.sync(),
    }

    for name, fn in fn_map.items():
        func_def = ontology.functions.get(name)
        if func_def:
            registry.register(name, lambda _fn=fn, **kw: _json_result(_fn(**kw)), func_def)
