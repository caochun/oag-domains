from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oag.ontology.repository import ObjectRepository
from oag.ontology.schema import Ontology

from .utils import _stable_id

def _upsert_runtime_rows(repository: ObjectRepository | None, object_type: str, rows: list[dict[str, Any]]) -> None:
    if repository is None:
        return
    id_field = repository.ontology.get_id_column(object_type)
    for row in rows:
        if id_field and row.get(id_field):
            existing = repository.query_by_id(object_type, row[id_field])
            if existing:
                repository.update_record(object_type, row[id_field], row)
                continue
        repository.insert_record(object_type, row)

class RuntimeMemoryAdapter:
    def __init__(self, ontology: Ontology, object_type: str, source, domain_dir: Path):
        self.ontology = ontology
        self.object_type = object_type
        self.source = source
        self.domain_dir = domain_dir
        self.id_field = source.id_field or ontology.get_id_column(object_type)
        self.rows: list[dict[str, Any]] = []

    @classmethod
    def factory(cls, domain_dir: str | Path):
        base_dir = Path(domain_dir).resolve()

        def build(ontology: Ontology, object_type: str, source, **kwargs):
            return cls(ontology, object_type, source, base_dir)

        return build

    def query(self, object_type: str, filters: dict[str, Any] | None = None,
              limit: int | None = None, order_by: str | None = None,
              offset: int | None = None) -> list[dict]:
        rows = self._apply_filters([dict(row) for row in self.rows], filters)
        rows = self._apply_order(rows, order_by)
        return self._apply_window(rows, limit, offset)

    def count(self, object_type: str, filters: dict[str, Any] | None = None) -> int:
        return len(self.query(object_type, filters))

    def query_by_id(self, object_type: str, id_value: Any) -> dict | None:
        if not self.id_field:
            return None
        rows = self.query(object_type, {self.id_field: id_value}, limit=1)
        return rows[0] if rows else None

    def search_text(self, keyword: str, object_types: list[str] | None = None,
                    limit: int = 20) -> list[dict]:
        if not keyword:
            return []
        obj_def = self.ontology.objects[self.object_type]
        text_cols = [name for name, prop in obj_def.properties.items() if prop.type == "str"]
        results = []
        for row in self.rows:
            matched = [col for col in text_cols if row.get(col) and keyword in str(row[col])]
            if matched:
                record = dict(row)
                record["_object_type"] = self.object_type
                record["_matched_field"] = ", ".join(matched)
                results.append(record)
            if len(results) >= limit:
                break
        return results

    def insert_record(self, object_type: str, data: dict) -> dict:
        record = self._project(data)
        if self.id_field and self.id_field not in record:
            record[self.id_field] = f"{object_type.lower()}_{_stable_id(json.dumps(record, ensure_ascii=False), 12)}"
        self.rows.append(record)
        return dict(record)

    def update_record(self, object_type: str, id_value: Any, data: dict) -> dict:
        if not self.id_field:
            raise ValueError(f"{object_type} 没有声明 id 字段，不能 update")
        patch = self._project(data)
        for row in self.rows:
            if row.get(self.id_field) == id_value:
                row.update({k: v for k, v in patch.items() if k != self.id_field})
                return {"updated": 1}
        return {"updated": 0}

    def delete_record(self, object_type: str, id_value: Any) -> dict:
        if not self.id_field:
            raise ValueError(f"{object_type} 没有声明 id 字段，不能 delete")
        before = len(self.rows)
        self.rows = [row for row in self.rows if row.get(self.id_field) != id_value]
        return {"deleted": before - len(self.rows)}

    def table_count(self, object_type: str) -> int:
        return len(self.rows)

    def _project(self, data: dict) -> dict:
        valid = set(self.ontology.objects[self.object_type].properties.keys())
        return {key: value for key, value in data.items() if key in valid}

    def _apply_filters(self, rows: list[dict], filters: dict[str, Any] | None) -> list[dict]:
        result = list(rows)
        for key, value in (filters or {}).items():
            field, op = key.split("__", 1) if "__" in key else (key, "eq")
            if op == "gt":
                result = [row for row in result if row.get(field) > value]
            elif op == "gte":
                result = [row for row in result if row.get(field) >= value]
            elif op == "lt":
                result = [row for row in result if row.get(field) < value]
            elif op == "lte":
                result = [row for row in result if row.get(field) <= value]
            elif op == "ne":
                result = [row for row in result if row.get(field) != value]
            elif op == "like":
                result = [row for row in result if value in str(row.get(field, ""))]
            else:
                result = [row for row in result if row.get(field) == value]
        return result

    def _apply_order(self, rows: list[dict], order_by: str | None) -> list[dict]:
        if not order_by:
            return rows
        reverse = order_by.startswith("-")
        field = order_by.lstrip("-")
        return sorted(rows, key=lambda row: row.get(field), reverse=reverse)

    def _apply_window(self, rows: list[dict], limit: int | None, offset: int | None) -> list[dict]:
        if offset:
            rows = rows[offset:]
        if limit:
            rows = rows[:limit]
        return rows
