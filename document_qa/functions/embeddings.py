from __future__ import annotations

import importlib.util
import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class EmbeddingConfig:
    enabled: bool
    provider: str
    api_key: str
    api_url: str
    model: str
    local_model_path: str
    batch_size: int
    max_chars: int
    min_similarity: float


def resolve_embedding_config() -> EmbeddingConfig:
    provider = os.getenv("DOCUMENT_QA_EMBEDDING_PROVIDER", "openai").strip().lower()
    default_model = "BAAI/bge-small-zh-v1.5" if provider == "local" else "text-embedding-3-small"
    return EmbeddingConfig(
        enabled=coerce_bool(os.getenv("DOCUMENT_QA_EMBEDDINGS", "")),
        provider=provider,
        api_key=os.getenv("DOCUMENT_QA_EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY", "sk-placeholder"),
        api_url=os.getenv("DOCUMENT_QA_EMBEDDING_API_URL") or os.getenv("LLM_API_URL", "http://localhost:8090/v1"),
        model=os.getenv("DOCUMENT_QA_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL", default_model),
        local_model_path=os.getenv("DOCUMENT_QA_EMBEDDING_LOCAL_PATH", ""),
        batch_size=max(1, min(int(os.getenv("DOCUMENT_QA_EMBEDDING_BATCH_SIZE", "32")), 128)),
        max_chars=max(300, min(int(os.getenv("DOCUMENT_QA_EMBEDDING_MAX_CHARS", "1800")), 6000)),
        min_similarity=max(0.0, min(float(os.getenv("DOCUMENT_QA_EMBEDDING_MIN_SIMILARITY", "0.55")), 1.0)),
    )


def cached_hf_snapshot(model_id: str) -> Path | None:
    if not model_id or "/" not in model_id:
        return None
    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [
        path for path in snapshots.iterdir()
        if path.is_dir() and (path / "config.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_faiss():
    if importlib.util.find_spec("faiss") is None:
        return None
    import faiss  # type: ignore

    return faiss


def embedding_text(row: dict[str, Any], max_chars: int) -> str:
    header = "\n".join(
        part for part in [
            f"标题：{row.get('title', '')}",
            f"类别：{row.get('category', '')}",
            f"机关：{row.get('agency', '')}",
            f"类型：{row.get('doc_type', '')}",
            f"小节：{row.get('heading', '')}",
        ]
        if part and not part.endswith("：")
    )
    content = re.sub(r"\s+", " ", str(row.get("content", "") or "")).strip()
    return f"{header}\n正文：{content}"[:max_chars]


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(item) / norm for item in vector]


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(data: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}f", data))


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def batched(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
