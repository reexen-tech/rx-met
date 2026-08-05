"""Shared identities for resumable quantization artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_source_signature(source_model: str | Path) -> dict[str, Any]:
    """Build a cheap, deterministic identity for one local HF checkpoint."""

    source = Path(source_model).expanduser().resolve()
    metadata_hashes: dict[str, str] = {}
    for filename in ("config.json", "model.safetensors.index.json"):
        path = source / filename
        if path.is_file():
            metadata_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint_files = []
    for pattern in ("*.safetensors", "*.bin"):
        for path in sorted(source.glob(pattern)):
            stat = path.stat()
            checkpoint_files.append(
                {
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "path": str(source),
        "metadata_sha256": metadata_hashes,
        "checkpoint_files": checkpoint_files,
    }


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
