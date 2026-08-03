"""Sharded packed-only storage for large Q4_0_64 GPTQ exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import torch

from .q4_0_64 import pack_q4_0_64, unpack_q4_0_64


FORMAT = "Q4_0_64"
VERSION = 2
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _compact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    packed = entry.get("packed")
    if not isinstance(packed, torch.Tensor):
        raise ValueError("GPTQ sidecar entry is missing packed bytes")
    shape = entry.get("shape")
    codes = entry.get("codes")
    if shape is None:
        if not isinstance(codes, torch.Tensor):
            logical_last = packed.shape[-1] // 34 * 64
            shape = (*packed.shape[:-1], logical_last)
        else:
            shape = tuple(codes.shape)
    result: dict[str, Any] = {
        "packed": packed.detach().to(device="cpu", dtype=torch.uint8).contiguous(),
        "shape": list(shape),
    }
    for key in ("gguf_name", "source_name", "method"):
        if key in entry:
            result[key] = entry[key]
    return result


def _expand_split_expert(
    name: str,
    entry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not name.endswith(".mlp.experts.gate_up_proj"):
        return {name: entry}
    match = _LAYER_RE.search(name)
    codes = entry.get("codes")
    scales = entry.get("scales")
    if not isinstance(codes, torch.Tensor) or not isinstance(scales, torch.Tensor):
        packed = entry.get("packed")
        shape = entry.get("shape")
        if not isinstance(packed, torch.Tensor) or not isinstance(shape, list):
            raise ValueError(f"cannot unpack fused expert sidecar entry: {name}")
        codes, scales = unpack_q4_0_64(
            packed, logical_size=int(shape[-1])
        )
    if match is None or not isinstance(codes, torch.Tensor) or not isinstance(
        scales, torch.Tensor
    ):
        raise ValueError(f"cannot split fused expert sidecar entry: {name}")
    if codes.ndim != 3 or codes.shape[-2] % 2:
        raise ValueError(f"invalid fused expert codes shape: {tuple(codes.shape)}")
    n_ff = codes.shape[-2] // 2
    layer = int(match.group(1))
    result: dict[str, dict[str, Any]] = {}
    for gguf_part, row_slice in (
        ("ffn_gate_exps", slice(0, n_ff)),
        ("ffn_up_exps", slice(n_ff, None)),
    ):
        split_codes = codes[:, row_slice, :].contiguous()
        split_scales = scales[:, row_slice, :].contiguous()
        gguf_name = f"blk.{layer}.{gguf_part}.weight"
        key = gguf_name
        result[key] = {
            "packed": pack_q4_0_64(split_codes, split_scales),
            "shape": list(split_codes.shape),
            "source_name": name,
            "gguf_name": gguf_name,
            "method": entry.get("method", "gptq"),
        }
    return result


class SidecarShardWriter:
    def __init__(
        self,
        directory: str | Path,
        *,
        source_model: str,
        resume: bool = False,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source_model = source_model
        self.resume = resume
        self._shards: list[dict[str, Any]] = []

    def write_layer(
        self,
        layer_index: int,
        tensors: dict[str, dict[str, Any]],
    ) -> str:
        filename = f"gptq_q4_0_64.layer{layer_index:03d}.pt"
        path = self.directory / filename
        if self.resume and path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                not isinstance(payload, dict)
                or payload.get("format") != FORMAT
                or payload.get("version") != VERSION
            ):
                raise ValueError(f"invalid existing sidecar shard: {path}")
            names = sorted(payload.get("tensors", {}))
        else:
            expanded: dict[str, dict[str, Any]] = {}
            for name, entry in tensors.items():
                expanded.update(_expand_split_expert(name, entry))
            compact = {
                name: _compact_entry(entry) for name, entry in expanded.items()
            }
            payload = {
                "format": FORMAT,
                "version": VERSION,
                "layer": layer_index,
                "tensors": compact,
            }
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            names = sorted(compact)
        self._shards.append({"file": filename, "tensors": names})
        return filename

    def finish(self) -> Path:
        manifest = {
            "format": FORMAT,
            "version": VERSION,
            "layout": "packed-only-sharded",
            "source_model": self.source_model,
            "shards": sorted(self._shards, key=lambda item: item["file"]),
        }
        path = self.directory / "gptq_q4_0_64.pt"
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(manifest, temporary)
        temporary.replace(path)
        return path


class SidecarShardReader:
    """Load at most one packed sidecar shard at a time."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path)
        manifest = torch.load(self.path, map_location="cpu", weights_only=True)
        if (
            not isinstance(manifest, dict)
            or manifest.get("format") != FORMAT
            or manifest.get("version") != VERSION
            or not isinstance(manifest.get("shards"), list)
        ):
            raise ValueError(f"invalid GPTQ sidecar v2 manifest: {self.path}")
        self.source_model = manifest.get("source_model")
        self._locations: dict[str, str] = {}
        for descriptor in manifest["shards"]:
            filename = descriptor.get("file")
            names = descriptor.get("tensors")
            if not isinstance(filename, str) or not isinstance(names, list):
                raise ValueError(f"invalid shard descriptor in {self.path}")
            for name in names:
                if name in self._locations:
                    raise ValueError(f"duplicate sidecar tensor: {name}")
                self._locations[name] = filename
        self._cached_file: str | None = None
        self._cached_tensors: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._locations)

    def __iter__(self):
        return iter(self._locations)

    def __getitem__(self, name: str) -> dict[str, Any]:
        entry = self.get(name)
        if entry is None:
            raise KeyError(name)
        return entry

    def get(self, name: str) -> dict[str, Any] | None:
        filename = self._locations.get(name)
        if filename is None:
            return None
        if filename != self._cached_file:
            shard_path = self.path.parent / filename
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            tensors = shard.get("tensors") if isinstance(shard, dict) else None
            if (
                not isinstance(shard, dict)
                or shard.get("format") != FORMAT
                or shard.get("version") != VERSION
                or not isinstance(tensors, dict)
            ):
                raise ValueError(f"invalid GPTQ sidecar shard: {shard_path}")
            self._cached_file = filename
            self._cached_tensors = tensors
        return self._cached_tensors[name]


def write_sidecar_v2(
    directory: str | Path,
    tensors: dict[str, dict[str, Any]],
    *,
    source_model: str,
    resume: bool = False,
) -> Path:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for name, entry in tensors.items():
        for expanded_name, expanded_entry in _expand_split_expert(
            name, entry
        ).items():
            match = _LAYER_RE.search(expanded_name)
            layer_index = int(match.group(1)) if match is not None else -1
            grouped.setdefault(layer_index, {})[expanded_name] = expanded_entry
    writer = SidecarShardWriter(directory, source_model=source_model, resume=resume)
    for layer_index in sorted(grouped):
        writer.write_layer(layer_index, grouped[layer_index])
    return writer.finish()


def iter_sidecar_v2(
    manifest_path: str | Path,
) -> Iterable[tuple[str, dict[str, Any]]]:
    path = Path(manifest_path)
    manifest = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != FORMAT
        or manifest.get("version") != VERSION
        or not isinstance(manifest.get("shards"), list)
    ):
        raise ValueError(f"invalid GPTQ sidecar v2 manifest: {path}")
    for descriptor in manifest["shards"]:
        shard_path = path.parent / descriptor["file"]
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(shard, dict)
            or shard.get("format") != FORMAT
            or shard.get("version") != VERSION
            or not isinstance(shard.get("tensors"), dict)
        ):
            raise ValueError(f"invalid GPTQ sidecar shard: {shard_path}")
        yield from shard["tensors"].items()
