"""Versioned packed sidecars for resumable GPTQ exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import torch

from llm_quant.artifacts import canonical_digest

from .formats import Q4_0_64, Q64Format, format_for_name


FORMAT = "Q4_0_64"
VERSION = 3
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def target_descriptors(
    targets: dict[str, Any] | dict[str, dict[str, Any]],
    block_format: Q64Format = Q4_0_64,
) -> list[dict[str, Any]]:
    result = []
    for name, value in sorted(targets.items()):
        if isinstance(value, dict):
            shape = value.get("shape")
            if shape is None and isinstance(value.get("codes"), torch.Tensor):
                shape = value["codes"].shape
            if shape is None and isinstance(value.get("packed"), torch.Tensor):
                packed = value["packed"]
                shape = (
                    *packed.shape[:-1],
                    packed.shape[-1]
                    // block_format.type_size
                    * block_format.group_size,
                )
        else:
            shape = value.shape
        if shape is None:
            raise ValueError(f"GPTQ target {name} has no logical shape")
        result.append({"name": name, "shape": list(shape)})
    return result


@torch.no_grad()
def restore_layer_tensors(
    entries: dict[str, dict[str, Any]],
    destinations: dict[str, torch.Tensor],
    block_format: Q64Format = Q4_0_64,
) -> None:
    """Restore source tensors from packed entries, including split experts."""

    restored_whole: set[str] = set()
    restored_slices: dict[str, set[tuple[int, int]]] = {}
    for entry_key, entry in entries.items():
        source_name = entry.get("source_name")
        if not isinstance(source_name, str) or source_name not in destinations:
            raise ValueError(
                f"GPTQ sidecar entry {entry_key} has unknown source {source_name!r}"
            )
        packed = entry.get("packed")
        shape = entry.get("shape")
        if not isinstance(packed, torch.Tensor) or not isinstance(shape, list):
            raise ValueError(f"invalid GPTQ sidecar entry: {entry_key}")
        codes, scales = block_format.unpack(
            packed, logical_size=int(shape[-1])
        )
        restored = block_format.dequantize(codes, scales)
        destination = destinations[source_name]
        source_slice = entry.get("source_slice")
        if source_slice is None:
            if tuple(restored.shape) != tuple(destination.shape):
                raise ValueError(
                    f"GPTQ restore shape mismatch for {source_name}: "
                    f"{tuple(restored.shape)} != {tuple(destination.shape)}"
                )
            destination.copy_(restored.to(destination))
            restored_whole.add(source_name)
            continue
        if (
            not isinstance(source_slice, list)
            or len(source_slice) != 2
            or not all(isinstance(index, int) for index in source_slice)
        ):
            raise ValueError(f"invalid source_slice for {entry_key}")
        start, stop = source_slice
        destination_slice = destination[..., start:stop, :]
        if tuple(restored.shape) != tuple(destination_slice.shape):
            raise ValueError(
                f"GPTQ restore slice mismatch for {source_name}: "
                f"{tuple(restored.shape)} != {tuple(destination_slice.shape)}"
            )
        destination_slice.copy_(restored.to(destination))
        restored_slices.setdefault(source_name, set()).add((start, stop))

    for name, destination in destinations.items():
        if name in restored_whole:
            continue
        slices = sorted(restored_slices.get(name, set()))
        expected_stop = int(destination.shape[-2])
        position = 0
        for start, stop in slices:
            if start != position or stop <= start:
                break
            position = stop
        if position != expected_stop:
            raise ValueError(f"incomplete GPTQ sidecar restore for {name}")


def _compact_entry(
    name: str,
    entry: dict[str, Any],
    block_format: Q64Format,
) -> dict[str, Any]:
    packed = entry.get("packed")
    if not isinstance(packed, torch.Tensor):
        raise ValueError("GPTQ sidecar entry is missing packed bytes")
    shape = entry.get("shape")
    codes = entry.get("codes")
    if shape is None:
        if not isinstance(codes, torch.Tensor):
            logical_last = (
                packed.shape[-1]
                // block_format.type_size
                * block_format.group_size
            )
            shape = (*packed.shape[:-1], logical_last)
        else:
            shape = tuple(codes.shape)
    result: dict[str, Any] = {
        "packed": packed.detach().to(device="cpu", dtype=torch.uint8).contiguous(),
        "shape": list(shape),
        "source_name": entry.get("source_name", name),
    }
    for key in ("gguf_name", "method", "source_slice"):
        if key in entry:
            result[key] = entry[key]
    return result


def _expand_split_expert(
    name: str,
    entry: dict[str, Any],
    block_format: Q64Format,
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
        codes, scales = block_format.unpack(
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
    for gguf_part, row_slice, source_slice in (
        ("ffn_gate_exps", slice(0, n_ff), [0, n_ff]),
        ("ffn_up_exps", slice(n_ff, None), [n_ff, 2 * n_ff]),
    ):
        split_codes = codes[:, row_slice, :].contiguous()
        split_scales = scales[:, row_slice, :].contiguous()
        gguf_name = f"blk.{layer}.{gguf_part}.weight"
        key = gguf_name
        result[key] = {
            "packed": block_format.pack(split_codes, split_scales),
            "shape": list(split_codes.shape),
            "source_name": name,
            "source_slice": source_slice,
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
        source_signature: dict[str, Any],
        quantization_config: dict[str, Any],
        resume: bool = False,
        format_name: str = FORMAT,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source_model = source_model
        self.source_signature = source_signature
        self.quantization_config = quantization_config
        self.block_format = format_for_name(format_name)
        self.format_name = self.block_format.name
        self.basename = self.block_format.basename
        self._shard_glob = f"{self.basename}.layer*.pt"
        self.manifest_path = self.directory / f"{self.basename}.pt"
        self._source_digest = canonical_digest(source_signature)
        self._config_digest = canonical_digest(quantization_config)
        self.resume = resume
        self._shards: dict[int, dict[str, Any]] = {}
        if not resume:
            for path in self.directory.glob(self._shard_glob):
                path.unlink()
            self.manifest_path.unlink(missing_ok=True)
        else:
            manifest_path = self.manifest_path
            if manifest_path.is_file():
                manifest = torch.load(
                    manifest_path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                if (
                    not isinstance(manifest, dict)
                    or manifest.get("format") != self.format_name
                    or manifest.get("version") != VERSION
                ):
                    version = (
                        manifest.get("version")
                        if isinstance(manifest, dict)
                        else None
                    )
                    raise ValueError(
                        f"unsupported GPTQ sidecar manifest "
                        f"{manifest_path}: expected version {VERSION}, "
                        f"got {version!r}"
                    )
                if manifest.get("source_digest") != self._source_digest:
                    raise ValueError(
                        f"GPTQ sidecar source mismatch: {manifest_path}"
                    )
                if manifest.get("config_digest") != self._config_digest:
                    raise ValueError(
                        f"GPTQ sidecar configuration mismatch: {manifest_path}"
                    )
            for path in sorted(self.directory.glob(self._shard_glob)):
                payload = self._load_and_validate(path)
                layer_index = int(payload["layer"])
                expected_filename = (
                    f"{self.basename}.layer{layer_index:03d}.pt"
                )
                if path.name != expected_filename:
                    raise ValueError(
                        f"GPTQ sidecar layer/file mismatch: {path}"
                    )
                if layer_index in self._shards:
                    raise ValueError(f"duplicate GPTQ sidecar layer {layer_index}")
                self._shards[layer_index] = self._descriptor(path.name, payload)
            completed = sorted(self._shards)
            if completed and completed != list(range(completed[-1] + 1)):
                raise ValueError(
                    f"GPTQ resume shards are not contiguous: {completed}"
                )

    def _load_and_validate(self, path: Path) -> dict[str, Any]:
        payload = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
        if (
            not isinstance(payload, dict)
            or payload.get("format") != self.format_name
            or payload.get("version") != VERSION
            or payload.get("complete") is not True
            or not isinstance(payload.get("tensors"), dict)
            or not isinstance(payload.get("targets"), list)
        ):
            version = payload.get("version") if isinstance(payload, dict) else None
            raise ValueError(
                f"unsupported GPTQ sidecar shard {path}: "
                f"expected version {VERSION}, got {version!r}"
            )
        if payload.get("source_digest") != self._source_digest:
            raise ValueError(f"GPTQ sidecar source mismatch: {path}")
        if payload.get("config_digest") != self._config_digest:
            raise ValueError(f"GPTQ sidecar configuration mismatch: {path}")
        return payload

    @staticmethod
    def _descriptor(filename: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "file": filename,
            "layer": int(payload["layer"]),
            "tensors": sorted(payload["tensors"]),
            "targets": payload["targets"],
        }

    def has_layer(self, layer_index: int) -> bool:
        return layer_index in self._shards

    def load_layer(
        self,
        layer_index: int,
        expected_targets: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        descriptor = self._shards.get(layer_index)
        if descriptor is None:
            raise KeyError(f"GPTQ sidecar layer {layer_index} does not exist")
        path = self.directory / descriptor["file"]
        payload = self._load_and_validate(path)
        if payload["targets"] != expected_targets:
            raise ValueError(f"GPTQ sidecar target mismatch: {path}")
        stats = payload.get("stats")
        if not isinstance(stats, dict):
            raise ValueError(f"GPTQ sidecar stats missing: {path}")
        return payload["tensors"], stats

    def write_layer(
        self,
        layer_index: int,
        tensors: dict[str, dict[str, Any]],
        stats: dict[str, Any] | None = None,
    ) -> str:
        filename = f"{self.basename}.layer{layer_index:03d}.pt"
        path = self.directory / filename
        if layer_index in self._shards or path.exists():
            raise ValueError(f"refusing to overwrite GPTQ sidecar shard: {path}")
        source_targets = target_descriptors(tensors, self.block_format)
        expanded: dict[str, dict[str, Any]] = {}
        for name, entry in tensors.items():
            expanded.update(
                _expand_split_expert(name, entry, self.block_format)
            )
        compact = {
            name: _compact_entry(name, entry, self.block_format)
            for name, entry in expanded.items()
        }
        payload = {
            "format": self.format_name,
            "version": VERSION,
            "complete": True,
            "layer": layer_index,
            "source_digest": self._source_digest,
            "config_digest": self._config_digest,
            "targets": source_targets,
            "stats": stats or {"layer": layer_index},
            "tensors": compact,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        self._shards[layer_index] = self._descriptor(filename, payload)
        return filename

    def finish(self) -> Path:
        manifest = {
            "format": self.format_name,
            "version": VERSION,
            "layout": "packed-only-sharded",
            "source_model": self.source_model,
            "source_signature": self.source_signature,
            "quantization_config": self.quantization_config,
            "source_digest": self._source_digest,
            "config_digest": self._config_digest,
            "shards": [
                self._shards[layer] for layer in sorted(self._shards)
            ],
        }
        path = self.manifest_path
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
            or manifest.get("version") != VERSION
            or not isinstance(manifest.get("shards"), list)
        ):
            version = manifest.get("version") if isinstance(manifest, dict) else None
            raise ValueError(
                f"unsupported GPTQ sidecar manifest {self.path}: "
                f"expected version {VERSION}, got {version!r}"
            )
        self.block_format = format_for_name(str(manifest.get("format")))
        self.format_name = self.block_format.name
        if self.path.name != f"{self.block_format.basename}.pt":
            raise ValueError(
                f"GPTQ sidecar format/file mismatch: {self.path}"
            )
        self.source_model = manifest.get("source_model")
        self.source_digest = manifest.get("source_digest")
        self.config_digest = manifest.get("config_digest")
        source_signature = manifest.get("source_signature")
        quantization_config = manifest.get("quantization_config")
        if (
            not isinstance(source_signature, dict)
            or canonical_digest(source_signature) != self.source_digest
            or not isinstance(quantization_config, dict)
            or canonical_digest(quantization_config) != self.config_digest
        ):
            raise ValueError(f"invalid GPTQ sidecar identity: {self.path}")
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
            shard = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            tensors = shard.get("tensors") if isinstance(shard, dict) else None
            if (
                not isinstance(shard, dict)
                or shard.get("format") != self.format_name
                or shard.get("version") != VERSION
                or shard.get("complete") is not True
                or shard.get("source_digest") != self.source_digest
                or shard.get("config_digest") != self.config_digest
                or not isinstance(tensors, dict)
            ):
                raise ValueError(f"invalid GPTQ sidecar shard: {shard_path}")
            self._cached_file = filename
            self._cached_tensors = tensors
        return self._cached_tensors[name]


def write_sidecar_v3(
    directory: str | Path,
    tensors: dict[str, dict[str, Any]],
    *,
    source_model: str,
    source_signature: dict[str, Any],
    quantization_config: dict[str, Any],
    resume: bool = False,
    format_name: str = FORMAT,
) -> Path:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for name, entry in tensors.items():
        match = _LAYER_RE.search(name)
        layer_index = int(match.group(1)) if match is not None else -1
        grouped.setdefault(layer_index, {})[name] = entry
    writer = SidecarShardWriter(
        directory,
        source_model=source_model,
        source_signature=source_signature,
        quantization_config=quantization_config,
        resume=resume,
        format_name=format_name,
    )
    for layer_index in sorted(grouped):
        if writer.has_layer(layer_index):
            writer.load_layer(
                layer_index,
                target_descriptors(
                    grouped[layer_index], writer.block_format
                ),
            )
        else:
            writer.write_layer(layer_index, grouped[layer_index])
    return writer.finish()


def iter_sidecar_v3(
    manifest_path: str | Path,
) -> Iterable[tuple[str, dict[str, Any]]]:
    path = Path(manifest_path)
    manifest = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != VERSION
        or not isinstance(manifest.get("shards"), list)
    ):
        raise ValueError(f"invalid GPTQ sidecar v3 manifest: {path}")
    block_format = format_for_name(str(manifest.get("format")))
    if path.name != f"{block_format.basename}.pt":
        raise ValueError(f"GPTQ sidecar format/file mismatch: {path}")
    source_digest = manifest.get("source_digest")
    config_digest = manifest.get("config_digest")
    for descriptor in manifest["shards"]:
        shard_path = path.parent / descriptor["file"]
        shard = torch.load(
            shard_path, map_location="cpu", weights_only=True, mmap=True
        )
        if (
            not isinstance(shard, dict)
            or shard.get("format") != block_format.name
            or shard.get("version") != VERSION
            or shard.get("complete") is not True
            or shard.get("source_digest") != source_digest
            or shard.get("config_digest") != config_digest
            or not isinstance(shard.get("tensors"), dict)
        ):
            raise ValueError(f"invalid GPTQ sidecar shard: {shard_path}")
        yield from shard["tensors"].items()
