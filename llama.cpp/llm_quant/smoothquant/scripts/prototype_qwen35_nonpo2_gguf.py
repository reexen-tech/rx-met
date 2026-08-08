#!/usr/bin/env python3
"""PROTOTYPE: apply requested non-PO2 Qwen3.5 MoE scales during GGUF conversion.

This experiment deliberately bypasses a smoothed BF16 Hugging Face checkpoint.
The stock converter promotes source BF16 tensors to FP32 before calling
``modify_tensors``. This script patches only the in-process Qwen3.5-MoE
converter class so the SmoothQuant transform is applied at that FP32 boundary.

The transformed expert/shared gate-up tensors are emitted as F32 in the
intermediate GGUF. ``llama-quantize`` can then quantize those F32 tensors
directly to Q64 without a BF16 or F16 materialization step.

This file is an experiment, not a production conversion interface.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[3]
N_LAYERS = 40
N_EMBD = 2048
N_EXPERTS = 256

_TARGET_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
_MULTIPLY_SUFFIXES = {
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_gate_inp_shexp.weight",
}
_DIVIDE_SUFFIXES = {"post_attention_norm.weight"}
_F32_INTERMEDIATE_SUFFIXES = {
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_index(model_path: Path) -> tuple[dict[str, str], Path]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing checkpoint index: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"invalid weight_map in {index_path}")
    return weight_map, index_path


def _tensor_key(layer: int, suffix: str) -> str:
    return f"model.language_model.layers.{layer}.{suffix}"


def _expert_weight_max(
    model_path: Path,
    weight_map: dict[str, str],
    layer: int,
    device: torch.device,
    expert_chunk_size: int,
) -> torch.Tensor:
    expert_key = _tensor_key(layer, "mlp.experts.gate_up_proj")
    expert_file = model_path / weight_map[expert_key]
    result = torch.zeros(N_EMBD, dtype=torch.float32, device=device)

    with safe_open(expert_file, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(expert_key)
        shape = tuple(tensor_slice.get_shape())
        if shape != (N_EXPERTS, 1024, N_EMBD):
            raise ValueError(f"unexpected {expert_key} shape: {shape}")
        for start in range(0, N_EXPERTS, expert_chunk_size):
            chunk = tensor_slice[start : start + expert_chunk_size, :, :]
            current = chunk.to(device=device).float().abs().amax(dim=(0, 1))
            torch.maximum(result, current, out=result)

    for suffix in (
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
    ):
        key = _tensor_key(layer, suffix)
        tensor_file = model_path / weight_map[key]
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
        if tuple(tensor.shape) != (512, N_EMBD):
            raise ValueError(f"unexpected {key} shape: {tuple(tensor.shape)}")
        current = tensor.to(device=device).float().abs().amax(dim=0)
        torch.maximum(result, current, out=result)

    return result.clamp_min_(1e-5)


def build_scale_manifest(
    model_path: Path,
    cache_path: Path,
    alpha: float,
    output_path: Path,
    *,
    device: torch.device,
    expert_chunk_size: int,
) -> dict[str, Any]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if expert_chunk_size <= 0:
        raise ValueError("expert_chunk_size must be positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not cache.get("complete"):
        raise RuntimeError("formal activation cache is not complete")
    act_scales = cache.get("act_scales")
    if not isinstance(act_scales, dict):
        raise ValueError("activation cache is missing act_scales")

    weight_map, index_path = _load_index(model_path)
    layers: dict[str, torch.Tensor] = {}
    stats: list[dict[str, Any]] = []
    started = time.time()

    for layer in range(N_LAYERS):
        act_key = f"model.layers.{layer}.mlp.shared_expert.gate_proj"
        activation = act_scales.get(act_key)
        if activation is None or tuple(activation.shape) != (N_EMBD,):
            shape = None if activation is None else tuple(activation.shape)
            raise ValueError(f"unexpected activation scale {act_key}: {shape}")
        activation = activation.to(device=device, dtype=torch.float32)
        weight_max = _expert_weight_max(
            model_path, weight_map, layer, device, expert_chunk_size
        )
        requested = (
            activation.pow(alpha) / weight_max.pow(1.0 - alpha)
        ).clamp_min_(1e-5)
        log2_error = (requested.log2() - requested.log2().round()).abs()
        layers[str(layer)] = requested.cpu()
        stats.append(
            {
                "layer": layer,
                "min": float(requested.min().item()),
                "max": float(requested.max().item()),
                "mean": float(requested.mean().item()),
                "non_po2_channels": int((log2_error > 1e-6).sum().item()),
                "max_log2_distance_to_po2": float(log2_error.max().item()),
            }
        )
        print(
            f"scale layer={layer:02d} min={stats[-1]['min']:.6g} "
            f"max={stats[-1]['max']:.6g} non_po2={stats[-1]['non_po2_channels']}"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "prototype": "qwen35_nonpo2_fp32_gguf",
        "model_path": str(model_path.resolve()),
        "model_index_sha256": _sha256(index_path),
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": _sha256(cache_path),
        "cache_source_model_fingerprint": cache.get("source_model_fingerprint"),
        "alpha": alpha,
        "scale_policy": "requested",
        "weight_max_definition": "routed gate_up + shared gate/up global per input channel",
        "layers": layers,
        "layer_stats": stats,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_torch_save(payload, output_path)
    return payload


def load_scale_manifest(
    path: Path, model_path: Path, cache_path: Path, alpha: float
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": 1,
        "prototype": "qwen35_nonpo2_fp32_gguf",
        "model_path": str(model_path.resolve()),
        "cache_path": str(cache_path.resolve()),
        "alpha": alpha,
        "scale_policy": "requested",
    }
    differences = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if payload.get("cache_sha256") != _sha256(cache_path):
        differences["cache_sha256"] = {
            "expected": _sha256(cache_path),
            "actual": payload.get("cache_sha256"),
        }
    layers = payload.get("layers")
    if not isinstance(layers, dict) or set(layers) != {str(i) for i in range(N_LAYERS)}:
        differences["layers"] = "expected exactly 40 layers"
    if differences:
        raise ValueError(f"scale manifest mismatch: {differences}")
    return payload


def _target(name: str) -> tuple[int, str] | None:
    match = _TARGET_RE.fullmatch(name)
    if match is None:
        return None
    layer = int(match.group(1))
    suffix = match.group(2)
    if layer >= N_LAYERS:
        return None
    if suffix in _MULTIPLY_SUFFIXES:
        return layer, "multiply"
    if suffix in _DIVIDE_SUFFIXES:
        return layer, "divide"
    return None


def transform_tensor(name: str, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    target = _target(name)
    if target is None:
        return tensor
    if tensor.shape[-1] != scale.numel():
        raise ValueError(
            f"scale shape mismatch for {name}: tensor={tuple(tensor.shape)} "
            f"scale={tuple(scale.shape)}"
        )
    shape = (1,) * (tensor.ndim - 1) + (scale.numel(),)
    scale = scale.to(dtype=tensor.dtype).reshape(shape)
    if target[1] == "multiply":
        return tensor * scale
    return tensor / scale


def force_f32_intermediate(name: str) -> bool:
    match = _TARGET_RE.fullmatch(name)
    return bool(match and match.group(2) in _F32_INTERMEDIATE_SUFFIXES)


def _import_converter(path: Path) -> ModuleType:
    converter_root = str(path.parent)
    if converter_root not in sys.path:
        sys.path.insert(0, converter_root)
    spec = importlib.util.spec_from_file_location("sq_prototype_hf_to_gguf", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import converter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def convert_with_manifest(
    converter_path: Path,
    model_path: Path,
    output_path: Path,
    manifest: dict[str, Any],
) -> dict[str, int]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module = _import_converter(converter_path)
    model_class = module.Qwen3_5MoeTextModel
    original_modify = model_class.modify_tensors
    original_force = model_class.tensor_force_quant
    scales = {int(key): value.float() for key, value in manifest["layers"].items()}
    lazy_scales = {
        layer: module.LazyTorchTensor(
            meta=module.LazyTorchTensor.meta_with_dtype_and_shape(
                value.dtype, tuple(value.shape)
            ),
            data=value,
        )
        for layer, value in scales.items()
    }
    transformed: dict[str, int] = {}
    forced_f32: dict[str, int] = {}

    def modify_tensors(
        self: Any, data_torch: torch.Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for new_name, tensor in original_modify(self, data_torch, name, bid):
            target = _target(new_name)
            if target is not None:
                layer, _ = target
                tensor = transform_tensor(new_name, tensor, lazy_scales[layer])
                transformed[new_name] = transformed.get(new_name, 0) + 1
            yield new_name, tensor

    def tensor_force_quant(
        self: Any, name: str, new_name: str, bid: int | None, n_dims: int
    ) -> Any:
        if force_f32_intermediate(new_name):
            forced_f32[new_name] = forced_f32.get(new_name, 0) + 1
            return module.gguf.GGMLQuantizationType.F32
        return original_force(self, name, new_name, bid, n_dims)

    model_class.modify_tensors = modify_tensors
    model_class.tensor_force_quant = tensor_force_quant
    previous_argv = sys.argv
    try:
        sys.argv = [
            str(converter_path),
            str(model_path),
            "--outfile",
            str(output_path),
            "--outtype",
            "f16",
        ]
        module.main()
    finally:
        sys.argv = previous_argv

    expected_transformed = N_LAYERS * (
        len(_MULTIPLY_SUFFIXES) + len(_DIVIDE_SUFFIXES)
    )
    expected_f32 = N_LAYERS * len(_F32_INTERMEDIATE_SUFFIXES)
    if len(transformed) != expected_transformed or set(transformed.values()) != {1}:
        raise RuntimeError(
            f"transformed tensor coverage mismatch: expected={expected_transformed} "
            f"actual={len(transformed)} duplicate_counts={set(transformed.values())}"
        )
    if len(forced_f32) != expected_f32 or set(forced_f32.values()) != {1}:
        raise RuntimeError(
            f"F32 tensor coverage mismatch: expected={expected_f32} "
            f"actual={len(forced_f32)} duplicate_counts={set(forced_f32.values())}"
        )
    return {"transformed_tensors": len(transformed), "forced_f32_tensors": len(forced_f32)}


def self_test() -> None:
    scale = torch.tensor([0.5, 1.25, 2.0, 3.0])
    matrix = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    expert = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    norm = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(
        transform_tensor("blk.0.ffn_gate_inp.weight", matrix, scale),
        matrix * scale,
    )
    torch.testing.assert_close(
        transform_tensor("blk.39.ffn_gate_exps.weight", expert, scale),
        expert * scale,
    )
    torch.testing.assert_close(
        transform_tensor("blk.7.post_attention_norm.weight", norm, scale),
        norm / scale,
    )
    assert force_f32_intermediate("blk.0.ffn_gate_exps.weight")
    assert force_f32_intermediate("blk.39.ffn_up_shexp.weight")
    assert not force_f32_intermediate("blk.0.ffn_gate_inp.weight")
    assert not force_f32_intermediate("blk.0.ffn_down_exps.weight")
    untouched = transform_tensor("blk.0.attn_q.weight", matrix, scale)
    assert untouched is matrix
    print("prototype self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--act-scales-cache", type=Path)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--scale-manifest", type=Path)
    parser.add_argument("--outfile", type=Path)
    parser.add_argument(
        "--converter", type=Path, default=REPO_ROOT / "convert_hf_to_gguf.py"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expert-chunk-size", type=int, default=8)
    parser.add_argument("--prepare-scales-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    required = {
        "model_path": args.model_path,
        "act_scales_cache": args.act_scales_cache,
        "alpha": args.alpha,
        "scale_manifest": args.scale_manifest,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    if not args.prepare_scales_only and args.outfile is None:
        raise ValueError("--outfile is required unless --prepare-scales-only is used")

    model_path = args.model_path.resolve()
    cache_path = args.act_scales_cache.resolve()
    manifest_path = args.scale_manifest.resolve()
    if manifest_path.exists():
        print(f"* Reusing scale manifest {manifest_path}")
        manifest = load_scale_manifest(
            manifest_path, model_path, cache_path, args.alpha
        )
    else:
        print(f"* Building requested non-PO2 scale manifest {manifest_path}")
        manifest = build_scale_manifest(
            model_path,
            cache_path,
            args.alpha,
            manifest_path,
            device=torch.device(args.device),
            expert_chunk_size=args.expert_chunk_size,
        )
    if args.prepare_scales_only:
        return

    started = time.time()
    counts = convert_with_manifest(
        args.converter.resolve(), model_path, args.outfile.resolve(), manifest
    )
    sidecar = {
        "prototype": "qwen35_nonpo2_fp32_gguf",
        "model_path": str(model_path),
        "cache_path": str(cache_path),
        "scale_manifest": str(manifest_path),
        "alpha": args.alpha,
        "scale_policy": "requested",
        "intermediate_policy": (
            "smoothed gate/up F32; other tensors stock f16 converter policy"
        ),
        "counts": counts,
        "output": str(args.outfile.resolve()),
        "output_bytes": args.outfile.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    sidecar_path = args.outfile.with_suffix(
        args.outfile.suffix + ".prototype.json"
    )
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2)
        handle.write("\n")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
