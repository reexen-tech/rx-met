"""JSON schema, defaults and validation for aimet_llama configs.

The schema is intentionally permissive (no hard `jsonschema` dependency):
we use plain dict-merge for defaults, and a small set of `_check_*`
helpers for the fields that must be syntactically correct before any
binary is invoked.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = "1.0"


VALID_GGML_TYPES = {
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "Q2_K", "Q2_K_S",
    "Q3_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_K", "Q4_K_S", "Q4_K_M",
    "Q5_K", "Q5_K_S", "Q5_K_M",
    "Q6_K",
    "IQ1_S", "IQ1_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_NL", "IQ4_XS",
    "TQ1_0", "TQ2_0",
    "MXFP4_MOE", "Q16_0",
    "F16", "BF16", "F32", "COPY",
}


# REEX block-64 quant types exposed by the in-tree (REEX) llama.cpp build.
# Every type is fully block=64 aligned (legacy: one scale / 64 elems; K-quant:
# super-block=256, sub-block=64). See ``llama.cpp/REEX_Q64_USAGE.md``.
# These are only valid when quantizing with the REEX-enabled binaries.
VALID_REEX_Q64_TYPES = {
    "Q4_0_64", "Q5_0_64", "Q8_0_64", "Q8_1_64",
    "Q4_1_64", "Q5_1_64",
    "Q2_K_64", "Q3_K_64", "Q4_K_64", "Q5_K_64", "Q6_K_64",
    "Q2_K_64S", "Q4_K_64S", "Q5_K_64S",
}

VALID_GGML_TYPES |= VALID_REEX_Q64_TYPES


# Lowercase KV-cache type tokens accepted by llama.cpp's `-ctk` / `-ctv`.
# Source: ggml-cuda kv-cache kernels (legacy + fp + iq4_nl supported).
VALID_KV_CACHE_TYPES = {
    "f32", "f16", "bf16",
    "q8_0",
    "q4_0", "q4_1",
    "q5_0", "q5_1",
    "iq4_nl",
}


DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,

    "experiment": {
        "name": "unnamed_experiment",
        "description": "",
        "output_dir": "./out/aimet_llama_run",
        "overwrite": False,
    },

    "binaries": {
        "llama_quantize": "llama-quantize",
        "llama_imatrix": "llama-imatrix",
        "llama_perplexity": "llama-perplexity",
        "convert_hf_to_gguf": None,
        "ld_library_path": None,
    },

    "model": {
        "hf_path": None,
        "gguf_fp16_path": None,
    },

    "calibration": {
        "enabled": False,
        "dataset_file": None,
        "imatrix_output": "imatrix.gguf",
        "output_format": "gguf",
        "n_chunks": -1,
        "from_chunk": 0,
        "context_length": 512,
        "n_gpu_layers": 99,
        "process_output": False,
        "no_ppl": True,
        "output_frequency": 10,
        "save_frequency": 0,
        "parse_special": False,
        "reuse_imatrix": None,
        "in_files": [],
    },

    "quantization": {
        "default_type": "Q4_K_M",
        "output_quantized": "model.quantized.gguf",
        "output_tensor_type": None,
        "token_embedding_type": None,
        "use_imatrix": True,
        "imatrix_include_weights": [],
        "imatrix_exclude_weights": [],
        "allow_requantize": False,
        "leave_output_tensor": False,
        "pure": False,
        "keep_split": False,
        "n_threads": 8,
        "tensor_type_overrides": [],
        "prune_layers": [],
        "override_kv": [],
        "dry_run": False,
    },

    "evaluation": {
        "enabled": False,
        "perplexity": {
            "dataset_file": None,
            "context_length": 2048,
            "n_gpu_layers": 99,
            "flash_attn": True,
            "parallel": 1,
            "chunks": -1,
            "cache_type_k": None,
            "cache_type_v": None,
            # REEX block-64 fixed-point Psum truncation bit-width. When > 0 the
            # pipeline exports REEX_Q64_PSUM_BITS=<value> for the perplexity
            # stage so the integer Psum is truncated to that many bits (CPU and
            # GPU stay numerically aligned). null / <= 0 disables truncation.
            "reex_psum_bits": None,
            "extra_args": [],
        },
    },

    # Optional Phase-2 stage: convert the quantized GGUF into a hardware-tiled
    # GGUF (weight_blocks layout) via the `reex-hw-convert` tool. Runs after
    # `quantize` and writes a sibling GGUF (default `<quantized>-hw.gguf`) where
    # matched Legacy block-64 GEMM weights are HW-tiled and all other tensors are
    # copied verbatim. The output is marked `reex.hw_layout=true` and must NOT be
    # fed to the stock llama.cpp inference path. A matched convertible tensor with
    # a non-tiling-divisible shape is a hard error (the stage fails).
    "hw_export": {
        "enabled": False,
        "binary": "reex-hw-convert",
        "ld_library_path": None,
        "input_gguf": None,          # null -> the quantized GGUF from this run
        "output_gguf": None,         # null -> "<quantized_stem>-hw.gguf"
        "patterns": [],              # empty -> tool defaults (attn/ffn GEMM weights)
        "only_tensor": None,
        "dump_dir": None,            # optional: also dump per-tensor weight_blocks.bin
    },

    "report": {
        "write_markdown": True,
        "markdown_name": "experiment_report.md",
        "save_resolved_config": True,
        "save_manifest": True,
    },
}


class ConfigError(ValueError):
    """Raised when an aimet_llama config is invalid."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON config file from disk (no validation, no merge)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``raw`` over ``DEFAULT_CONFIG`` and run static validation."""
    merged = _deep_merge(DEFAULT_CONFIG, raw)
    validate_config(merged)
    return merged


def _check_ggml_type(value: Optional[str], field: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string, got {type(value).__name__}")
    if value.upper() not in VALID_GGML_TYPES:
        raise ConfigError(
            f"{field}={value!r} is not a recognised ggml type. "
            f"Valid types: {sorted(VALID_GGML_TYPES)}"
        )


def _check_tensor_overrides(overrides: Any) -> None:
    if not isinstance(overrides, list):
        raise ConfigError("quantization.tensor_type_overrides must be a list")
    for i, item in enumerate(overrides):
        if not isinstance(item, dict):
            raise ConfigError(
                f"quantization.tensor_type_overrides[{i}] must be an object"
            )
        if "pattern" not in item or "type" not in item:
            raise ConfigError(
                f"quantization.tensor_type_overrides[{i}] must have "
                "'pattern' and 'type' keys"
            )
        try:
            re.compile(item["pattern"])
        except re.error as exc:
            raise ConfigError(
                f"quantization.tensor_type_overrides[{i}].pattern is not a "
                f"valid regex: {exc}"
            ) from exc
        _check_ggml_type(
            item["type"],
            f"quantization.tensor_type_overrides[{i}].type",
        )


def _check_kv_cache_type(value: Optional[str], field: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string, got {type(value).__name__}")
    if value.lower() not in VALID_KV_CACHE_TYPES:
        raise ConfigError(
            f"{field}={value!r} is not a recognised KV-cache type. "
            f"Valid types: {sorted(VALID_KV_CACHE_TYPES)}"
        )


def _check_reex_psum_bits(value: Any, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer, got {type(value).__name__}")
    if value > 32:
        raise ConfigError(f"{field}={value} is out of range (expected <= 32)")


def _check_path(value: Optional[str], field: str, must_exist: bool) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string path")
    if must_exist and not Path(value).exists():
        raise ConfigError(f"{field}: path does not exist: {value}")


def validate_config(cfg: Dict[str, Any]) -> None:
    """Static checks. Raises ConfigError on failure."""
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(
            f"Unsupported schema_version: {cfg.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION})"
        )

    # model.gguf_fp16_path is required for quantization
    gguf_in = cfg["model"].get("gguf_fp16_path")
    hf_path = cfg["model"].get("hf_path")
    if not gguf_in and not hf_path:
        raise ConfigError(
            "Either model.gguf_fp16_path or model.hf_path must be specified"
        )
    if gguf_in:
        _check_path(gguf_in, "model.gguf_fp16_path", must_exist=True)

    # quantization
    q = cfg["quantization"]
    _check_ggml_type(q["default_type"], "quantization.default_type")
    _check_ggml_type(q.get("output_tensor_type"), "quantization.output_tensor_type")
    _check_ggml_type(q.get("token_embedding_type"), "quantization.token_embedding_type")
    _check_tensor_overrides(q["tensor_type_overrides"])

    inc = q.get("imatrix_include_weights") or []
    exc = q.get("imatrix_exclude_weights") or []
    if inc and exc:
        raise ConfigError(
            "quantization.imatrix_include_weights and "
            "imatrix_exclude_weights are mutually exclusive"
        )

    # calibration
    cal = cfg["calibration"]
    if cal["enabled"]:
        if not cal.get("reuse_imatrix") and not cal.get("dataset_file"):
            raise ConfigError(
                "calibration.enabled=true requires either dataset_file or "
                "reuse_imatrix"
            )
        if cal.get("dataset_file"):
            _check_path(cal["dataset_file"], "calibration.dataset_file", must_exist=True)
        if cal.get("reuse_imatrix"):
            _check_path(
                cal["reuse_imatrix"], "calibration.reuse_imatrix", must_exist=True
            )
        if cal.get("output_format") not in ("gguf", "dat"):
            raise ConfigError(
                f"calibration.output_format must be 'gguf' or 'dat', "
                f"got {cal.get('output_format')!r}"
            )

    # evaluation
    ev = cfg["evaluation"]
    if ev["enabled"]:
        ppl_cfg = ev.get("perplexity", {})
        ppl_ds = ppl_cfg.get("dataset_file")
        if not ppl_ds:
            raise ConfigError(
                "evaluation.enabled=true requires evaluation.perplexity.dataset_file"
            )
        _check_path(ppl_ds, "evaluation.perplexity.dataset_file", must_exist=True)
        _check_kv_cache_type(
            ppl_cfg.get("cache_type_k"), "evaluation.perplexity.cache_type_k"
        )
        _check_kv_cache_type(
            ppl_cfg.get("cache_type_v"), "evaluation.perplexity.cache_type_v"
        )
        _check_reex_psum_bits(
            ppl_cfg.get("reex_psum_bits"), "evaluation.perplexity.reex_psum_bits"
        )

    # hw_export (optional Phase-2 stage)
    he = cfg.get("hw_export", {})
    if he.get("enabled"):
        if not isinstance(he.get("binary"), str) or not he["binary"]:
            raise ConfigError("hw_export.binary must be a non-empty string")
        pats = he.get("patterns", [])
        if not isinstance(pats, list):
            raise ConfigError("hw_export.patterns must be a list")
        for i, p in enumerate(pats):
            if not isinstance(p, str):
                raise ConfigError(f"hw_export.patterns[{i}] must be a string")
            try:
                re.compile(p)
            except re.error as exc:
                raise ConfigError(
                    f"hw_export.patterns[{i}] is not a valid regex: {exc}"
                ) from exc
        ig = he.get("input_gguf")
        if ig is not None:
            # produced by the quantize stage; may not exist yet at validation time
            if not isinstance(ig, str):
                raise ConfigError("hw_export.input_gguf must be a string path")
        og = he.get("output_gguf")
        if og is not None and not isinstance(og, str):
            raise ConfigError("hw_export.output_gguf must be a string path")
        ot = he.get("only_tensor")
        if ot is not None and not isinstance(ot, str):
            raise ConfigError("hw_export.only_tensor must be a string")
        dd = he.get("dump_dir")
        if dd is not None and not isinstance(dd, str):
            raise ConfigError("hw_export.dump_dir must be a string path")


def imatrix_path(cfg: Dict[str, Any]) -> Optional[Path]:
    """Return absolute path of the imatrix file this run will produce / reuse."""
    cal = cfg["calibration"]
    if not cal["enabled"]:
        return None
    if cal.get("reuse_imatrix"):
        return Path(cal["reuse_imatrix"]).resolve()
    out_dir = Path(cfg["experiment"]["output_dir"]).resolve()
    return out_dir / cal["imatrix_output"]


def quantized_path(cfg: Dict[str, Any]) -> Path:
    """Return absolute path of the output quantized GGUF file."""
    out_dir = Path(cfg["experiment"]["output_dir"]).resolve()
    return out_dir / cfg["quantization"]["output_quantized"]


def hw_export_input(cfg: Dict[str, Any]) -> Path:
    """Return the GGUF the hw_export stage reads from (defaults to the
    quantized GGUF produced by this run)."""
    ig = cfg.get("hw_export", {}).get("input_gguf")
    return Path(ig).resolve() if ig else quantized_path(cfg)


def hw_export_output_gguf(cfg: Dict[str, Any]) -> Path:
    """Return the HW-tiled GGUF the hw_export stage writes.

    Defaults to a sibling of the quantized GGUF with a ``-hw`` suffix, e.g.
    ``out.gguf`` -> ``out-hw.gguf``.
    """
    og = cfg.get("hw_export", {}).get("output_gguf")
    if og:
        p = Path(og)
        if not p.is_absolute():
            p = Path(cfg["experiment"]["output_dir"]).resolve() / og
        return p
    q = quantized_path(cfg)
    return q.with_name(q.stem + "-hw" + q.suffix)
