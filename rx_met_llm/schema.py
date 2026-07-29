"""JSON schema, defaults and validation for rx_met_llm configs.

User-facing config (v2) is minimal::

    {"model": "/path/to/hf_model_dir", "quant": "Q4_K_64"}

Resolved pipeline config expands defaults, injects binary paths, and is
written to ``resolved_config.json`` for audit.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "2.0"
LEGACY_SCHEMA_VERSION = "1.0"


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

VALID_REEX_Q64_TYPES = {
    "Q4_0_64", "Q5_0_64", "Q8_0_64", "Q8_1_64",
    "Q4_1_64", "Q5_1_64",
    "Q2_K_64", "Q3_K_64", "Q4_K_64", "Q5_K_64", "Q6_K_64",
    "Q2_K_64S", "Q4_K_64S", "Q5_K_64S",
}

VALID_GGML_TYPES |= VALID_REEX_Q64_TYPES

VALID_KV_CACHE_TYPES = {
    "f32", "f16", "bf16",
    "q8_0",
    "q4_0", "q4_1",
    "q5_0", "q5_1",
    "iq4_nl",
}


def _pipeline_defaults() -> Dict[str, Any]:
    n_threads = os.cpu_count() or 8
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "name": "unnamed_experiment",
            "description": "",
            "output_dir": "./runs/unnamed",
            "overwrite": False,
        },
        "binaries": {},
        "model": {
            "hf_path": None,
            "gguf_fp16_path": None,
        },
        "calibration": {
            "enabled": False,
            "dataset_file": None,
            "imatrix_output": "imatrix.gguf",
            "output_format": "gguf",
            "n_chunks": 100,
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
            "use_imatrix": False,
            "imatrix_include_weights": [],
            "imatrix_exclude_weights": [],
            "allow_requantize": False,
            "leave_output_tensor": False,
            "pure": False,
            "keep_split": False,
            "n_threads": n_threads,
            "tensor_type_overrides": [],
            "prune_layers": [],
            "override_kv": [],
            "dry_run": False,
        },
        "evaluation": {
            "enabled": False,
            "perplexity": {
                "dataset_file": None,
                "context_length": 512,
                "n_gpu_layers": 99,
                "flash_attn": True,
                "parallel": 1,
                "chunks": 8,
                "cache_type_k": None,
                "cache_type_v": None,
                "reex_psum_bits": None,
                "extra_args": [],
            },
        },
        "hw_export": {
            "enabled": False,
            "binary": "",
            "ld_library_path": None,
            "input_gguf": None,
            "output_gguf": None,
            "patterns": [],
            "only_tensor": None,
            "dump_dir": None,
        },
        "report": {
            "write_markdown": True,
            "markdown_name": "experiment_report.md",
            "save_resolved_config": True,
            "save_manifest": True,
        },
    }


DEFAULT_CONFIG = _pipeline_defaults()


class ConfigError(ValueError):
    """Raised when an rx_met_llm config is invalid."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def is_legacy_v1(raw: Dict[str, Any]) -> bool:
    return raw.get("schema_version") == LEGACY_SCHEMA_VERSION or (
        isinstance(raw.get("experiment"), dict)
        and "quant" not in raw
        and isinstance(raw.get("model"), dict)
    )


def _model_stem(path: Path) -> str:
    if path.is_dir():
        return path.name
    return path.stem


def normalize_user_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert user-facing v2 JSON to pipeline-shaped dict (pre-defaults)."""
    if "model" not in raw:
        raise ConfigError("'model' is required (HF model directory or GGUF file path)")
    if "quant" not in raw:
        raise ConfigError("'quant' is required (e.g. Q4_K_64)")

    model_in = raw["model"]
    if not isinstance(model_in, str):
        raise ConfigError("'model' must be a string path")

    quant = raw["quant"]
    if not isinstance(quant, str):
        raise ConfigError("'quant' must be a string")

    model_path = Path(model_in)
    hf_path: Optional[str] = None
    gguf_fp16_path: Optional[str] = None

    if model_path.suffix.lower() == ".gguf":
        gguf_fp16_path = str(model_path)
    else:
        hf_path = str(model_path)

    stem = _model_stem(model_path)
    output_dir = raw.get("output") or f"./runs/{stem}-{quant}"
    output_quantized = f"{stem}-{quant}.gguf"

    calib = raw.get("calib") or {}
    eval_cfg = raw.get("eval") or {}

    calib_dataset = calib.get("dataset")
    eval_dataset = eval_cfg.get("dataset")

    mixed = raw.get("mixed_precision") or []
    if not isinstance(mixed, list):
        raise ConfigError("'mixed_precision' must be a list")

    hw_raw = raw.get("hw_export", False)
    if isinstance(hw_raw, bool):
        hw_export: Dict[str, Any] = {"enabled": hw_raw}
    elif isinstance(hw_raw, dict):
        hw_export = {"enabled": True, **hw_raw}
    else:
        raise ConfigError("'hw_export' must be a boolean or object")

    advanced = raw.get("advanced") or {}

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "name": f"{stem}-{quant}",
            "output_dir": output_dir,
        },
        "model": {
            "hf_path": hf_path,
            "gguf_fp16_path": gguf_fp16_path,
        },
        "calibration": {
            "enabled": bool(calib_dataset),
            "dataset_file": calib_dataset,
            "n_chunks": calib.get("chunks", 100),
            "reuse_imatrix": calib.get("reuse_imatrix"),
        },
        "quantization": {
            "default_type": quant.upper(),
            "output_quantized": output_quantized,
            "use_imatrix": bool(calib_dataset),
            "tensor_type_overrides": mixed,
            "output_tensor_type": advanced.get("output_tensor_type"),
            "token_embedding_type": advanced.get("token_embedding_type"),
            "prune_layers": advanced.get("prune_layers") or [],
            "override_kv": advanced.get("override_kv") or [],
        },
        "evaluation": {
            "enabled": bool(eval_dataset),
            "perplexity": {
                "dataset_file": eval_dataset,
                "context_length": eval_cfg.get("context", 512),
                "chunks": eval_cfg.get("chunks", 8),
                "cache_type_k": eval_cfg.get("cache_type_k"),
                "cache_type_v": eval_cfg.get("cache_type_v"),
            },
        },
        "hw_export": hw_export,
    }


def _inject_binaries(cfg: Dict[str, Any]) -> None:
    from .paths import resolve_binaries

    paths = resolve_binaries()
    cfg["binaries"] = {
        "llama_quantize": paths["llama_quantize"],
        "llama_imatrix": paths["llama_imatrix"],
        "llama_perplexity": paths["llama_perplexity"],
        "convert_hf_to_gguf": paths["convert_hf_to_gguf"],
        "ld_library_path": paths["ld_library_path"],
    }
    if cfg.get("hw_export", {}).get("enabled"):
        cfg["hw_export"]["binary"] = paths["hw_export"]
        cfg["hw_export"].setdefault("ld_library_path", paths["ld_library_path"])


def resolve_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    if is_legacy_v1(raw):
        merged = _deep_merge(_pipeline_defaults(), raw)
    else:
        normalized = normalize_user_config(raw)
        merged = _deep_merge(_pipeline_defaults(), normalized)

    _inject_binaries(merged)
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
        raise ConfigError("mixed_precision / tensor_type_overrides must be a list")
    for i, item in enumerate(overrides):
        if not isinstance(item, dict):
            raise ConfigError(f"mixed_precision[{i}] must be an object")
        if "pattern" not in item or "type" not in item:
            raise ConfigError(f"mixed_precision[{i}] must have 'pattern' and 'type' keys")
        try:
            re.compile(item["pattern"])
        except re.error as exc:
            raise ConfigError(f"mixed_precision[{i}].pattern invalid regex: {exc}") from exc
        _check_ggml_type(item["type"], f"mixed_precision[{i}].type")


def _check_kv_cache_type(value: Optional[str], field: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string")
    if value.lower() not in VALID_KV_CACHE_TYPES:
        raise ConfigError(f"{field}={value!r} is not a recognised KV-cache type")


def _check_path(value: Optional[str], field: str, must_exist: bool) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string path")
    if must_exist and not Path(value).exists():
        raise ConfigError(f"{field}: path does not exist: {value}")


def validate_config(cfg: Dict[str, Any]) -> None:
    gguf_in = cfg["model"].get("gguf_fp16_path")
    hf_path = cfg["model"].get("hf_path")
    if not gguf_in and not hf_path:
        raise ConfigError("'model' must be a HuggingFace directory or GGUF file path")
    if gguf_in:
        _check_path(gguf_in, "model", must_exist=True)
    if hf_path:
        p = Path(hf_path)
        if not p.is_dir():
            raise ConfigError(f"model: HuggingFace path is not a directory: {hf_path}")
        if not (p / "config.json").is_file():
            raise ConfigError(
                f"model: directory missing config.json (not a HF model dir?): {hf_path}"
            )

    q = cfg["quantization"]
    _check_ggml_type(q["default_type"], "quant")
    _check_ggml_type(q.get("output_tensor_type"), "quant.output_tensor_type")
    _check_ggml_type(q.get("token_embedding_type"), "quant.token_embedding_type")
    _check_tensor_overrides(q["tensor_type_overrides"])

    inc = q.get("imatrix_include_weights") or []
    exc = q.get("imatrix_exclude_weights") or []
    if inc and exc:
        raise ConfigError("imatrix include/exclude weights are mutually exclusive")

    cal = cfg["calibration"]
    if cal["enabled"]:
        if not cal.get("reuse_imatrix") and not cal.get("dataset_file"):
            raise ConfigError("calib.dataset required when calibration runs")
        if cal.get("dataset_file"):
            _check_path(cal["dataset_file"], "calib.dataset", must_exist=True)
        if cal.get("reuse_imatrix"):
            _check_path(cal["reuse_imatrix"], "calib.reuse_imatrix", must_exist=True)

    ev = cfg["evaluation"]
    if ev["enabled"]:
        ppl_cfg = ev.get("perplexity", {})
        ppl_ds = ppl_cfg.get("dataset_file")
        if not ppl_ds:
            raise ConfigError("eval.dataset required when evaluation runs")
        _check_path(ppl_ds, "eval.dataset", must_exist=True)
        _check_kv_cache_type(ppl_cfg.get("cache_type_k"), "eval.cache_type_k")
        _check_kv_cache_type(ppl_cfg.get("cache_type_v"), "eval.cache_type_v")

    he = cfg.get("hw_export", {})
    if he.get("enabled"):
        if not he.get("binary"):
            raise ConfigError("hw_export binary not resolved")
        pats = he.get("patterns", [])
        if not isinstance(pats, list):
            raise ConfigError("hw_export.patterns must be a list")
        for i, pat in enumerate(pats):
            if not isinstance(pat, str):
                raise ConfigError(f"hw_export.patterns[{i}] must be a string")
            try:
                re.compile(pat)
            except re.error as exc:
                raise ConfigError(f"hw_export.patterns[{i}] invalid regex: {exc}") from exc


def imatrix_path(cfg: Dict[str, Any]) -> Optional[Path]:
    cal = cfg["calibration"]
    if not cal["enabled"]:
        return None
    if cal.get("reuse_imatrix"):
        return Path(cal["reuse_imatrix"]).resolve()
    out_dir = Path(cfg["experiment"]["output_dir"]).resolve()
    return out_dir / cal["imatrix_output"]


def quantized_path(cfg: Dict[str, Any]) -> Path:
    out_dir = Path(cfg["experiment"]["output_dir"]).resolve()
    return out_dir / cfg["quantization"]["output_quantized"]


def hf_gguf_intermediate_path(cfg: Dict[str, Any]) -> Path:
    """FP16 GGUF produced from HF conversion inside output_dir."""
    out_dir = Path(cfg["experiment"]["output_dir"]).resolve()
    stem = Path(cfg["model"]["hf_path"]).name if cfg["model"].get("hf_path") else "model"
    return out_dir / f"{stem}-f16.gguf"


def hw_export_input(cfg: Dict[str, Any]) -> Path:
    ig = cfg.get("hw_export", {}).get("input_gguf")
    return Path(ig).resolve() if ig else quantized_path(cfg)


def hw_export_output_gguf(cfg: Dict[str, Any]) -> Path:
    og = cfg.get("hw_export", {}).get("output_gguf")
    if og:
        p = Path(og)
        if not p.is_absolute():
            p = Path(cfg["experiment"]["output_dir"]).resolve() / og
        return p
    q = quantized_path(cfg)
    return q.with_name(q.stem + "-hw" + q.suffix)
