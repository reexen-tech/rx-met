"""Orchestration of SmoothQuant Step 1: calibrate, then smooth in place."""

# === REEX_SMOOTHQUANT BEGIN: calibrate + smooth orchestration with scale cache ===

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .calibration import get_act_scales, resolve_calib_dataset
from .smooth import smooth_lm


@dataclass
class SmoothQuantResult:
    """Everything the exporter needs to describe one smoothing run."""

    act_scales: dict[str, torch.Tensor]
    alpha: float
    n_layers_smoothed: int
    n_groups_smoothed: int
    calib_data: str
    n_samples: int
    seqlen: int
    act_scales_from_cache: bool = False
    act_scales_cache: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def run_smoothquant(
    model: nn.Module,
    tokenizer,
    *,
    alpha: float = 0.85,
    n_samples: int = 512,
    seqlen: int = 512,
    calib_data: str = "pileval",
    model_path: str | None = None,
    act_scales_cache: str | Path | None = None,
    reuse_act_scales: bool = False,
) -> SmoothQuantResult:
    """Calibrate activation scales and smooth ``model`` in place.

    ``alpha`` is deliberately excluded from the cache key so that a single
    calibration pass can be reused while sweeping alpha.
    """
    calib_path = resolve_calib_dataset(calib_data)
    meta = _cache_meta(
        model_path=model_path,
        calib_path=calib_path,
        n_samples=n_samples,
        seqlen=seqlen,
    )
    cache_path = Path(act_scales_cache).expanduser() if act_scales_cache else None

    act_scales = None
    from_cache = False
    if reuse_act_scales:
        if cache_path is None:
            raise ValueError("--reuse_act_scales requires --act_scales_cache")
        act_scales = _load_act_scales(cache_path, meta)
        from_cache = True

    if act_scales is None:
        print(
            f" * SmoothQuant calibration: {n_samples} samples x {seqlen} tokens "
            f"from {calib_path}",
            flush=True,
        )
        act_scales = get_act_scales(
            model, tokenizer, calib_path, num_samples=n_samples, seq_len=seqlen
        )
        if cache_path is not None:
            _save_act_scales(cache_path, meta, act_scales)

    print(f" * SmoothQuant smoothing with alpha={alpha}", flush=True)
    n_layers, n_groups = smooth_lm(model, act_scales, alpha)
    print(f" * Smoothed {n_groups} groups across {n_layers} decoder layers", flush=True)

    return SmoothQuantResult(
        act_scales=act_scales,
        alpha=alpha,
        n_layers_smoothed=n_layers,
        n_groups_smoothed=n_groups,
        calib_data=str(calib_path),
        n_samples=n_samples,
        seqlen=seqlen,
        act_scales_from_cache=from_cache,
        act_scales_cache=str(cache_path) if cache_path else None,
        meta=meta,
    )


def _cache_meta(
    *,
    model_path: str | None,
    calib_path: Path,
    n_samples: int,
    seqlen: int,
) -> dict[str, Any]:
    stat = calib_path.stat()
    return {
        "model_path": str(Path(model_path).resolve()) if model_path else None,
        "model_config_sha256": _model_config_sha256(model_path),
        "calib_data": str(calib_path.resolve()),
        "calib_size": stat.st_size,
        "n_samples": n_samples,
        "seqlen": seqlen,
    }


def _model_config_sha256(model_path: str | None) -> str | None:
    if not model_path:
        return None
    config = Path(model_path).expanduser() / "config.json"
    if not config.is_file():
        return None
    return hashlib.sha256(config.read_bytes()).hexdigest()


def _load_act_scales(
    cache_path: Path, meta: dict[str, Any]
) -> dict[str, torch.Tensor] | None:
    if not cache_path.is_file():
        return None
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "act_scales" not in payload:
        raise ValueError(
            f"{cache_path} is not a SmoothQuant act_scales cache "
            "(expected keys 'meta' and 'act_scales')"
        )
    cached_meta = payload.get("meta", {})
    mismatched = {
        key: (cached_meta.get(key), value)
        for key, value in meta.items()
        if value is not None and cached_meta.get(key) != value
    }
    if mismatched:
        raise ValueError(
            f"act_scales cache {cache_path} does not match this run: "
            + json.dumps(mismatched, ensure_ascii=False)
            + " (delete the cache or drop --reuse_act_scales)"
        )
    print(f" * Reusing cached act_scales from {cache_path}", flush=True)
    return payload["act_scales"]


def _save_act_scales(
    cache_path: Path, meta: dict[str, Any], act_scales: dict[str, torch.Tensor]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "act_scales": act_scales}, cache_path)
    print(f" * Saved act_scales to {cache_path}", flush=True)

# === REEX_SMOOTHQUANT END ===
