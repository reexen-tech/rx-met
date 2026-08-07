"""Orchestration of SmoothQuant calibration, cache, and in-place smoothing."""

# === REEX_SMOOTHQUANT BEGIN: calibrate + smooth orchestration with scale cache ===

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .adapters import qwen35_calibration_targets
from .calibration import (
    CalibrationState,
    calibrate_progressively,
    expert_coverage_summary,
    resolve_calib_dataset,
)
from .smooth import smooth_lm

CACHE_SCHEMA_VERSION = 2
ADAPTER_VERSION = "smoothquant-qwen35-moe-v1"
CALIBRATION_SEED = 42


@dataclass(frozen=True)
class CalibrationPolicy:
    """Resolved sample and coverage policy for one calibration run."""

    mode: str
    min_samples: int
    max_samples: int
    seqlen: int
    require_full_coverage: bool


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
    calibration_resumed: bool = False
    coverage: dict[str, Any] = field(default_factory=dict)
    group_stats: list[dict[str, Any]] = field(default_factory=list)
    smoothing_policy: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def resolve_calibration_policy(
    mode: str,
    *,
    n_samples: int,
    seqlen: int,
    min_samples: int | None = None,
    max_samples: int | None = None,
) -> CalibrationPolicy:
    """Resolve fixed/dense compatibility and Qwen3.5 smoke/formal policies."""
    if mode == "smoke":
        if min_samples not in (None, 32) or max_samples not in (None, 32):
            raise ValueError("smoke calibration is fixed at 32 samples")
        return CalibrationPolicy("smoke", 32, 32, 128, False)
    if mode == "formal":
        if min_samples not in (None, 512) or max_samples not in (None, 2048):
            raise ValueError("formal calibration is fixed at min=512, max=2048")
        return CalibrationPolicy("formal", 512, 2048, 512, True)
    if mode != "fixed":
        raise ValueError(f"unknown calibration mode {mode!r}")

    resolved_min = n_samples if min_samples is None else min_samples
    resolved_max = n_samples if max_samples is None else max_samples
    if resolved_min <= 0 or resolved_max < resolved_min:
        raise ValueError(
            f"invalid fixed calibration bounds: min={resolved_min}, max={resolved_max}"
        )
    return CalibrationPolicy("fixed", resolved_min, resolved_max, seqlen, False)


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
    resume_act_scales: bool = False,
    calibration_mode: str = "fixed",
    min_samples: int | None = None,
    max_samples: int | None = None,
    checkpoint_every: int = 32,
    smoothing_scope: str = "all",
    scale_policy: str = "requested",
    norm_materialization: str = "target_dtype",
) -> SmoothQuantResult:
    """Calibrate activation scales and smooth ``model`` in place.

    ``alpha`` is deliberately excluded from the cache metadata so a complete
    calibration can be used by freshly loaded models at multiple alpha values.
    """
    if reuse_act_scales and resume_act_scales:
        raise ValueError("reuse_act_scales and resume_act_scales are mutually exclusive")

    calib_path = resolve_calib_dataset(calib_data)
    policy = resolve_calibration_policy(
        calibration_mode,
        n_samples=n_samples,
        seqlen=seqlen,
        min_samples=min_samples,
        max_samples=max_samples,
    )
    calibration_targets = qwen35_calibration_targets(model)
    if policy.mode in {"smoke", "formal"} and calibration_targets is None:
        raise TypeError(f"{policy.mode} calibration mode requires Qwen3.5 MoE")
    target_modules = (
        calibration_targets.act_module_names if calibration_targets else None
    )
    routing_targets = calibration_targets.routers if calibration_targets else ()
    meta = _cache_meta(
        model_path=model_path,
        calib_path=calib_path,
        policy=policy,
        target_modules=target_modules,
        routing_layer_names=tuple(target.layer_name for target in routing_targets),
    )
    cache_path = Path(act_scales_cache).expanduser() if act_scales_cache else None

    state: CalibrationState | None = None
    from_cache = False
    resumed = False
    if reuse_act_scales or resume_act_scales:
        if cache_path is None:
            flag = "reuse_act_scales" if reuse_act_scales else "resume_act_scales"
            raise ValueError(f"{flag} requires act_scales_cache")
        state = _load_calibration_cache(
            cache_path,
            meta,
            allow_partial=resume_act_scales,
        )
        from_cache = state.complete
        resumed = not state.complete

    if state is None or not state.complete:
        print(
            f" * SmoothQuant {policy.mode} calibration: "
            f"min={policy.min_samples} max={policy.max_samples} "
            f"x {policy.seqlen} tokens from {calib_path}",
            flush=True,
        )

        def save_partial(current: CalibrationState) -> None:
            if cache_path is not None:
                _save_calibration_cache(cache_path, meta, current, complete=False)

        state = calibrate_progressively(
            model,
            tokenizer,
            calib_path,
            min_samples=policy.min_samples,
            max_samples=policy.max_samples,
            seq_len=policy.seqlen,
            seed=CALIBRATION_SEED,
            target_modules=target_modules,
            routing_targets=routing_targets,
            require_full_coverage=policy.require_full_coverage,
            initial_state=state,
            checkpoint_every=checkpoint_every,
            checkpoint_callback=save_partial,
        )
        if cache_path is not None:
            _save_calibration_cache(cache_path, meta, state, complete=state.complete)

    coverage = expert_coverage_summary(state)
    if calibration_targets and coverage["missing_pairs"]:
        message = (
            f"Qwen3.5 calibration has {coverage['missing_pairs']} missing "
            f"layer-expert pairs after {state.actual_samples} samples"
        )
        if policy.require_full_coverage:
            raise RuntimeError(message + "; refusing to smooth/export")
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    if not state.complete:
        raise RuntimeError(
            f"calibration did not satisfy policy: {state.stop_reason}; "
            f"actual_samples={state.actual_samples}"
        )

    print(f" * SmoothQuant smoothing with alpha={alpha}", flush=True)
    n_layers, n_groups, group_stats = smooth_lm(
        model,
        state.act_scales,
        alpha,
        return_group_stats=True,
        scope=smoothing_scope,
        scale_policy=scale_policy,
        norm_materialization=norm_materialization,
    )
    print(f" * Smoothed {n_groups} groups across {n_layers} decoder layers", flush=True)

    return SmoothQuantResult(
        act_scales=state.act_scales,
        alpha=alpha,
        n_layers_smoothed=n_layers,
        n_groups_smoothed=n_groups,
        calib_data=str(calib_path),
        n_samples=state.actual_samples,
        seqlen=policy.seqlen,
        act_scales_from_cache=from_cache,
        act_scales_cache=str(cache_path) if cache_path else None,
        calibration_resumed=resumed,
        coverage=coverage,
        group_stats=group_stats,
        smoothing_policy={
            "scope": smoothing_scope,
            "scale_policy": scale_policy,
            "norm_materialization": norm_materialization,
        },
        meta=meta,
    )


def _cache_meta(
    *,
    model_path: str | None,
    calib_path: Path,
    policy: CalibrationPolicy,
    target_modules: tuple[str, ...] | None,
    routing_layer_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "source_model_fingerprint": _source_model_fingerprint(model_path),
        "calibration_file_fingerprint": _file_fingerprint(calib_path),
        "seed": CALIBRATION_SEED,
        "min_samples": policy.min_samples,
        "max_samples": policy.max_samples,
        "seqlen": policy.seqlen,
        "calibration_mode": policy.mode,
        "target_modules": list(target_modules) if target_modules is not None else None,
        "routing_layer_names": list(routing_layer_names),
        "coverage_policy": {
            "require_full_coverage": policy.require_full_coverage,
            "minimum_hit_count": 1,
        },
    }


def _source_model_fingerprint(model_path: str | None) -> dict[str, Any] | None:
    if not model_path:
        return None
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        return {"path": str(root), "exists": False}

    config = root / "config.json"
    index = root / "model.safetensors.index.json"
    weight_files = sorted(root.glob("*.safetensors"))
    record: dict[str, Any] = {
        "path": str(root),
        "config_sha256": _sha256_file(config) if config.is_file() else None,
        "index_sha256": _sha256_file(index) if index.is_file() else None,
        "weight_files": [
            {"name": path.name, "size_bytes": path.stat().st_size}
            for path in weight_files
        ],
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return record


def _file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _sha256_file(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_differences(cached: dict[str, Any], expected: dict[str, Any]) -> dict:
    return {
        key: {"cached": cached.get(key), "expected": value}
        for key, value in expected.items()
        if cached.get(key) != value
    }


def _load_calibration_cache(
    cache_path: Path,
    meta: dict[str, Any],
    *,
    allow_partial: bool,
) -> CalibrationState:
    if not cache_path.is_file():
        raise FileNotFoundError(f"SmoothQuant calibration cache not found: {cache_path}")
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"failed to load SmoothQuant cache {cache_path}: {error}") from error
    required = {
        "schema_version",
        "adapter_version",
        "act_scales",
        "actual_samples",
        "next_dataset_index",
        "complete",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(
            f"{cache_path} is not a schema-{CACHE_SCHEMA_VERSION} SmoothQuant cache; "
            f"required keys: {sorted(required)}"
        )
    mismatched = _cache_differences(payload, meta)
    if mismatched:
        raise ValueError(
            f"act_scales cache {cache_path} does not match this run: "
            + json.dumps(mismatched, ensure_ascii=False, sort_keys=True)
        )
    if not payload["complete"] and not allow_partial:
        raise ValueError(
            f"act_scales cache {cache_path} is partial; use resume_act_scales, "
            "not reuse_act_scales"
        )
    print(
        f" * {'Reusing complete' if payload['complete'] else 'Resuming partial'} "
        f"calibration cache from {cache_path}",
        flush=True,
    )
    return CalibrationState(
        act_scales=payload["act_scales"],
        expert_hit_counts=payload.get("expert_hit_counts"),
        actual_samples=int(payload["actual_samples"]),
        next_dataset_index=int(payload["next_dataset_index"]),
        complete=bool(payload["complete"]),
        stop_reason=payload.get("stop_reason", "cached"),
        routing_layer_names=tuple(payload.get("routing_layer_names", ())),
    )


def _save_calibration_cache(
    cache_path: Path,
    meta: dict[str, Any],
    state: CalibrationState,
    *,
    complete: bool,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        "actual_samples": state.actual_samples,
        "next_dataset_index": state.next_dataset_index,
        "act_scales": state.act_scales,
        "expert_hit_counts": state.expert_hit_counts,
        "coverage_result": expert_coverage_summary(state),
        "stop_reason": state.stop_reason,
        "complete": bool(complete),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            dir=cache_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print(
        f" * Saved {'complete' if complete else 'partial'} calibration cache "
        f"to {cache_path}",
        flush=True,
    )


# === REEX_SMOOTHQUANT END ===
