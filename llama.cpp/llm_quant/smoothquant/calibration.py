"""Activation-scale calibration for SmoothQuant.

Adapted from https://github.com/mit-han-lab/smoothquant (MIT License),
file ``smoothquant/calibration.py``.

Unlike ``llm_quant.awq.utils.calib_data``, samples are tokenized and truncated
individually instead of being concatenated into fixed-size blocks; this keeps
the collected scales numerically identical to the reference implementation.
"""

# === REEX_SMOOTHQUANT BEGIN: per-input-channel activation scale collection ===

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Collection, Sequence

import torch
import torch.nn as nn
from datasets import load_dataset

from .adapters import RoutingTarget

_CALIB_PATH_ENV_VARS = ("SMOOTHQUANT_PILEVAL_PATH", "AWQ_PILEVAL_PATH")


@dataclass
class CalibrationState:
    """Serializable progressive calibration state."""

    act_scales: dict[str, torch.Tensor] = field(default_factory=dict)
    expert_hit_counts: torch.Tensor | None = None
    actual_samples: int = 0
    next_dataset_index: int = 0
    complete: bool = False
    stop_reason: str = "not_started"
    routing_layer_names: tuple[str, ...] = ()


def resolve_calib_dataset(data: str = "pileval") -> Path:
    """Resolve ``--calib_data`` to a local .jsonl file.

    ``"pileval"`` falls back to ``SMOOTHQUANT_PILEVAL_PATH`` and then to the
    ``AWQ_PILEVAL_PATH`` already used by the AWQ exporter.
    """
    if data == "pileval":
        value = next(
            (os.environ[var] for var in _CALIB_PATH_ENV_VARS if os.environ.get(var)),
            None,
        )
        if not value:
            raise ValueError(
                "Set SMOOTHQUANT_PILEVAL_PATH (or AWQ_PILEVAL_PATH) to a local "
                "PileVal val.jsonl file, or pass the local JSONL path through "
                "--calib_data."
            )
    else:
        value = data

    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Local calibration dataset not found: {path}.")
    if path.suffix != ".jsonl":
        raise ValueError(f"Expected a .jsonl calibration dataset, got: {path}")
    return path


@torch.no_grad()
def get_act_scales(
    model: nn.Module,
    tokenizer,
    dataset_path: str | Path,
    num_samples: int = 512,
    seq_len: int = 512,
    *,
    progress_every: int = 64,
    target_modules: Collection[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Collect per-input-channel maxima, defaulting to every ``nn.Linear``."""
    state = calibrate_progressively(
        model,
        tokenizer,
        dataset_path,
        min_samples=num_samples,
        max_samples=num_samples,
        seq_len=seq_len,
        progress_every=progress_every,
        target_modules=target_modules,
    )
    return state.act_scales


@torch.no_grad()
def calibrate_progressively(
    model: nn.Module,
    tokenizer,
    dataset_path: str | Path,
    *,
    min_samples: int,
    max_samples: int,
    seq_len: int,
    seed: int = 42,
    target_modules: Collection[str] | None = None,
    routing_targets: Sequence[RoutingTarget] = (),
    require_full_coverage: bool = False,
    initial_state: CalibrationState | None = None,
    checkpoint_every: int = 32,
    checkpoint_callback: Callable[[CalibrationState], None] | None = None,
    progress_every: int = 64,
) -> CalibrationState:
    """Collect activation maxima and routing hits with deterministic resume."""
    if min_samples <= 0 or max_samples < min_samples:
        raise ValueError(
            f"invalid calibration sample bounds: min={min_samples}, max={max_samples}"
        )
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if require_full_coverage and not routing_targets:
        raise ValueError("full expert coverage requires routing_targets")

    model.eval()
    device = next(model.parameters()).device
    state = initial_state or CalibrationState()
    act_scales = state.act_scales

    named_modules = dict(model.named_modules())
    if target_modules is None:
        selected_modules = {
            name: module
            for name, module in named_modules.items()
            if isinstance(module, nn.Linear)
        }
    else:
        target_names = tuple(dict.fromkeys(target_modules))
        missing = [name for name in target_names if name not in named_modules]
        if missing:
            raise KeyError(f"activation target modules not found: {missing}")
        non_linear = [
            name for name in target_names if not isinstance(named_modules[name], nn.Linear)
        ]
        if non_linear:
            raise TypeError(f"activation targets must be nn.Linear modules: {non_linear}")
        selected_modules = {name: named_modules[name] for name in target_names}

    routing_layer_names = tuple(target.layer_name for target in routing_targets)
    if len(set(routing_layer_names)) != len(routing_layer_names):
        raise ValueError(f"duplicate routing layer names: {routing_layer_names}")
    if routing_targets:
        expert_counts = {target.num_experts for target in routing_targets}
        if len(expert_counts) != 1:
            raise ValueError(
                "all routed layers must have the same expert count, got "
                f"{sorted(expert_counts)}"
            )
        expected_shape = (len(routing_targets), expert_counts.pop())
        if state.expert_hit_counts is None:
            state.expert_hit_counts = torch.zeros(expected_shape, dtype=torch.int64)
        elif tuple(state.expert_hit_counts.shape) != expected_shape:
            raise ValueError(
                "resume expert_hit_counts shape mismatch: "
                f"cached={tuple(state.expert_hit_counts.shape)} expected={expected_shape}"
            )
        if state.routing_layer_names and state.routing_layer_names != routing_layer_names:
            raise ValueError(
                "resume routing layer order mismatch: "
                f"cached={state.routing_layer_names} expected={routing_layer_names}"
            )
        state.routing_layer_names = routing_layer_names
    elif state.expert_hit_counts is not None:
        raise ValueError("resume state contains routing counts but no routing_targets")

    def stat_tensor(name: str, tensor: torch.Tensor) -> None:
        hidden_dim = tensor.shape[-1]
        tensor = tensor.detach().reshape(-1, hidden_dim).float().abs()
        comming_max = torch.max(tensor, dim=0)[0].cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = [
        module.register_forward_hook(
            lambda m, x, y, name=name: stat_input_hook(m, x, y, name)
        )
        for name, module in selected_modules.items()
    ]

    def routing_hook(_module, _inputs, output, *, layer_index: int) -> None:
        if not isinstance(output, tuple) or len(output) < 3:
            raise TypeError(
                "Qwen3.5 router hook expected (logits, scores, topk_indices), "
                f"got {type(output).__name__}"
            )
        indices = output[2]
        if not isinstance(indices, torch.Tensor) or indices.ndim < 1:
            raise TypeError(
                f"router top-k indices must be a Tensor, got {type(indices).__name__}"
            )
        counts = torch.bincount(
            indices.detach().reshape(-1).to(device="cpu", dtype=torch.int64),
            minlength=state.expert_hit_counts.shape[1],
        )
        state.expert_hit_counts[layer_index].add_(counts)

    hooks.extend(
        target.module.register_forward_hook(
            lambda module, inputs, output, layer_index=layer_index: routing_hook(
                module, inputs, output, layer_index=layer_index
            )
        )
        for layer_index, target in enumerate(routing_targets)
    )

    try:
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        dataset = dataset.shuffle(seed=seed)
        upper_bound = min(max_samples, len(dataset))
        if state.next_dataset_index > upper_bound:
            raise ValueError(
                "resume next_dataset_index exceeds available calibration data: "
                f"{state.next_dataset_index} > {upper_bound}"
            )
        for i in range(state.next_dataset_index, upper_bound):
            input_ids = tokenizer(
                dataset[i]["text"],
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
            ).input_ids.to(device)
            model(input_ids)
            state.actual_samples += 1
            state.next_dataset_index = i + 1
            state.complete = False
            state.stop_reason = "in_progress"
            if checkpoint_callback and checkpoint_every and state.actual_samples % checkpoint_every == 0:
                checkpoint_callback(state)
            if progress_every and state.actual_samples % progress_every == 0:
                print(
                    f" * calibrated {state.actual_samples}/{max_samples} samples",
                    flush=True,
                )
            coverage_complete = (
                state.expert_hit_counts is None
                or bool(torch.all(state.expert_hit_counts > 0).item())
            )
            if state.actual_samples >= min_samples and (
                not require_full_coverage or coverage_complete
            ):
                state.complete = True
                state.stop_reason = (
                    "minimum_samples_and_full_coverage"
                    if require_full_coverage
                    else "sample_target_reached"
                )
                break
    finally:
        for hook in hooks:
            hook.remove()

    if not state.complete:
        coverage_complete = (
            state.expert_hit_counts is None
            or bool(torch.all(state.expert_hit_counts > 0).item())
        )
        state.complete = state.actual_samples >= min_samples and (
            not require_full_coverage or coverage_complete
        )
        if state.complete:
            state.stop_reason = "sample_target_reached"
        elif state.actual_samples >= max_samples:
            state.stop_reason = "maximum_samples_without_full_coverage"
        else:
            state.stop_reason = "dataset_exhausted"
    return state


def expert_coverage_summary(state: CalibrationState) -> dict:
    """Return JSON-serializable coverage and hit-count statistics."""
    counts = state.expert_hit_counts
    if counts is None:
        return {
            "enabled": False,
            "actual_samples": state.actual_samples,
            "complete": state.complete,
            "stop_reason": state.stop_reason,
        }

    missing_indices = torch.nonzero(counts == 0, as_tuple=False)
    missing = [
        {
            "layer": state.routing_layer_names[int(layer)],
            "expert": int(expert),
        }
        for layer, expert in missing_indices.tolist()
    ]
    flat = counts.flatten().float()
    per_layer = (counts > 0).float().mean(dim=1)
    return {
        "enabled": True,
        "actual_samples": state.actual_samples,
        "complete": state.complete,
        "stop_reason": state.stop_reason,
        "total_pairs": counts.numel(),
        "covered_pairs": int((counts > 0).sum().item()),
        "missing_pairs": len(missing),
        "missing": missing,
        "per_layer_coverage": {
            name: float(per_layer[i].item())
            for i, name in enumerate(state.routing_layer_names)
        },
        "hit_count": {
            "min": int(flat.min().item()),
            "p10": float(torch.quantile(flat, 0.10).item()),
            "median": float(torch.median(flat).item()),
            "max": int(flat.max().item()),
        },
    }

# === REEX_SMOOTHQUANT END ===
