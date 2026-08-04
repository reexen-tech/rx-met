"""Serial GPTQ for routed expert weights stored as 3D parameters."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hessian import GPTQResult, quantizer_class_for_bits

ExpertHessianWeighting = Literal["none", "route_squared"]


@dataclass(frozen=True)
class ExpertCalibrationBatch:
    hidden_states: torch.Tensor
    top_k_index: torch.Tensor
    top_k_weights: torch.Tensor


@dataclass(frozen=True)
class ExpertQuantResult:
    tensor_data: dict[str, dict[str, Any]]
    stats: dict[str, Any]
    fixed_point_ok: bool


def is_routed_experts(module: nn.Module) -> bool:
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    return (
        isinstance(gate_up, nn.Parameter)
        and isinstance(down, nn.Parameter)
        and gate_up.ndim == 3
        and down.ndim == 3
        and callable(getattr(module, "act_fn", None))
    )


def get_routed_experts(module: nn.Module) -> dict[str, nn.Module]:
    return {
        name: child
        for name, child in module.named_modules()
        if name and is_routed_experts(child)
    }


def capture_expert_batch(inputs: tuple[Any, ...]) -> ExpertCalibrationBatch:
    if len(inputs) < 3:
        raise TypeError("routed experts forward did not receive routing inputs")
    hidden_states, top_k_index, top_k_weights = inputs[:3]
    if not isinstance(hidden_states, torch.Tensor) or not isinstance(
        top_k_index, torch.Tensor
    ) or not isinstance(top_k_weights, torch.Tensor):
        raise TypeError("routed expert inputs must be tensors")
    hidden_states = hidden_states.detach().reshape(-1, hidden_states.shape[-1]).cpu()
    top_k_index = top_k_index.detach().reshape(hidden_states.shape[0], -1).cpu()
    top_k_weights = top_k_weights.detach().reshape(hidden_states.shape[0], -1).cpu()
    return ExpertCalibrationBatch(hidden_states, top_k_index, top_k_weights)


def _selected_inputs(
    batches: list[ExpertCalibrationBatch],
    expert_index: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    selected: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch in batches:
        matches = batch.top_k_index == expert_index
        token_mask = matches.any(dim=-1)
        if token_mask.any():
            weights = (batch.top_k_weights * matches).sum(dim=-1)
            selected.append(
                (
                    batch.hidden_states[token_mask].to(device),
                    weights[token_mask].to(device),
                )
            )
    return selected


def _temporary_linear(weight: torch.Tensor) -> nn.Linear:
    layer = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device=weight.device,
        dtype=weight.dtype,
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def _result_stats(result: GPTQResult) -> dict[str, Any]:
    passed = int(result.fixed_point.sum().item())
    total = result.fixed_point.numel()
    return {
        "shape": list(result.codes.shape),
        "loss": result.loss,
        "dead_columns": result.dead_columns,
        "damp": result.damp,
        "method": result.method,
        "fallback_reason": result.fallback_reason,
        "fixed_point_blocks": passed,
        "total_blocks": total,
    }


@torch.no_grad()
def quantize_routed_experts(
    module: nn.Module,
    batches: list[ExpertCalibrationBatch],
    *,
    tensor_prefix: str,
    lazy_block_size: int = 128,
    damp_percent: float = 0.01,
    packed_only: bool = False,
    hessian_weighting: ExpertHessianWeighting = "route_squared",
    bits: int = 4,
) -> ExpertQuantResult:
    """Quantize one 3D expert collection with one live Hessian at a time."""

    if not is_routed_experts(module):
        raise TypeError("module does not expose supported 3D routed expert weights")
    if hessian_weighting not in {"none", "route_squared"}:
        raise ValueError(
            f"unsupported expert Hessian weighting: {hessian_weighting}"
        )
    quantizer_class = quantizer_class_for_bits(bits)
    gate_up = module.gate_up_proj
    down = module.down_proj
    if gate_up.shape[0] != down.shape[0]:
        raise ValueError("gate_up_proj and down_proj expert counts differ")
    if gate_up.shape[1] != 2 * down.shape[2]:
        raise ValueError("gate_up_proj is not a fused gate/up tensor")
    if gate_up.shape[2] != down.shape[1]:
        raise ValueError("expert hidden dimensions do not match")

    gate_entries: dict[str, list[torch.Tensor]] = {
        "codes": [],
        "scales": [],
        "packed": [],
    }
    down_entries: dict[str, list[torch.Tensor]] = {
        "codes": [],
        "scales": [],
        "packed": [],
    }
    expert_stats: list[dict[str, Any]] = []
    fixed_point_ok = True
    device = gate_up.device

    for expert_index in range(gate_up.shape[0]):
        inputs = _selected_inputs(batches, expert_index, device)
        original_gate_up = gate_up[expert_index].detach().clone()

        gate_layer = _temporary_linear(original_gate_up)
        gate_quantizer = quantizer_class(
            gate_layer,
            name=f"{tensor_prefix}.{expert_index}.gate_up_proj",
        )
        for selected, routing_weight in inputs:
            weighted = selected
            if hessian_weighting == "route_squared":
                weighted = selected * routing_weight.unsqueeze(-1).to(
                    selected.dtype
                )
            gate_quantizer.add_batch(weighted)
        gate_result = gate_quantizer.fasterquant(
            lazy_block_size=lazy_block_size, damp_percent=damp_percent
        )
        gate_up[expert_index].copy_(gate_result.dequant.to(gate_up.dtype))
        gate_quantizer.free()
        del gate_quantizer, gate_layer

        down_layer = _temporary_linear(down[expert_index].detach())
        down_quantizer = quantizer_class(
            down_layer,
            name=f"{tensor_prefix}.{expert_index}.down_proj",
        )
        for selected, routing_weight in inputs:
            # true_sequential=false: down calibration always uses the original
            # gate/up weight, not the just-quantized gate_up_proj slice.
            gate, up = F.linear(
                selected.to(torch.float32), original_gate_up.to(torch.float32)
            ).chunk(2, dim=-1)
            middle = module.act_fn(gate) * up
            if hessian_weighting == "route_squared":
                middle = middle * routing_weight.unsqueeze(-1).to(middle.dtype)
            down_quantizer.add_batch(middle)
        down_result = down_quantizer.fasterquant(
            lazy_block_size=lazy_block_size, damp_percent=damp_percent
        )
        down[expert_index].copy_(down_result.dequant.to(down.dtype))
        down_quantizer.free()
        del down_quantizer, down_layer, original_gate_up, inputs

        for result in (gate_result, down_result):
            fixed_point_ok = fixed_point_ok and bool(result.fixed_point.all().item())
        expert_stats.append(
            {
                "expert": expert_index,
                "gate_up_proj": _result_stats(gate_result),
                "down_proj": _result_stats(down_result),
            }
        )
        for destination, result in (
            (gate_entries, gate_result),
            (down_entries, down_result),
        ):
            destination["packed"].append(result.packed.cpu())
            if not packed_only:
                destination["codes"].append(result.codes.cpu())
                destination["scales"].append(result.scales.cpu())
        del gate_result, down_result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tensor_data: dict[str, dict[str, Any]] = {}
    for suffix, entries, shape in (
        ("gate_up_proj", gate_entries, gate_up.shape),
        ("down_proj", down_entries, down.shape),
    ):
        output: dict[str, Any] = {
            "packed": torch.stack(entries["packed"]),
            "shape": list(shape),
            "method": "gptq",
        }
        if not packed_only:
            output["codes"] = torch.stack(entries["codes"])
            output["scales"] = torch.stack(entries["scales"])
        tensor_data[f"{tensor_prefix}.{suffix}"] = output
    return ExpertQuantResult(
        tensor_data=tensor_data,
        stats={"experts": expert_stats},
        fixed_point_ok=fixed_point_ok,
    )
