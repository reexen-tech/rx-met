"""Layer-sequential GPTQ runner for supported Qwen text towers."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from llm_quant.awq.utils.calib_data import get_calib_dataset
from llm_quant.targets import ModelTargetPolicy, get_model_target_policy

from .adapters import QwenGPTQAdapter, get_model_adapter
from .expert_quant import (
    ExpertHessianWeighting,
    ExpertCalibrationBatch,
    capture_expert_batch,
    get_routed_experts,
    quantize_routed_experts,
)
from .formats import format_for_bits
from .hessian import GPTQQuantizer, quantizer_class_for_bits
from .sidecar import (
    SidecarShardWriter,
    restore_layer_tensors,
    target_descriptors,
)

CALIBRATION_SAMPLES = 128
CALIBRATION_SEQUENCE_LENGTH = 512


@dataclass
class _CalibrationBatch:
    hidden_states: torch.Tensor
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class GPTQRunResult:
    tensor_data: dict[str, dict[str, Any]]
    layer_stats: list[dict[str, Any]]
    calibration_sequences: int
    fixed_point_ok: bool


class _StopForward(Exception):
    pass


def _move_tree(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    return value


def _hidden_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"unsupported transformer layer output: {type(output).__name__}")


def _layer_destinations(
    adapter: QwenGPTQAdapter,
    layer_index: int,
    named_linears: dict[str, nn.Linear],
    routed_experts: dict[str, nn.Module],
) -> dict[str, torch.Tensor]:
    prefix = f"{adapter.tensor_prefix}.{layer_index}"
    destinations = {
        f"{prefix}.{name}.weight": linear.weight
        for name, linear in named_linears.items()
    }
    for name, experts in routed_experts.items():
        destinations[f"{prefix}.{name}.gate_up_proj"] = experts.gate_up_proj
        destinations[f"{prefix}.{name}.down_proj"] = experts.down_proj
    return destinations


@torch.no_grad()
def _propagate_layer(
    layer: nn.Module,
    batches: list[_CalibrationBatch],
    target_device: torch.device,
) -> list[_CalibrationBatch]:
    next_batches: list[_CalibrationBatch] = []
    for batch in batches:
        output = _hidden_output(
            layer(
                batch.hidden_states.to(target_device),
                **_move_tree(batch.kwargs, target_device),
            )
        )
        next_batches.append(
            _CalibrationBatch(
                hidden_states=output.detach().cpu(),
                kwargs=batch.kwargs,
            )
        )
    return next_batches


def _stats_fixed_point_ok(stats: dict[str, Any]) -> bool:
    groups = list(stats.get("linears", {}).values())
    for routed in stats.get("routed_experts", {}).values():
        for expert in routed.get("experts", []):
            groups.extend(
                value for key, value in expert.items() if key != "expert"
            )
    return all(
        item.get("fixed_point_blocks") == item.get("total_blocks")
        for item in groups
    )


def _validate_model(model: nn.Module) -> None:
    adapter = get_model_adapter(model)
    if model.__class__.__name__ != "Qwen2ForCausalLM":
        return
    config = adapter.text_config
    expected = {
        "model_type": "qwen2",
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
    }
    mismatches = {
        name: (getattr(config, name, None), value)
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if mismatches:
        detail = ", ".join(
            f"{name}={actual!r} (expected {wanted!r})"
            for name, (actual, wanted) in mismatches.items()
        )
        raise NotImplementedError(f"the MVP is fixed to Qwen2.5-0.5B: {detail}")


@torch.no_grad()
def _capture_first_layer_inputs(
    adapter: QwenGPTQAdapter,
    tokenizer: Any,
    *,
    device: torch.device,
) -> list[_CalibrationBatch]:
    layers = adapter.layers
    samples = get_calib_dataset(
        data="pileval",
        tokenizer=tokenizer,
        n_samples=CALIBRATION_SAMPLES,
        block_size=CALIBRATION_SEQUENCE_LENGTH,
        exact_blocks=True,
    )
    if len(samples) != CALIBRATION_SAMPLES:
        raise RuntimeError(
            f"calibration loader returned {len(samples)} sequences, "
            f"expected {CALIBRATION_SAMPLES}"
        )

    captured: list[_CalibrationBatch] = []
    original = layers[0]

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, inp: torch.Tensor, **kwargs: Any) -> None:
            captured.append(
                _CalibrationBatch(
                    hidden_states=inp.detach().cpu(),
                    kwargs=_move_tree(kwargs, "cpu"),
                )
            )
            raise _StopForward

    layers[0] = original.to(device)
    adapter.move_text_inputs(device)
    layers[0] = Catcher(layers[0])
    try:
        for sample in samples:
            try:
                adapter.forward_text(sample.to(device))
            except _StopForward:
                pass
    finally:
        layers[0] = layers[0].module.cpu()
        adapter.move_text_inputs("cpu")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return captured


@torch.no_grad()
def run_gptq(
    model: nn.Module,
    tokenizer: Any,
    *,
    device: str | torch.device = "cuda",
    packed_only: bool = False,
    sidecar_writer: SidecarShardWriter | None = None,
    keep_tensor_data: bool = True,
    max_layers: int | None = None,
    target_policy: ModelTargetPolicy | None = None,
    expert_hessian_weighting: ExpertHessianWeighting = "route_squared",
    bits: int = 4,
) -> GPTQRunResult:
    """Quantize supported Qwen text towers in place with W4A8 or W8A8/G64."""

    if expert_hessian_weighting not in {"none", "route_squared"}:
        raise ValueError(
            f"unsupported expert Hessian weighting: "
            f"{expert_hessian_weighting}"
        )
    quantizer_class = quantizer_class_for_bits(bits)
    block_format = format_for_bits(bits)
    if (
        sidecar_writer is not None
        and sidecar_writer.format_name != block_format.name
    ):
        raise ValueError(
            f"GPTQ bits={bits} requires {block_format.name} sidecar, got "
            f"{sidecar_writer.format_name}"
        )
    _validate_model(model)
    adapter = get_model_adapter(model)
    policy = target_policy or get_model_target_policy(adapter.model_family)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    cache_configs = adapter.set_use_cache(False)

    batches = _capture_first_layer_inputs(
        adapter, tokenizer, device=target_device
    )
    layers = adapter.layers
    tensor_data: dict[str, dict[str, Any]] = {}
    layer_stats: list[dict[str, Any]] = []
    fixed_point_ok = True

    try:
        selected_layers = layers if max_layers is None else layers[:max_layers]
        pbar = tqdm(
            enumerate(selected_layers),
            total=len(selected_layers),
            desc="GPTQ",
            unit="layer",
        )
        for layer_index, layer in pbar:
            layer = layer.to(target_device)
            named_linears = policy.select_linears(layer)
            routed_experts = get_routed_experts(layer)
            policy.validate_routed_experts(set(routed_experts))
            destinations = _layer_destinations(
                adapter, layer_index, named_linears, routed_experts
            )

            if sidecar_writer is not None and sidecar_writer.has_layer(layer_index):
                entries, current_stats = sidecar_writer.load_layer(
                    layer_index,
                    target_descriptors(destinations, block_format),
                )
                restore_layer_tensors(entries, destinations, block_format)
                batches = _propagate_layer(layer, batches, target_device)
                layer_stats.append(current_stats)
                fixed_point_ok = (
                    fixed_point_ok and _stats_fixed_point_ok(current_stats)
                )
                layers[layer_index] = layer.cpu()
                del layer, destinations, entries
                gc.collect()
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            quantizers = {
                name: quantizer_class(
                    linear,
                    name=f"{adapter.tensor_prefix}.{layer_index}.{name}.weight",
                )
                for name, linear in named_linears.items()
            }
            expert_batches: dict[str, list[ExpertCalibrationBatch]] = {
                name: [] for name in routed_experts
            }
            handles = []
            for name, linear in named_linears.items():
                quantizer = quantizers[name]

                def add_batch(
                    _module: nn.Module,
                    inputs: tuple[torch.Tensor, ...],
                    _output: torch.Tensor,
                    *,
                    target: GPTQQuantizer = quantizer,
                ) -> None:
                    target.add_batch(inputs[0])

                handles.append(linear.register_forward_hook(add_batch))

            for name, experts in routed_experts.items():
                captured = expert_batches[name]

                def add_expert_batch(
                    _module: nn.Module,
                    inputs: tuple[Any, ...],
                    _output: torch.Tensor,
                    *,
                    target: list[ExpertCalibrationBatch] = captured,
                ) -> None:
                    target.append(capture_expert_batch(inputs))

                handles.append(experts.register_forward_hook(add_expert_batch))

            try:
                for batch in batches:
                    layer(
                        batch.hidden_states.to(target_device),
                        **_move_tree(batch.kwargs, target_device),
                    )
            finally:
                for handle in handles:
                    handle.remove()

            current_stats: dict[str, Any] = {
                "layer": layer_index,
                "linears": {},
            }
            layer_tensor_data: dict[str, dict[str, Any]] = {}
            for name, quantizer in quantizers.items():
                result = quantizer.fasterquant(
                    lazy_block_size=128, damp_percent=0.01
                )
                full_name = (
                    f"{adapter.tensor_prefix}.{layer_index}.{name}.weight"
                )
                entry: dict[str, Any] = {
                    "packed": result.packed.cpu(),
                    "shape": list(result.codes.shape),
                    "method": result.method,
                }
                if not packed_only:
                    entry["codes"] = result.codes.cpu()
                    entry["scales"] = result.scales.cpu()
                layer_tensor_data[full_name] = entry
                passed = int(result.fixed_point.sum().item())
                total = result.fixed_point.numel()
                fixed_point_ok = fixed_point_ok and passed == total
                current_stats["linears"][name] = {
                    "shape": list(result.codes.shape),
                    "loss": result.loss,
                    "dead_columns": result.dead_columns,
                    "damp": result.damp,
                    "method": result.method,
                    "fallback_reason": result.fallback_reason,
                    "fixed_point_blocks": passed,
                    "total_blocks": total,
                }
                quantizer.free()

            current_stats["routed_experts"] = {}
            for name, experts in routed_experts.items():
                result = quantize_routed_experts(
                    experts,
                    expert_batches[name],
                    tensor_prefix=(
                        f"{adapter.tensor_prefix}.{layer_index}.{name}"
                    ),
                    lazy_block_size=128,
                    damp_percent=0.01,
                    packed_only=packed_only,
                    hessian_weighting=expert_hessian_weighting,
                    bits=bits,
                )
                layer_tensor_data.update(result.tensor_data)
                fixed_point_ok = fixed_point_ok and result.fixed_point_ok
                current_stats["routed_experts"][name] = result.stats

            batches = _propagate_layer(layer, batches, target_device)
            layer_stats.append(current_stats)
            linear_stats = current_stats.get("linears", {})
            if linear_stats:
                passed = sum(s["fixed_point_blocks"] for s in linear_stats.values())
                total = sum(s["total_blocks"] for s in linear_stats.values())
                pbar.set_postfix(fixed_point=f"{passed}/{total}")
            if sidecar_writer is not None:
                sidecar_writer.write_layer(
                    layer_index, layer_tensor_data, current_stats
                )
            if keep_tensor_data:
                tensor_data.update(layer_tensor_data)
            layers[layer_index] = layer.cpu()
            del quantizers, layer, layer_tensor_data
            gc.collect()
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for config, original_value in cache_configs:
            config.use_cache = original_value

    return GPTQRunResult(
        tensor_data=tensor_data,
        layer_stats=layer_stats,
        calibration_sequences=len(batches),
        fixed_point_ok=fixed_point_ok,
    )
