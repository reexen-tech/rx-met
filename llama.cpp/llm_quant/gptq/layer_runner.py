"""Layer-sequential GPTQ runner for supported Qwen text towers."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from llm_quant.awq.utils.calib_data import get_calib_dataset

from .adapters import QwenGPTQAdapter, get_model_adapter
from .expert_quant import (
    ExpertCalibrationBatch,
    capture_expert_batch,
    get_routed_experts,
    quantize_routed_experts,
)
from .hessian import GPTQQ4064

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
    layer_sink: Callable[
        [int, dict[str, dict[str, Any]], dict[str, Any]], None
    ]
    | None = None,
    keep_tensor_data: bool = True,
    max_layers: int | None = None,
) -> GPTQRunResult:
    """Quantize supported Qwen text towers in place with W4A8/G64."""

    _validate_model(model)
    adapter = get_model_adapter(model)
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
        for layer_index, layer in enumerate(selected_layers):
            layer = layer.to(target_device)
            named_linears = adapter.named_linears(layer)
            routed_experts = get_routed_experts(layer)
            if not named_linears and not routed_experts:
                raise RuntimeError(
                    f"layer {layer_index} contains no quantizable weights"
                )

            quantizers = {
                name: GPTQQ4064(linear) for name, linear in named_linears.items()
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
                    target: GPTQQ4064 = quantizer,
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
                )
                layer_tensor_data.update(result.tensor_data)
                fixed_point_ok = fixed_point_ok and result.fixed_point_ok
                current_stats["routed_experts"][name] = result.stats

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
            batches = next_batches
            layer_stats.append(current_stats)
            if layer_sink is not None:
                layer_sink(layer_index, layer_tensor_data, current_stats)
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
