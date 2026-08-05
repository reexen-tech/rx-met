from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_quant.gptq.expert_quant import (
    ExpertCalibrationBatch,
    quantize_routed_experts,
)
from llm_quant.gptq.hessian import GPTQQ4064


class _TinyExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2, 128, 64) * 0.05)
        self.down_proj = nn.Parameter(torch.randn(2, 64, 64) * 0.05)
        self.act_fn = F.silu


def test_experts_are_serially_quantized_with_original_gate_up(
    monkeypatch,
) -> None:
    experts = _TinyExperts()
    hidden = torch.randn(6, 64)
    routes = torch.tensor([[0], [0], [0], [1], [1], [1]])
    routing_weights = torch.ones(6, 1)
    batches = [ExpertCalibrationBatch(hidden, routes, routing_weights)]
    original_gate_up = experts.gate_up_proj[0].detach().clone()
    expected_gate, expected_up = F.linear(
        hidden[:3], original_gate_up
    ).chunk(2, dim=-1)
    expected_down_input = F.silu(expected_gate) * expected_up

    down_inputs: list[torch.Tensor] = []
    original_add_batch = GPTQQ4064.add_batch

    def record_add_batch(self: GPTQQ4064, inp: torch.Tensor) -> None:
        if self.layer.out_features == 64:
            down_inputs.append(inp.detach().clone())
        original_add_batch(self, inp)

    monkeypatch.setattr(GPTQQ4064, "add_batch", record_add_batch)
    result = quantize_routed_experts(
        experts,
        batches,
        tensor_prefix="model.language_model.layers.0.mlp.experts",
    )

    torch.testing.assert_close(down_inputs[0], expected_down_input)
    assert result.tensor_data[
        "model.language_model.layers.0.mlp.experts.gate_up_proj"
    ]["codes"].shape == (2, 128, 64)
    assert result.tensor_data[
        "model.language_model.layers.0.mlp.experts.down_proj"
    ]["codes"].shape == (2, 64, 64)
    assert all(
        expert["gate_up_proj"]["method"] == "gptq"
        and expert["down_proj"]["method"] == "gptq"
        for expert in result.stats["experts"]
    )


def test_expert_hessian_weighting_modes(monkeypatch) -> None:
    hidden = torch.randn(8, 64)
    routes = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1]])
    routing_weights = torch.linspace(0.2, 0.9, 8).reshape(8, 1)
    batches = [ExpertCalibrationBatch(hidden, routes, routing_weights)]
    captured: list[torch.Tensor] = []
    original_add_batch = GPTQQ4064.add_batch

    def record_gate_inputs(self: GPTQQ4064, inp: torch.Tensor) -> None:
        if self.layer.out_features == 128:
            captured.append(inp.detach().clone())
        original_add_batch(self, inp)

    monkeypatch.setattr(GPTQQ4064, "add_batch", record_gate_inputs)
    quantize_routed_experts(
        _TinyExperts(),
        batches,
        tensor_prefix="model.layers.0.mlp.experts",
        hessian_weighting="none",
    )
    unweighted = captured.copy()
    captured.clear()
    quantize_routed_experts(
        _TinyExperts(),
        batches,
        tensor_prefix="model.layers.0.mlp.experts",
        hessian_weighting="route_squared",
    )
    weighted = captured.copy()

    torch.testing.assert_close(unweighted[0], hidden[routes[:, 0] == 0])
    torch.testing.assert_close(
        weighted[0],
        hidden[routes[:, 0] == 0]
        * routing_weights[routes[:, 0] == 0],
    )


def test_experts_support_q8_packed_output() -> None:
    hidden = torch.randn(6, 64)
    routes = torch.tensor([[0], [0], [0], [1], [1], [1]])
    batches = [
        ExpertCalibrationBatch(hidden, routes, torch.ones(6, 1))
    ]
    result = quantize_routed_experts(
        _TinyExperts(),
        batches,
        tensor_prefix="model.layers.0.mlp.experts",
        bits=8,
        packed_only=True,
    )
    gate = result.tensor_data[
        "model.layers.0.mlp.experts.gate_up_proj"
    ]
    down = result.tensor_data[
        "model.layers.0.mlp.experts.down_proj"
    ]
    assert gate["packed"].shape == (2, 128, 66)
    assert down["packed"].shape == (2, 64, 66)
