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
    routes = torch.zeros(6, 1, dtype=torch.long)
    routing_weights = torch.ones(6, 1)
    batches = [ExpertCalibrationBatch(hidden, routes, routing_weights)]
    original_gate_up = experts.gate_up_proj[0].detach().clone()
    expected_gate, expected_up = F.linear(hidden, original_gate_up).chunk(2, dim=-1)
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
    unrouted = result.stats["experts"][1]
    assert unrouted["gate_up_proj"]["fallback_reason"] == "zero_samples"
    assert unrouted["down_proj"]["fallback_reason"] == "zero_samples"
