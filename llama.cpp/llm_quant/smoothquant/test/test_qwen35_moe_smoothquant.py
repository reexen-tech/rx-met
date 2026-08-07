"""Tiny Qwen3.5 MoE tests for adapter boundaries and smoothing math."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_quant.smoothquant.adapters import smooth_groups
from llm_quant.smoothquant.smooth import (
    chunked_absmax_last_dim,
    smooth_lm,
    smooth_ln_fcs,
)


class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * x.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        return (normalized * (1.0 + self.weight.float())).to(x.dtype)


class Qwen3_5MoeTopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor):
        logits = F.linear(hidden_states.reshape(-1, hidden_states.shape[-1]), self.weight)
        probabilities = logits.softmax(dim=-1, dtype=torch.float32)
        scores, indices = probabilities.topk(self.top_k, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True)
        return probabilities, scores, indices


class _Experts(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size * 2, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )

    def forward(self, x: torch.Tensor, indices: torch.Tensor, scores: torch.Tensor):
        output = torch.zeros_like(x)
        for token in range(x.shape[0]):
            for topk_pos in range(indices.shape[1]):
                expert = int(indices[token, topk_pos])
                gate, up = F.linear(x[token], self.gate_up_proj[expert]).chunk(2)
                current = F.linear(F.silu(gate) * up, self.down_proj[expert])
                output[token] += scores[token, topk_pos].to(x.dtype) * current
        return output


class _SharedExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _SparseMoe(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 6) -> None:
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter(hidden_size, num_experts=4, top_k=2)
        self.experts = _Experts(hidden_size, intermediate_size, num_experts=4)
        self.shared_expert = _SharedExpert(hidden_size, intermediate_size)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        _, scores, indices = self.gate(flat)
        routed = self.experts(flat, indices, scores)
        shared = torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return (routed + shared).reshape(shape)


class _FullAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        return self.o_proj(mixed)


class _LinearAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.in_proj_qkv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, hidden_size, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = sum(
            fc(x)
            for fc in (
                self.in_proj_qkv,
                self.in_proj_z,
                self.in_proj_b,
                self.in_proj_a,
            )
        )
        return self.out_proj(mixed)


class Qwen3_5MoeDecoderLayer(nn.Module):
    def __init__(self, layer_type: str, hidden_size: int = 8) -> None:
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "full_attention":
            self.self_attn = _FullAttention(hidden_size)
        elif layer_type == "linear_attention":
            self.linear_attn = _LinearAttention(hidden_size)
        self.mlp = _SparseMoe(hidden_size)
        self.input_layernorm = Qwen3_5MoeRMSNorm(hidden_size)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            x = x + self.self_attn(normalized)
        else:
            x = x + self.linear_attn(normalized)
        return x + self.mlp(self.post_attention_layernorm(x))


class _TinyQwen35(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList(
            [
                Qwen3_5MoeDecoderLayer("linear_attention"),
                Qwen3_5MoeDecoderLayer("full_attention"),
            ]
        )
        self.norm = Qwen3_5MoeRMSNorm(8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        torch.manual_seed(7)
        for parameter in self.parameters():
            parameter.data.uniform_(-0.4, 0.4)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def _scales(model: _TinyQwen35) -> dict[str, torch.Tensor]:
    return {
        f"layers.{i}.{key}": torch.linspace(0.3 + i * 0.1, 1.7 + i * 0.1, 8)
        for i, layer in enumerate(model.layers)
        for key in (
            "self_attn.q_proj"
            if layer.layer_type == "full_attention"
            else "linear_attn.in_proj_qkv",
            "mlp.shared_expert.gate_proj",
        )
    }


def test_qwen35_adapter_dispatches_full_and_linear_attention() -> None:
    model = _TinyQwen35()
    linear = smooth_groups(model.layers[0], "layers.0")
    full = smooth_groups(model.layers[1], "layers.1")

    assert [fc for fc in linear[0].fcs] == [
        model.layers[0].linear_attn.in_proj_qkv,
        model.layers[0].linear_attn.in_proj_z,
        model.layers[0].linear_attn.in_proj_b,
        model.layers[0].linear_attn.in_proj_a,
    ]
    assert [fc for fc in full[0].fcs] == [
        model.layers[1].self_attn.q_proj,
        model.layers[1].self_attn.k_proj,
        model.layers[1].self_attn.v_proj,
    ]
    assert linear[1].expert_weight is model.layers[0].mlp.experts.gate_up_proj
    assert linear[0].offset_one_norm and linear[1].offset_one_norm


def test_qwen35_transform_is_equivalent_and_preserves_router_topk() -> None:
    model = _TinyQwen35().eval()
    input_ids = torch.tensor([[1, 5, 9, 3]])
    with torch.no_grad():
        expected = model(input_ids)
        hidden = model.layers[0].post_attention_layernorm(model.embed_tokens(input_ids))
        topk_before = model.layers[0].mlp.gate(hidden)[2]

    untouched = {
        "linear_out": model.layers[0].linear_attn.out_proj.weight.detach().clone(),
        "full_out": model.layers[1].self_attn.o_proj.weight.detach().clone(),
        "expert_down": model.layers[0].mlp.experts.down_proj.detach().clone(),
        "shared_down": model.layers[0].mlp.shared_expert.down_proj.weight.detach().clone(),
        "embedding": model.embed_tokens.weight.detach().clone(),
        "lm_head": model.lm_head.weight.detach().clone(),
    }

    assert smooth_lm(model, _scales(model), alpha=0.85) == (2, 4)

    with torch.no_grad():
        actual = model(input_ids)
        hidden_after = model.layers[0].post_attention_layernorm(model.embed_tokens(input_ids))
        topk_after = model.layers[0].mlp.gate(hidden_after)[2]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(topk_after, topk_before)
    torch.testing.assert_close(model.layers[0].linear_attn.out_proj.weight, untouched["linear_out"])
    torch.testing.assert_close(model.layers[1].self_attn.o_proj.weight, untouched["full_out"])
    torch.testing.assert_close(model.layers[0].mlp.experts.down_proj, untouched["expert_down"])
    torch.testing.assert_close(model.layers[0].mlp.shared_expert.down_proj.weight, untouched["shared_down"])
    torch.testing.assert_close(model.embed_tokens.weight, untouched["embedding"])
    torch.testing.assert_close(model.lm_head.weight, untouched["lm_head"])


def test_offset_one_and_moe_joint_weight_max_excludes_compensation() -> None:
    norm = Qwen3_5MoeRMSNorm(4)
    norm.weight.data.copy_(torch.tensor([-0.2, 0.0, 0.2, 0.4]))
    old_norm = norm.weight.detach().clone()
    shared = [nn.Linear(4, 3, bias=False), nn.Linear(4, 3, bias=False)]
    for fc in shared:
        fc.weight.data.fill_(2.0)
    expert = nn.Parameter(torch.full((3, 6, 4), 4.0))
    router = nn.Parameter(torch.full((3, 4), 1000.0))
    shared_gate = nn.Parameter(torch.full((1, 4), 2000.0))

    scales = smooth_ln_fcs(
        norm,
        shared,
        torch.ones(4),
        alpha=0.5,
        offset_one_norm=True,
        expert_weight=expert,
        compensation_weights=(router, shared_gate),
        expert_chunk_size=1,
    )

    expected_scale = torch.full((4,), 0.5)
    torch.testing.assert_close(scales, expected_scale)
    torch.testing.assert_close(norm.weight, (1.0 + old_norm) / expected_scale - 1.0)
    torch.testing.assert_close(expert, torch.full_like(expert, 2.0))
    torch.testing.assert_close(router, torch.full_like(router, 500.0))
    torch.testing.assert_close(shared_gate, torch.full_like(shared_gate, 1000.0))


def test_chunked_expert_max_matches_direct_reference() -> None:
    torch.manual_seed(11)
    weight = torch.randn(7, 10, 9, dtype=torch.bfloat16)
    direct = weight.float().abs().amax(dim=(0, 1))
    for chunk_size in (1, 2, 4, 20):
        torch.testing.assert_close(
            chunked_absmax_last_dim(weight, chunk_size=chunk_size),
            direct,
            rtol=0,
            atol=0,
        )


def test_same_model_instance_cannot_be_smoothed_twice() -> None:
    model = _TinyQwen35()
    smooth_lm(model, _scales(model), alpha=0.80)
    with pytest.raises(RuntimeError, match="reload the original checkpoint"):
        smooth_lm(model, _scales(model), alpha=0.85)


def test_qwen35_adapter_rejects_unknown_layer_type() -> None:
    layer = Qwen3_5MoeDecoderLayer("unexpected")
    with pytest.raises(NotImplementedError, match="layer_type"):
        smooth_groups(layer, "layers.0")
