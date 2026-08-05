from __future__ import annotations

import pytest
import torch.nn as nn

from llm_quant.targets import get_model_target_policy


def _add_linear(root: nn.Module, path: str) -> None:
    parent = root
    parts = path.split(".")
    for part in parts[:-1]:
        child = getattr(parent, part, None)
        if child is None:
            child = nn.Module()
            parent.add_module(part, child)
        parent = child
    parent.add_module(parts[-1], nn.Linear(64, 64, bias=False))


def test_qwen35_policy_accepts_both_attention_signatures() -> None:
    policy = get_model_target_policy("qwen3_5_moe")
    signatures = (
        {
            "linear_attn.in_proj_a",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
        },
        {
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        },
    )
    shared = {
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
    }
    for attention in signatures:
        layer = nn.Module()
        for name in attention | shared | {"mlp.shared_expert_gate"}:
            _add_linear(layer, name)
        selected = policy.select_linears(layer)
        assert set(selected) == attention | shared
        policy.validate_routed_experts({"mlp.experts"})


def test_target_policy_rejects_missing_or_unexpected_linears() -> None:
    policy = get_model_target_policy("qwen2")
    layer = nn.Module()
    for name in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
    ):
        _add_linear(layer, name)
    _add_linear(layer, "unexpected")
    with pytest.raises(RuntimeError, match="target mismatch"):
        policy.select_linears(layer)


def test_target_policy_rejects_wrong_routed_expert_set() -> None:
    policy = get_model_target_policy("qwen3_5_moe")
    with pytest.raises(RuntimeError, match="routed expert target mismatch"):
        policy.validate_routed_experts(set())
