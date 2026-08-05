"""Algorithm-independent model weight selection policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


_DENSE_TRANSFORMER = frozenset(
    {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
)
_QWEN35_SHARED_EXPERT = frozenset(
    {
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
    }
)
_QWEN35_FULL_ATTENTION = frozenset(
    {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    }
) | _QWEN35_SHARED_EXPERT
_QWEN35_LINEAR_ATTENTION = frozenset(
    {
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
    }
) | _QWEN35_SHARED_EXPERT


@dataclass(frozen=True)
class ModelTargetPolicy:
    """Exact per-layer target signatures for one model family."""

    family: str
    linear_signatures: tuple[frozenset[str], ...]
    excluded_linears: frozenset[str] = frozenset()
    routed_experts: frozenset[str] = frozenset()

    def select_linears(self, layer: nn.Module) -> dict[str, nn.Linear]:
        discovered = {
            name: module
            for name, module in layer.named_modules()
            if name and isinstance(module, nn.Linear)
        }
        candidates = frozenset(discovered) - self.excluded_linears
        if candidates not in self.linear_signatures:
            expected = " or ".join(
                "{" + ", ".join(sorted(signature)) + "}"
                for signature in self.linear_signatures
            )
            missing = sorted(
                min(
                    self.linear_signatures,
                    key=lambda signature: len(signature - candidates),
                )
                - candidates
            )
            allowed = frozenset().union(*self.linear_signatures)
            unexpected = sorted(candidates - allowed)
            raise RuntimeError(
                f"{self.family} layer target mismatch: "
                f"missing={missing}, unexpected={unexpected}; expected {expected}"
            )
        return {name: discovered[name] for name in sorted(candidates)}

    def validate_routed_experts(self, names: set[str]) -> None:
        if names != set(self.routed_experts):
            raise RuntimeError(
                f"{self.family} routed expert target mismatch: "
                f"expected={sorted(self.routed_experts)}, actual={sorted(names)}"
            )


_POLICIES = {
    "qwen2": ModelTargetPolicy(
        family="qwen2",
        linear_signatures=(_DENSE_TRANSFORMER,),
    ),
    "qwen3": ModelTargetPolicy(
        family="qwen3",
        linear_signatures=(_DENSE_TRANSFORMER,),
    ),
    "qwen3_5_moe": ModelTargetPolicy(
        family="qwen3_5_moe",
        linear_signatures=(
            _QWEN35_LINEAR_ATTENTION,
            _QWEN35_FULL_ATTENTION,
        ),
        excluded_linears=frozenset({"mlp.shared_expert_gate"}),
        routed_experts=frozenset({"mlp.experts"}),
    ),
}


def get_model_target_policy(model_family: str) -> ModelTargetPolicy:
    try:
        return _POLICIES[model_family]
    except KeyError as exc:
        raise NotImplementedError(
            f"unsupported quantization target policy: {model_family}"
        ) from exc
