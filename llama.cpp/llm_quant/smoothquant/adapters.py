"""Per-architecture description of which norm/Linear pairs SmoothQuant smooths.

The reference implementation dispatches on ``isinstance`` against transformers
classes, which breaks whenever transformers moves a class. Here we dispatch on
the decoder layer class name and then validate the structure, so a renamed but
structurally identical layer still works and an unexpected one fails loudly.
"""

# === REEX_SMOOTHQUANT BEGIN: decoder-layer adapters for SmoothQuant smoothing ===

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Dense "llama-like" decoder layers: pre-attention RMSNorm feeding q/k/v_proj,
# pre-MLP RMSNorm feeding gate/up_proj.
_DENSE_DECODER_LAYERS = frozenset(
    {
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
        "Qwen2DecoderLayer",
        "Qwen3DecoderLayer",
    }
)

# Recognised but deliberately unsupported in Phase 1 (see plan: MoE smoothing
# must follow the llama-quantize tensor allow-list, deferred to Qwen3.5).
_MOE_DECODER_LAYERS = frozenset(
    {
        "MixtralDecoderLayer",
        "Qwen2MoeDecoderLayer",
        "Qwen3MoeDecoderLayer",
        "Qwen3NextDecoderLayer",
    }
)

_QWEN35_MOE_DECODER_LAYER = "Qwen3_5MoeDecoderLayer"


@dataclass(frozen=True)
class SmoothGroup:
    """One norm layer plus the Linears consuming its output."""

    norm: nn.Module
    fcs: list[nn.Linear]
    act_scales_key: str
    offset_one_norm: bool = False
    expert_weight: torch.Tensor | None = None
    compensation_weights: tuple[torch.Tensor, ...] = ()

    @property
    def is_moe(self) -> bool:
        """Whether this group contains the routed 3-D expert projection."""
        return self.expert_weight is not None


@dataclass(frozen=True)
class RoutingTarget:
    """One Qwen3.5 router whose final top-k indices must be counted."""

    layer_name: str
    module: nn.Module
    num_experts: int


@dataclass(frozen=True)
class CalibrationTargets:
    """Representative activation hooks and routed-expert coverage hooks."""

    act_module_names: tuple[str, ...]
    routers: tuple[RoutingTarget, ...]


def is_decoder_layer(module: nn.Module) -> bool:
    """Whether ``module`` is a decoder layer SmoothQuant should look at."""
    name = type(module).__name__
    return (
        name in _DENSE_DECODER_LAYERS
        or name in _MOE_DECODER_LAYERS
        or name == _QWEN35_MOE_DECODER_LAYER
    )


def smooth_groups(layer: nn.Module, name: str) -> list[SmoothGroup]:
    """Return the smoothing groups of a decoder layer named ``name``."""
    cls = type(layer).__name__
    if cls == _QWEN35_MOE_DECODER_LAYER:
        return _qwen35_moe_groups(layer, name)
    if cls in _MOE_DECODER_LAYERS:
        raise NotImplementedError(
            f"{cls} is a MoE decoder layer; SmoothQuant MoE support is planned "
            "for the next phase and is not implemented in Phase 1"
        )
    if cls not in _DENSE_DECODER_LAYERS:
        raise NotImplementedError(
            f"unsupported decoder layer {cls}; add it to "
            "_DENSE_DECODER_LAYERS in llm_quant/smoothquant/adapters.py "
            "after checking it is structurally llama-like"
        )

    attn = _require(layer, "self_attn", name)
    mlp = _require(layer, "mlp", name)
    return [
        SmoothGroup(
            norm=_require(layer, "input_layernorm", name),
            fcs=[
                _require(attn, "q_proj", f"{name}.self_attn"),
                _require(attn, "k_proj", f"{name}.self_attn"),
                _require(attn, "v_proj", f"{name}.self_attn"),
            ],
            act_scales_key=f"{name}.self_attn.q_proj",
        ),
        SmoothGroup(
            norm=_require(layer, "post_attention_layernorm", name),
            fcs=[
                _require(mlp, "gate_proj", f"{name}.mlp"),
                _require(mlp, "up_proj", f"{name}.mlp"),
            ],
            act_scales_key=f"{name}.mlp.gate_proj",
        ),
    ]


def qwen35_calibration_targets(model: nn.Module) -> CalibrationTargets | None:
    """Return the minimal hook allowlist for a Qwen3.5 MoE text model."""
    act_names: list[str] = []
    routers: list[RoutingTarget] = []
    for name, module in model.named_modules():
        if type(module).__name__ != _QWEN35_MOE_DECODER_LAYER:
            continue
        groups = _qwen35_moe_groups(module, name)
        act_names.extend(group.act_scales_key for group in groups)
        router = _require(_require(module, "mlp", name), "gate", f"{name}.mlp")
        num_experts = int(groups[1].expert_weight.shape[0])
        routers.append(
            RoutingTarget(layer_name=name, module=router, num_experts=num_experts)
        )

    if not routers:
        return None
    return CalibrationTargets(tuple(act_names), tuple(routers))


def _qwen35_moe_groups(layer: nn.Module, name: str) -> list[SmoothGroup]:
    """Describe the two Qwen3.5 boundaries without hiding 3-D expert weights."""
    layer_type = getattr(layer, "layer_type", None)
    input_norm = _require(layer, "input_layernorm", name)
    post_attn_norm = _require(layer, "post_attention_layernorm", name)
    mlp = _require(layer, "mlp", name)

    if layer_type == "full_attention":
        attn = _require(layer, "self_attn", name)
        attn_fcs = [
            _require_linear(attn, attr, f"{name}.self_attn")
            for attr in ("q_proj", "k_proj", "v_proj")
        ]
        act_scales_key = f"{name}.self_attn.q_proj"
    elif layer_type == "linear_attention":
        attn = _require(layer, "linear_attn", name)
        attn_fcs = [
            _require_linear(attn, attr, f"{name}.linear_attn")
            for attr in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        ]
        act_scales_key = f"{name}.linear_attn.in_proj_qkv"
    else:
        raise NotImplementedError(
            f"unsupported Qwen3.5 layer_type {layer_type!r} at '{name}'; "
            "expected 'full_attention' or 'linear_attention'"
        )

    experts = _require(mlp, "experts", f"{name}.mlp")
    expert_weight = getattr(experts, "gate_up_proj", None)
    if not isinstance(expert_weight, torch.Tensor) or expert_weight.ndim != 3:
        shape = getattr(expert_weight, "shape", None)
        raise TypeError(
            f"{name}.mlp.experts.gate_up_proj must be a 3-D Parameter, got "
            f"{type(expert_weight).__name__} with shape {shape}"
        )

    shared = _require(mlp, "shared_expert", f"{name}.mlp")
    shared_fcs = [
        _require_linear(shared, attr, f"{name}.mlp.shared_expert")
        for attr in ("gate_proj", "up_proj")
    ]
    router = _require(mlp, "gate", f"{name}.mlp")
    router_weight = getattr(router, "weight", None)
    if not isinstance(router_weight, torch.Tensor) or router_weight.ndim != 2:
        raise TypeError(
            f"{name}.mlp.gate.weight must be a 2-D Parameter, got "
            f"{type(router_weight).__name__} with shape "
            f"{getattr(router_weight, 'shape', None)}"
        )
    shared_gate = _require_linear(mlp, "shared_expert_gate", f"{name}.mlp")

    hidden_size = input_norm.weight.numel()
    if expert_weight.shape[-1] != hidden_size:
        raise ValueError(
            f"{name}.mlp.experts.gate_up_proj last dimension "
            f"{expert_weight.shape[-1]} does not match norm size {hidden_size}"
        )
    named_fcs = [
        *((f"attention[{i}]", fc) for i, fc in enumerate(attn_fcs)),
        *((f"shared_expert[{i}]", fc) for i, fc in enumerate(shared_fcs)),
        ("gate", router_weight),
        ("shared_expert_gate", shared_gate),
    ]
    for path, fc_or_weight in named_fcs:
        input_size = (
            fc_or_weight.in_features
            if isinstance(fc_or_weight, nn.Linear)
            else fc_or_weight.shape[-1]
        )
        if input_size != hidden_size:
            raise ValueError(
                f"{name}.{path} input size {input_size} does not match "
                f"norm size {hidden_size}"
            )

    return [
        SmoothGroup(
            norm=input_norm,
            fcs=attn_fcs,
            act_scales_key=act_scales_key,
            offset_one_norm=True,
        ),
        SmoothGroup(
            norm=post_attn_norm,
            fcs=shared_fcs,
            act_scales_key=f"{name}.mlp.shared_expert.gate_proj",
            offset_one_norm=True,
            expert_weight=expert_weight,
            compensation_weights=(router_weight, shared_gate.weight),
        ),
    ]


def _require(module: nn.Module, attr: str, path: str) -> nn.Module:
    child = getattr(module, attr, None)
    if child is None:
        raise NotImplementedError(
            f"{type(module).__name__} at '{path}' has no '{attr}'; it is not "
            "structurally llama-like and cannot be smoothed by this adapter"
        )
    return child


def _require_linear(module: nn.Module, attr: str, path: str) -> nn.Linear:
    child = _require(module, attr, path)
    if not isinstance(child, nn.Linear):
        raise TypeError(
            f"{type(module).__name__} at '{path}' has non-Linear '{attr}': "
            f"{type(child).__name__}"
        )
    return child

# === REEX_SMOOTHQUANT END ===
