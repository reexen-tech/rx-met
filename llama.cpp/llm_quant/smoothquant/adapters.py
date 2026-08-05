"""Per-architecture description of which norm/Linear pairs SmoothQuant smooths.

The reference implementation dispatches on ``isinstance`` against transformers
classes, which breaks whenever transformers moves a class. Here we dispatch on
the decoder layer class name and then validate the structure, so a renamed but
structurally identical layer still works and an unexpected one fails loudly.
"""

# === REEX_SMOOTHQUANT BEGIN: decoder-layer adapters for SmoothQuant smoothing ===

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class SmoothGroup:
    """One norm layer plus the Linears consuming its output."""

    norm: nn.Module
    fcs: list[nn.Linear]
    act_scales_key: str


def is_decoder_layer(module: nn.Module) -> bool:
    """Whether ``module`` is a decoder layer SmoothQuant should look at."""
    name = type(module).__name__
    return name in _DENSE_DECODER_LAYERS or name in _MOE_DECODER_LAYERS


def smooth_groups(layer: nn.Module, name: str) -> list[SmoothGroup]:
    """Return the smoothing groups of a decoder layer named ``name``."""
    cls = type(layer).__name__
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


def _require(module: nn.Module, attr: str, path: str) -> nn.Module:
    child = getattr(module, attr, None)
    if child is None:
        raise NotImplementedError(
            f"{type(module).__name__} at '{path}' has no '{attr}'; it is not "
            "structurally llama-like and cannot be smoothed by this adapter"
        )
    return child

# === REEX_SMOOTHQUANT END ===
