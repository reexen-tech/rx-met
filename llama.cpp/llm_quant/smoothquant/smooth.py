"""SmoothQuant offline smoothing transform used before Q64 quantization.

Adapted from https://github.com/mit-han-lab/smoothquant (MIT License),
file ``smoothquant/smooth.py``.

Only the mathematically equivalent smoothing transform is ported. The fake-
quant modules of the reference implementation are intentionally *not* ported:
the selected Q64 quantization is performed later by ``llama-quantize`` on the
exported GGUF.
"""

# === REEX_SMOOTHQUANT BEGIN: LayerNorm/RMSNorm <-> Linear equivalent smoothing ===

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .adapters import is_decoder_layer, smooth_groups


@torch.no_grad()
def smooth_ln_fcs(
    ln: nn.Module,
    fcs: nn.Linear | list[nn.Linear],
    act_scales: torch.Tensor,
    alpha: float = 0.5,
    *,
    offset_one_norm: bool = False,
    expert_weight: torch.Tensor | None = None,
    compensation_weights: tuple[torch.Tensor, ...] = (),
    expert_chunk_size: int = 8,
    scale_policy: str = "requested",
    norm_materialization: str = "target_dtype",
) -> torch.Tensor:
    """Migrate activation outliers from ``ln``'s output into ``fcs``' weights.

    Divides the norm weight (and bias, for LayerNorm) by a per-input-channel
    factor and multiplies the same factor into every downstream Linear, which
    leaves the float forward pass unchanged. Returns the applied factors.
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        if not isinstance(fc, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(fc).__name__}")
        if not (ln.weight.numel() == fc.in_features == act_scales.numel()):
            raise ValueError(
                "smoothing shape mismatch: "
                f"ln={ln.weight.numel()} fc.in_features={fc.in_features} "
                f"act_scales={act_scales.numel()}"
            )

    for weight in compensation_weights:
        if weight.ndim != 2 or weight.shape[-1] != act_scales.numel():
            raise ValueError(
                "smoothing shape mismatch: "
                f"compensation weight={tuple(weight.shape)} "
                f"act_scales={act_scales.numel()}"
            )

    if expert_weight is not None:
        if expert_weight.ndim != 3 or expert_weight.shape[-1] != act_scales.numel():
            raise ValueError(
                "smoothing shape mismatch: "
                f"expert_weight={tuple(expert_weight.shape)} "
                f"act_scales={act_scales.numel()}"
            )
        device = expert_weight.device
        weight_scales = chunked_absmax_last_dim(
            expert_weight, chunk_size=expert_chunk_size
        )
    else:
        device = fcs[0].weight.device
        weight_scales = torch.zeros(
            act_scales.numel(), device=device, dtype=torch.float32
        )

    for fc in fcs:
        current = fc.weight.detach().to(device=device, dtype=torch.float32).abs().amax(dim=0)
        torch.maximum(weight_scales, current, out=weight_scales)
    weight_scales.clamp_(min=1e-5)

    if scale_policy not in {"requested", "power_of_two"}:
        raise ValueError(f"unsupported SmoothQuant scale policy: {scale_policy}")
    if norm_materialization not in {"target_dtype", "fp32_offset"}:
        raise ValueError(
            f"unsupported SmoothQuant norm materialization: {norm_materialization}"
        )
    if norm_materialization == "fp32_offset" and not offset_one_norm:
        raise ValueError("FP32 offset materialization requires an offset-one norm")

    act_scales_fp32 = act_scales.detach().to(device=device, dtype=torch.float32)
    requested_scales = (
        act_scales_fp32.pow(alpha) / weight_scales.pow(1.0 - alpha)
    ).clamp(min=1e-5)
    scales = requested_scales
    if scale_policy == "power_of_two":
        scales = torch.pow(2.0, requested_scales.log2().round())

    norm_scales = scales.to(ln.weight.device, torch.float32)
    if offset_one_norm:
        new_offset = (1.0 + ln.weight.float()) / norm_scales - 1.0
        if norm_materialization == "fp32_offset":
            ln.weight = nn.Parameter(
                new_offset,
                requires_grad=ln.weight.requires_grad,
            )
        else:
            ln.weight.copy_(new_offset.to(ln.weight.dtype))
    else:
        ln.weight.div_(norm_scales.to(ln.weight.dtype))
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(scales.to(ln.bias.device, ln.bias.dtype))

    for fc in fcs:
        fc.weight.mul_(scales.to(fc.weight.device, fc.weight.dtype).view(1, -1))
    for weight in compensation_weights:
        weight.mul_(scales.to(weight.device, weight.dtype).view(1, -1))

    if expert_weight is not None:
        expert_weight.mul_(
            scales.to(expert_weight.device, expert_weight.dtype).view(1, 1, -1)
        )

    return scales


@torch.no_grad()
def chunked_absmax_last_dim(
    weight: torch.Tensor, *, chunk_size: int = 8
) -> torch.Tensor:
    """Return FP32 per-last-dimension absmax without a full-sized temporary."""
    if weight.ndim < 2:
        raise ValueError(f"expected at least 2-D weight, got shape {tuple(weight.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    accumulator = torch.zeros(
        weight.shape[-1], device=weight.device, dtype=torch.float32
    )
    reduce_dims = tuple(range(weight.ndim - 1))
    for start in range(0, weight.shape[0], chunk_size):
        chunk = weight[start : start + chunk_size]
        current = chunk.detach().float().abs().amax(dim=reduce_dims)
        torch.maximum(accumulator, current, out=accumulator)
    return accumulator


@torch.no_grad()
def smooth_lm(
    model: nn.Module,
    act_scales: dict[str, torch.Tensor],
    alpha: float = 0.5,
    *,
    return_group_stats: bool = False,
    scope: str = "all",
    scale_policy: str = "requested",
    norm_materialization: str = "target_dtype",
) -> tuple[int, int] | tuple[int, int, list[dict[str, Any]]]:
    """Smooth every supported decoder layer of ``model`` in place.

    Returns ``(n_layers, n_groups)`` describing how much was actually smoothed
    so callers can fail loudly when an architecture silently matched nothing.
    """
    if hasattr(model, "_smoothquant_applied"):
        raise RuntimeError(
            "SmoothQuant was already applied to this model instance; reload the "
            "original checkpoint before applying another alpha"
        )
    if scope not in {"all", "moe"}:
        raise ValueError(f"unsupported SmoothQuant scope: {scope}")

    n_layers = 0
    n_groups = 0
    group_stats: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not is_decoder_layer(module):
            continue
        n_layers += 1
        for group in smooth_groups(module, name):
            if scope == "moe" and not group.is_moe:
                continue
            if group.act_scales_key not in act_scales:
                raise KeyError(
                    f"missing activation scale '{group.act_scales_key}'; "
                    "calibration and smoothing must use the same model"
                )
            scales = smooth_ln_fcs(
                group.norm,
                group.fcs,
                act_scales[group.act_scales_key],
                alpha,
                offset_one_norm=group.offset_one_norm,
                expert_weight=group.expert_weight,
                compensation_weights=group.compensation_weights,
                scale_policy=scale_policy,
                norm_materialization=norm_materialization,
            )
            group_stats.append(
                {
                    "layer": name,
                    "act_scales_key": group.act_scales_key,
                    "kind": "moe" if group.is_moe else "attention_or_dense_mlp",
                    "scale_min": float(scales.min().item()),
                    "scale_max": float(scales.max().item()),
                    "scale_mean": float(scales.mean().item()),
                    "scale_policy": scale_policy,
                    "norm_materialization": norm_materialization,
                    "scale_log2_rounding_error_max": (
                        float(
                            (scales.log2() - scales.log2().round()).abs().max().item()
                        )
                        if scale_policy == "power_of_two" else None
                    ),
                }
            )
            n_groups += 1

    if n_layers == 0:
        raise NotImplementedError(
            f"no supported decoder layer found in {type(model).__name__}; "
            "see llm_quant/smoothquant/adapters.py"
        )
    if n_groups == 0:
        raise RuntimeError(f"SmoothQuant scope {scope!r} selected no groups")
    model._smoothquant_applied = {
        "alpha": alpha,
        "n_groups": n_groups,
        "scope": scope,
        "scale_policy": scale_policy,
        "norm_materialization": norm_materialization,
    }
    if return_group_stats:
        return n_layers, n_groups, group_stats
    return n_layers, n_groups

# === REEX_SMOOTHQUANT END ===
