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

import torch
import torch.nn as nn

from .adapters import is_decoder_layer, smooth_groups


@torch.no_grad()
def smooth_ln_fcs(
    ln: nn.Module,
    fcs: nn.Linear | list[nn.Linear],
    act_scales: torch.Tensor,
    alpha: float = 0.5,
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

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device=device, dtype=dtype)
    )

    ln.weight.div_(scales.to(ln.weight.device, ln.weight.dtype))
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(scales.to(ln.bias.device, ln.bias.dtype))

    for fc in fcs:
        fc.weight.mul_(scales.to(fc.weight.device, fc.weight.dtype).view(1, -1))

    return scales


@torch.no_grad()
def smooth_lm(
    model: nn.Module,
    act_scales: dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> tuple[int, int]:
    """Smooth every supported decoder layer of ``model`` in place.

    Returns ``(n_layers, n_groups)`` describing how much was actually smoothed
    so callers can fail loudly when an architecture silently matched nothing.
    """
    n_layers = 0
    n_groups = 0
    for name, module in model.named_modules():
        if not is_decoder_layer(module):
            continue
        n_layers += 1
        for group in smooth_groups(module, name):
            if group.act_scales_key not in act_scales:
                raise KeyError(
                    f"missing activation scale '{group.act_scales_key}'; "
                    "calibration and smoothing must use the same model"
                )
            smooth_ln_fcs(
                group.norm, group.fcs, act_scales[group.act_scales_key], alpha
            )
            n_groups += 1

    if n_layers == 0:
        raise NotImplementedError(
            f"no supported decoder layer found in {type(model).__name__}; "
            "see llm_quant/smoothquant/adapters.py"
        )
    return n_layers, n_groups

# === REEX_SMOOTHQUANT END ===
