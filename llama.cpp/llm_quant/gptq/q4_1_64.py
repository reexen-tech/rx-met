"""Q4_1_64 affine quantization matching ggml's REEX block layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch


GROUP_SIZE = 64
TYPE_SIZE = 36
QMIN = 0
QMAX = 15
PARAMETER_COUNT = 2


@dataclass(frozen=True)
class Q4164Result:
    dequant: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor
    packed: torch.Tensor


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")


def round_affine(values: torch.Tensor) -> torch.Tensor:
    """Round affine codes like the C reference's positive x + 0.5."""

    values = values.to(torch.float32)
    _require_finite(values, "round input")
    return torch.floor(values + 0.5).clamp(QMIN, QMAX).to(torch.uint8)


def find_scales(groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 and stored FP16 [delta, minimum] parameters."""

    if groups.ndim < 2 or groups.shape[-1] != GROUP_SIZE:
        raise ValueError(f"expected groups with last dimension {GROUP_SIZE}")
    groups_f32 = groups.to(torch.float32)
    _require_finite(groups_f32, "weights")
    minimum = groups_f32.amin(dim=-1)
    maximum = groups_f32.amax(dim=-1)
    parameters = torch.stack(((maximum - minimum) / QMAX, minimum), dim=-1)
    return parameters, parameters.to(torch.float16)


def quantize_with_scales(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values_f32 = values.to(torch.float32)
    params_f32 = scales.to(torch.float16).to(torch.float32)
    if params_f32.shape[-1] != PARAMETER_COUNT:
        raise ValueError("Q4_1_64 parameters must end with [delta, minimum]")
    _require_finite(values_f32, "weights")
    _require_finite(params_f32, "parameters")
    delta, minimum = params_f32.unbind(dim=-1)
    inv = torch.where(delta != 0, delta.reciprocal(), 0.0)
    return round_affine((values_f32 - minimum) * inv)


def dequantize_with_scales(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    params_f32 = scales.to(torch.float16).to(torch.float32)
    if params_f32.shape[-1] != PARAMETER_COUNT:
        raise ValueError("Q4_1_64 parameters must end with [delta, minimum]")
    delta, minimum = params_f32.unbind(dim=-1)
    return codes.to(torch.float32) * delta + minimum


def pack_q4_1_64(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack unsigned codes plus FP16 delta/min into 36-byte blocks."""

    if codes.ndim < 2 or not codes.shape[-1] or codes.shape[-1] % GROUP_SIZE:
        raise ValueError("codes last dimension must be a non-zero multiple of 64")
    if codes.numel() and (torch.any(codes < QMIN) or torch.any(codes > QMAX)):
        raise ValueError("codes must be in [0, 15]")
    n_groups = codes.shape[-1] // GROUP_SIZE
    expected = (*codes.shape[:-1], n_groups, PARAMETER_COUNT)
    if tuple(scales.shape) != expected:
        raise ValueError(f"expected scales shape {expected}, got {tuple(scales.shape)}")

    grouped = codes.to(torch.uint8).reshape(
        *codes.shape[:-1], n_groups, GROUP_SIZE
    )
    quant_bytes = grouped[..., :32] | (grouped[..., 32:] << 4)
    parameter_bytes = (
        scales.detach()
        .to(device="cpu", dtype=torch.float16)
        .contiguous()
        .view(torch.uint8)
        .reshape(*scales.shape[:-1], 2 * PARAMETER_COUNT)
        .to(codes.device)
    )
    return torch.cat((parameter_bytes, quant_bytes), dim=-1).reshape(
        *codes.shape[:-1], n_groups * TYPE_SIZE
    )


def unpack_q4_1_64(
    packed: torch.Tensor, *, logical_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if packed.ndim < 1 or not packed.shape[-1] or packed.shape[-1] % TYPE_SIZE:
        raise ValueError("packed last dimension must be a non-zero multiple of 36")
    packed_cpu = packed.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    n_groups = packed_cpu.shape[-1] // TYPE_SIZE
    blocks = packed_cpu.reshape(*packed_cpu.shape[:-1], n_groups, TYPE_SIZE)
    scales = (
        blocks[..., :4]
        .contiguous()
        .view(torch.float16)
        .reshape(*packed_cpu.shape[:-1], n_groups, PARAMETER_COUNT)
    )
    quant_bytes = blocks[..., 4:]
    codes = torch.cat((quant_bytes & 0x0F, (quant_bytes >> 4) & 0x0F), dim=-1)
    codes = codes.reshape(*packed_cpu.shape[:-1], n_groups * GROUP_SIZE)
    if logical_size is not None and logical_size != codes.shape[-1]:
        raise ValueError(
            f"logical size {logical_size} does not match packed size {codes.shape[-1]}"
        )
    return codes.to(packed.device), scales.to(packed.device)


def dequantize_q4_1_64(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % GROUP_SIZE:
        raise ValueError("codes last dimension must be a multiple of 64")
    n_groups = codes.shape[-1] // GROUP_SIZE
    expected = (*codes.shape[:-1], n_groups, PARAMETER_COUNT)
    if tuple(scales.shape) != expected:
        raise ValueError(f"expected scales shape {expected}, got {tuple(scales.shape)}")
    grouped = codes.reshape(*codes.shape[:-1], n_groups, GROUP_SIZE)
    return dequantize_with_scales(grouped, scales.unsqueeze(-2)).reshape(codes.shape)


def quantize_q4_1_64(weights: torch.Tensor) -> Q4164Result:
    """Quantize rows like quantize_row_q4_1_64_ref."""

    if weights.ndim < 1 or not weights.shape[-1] or weights.shape[-1] % GROUP_SIZE:
        raise ValueError("weights last dimension must be a non-zero multiple of 64")
    original_shape = weights.shape
    n_groups = original_shape[-1] // GROUP_SIZE
    groups = weights.to(torch.float32).reshape(
        *original_shape[:-1], n_groups, GROUP_SIZE
    )
    params_f32, scales = find_scales(groups)
    delta, minimum = params_f32.unbind(dim=-1)
    inv = torch.where(delta != 0, delta.reciprocal(), 0.0)
    codes = round_affine(
        (groups - minimum.unsqueeze(-1)) * inv.unsqueeze(-1)
    ).reshape(original_shape)
    dequant = dequantize_q4_1_64(codes, scales).to(weights.dtype)
    packed = pack_q4_1_64(codes, scales)
    return Q4164Result(dequant=dequant, codes=codes, scales=scales, packed=packed)


def fixed_point_mask(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Return one boolean per block indicating fake-FP16 RTN byte stability."""

    dequant = dequantize_q4_1_64(codes, scales).to(torch.float16)
    requant = quantize_q4_1_64(dequant)
    expected = pack_q4_1_64(codes, scales).reshape(-1, TYPE_SIZE)
    actual = requant.packed.reshape(-1, TYPE_SIZE)
    return (expected == actual).all(dim=-1).reshape(*scales.shape[:-1])
