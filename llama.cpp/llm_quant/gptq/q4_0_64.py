"""Q4_0_64 quantization primitives matching ggml's REEX reference layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch


GROUP_SIZE = 64
TYPE_SIZE = 34
QMIN = -8
QMAX = 7


@dataclass(frozen=True)
class Q4064Result:
    dequant: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor
    packed: torch.Tensor


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")


def nearest_int(values: torch.Tensor) -> torch.Tensor:
    """Round float32 values to nearest integer, ties to even.

    ``reex_nearest_int`` relies on the default IEEE round-to-nearest-even mode.
    PyTorch's float32 ``round`` has the same tie rule on CPU and CUDA.
    """

    values = values.to(torch.float32)
    if torch.any(values.abs() > 4_194_303):
        raise ValueError("nearest_int input exceeds reex_nearest_int range")
    return torch.round(values).to(torch.int32)


def find_scales(groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the C++ search scale in FP32 and its stored FP16 representation."""

    if groups.ndim < 2 or groups.shape[-1] != GROUP_SIZE:
        raise ValueError(f"expected groups with last dimension {GROUP_SIZE}")
    groups_f32 = groups.to(torch.float32)
    _require_finite(groups_f32, "weights")
    max_indices = groups_f32.abs().argmax(dim=-1, keepdim=True)
    signed_max = torch.gather(groups_f32, -1, max_indices)
    scale_f32 = signed_max / -8.0
    return scale_f32, scale_f32.to(torch.float16)


def quantize_with_scales(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Quantize values to the runtime grid represented by stored FP16 scales."""

    values_f32 = values.to(torch.float32)
    scales_f32 = scales.to(torch.float16).to(torch.float32)
    inv = torch.where(scales_f32 != 0, scales_f32.reciprocal(), 0.0)
    return nearest_int(values_f32 * inv).clamp(QMIN, QMAX).to(torch.int8)


def pack_q4_0_64(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack signed codes and FP16 scales into 34-byte Q4_0_64 blocks."""

    if codes.ndim < 2 or codes.shape[-1] % GROUP_SIZE:
        raise ValueError("codes last dimension must be a non-zero multiple of 64")
    if codes.numel() and (torch.any(codes < QMIN) or torch.any(codes > QMAX)):
        raise ValueError("codes must be in [-8, 7]")

    n_groups = codes.shape[-1] // GROUP_SIZE
    expected_scale_shape = (*codes.shape[:-1], n_groups)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"expected scales shape {expected_scale_shape}, got {tuple(scales.shape)}"
        )

    device = codes.device
    grouped = codes.to(torch.int16).reshape(*codes.shape[:-1], n_groups, GROUP_SIZE)
    low = grouped[..., : GROUP_SIZE // 2] & 0x0F
    high = grouped[..., GROUP_SIZE // 2 :] & 0x0F
    quant_bytes = (low | (high << 4)).to(torch.uint8)

    # torch does not expose a portable float16->uint8 view on every backend.
    scale_bytes = (
        scales.detach()
        .to(device="cpu", dtype=torch.float16)
        .contiguous()
        .view(torch.uint8)
        .reshape(*scales.shape, 2)
        .to(device)
    )
    blocks = torch.cat((scale_bytes, quant_bytes), dim=-1)
    return blocks.reshape(*codes.shape[:-1], n_groups * TYPE_SIZE)


def unpack_q4_0_64(
    packed: torch.Tensor, *, logical_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack Q4_0_64 bytes into signed codes and FP16 scales."""

    if packed.ndim < 1 or packed.shape[-1] % TYPE_SIZE:
        raise ValueError("packed last dimension must be a multiple of 34")
    packed_cpu = packed.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    n_groups = packed_cpu.shape[-1] // TYPE_SIZE
    blocks = packed_cpu.reshape(*packed_cpu.shape[:-1], n_groups, TYPE_SIZE)
    scales = (
        blocks[..., :2]
        .contiguous()
        .view(torch.float16)
        .reshape(*packed_cpu.shape[:-1], n_groups)
    )
    quant_bytes = blocks[..., 2:]
    low = (quant_bytes & 0x0F).to(torch.int8)
    high = ((quant_bytes >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    codes = torch.cat((low, high), dim=-1).reshape(
        *packed_cpu.shape[:-1], n_groups * GROUP_SIZE
    )
    if logical_size is not None and logical_size != codes.shape[-1]:
        raise ValueError(
            f"logical size {logical_size} does not match packed size {codes.shape[-1]}"
        )
    return codes.to(packed.device), scales.to(packed.device)


def dequantize_q4_0_64(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % GROUP_SIZE:
        raise ValueError("codes last dimension must be a multiple of 64")
    n_groups = codes.shape[-1] // GROUP_SIZE
    if tuple(scales.shape) != (*codes.shape[:-1], n_groups):
        raise ValueError("scales shape does not match codes")
    grouped = codes.to(torch.float32).reshape(
        *codes.shape[:-1], n_groups, GROUP_SIZE
    )
    return (
        grouped * scales.to(torch.float16).to(torch.float32).unsqueeze(-1)
    ).reshape(codes.shape)


def quantize_q4_0_64(weights: torch.Tensor) -> Q4064Result:
    """Quantize rows exactly like ``quantize_row_q4_0_64_ref``.

    C++ computes codes with the unrounded FP32 search scale, but stores the scale
    in FP16 and uses that stored value for dequantization.
    """

    if weights.ndim < 1 or not weights.shape[-1] or weights.shape[-1] % GROUP_SIZE:
        raise ValueError("weights last dimension must be a non-zero multiple of 64")

    original_shape = weights.shape
    n_groups = original_shape[-1] // GROUP_SIZE
    groups = weights.to(torch.float32).reshape(
        *original_shape[:-1], n_groups, GROUP_SIZE
    )
    scale_f32, scales_f16 = find_scales(groups)
    inv = torch.where(scale_f32 != 0, scale_f32.reciprocal(), 0.0)
    codes = nearest_int(groups * inv).clamp(QMIN, QMAX).to(torch.int8)
    codes = codes.reshape(original_shape)
    scales = scales_f16.squeeze(-1)
    dequant = dequantize_q4_0_64(codes, scales).to(weights.dtype)
    packed = pack_q4_0_64(codes, scales)
    return Q4064Result(dequant=dequant, codes=codes, scales=scales, packed=packed)


def fixed_point_mask(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Return one boolean per block indicating fake-FP16 RTN byte stability."""

    dequant = dequantize_q4_0_64(codes, scales).to(torch.float16)
    requant = quantize_q4_0_64(dequant)
    expected = pack_q4_0_64(codes, scales).reshape(-1, TYPE_SIZE)
    actual = requant.packed.reshape(-1, TYPE_SIZE)
    return (expected == actual).all(dim=-1).reshape(*scales.shape)
