"""Physical Q64 block-format backends shared by GPTQ and sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from . import q4_0_64, q4_1_64, q8_0_64


@dataclass(frozen=True)
class Q64Format:
    name: str
    bits: int
    group_size: int
    type_size: int
    parameter_count: int
    find_scales: Callable[..., Any]
    quantize_with_scales: Callable[..., Any]
    dequantize_with_scales: Callable[..., Any]
    quantize: Callable[..., Any]
    pack: Callable[..., Any]
    unpack: Callable[..., Any]
    dequantize: Callable[..., Any]
    fixed_point_mask: Callable[..., Any]

    @property
    def basename(self) -> str:
        return f"gptq_{self.name.lower()}"


Q4_0_64 = Q64Format(
    name="Q4_0_64",
    bits=4,
    group_size=q4_0_64.GROUP_SIZE,
    type_size=q4_0_64.TYPE_SIZE,
    parameter_count=1,
    find_scales=q4_0_64.find_scales,
    quantize_with_scales=q4_0_64.quantize_with_scales,
    dequantize_with_scales=lambda codes, scales: (
        codes.to(torch.float32) * scales.to(torch.float16).to(torch.float32)
    ),
    quantize=q4_0_64.quantize_q4_0_64,
    pack=q4_0_64.pack_q4_0_64,
    unpack=q4_0_64.unpack_q4_0_64,
    dequantize=q4_0_64.dequantize_q4_0_64,
    fixed_point_mask=q4_0_64.fixed_point_mask,
)

Q4_1_64 = Q64Format(
    name="Q4_1_64",
    bits=4,
    group_size=q4_1_64.GROUP_SIZE,
    type_size=q4_1_64.TYPE_SIZE,
    parameter_count=q4_1_64.PARAMETER_COUNT,
    find_scales=q4_1_64.find_scales,
    quantize_with_scales=q4_1_64.quantize_with_scales,
    dequantize_with_scales=q4_1_64.dequantize_with_scales,
    quantize=q4_1_64.quantize_q4_1_64,
    pack=q4_1_64.pack_q4_1_64,
    unpack=q4_1_64.unpack_q4_1_64,
    dequantize=q4_1_64.dequantize_q4_1_64,
    fixed_point_mask=q4_1_64.fixed_point_mask,
)

Q8_0_64 = Q64Format(
    name="Q8_0_64",
    bits=8,
    group_size=q8_0_64.GROUP_SIZE,
    type_size=q8_0_64.TYPE_SIZE,
    parameter_count=1,
    find_scales=q8_0_64.find_scales,
    quantize_with_scales=q8_0_64.quantize_with_scales,
    dequantize_with_scales=lambda codes, scales: (
        codes.to(torch.float32) * scales.to(torch.float16).to(torch.float32)
    ),
    quantize=q8_0_64.quantize_q8_0_64,
    pack=q8_0_64.pack_q8_0_64,
    unpack=q8_0_64.unpack_q8_0_64,
    dequantize=q8_0_64.dequantize_q8_0_64,
    fixed_point_mask=q8_0_64.fixed_point_mask,
)

_BY_BITS = {item.bits: item for item in (Q4_0_64, Q8_0_64)}
_BY_NAME = {item.name: item for item in (Q4_0_64, Q4_1_64, Q8_0_64)}


def format_for_bits(bits: int) -> Q64Format:
    try:
        return _BY_BITS[bits]
    except KeyError as exc:
        raise ValueError(f"unsupported GPTQ weight bits: {bits}") from exc


def format_for_name(name: str) -> Q64Format:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unsupported GPTQ format: {name!r}") from exc
