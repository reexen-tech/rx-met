from __future__ import annotations

import torch
import torch.nn as nn

from llm_quant.gptq.hessian import GPTQQ4064
from llm_quant.gptq.q4_0_64 import (
    TYPE_SIZE,
    fixed_point_mask,
    pack_q4_0_64,
    quantize_q4_0_64,
)


def test_gptq_output_exposes_non_fixed_point_blocks() -> None:
    generator = torch.Generator().manual_seed(0)
    layer = nn.Linear(128, 32, bias=False)
    with torch.no_grad():
        layer.weight.copy_(
            torch.randn(layer.weight.shape, generator=generator) * 0.05
        )
    inputs = torch.randn((4, 16, 128), generator=generator)
    quantizer = GPTQQ4064(layer)
    quantizer.add_batch(inputs)
    result = quantizer.fasterquant()

    expected = pack_q4_0_64(result.codes, result.scales).reshape(-1, TYPE_SIZE)
    fake_fp16 = result.dequant.to(torch.float16)
    actual = quantize_q4_0_64(fake_fp16).packed.reshape(-1, TYPE_SIZE)
    measured = (expected == actual).all(dim=-1).reshape(result.scales.shape)
    assert torch.equal(measured, result.fixed_point)
    assert not measured.all(), "the sidecar path is unnecessary if every block is fixed"


def test_negative_eight_anchor_is_sufficient_for_simple_grid_block() -> None:
    codes = torch.arange(-8, 8, dtype=torch.int8).repeat(4).reshape(1, 64)
    scales = torch.tensor([[0.125]], dtype=torch.float16)
    assert fixed_point_mask(codes, scales).item()


def test_missing_anchor_requires_direct_write() -> None:
    codes = torch.full((1, 64), 3, dtype=torch.int8)
    scales = torch.tensor([[-0.25]], dtype=torch.float16)
    assert not fixed_point_mask(codes, scales).item()
