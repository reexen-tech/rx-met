from __future__ import annotations

import torch
import torch.nn as nn

from llm_quant.gptq.hessian import GPTQQ4164
from llm_quant.gptq.q4_1_64 import (
    dequantize_q4_1_64,
    pack_q4_1_64,
    quantize_q4_1_64,
    unpack_q4_1_64,
)


def test_q4_1_64_pack_round_trip() -> None:
    codes = (torch.arange(2 * 128).reshape(2, 128) % 16).to(torch.uint8)
    scales = torch.tensor(
        [[[0.25, -1.0], [0.5, -2.0]], [[0.125, 0.0], [1.0, -8.0]]],
        dtype=torch.float16,
    )
    packed = pack_q4_1_64(codes, scales)
    assert packed.shape == (2, 72)
    restored_codes, restored_scales = unpack_q4_1_64(packed)
    assert torch.equal(restored_codes, codes)
    assert torch.equal(restored_scales, scales)
    torch.testing.assert_close(
        dequantize_q4_1_64(restored_codes, restored_scales),
        dequantize_q4_1_64(codes, scales),
        rtol=0,
        atol=0,
    )


def test_q4_1_64_reference_layout_and_constant_group() -> None:
    weights = torch.linspace(-1.0, 2.0, 64).reshape(1, 64)
    result = quantize_q4_1_64(weights)
    parameter_bytes = (
        result.scales.contiguous().view(torch.uint8).reshape(1, 4)
    )
    assert torch.equal(result.packed[:, :4], parameter_bytes)
    assert result.codes.min() == 0
    assert result.codes.max() == 15

    constant = quantize_q4_1_64(torch.full((2, 64), 0.75))
    assert torch.count_nonzero(constant.codes) == 0
    assert torch.all(constant.scales[..., 0] == 0)
    torch.testing.assert_close(
        constant.dequant,
        torch.full((2, 64), 0.75),
        rtol=0,
        atol=0,
    )


def test_q4_1_64_gptq_uses_affine_parameters() -> None:
    generator = torch.Generator().manual_seed(17)
    layer = nn.Linear(128, 16, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(
            torch.randn(layer.weight.shape, generator=generator) * 0.05 + 0.02
        )
    inputs = torch.randn((4, 16, 128), generator=generator)
    quantizer = GPTQQ4164(layer)
    quantizer.add_batch(inputs)
    result = quantizer.fasterquant()

    assert result.codes.dtype == torch.int8
    assert result.codes.shape == (16, 128)
    assert result.scales.shape == (16, 2, 2)
    assert result.scales.dtype == torch.float16
    assert result.packed.shape == (16, 72)
    assert torch.all(result.codes <= 15)
    torch.testing.assert_close(layer.weight, result.dequant, rtol=0, atol=0)
