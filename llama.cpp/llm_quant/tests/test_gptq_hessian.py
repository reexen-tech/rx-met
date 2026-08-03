from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llm_quant.gptq.hessian import GPTQQ4064
from llm_quant.gptq.q4_0_64 import quantize_q4_0_64


def _layer_and_input(seed: int = 0) -> tuple[nn.Linear, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    layer = nn.Linear(128, 32, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(
            torch.randn(layer.weight.shape, generator=generator) * 0.05
        )
    inputs = torch.randn((4, 16, 128), generator=generator)
    return layer, inputs


def test_hessian_single_and_multiple_batches_match() -> None:
    layer_a, inputs = _layer_and_input()
    layer_b = nn.Linear(128, 32, bias=False)
    with torch.no_grad():
        layer_b.weight.copy_(layer_a.weight)
    one = GPTQQ4064(layer_a)
    split = GPTQQ4064(layer_b)
    one.add_batch(inputs)
    split.add_batch(inputs[:1])
    split.add_batch(inputs[1:3])
    split.add_batch(inputs[3:])
    assert one.n_tokens == split.n_tokens == 64
    torch.testing.assert_close(one.H_sum, split.H_sum, rtol=1e-5, atol=1e-5)


def test_gptq_writes_valid_q4_0_64_weights() -> None:
    layer, inputs = _layer_and_input(1)
    quantizer = GPTQQ4064(layer)
    quantizer.add_batch(inputs)
    result = quantizer.fasterquant()
    assert result.codes.dtype == torch.int8
    assert result.scales.dtype == torch.float16
    assert result.codes.shape == layer.weight.shape
    assert result.scales.shape == (32, 2)
    assert result.packed.shape == (32, 68)
    torch.testing.assert_close(layer.weight, result.dequant, rtol=0, atol=0)


def test_gptq_output_error_is_no_worse_than_rtn() -> None:
    layer, inputs = _layer_and_input(3)
    original = layer.weight.detach().clone()
    x = inputs.reshape(-1, 128).transpose(0, 1)
    rtn = quantize_q4_0_64(original).dequant
    rtn_error = ((original - rtn) @ x).square().sum()

    quantizer = GPTQQ4064(layer)
    quantizer.add_batch(inputs)
    result = quantizer.fasterquant()
    gptq_error = ((original - result.dequant) @ x).square().sum()
    assert gptq_error <= rtn_error * 1.0001


def test_dead_columns_are_reported() -> None:
    layer, inputs = _layer_and_input(4)
    inputs[..., 7] = 0
    quantizer = GPTQQ4064(layer)
    quantizer.add_batch(inputs)
    result = quantizer.fasterquant()
    assert result.dead_columns == 1
    assert torch.count_nonzero(result.dequant[:, 7]) == 0


def test_cholesky_failure_falls_back_to_rtn_and_records_reason() -> None:
    layer, _ = _layer_and_input(5)
    original = layer.weight.detach().clone()
    repeated = torch.ones((1, 2, 128))
    quantizer = GPTQQ4064(layer)
    quantizer.add_batch(repeated)
    result = quantizer.fasterquant(damp_percent=0)
    expected = quantize_q4_0_64(original)
    assert result.method == "rtn"
    assert result.fallback_reason is not None
    assert result.fallback_reason.startswith("cholesky_failure:")
    torch.testing.assert_close(result.dequant, expected.dequant)


def test_zero_samples_falls_back_to_rtn_and_records_reason() -> None:
    layer, _ = _layer_and_input(7)
    original = layer.weight.detach().clone()
    result = GPTQQ4064(layer).fasterquant()
    expected = quantize_q4_0_64(original)
    assert result.method == "rtn"
    assert result.fallback_reason == "zero_samples"
    torch.testing.assert_close(result.dequant, expected.dequant)


def test_invalid_lazy_block_rejected() -> None:
    layer, inputs = _layer_and_input(6)
    quantizer = GPTQQ4064(layer)
    quantizer.add_batch(inputs)
    with pytest.raises(ValueError, match="multiple of 64"):
        quantizer.fasterquant(lazy_block_size=96)
