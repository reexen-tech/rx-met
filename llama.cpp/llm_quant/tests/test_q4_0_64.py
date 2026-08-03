from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402

from llm_quant.gptq.q4_0_64 import (  # noqa: E402
    fixed_point_mask,
    pack_q4_0_64,
    quantize_q4_0_64,
    unpack_q4_0_64,
)


def test_torch_quant_matches_gguf_python_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    weights = torch.randn((17, 256), generator=generator, dtype=torch.float32)
    weights[0] = 0
    weights[1, 0] = 1
    weights[1, 1] = -1

    result = quantize_q4_0_64(weights)
    expected = gguf.quantize(
        weights.numpy(), gguf.GGMLQuantizationType.Q4_0_64
    )
    np.testing.assert_array_equal(result.packed.numpy(), expected)


def test_pack_unpack_round_trip_all_codes_and_scale_signs() -> None:
    codes = torch.arange(-8, 8, dtype=torch.int8).repeat(4).reshape(1, 64)
    codes = torch.cat((codes, codes.flip(-1)), dim=-1)
    scales = torch.tensor([[0.125, -0.25]], dtype=torch.float16)
    packed = pack_q4_0_64(codes, scales)
    unpacked_codes, unpacked_scales = unpack_q4_0_64(packed)
    assert torch.equal(unpacked_codes, codes)
    assert torch.equal(unpacked_scales, scales)


@pytest.mark.parametrize(
    "values",
    [
        torch.zeros(64),
        torch.full((64,), 0.125),
        torch.tensor([1.0, -1.0] + [0.0] * 62),
        torch.linspace(-1.0, 1.0, 64),
    ],
)
def test_special_blocks_match_reference(values: torch.Tensor) -> None:
    result = quantize_q4_0_64(values.reshape(1, 64))
    expected = gguf.quantize(
        values.reshape(1, 64).numpy(), gguf.GGMLQuantizationType.Q4_0_64
    )
    np.testing.assert_array_equal(result.packed.numpy(), expected)


def test_fixed_point_detects_missing_negative_eight_anchor() -> None:
    codes = torch.ones((1, 64), dtype=torch.int8)
    scales = torch.ones((1, 1), dtype=torch.float16)
    assert not fixed_point_mask(codes, scales).item()


def test_invalid_shape_rejected() -> None:
    with pytest.raises(ValueError, match="multiple of 64"):
        quantize_q4_0_64(torch.zeros(1, 63))
