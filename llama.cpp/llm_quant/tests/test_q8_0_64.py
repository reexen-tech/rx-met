from __future__ import annotations

import pytest
import torch

from llm_quant.gptq.formats import format_for_bits, format_for_name
from llm_quant.gptq.q8_0_64 import (
    fixed_point_mask,
    pack_q8_0_64,
    quantize_q8_0_64,
    round_away_from_zero,
    unpack_q8_0_64,
)


def test_round_away_from_zero_matches_c_roundf_boundaries() -> None:
    values = torch.tensor(
        [-126.5, -1.5, -0.5, -0.49, 0.0, 0.49, 0.5, 1.5, 126.5]
    )
    expected = torch.tensor(
        [-127, -2, -1, 0, 0, 0, 1, 2, 127], dtype=torch.int32
    )
    assert torch.equal(round_away_from_zero(values), expected)


def test_quantize_matches_frozen_c_reference_block() -> None:
    # The +/-127 anchors force d=1.0. The remaining values exercise C roundf.
    values = torch.zeros(64, dtype=torch.float32)
    values[:8] = torch.tensor(
        [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5]
    )
    result = quantize_q8_0_64(values.reshape(1, 64))

    expected_codes = torch.zeros(64, dtype=torch.int8)
    expected_codes[:8] = torch.tensor(
        [127, -127, 1, -1, 2, -2, 127, -127], dtype=torch.int8
    )
    expected = torch.cat(
        (
            torch.tensor([0x00, 0x3C], dtype=torch.uint8),
            expected_codes.view(torch.uint8),
        )
    ).reshape(1, 66)
    assert torch.equal(result.packed, expected)


def test_pack_unpack_round_trip() -> None:
    codes = torch.tensor(
        [-127, -126, -2, -1, 0, 1, 2, 126, 127], dtype=torch.int8
    ).repeat(8)[:64]
    codes = torch.stack((codes, codes.flip(-1)), dim=0)
    scales = torch.tensor([[0.125], [0.25]], dtype=torch.float16)
    packed = pack_q8_0_64(codes, scales)
    unpacked_codes, unpacked_scales = unpack_q8_0_64(packed)
    assert packed.shape == (2, 66)
    assert torch.equal(unpacked_codes, codes)
    assert torch.equal(unpacked_scales, scales)


def test_zero_block_and_fixed_point_behavior() -> None:
    zero = quantize_q8_0_64(torch.zeros(1, 64))
    assert torch.equal(zero.codes, torch.zeros_like(zero.codes))
    assert torch.equal(zero.scales, torch.zeros_like(zero.scales))
    assert fixed_point_mask(zero.codes, zero.scales).item()

    codes = torch.ones((1, 64), dtype=torch.int8)
    scales = torch.ones((1, 1), dtype=torch.float16)
    assert not fixed_point_mask(codes, scales).item()


def test_format_registry() -> None:
    assert format_for_bits(4).name == "Q4_0_64"
    assert format_for_bits(8).name == "Q8_0_64"
    assert format_for_name("Q8_0_64").type_size == 66
    with pytest.raises(ValueError, match="unsupported GPTQ weight bits"):
        format_for_bits(3)


def test_invalid_shape_rejected() -> None:
    with pytest.raises(ValueError, match="multiple of 64"):
        quantize_q8_0_64(torch.zeros(1, 63))
