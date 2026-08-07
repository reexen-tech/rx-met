"""Qwen3.5 representative activation hooks and routing coverage tests."""

from __future__ import annotations

import torch

from llm_quant.smoothquant.adapters import qwen35_calibration_targets
from llm_quant.smoothquant.calibration import (
    calibrate_progressively,
    expert_coverage_summary,
)

from .test_qwen35_moe_smoothquant import _TinyQwen35
from .test_smoothquant_act_scales import _StubTokenizer, write_calib_jsonl


def _calib_file(tmp_path, count: int = 5):
    return write_calib_jsonl(
        tmp_path / "calib.jsonl",
        [chr(ord("a") + i) * (i + 2) for i in range(count)],
    )


def test_qwen35_targets_use_one_representative_per_boundary(tmp_path) -> None:
    model = _TinyQwen35()
    targets = qwen35_calibration_targets(model)
    assert targets is not None
    assert targets.act_module_names == (
        "layers.0.linear_attn.in_proj_qkv",
        "layers.0.mlp.shared_expert.gate_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.mlp.shared_expert.gate_proj",
    )

    state = calibrate_progressively(
        model,
        _StubTokenizer(32),
        _calib_file(tmp_path),
        min_samples=2,
        max_samples=5,
        seq_len=8,
        target_modules=targets.act_module_names,
        routing_targets=targets.routers,
        progress_every=0,
    )

    assert set(state.act_scales) == set(targets.act_module_names)
    assert state.expert_hit_counts.shape == (2, 4)
    assert state.actual_samples == state.next_dataset_index == 2
    assert state.complete
    assert all(int(row.sum()) > 0 for row in state.expert_hit_counts)


def test_partial_callback_every_32_equivalent_interval(tmp_path) -> None:
    model = _TinyQwen35()
    targets = qwen35_calibration_targets(model)
    checkpoints = []

    calibrate_progressively(
        model,
        _StubTokenizer(32),
        _calib_file(tmp_path, count=5),
        min_samples=5,
        max_samples=5,
        seq_len=8,
        target_modules=targets.act_module_names,
        routing_targets=targets.routers,
        checkpoint_every=2,
        checkpoint_callback=lambda state: checkpoints.append(
            (state.actual_samples, state.next_dataset_index, state.complete)
        ),
        progress_every=0,
    )

    assert checkpoints == [(2, 2, False), (4, 4, False)]


def test_coverage_policy_warns_for_smoke_but_blocks_formal(tmp_path) -> None:
    model = _TinyQwen35()
    for layer in model.layers:
        layer.mlp.gate.weight.data.zero_()
    targets = qwen35_calibration_targets(model)
    calib = _calib_file(tmp_path, count=3)

    smoke = calibrate_progressively(
        model,
        _StubTokenizer(32),
        calib,
        min_samples=2,
        max_samples=3,
        seq_len=8,
        target_modules=targets.act_module_names,
        routing_targets=targets.routers,
        require_full_coverage=False,
        progress_every=0,
    )
    assert smoke.complete and smoke.actual_samples == 2
    assert expert_coverage_summary(smoke)["missing_pairs"] > 0

    formal = calibrate_progressively(
        model,
        _StubTokenizer(32),
        calib,
        min_samples=2,
        max_samples=3,
        seq_len=8,
        target_modules=targets.act_module_names,
        routing_targets=targets.routers,
        require_full_coverage=True,
        progress_every=0,
    )
    summary = expert_coverage_summary(formal)
    assert not formal.complete and formal.actual_samples == 3
    assert formal.stop_reason == "maximum_samples_without_full_coverage"
    assert summary["covered_pairs"] < summary["total_pairs"]
