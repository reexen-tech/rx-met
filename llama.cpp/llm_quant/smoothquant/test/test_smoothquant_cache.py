"""Versioned partial/complete SmoothQuant calibration cache tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from llm_quant.smoothquant.calibration import CalibrationState
from llm_quant.smoothquant.runner import (
    ADAPTER_VERSION,
    CACHE_SCHEMA_VERSION,
    _save_calibration_cache,
    run_smoothquant,
)

from .test_qwen35_moe_smoothquant import _TinyQwen35
from .test_smoothquant_act_scales import _StubTokenizer, write_calib_jsonl


def _source_dir(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen3_5_moe"}\n')
    return source


def _run(tmp_path: Path, model, calib: Path, **overrides):
    kwargs = {
        "alpha": 0.80,
        "n_samples": 4,
        "seqlen": 8,
        "calib_data": str(calib),
        "model_path": str(tmp_path / "source"),
        "act_scales_cache": str(tmp_path / "act_scales.pt"),
        "checkpoint_every": 2,
    }
    kwargs.update(overrides)
    return run_smoothquant(model, _StubTokenizer(32), **kwargs)


def test_partial_cache_resumes_after_interrupted_forward(tmp_path) -> None:
    _source_dir(tmp_path)
    calib = write_calib_jsonl(
        tmp_path / "calib.jsonl", ["aa", "bbb", "cccc", "ddddd"]
    )
    model = _TinyQwen35()
    original_forward = model.forward
    calls = 0

    def interrupted(input_ids):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("interrupted")
        return original_forward(input_ids)

    model.forward = interrupted
    with pytest.raises(RuntimeError, match="interrupted"):
        _run(tmp_path, model, calib)

    partial = torch.load(tmp_path / "act_scales.pt", weights_only=False)
    assert partial["complete"] is False
    assert partial["actual_samples"] == partial["next_dataset_index"] == 2

    result = _run(
        tmp_path,
        _TinyQwen35(),
        calib,
        alpha=0.85,
        resume_act_scales=True,
    )
    complete = torch.load(tmp_path / "act_scales.pt", weights_only=False)
    assert result.calibration_resumed
    assert result.n_samples == 4
    assert complete["complete"] is True
    assert complete["actual_samples"] == complete["next_dataset_index"] == 4


def test_complete_cache_reused_across_alpha_without_alpha_in_key(tmp_path) -> None:
    _source_dir(tmp_path)
    calib = write_calib_jsonl(tmp_path / "calib.jsonl", ["one", "two"])
    first = _run(tmp_path, _TinyQwen35(), calib, n_samples=2, alpha=0.80)

    class _TokenizerMustNotRun:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("complete cache should skip tokenization")

    second = run_smoothquant(
        _TinyQwen35(),
        _TokenizerMustNotRun(),
        alpha=0.85,
        n_samples=2,
        seqlen=8,
        calib_data=str(calib),
        model_path=str(tmp_path / "source"),
        act_scales_cache=str(tmp_path / "act_scales.pt"),
        reuse_act_scales=True,
    )

    payload = torch.load(tmp_path / "act_scales.pt", weights_only=False)
    assert first.alpha == 0.80 and second.alpha == 0.85
    assert second.act_scales_from_cache
    assert "alpha" not in payload
    assert payload["schema_version"] == CACHE_SCHEMA_VERSION
    assert payload["adapter_version"] == ADAPTER_VERSION


def test_cache_fingerprint_detects_same_size_data_change(tmp_path) -> None:
    _source_dir(tmp_path)
    calib = write_calib_jsonl(tmp_path / "calib.jsonl", ["a"])
    _run(tmp_path, _TinyQwen35(), calib, n_samples=1)
    original_size = calib.stat().st_size
    write_calib_jsonl(calib, ["b"])
    assert calib.stat().st_size == original_size

    with pytest.raises(ValueError, match="calibration_file_fingerprint"):
        _run(
            tmp_path,
            _TinyQwen35(),
            calib,
            n_samples=1,
            reuse_act_scales=True,
        )


def test_partial_cache_cannot_be_reused_as_complete(tmp_path) -> None:
    _source_dir(tmp_path)
    calib = write_calib_jsonl(tmp_path / "calib.jsonl", ["aa", "bb"])
    model = _TinyQwen35()
    original_forward = model.forward
    calls = 0

    def interrupted(input_ids):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stop")
        return original_forward(input_ids)

    model.forward = interrupted
    with pytest.raises(RuntimeError, match="stop"):
        _run(tmp_path, model, calib, n_samples=2, checkpoint_every=1)

    with pytest.raises(ValueError, match="partial"):
        _run(
            tmp_path,
            _TinyQwen35(),
            calib,
            n_samples=2,
            checkpoint_every=1,
            reuse_act_scales=True,
        )


def test_atomic_cache_save_replaces_from_same_directory(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "nested" / "cache.pt"
    state = CalibrationState(
        act_scales={"x": torch.ones(3)},
        actual_samples=2,
        next_dataset_index=2,
        complete=False,
    )
    real_replace = os.replace
    replacements = []

    def record_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("llm_quant.smoothquant.runner.os.replace", record_replace)
    _save_calibration_cache(
        cache,
        {"schema_version": CACHE_SCHEMA_VERSION, "adapter_version": ADAPTER_VERSION},
        state,
        complete=False,
    )

    assert cache.is_file()
    assert len(replacements) == 1
    temporary, destination = replacements[0]
    assert temporary.parent == destination.parent == cache.parent
    assert temporary.suffix == ".tmp"
    assert not temporary.exists()
    assert torch.load(cache, weights_only=False)["complete"] is False


def test_corrupt_cache_fails_loudly(tmp_path) -> None:
    _source_dir(tmp_path)
    calib = write_calib_jsonl(tmp_path / "calib.jsonl", ["x"])
    (tmp_path / "act_scales.pt").write_bytes(b"not a torch cache")

    with pytest.raises(ValueError, match="failed to load SmoothQuant cache"):
        _run(
            tmp_path,
            _TinyQwen35(),
            calib,
            n_samples=1,
            reuse_act_scales=True,
        )
