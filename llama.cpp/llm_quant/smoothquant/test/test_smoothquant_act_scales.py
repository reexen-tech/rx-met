# === REEX_SMOOTHQUANT BEGIN: calibration hook and calib-path resolution tests ===
"""Tests for activation calibration and local dataset resolution."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from llm_quant.smoothquant.calibration import get_act_scales, resolve_calib_dataset


class _StubTokenizer:
    """Maps characters to ids so calibration needs no real tokenizer files."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, text, *, return_tensors=None, max_length=None, truncation=False):
        ids = [ord(c) % self.vocab_size for c in text] or [0]
        if truncation and max_length:
            ids = ids[:max_length]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    def save_pretrained(self, output) -> None:
        Path(output, "tokenizer_stub.json").write_text("{}")


def write_calib_jsonl(path: Path, texts: list[str]) -> Path:
    path.write_text("\n".join(json.dumps({"text": t}) for t in texts) + "\n")
    return path


class _TwoLinears(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, 8)
        self.first = nn.Linear(8, 6)
        self.second = nn.Linear(6, 4)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(self.embed(input_ids)))


def test_act_scales_are_per_input_channel_maxima(tmp_path) -> None:
    calib = write_calib_jsonl(tmp_path / "calib.jsonl", ["hello", "smoothquant"])
    model = _TwoLinears()

    scales = get_act_scales(
        model, _StubTokenizer(64), calib, num_samples=2, seq_len=8, progress_every=0
    )

    assert set(scales) == {"first", "second"}
    assert scales["first"].shape == (8,)
    assert scales["second"].shape == (6,)

    tokenizer = _StubTokenizer(64)
    expected = torch.zeros(8)
    with torch.no_grad():
        for text in ["hello", "smoothquant"]:
            ids = tokenizer(text, max_length=8, truncation=True).input_ids
            expected = torch.maximum(
                expected, model.embed(ids).view(-1, 8).abs().max(dim=0)[0]
            )
    torch.testing.assert_close(scales["first"], expected)


def test_hooks_are_removed_even_when_forward_fails(tmp_path) -> None:
    calib = write_calib_jsonl(tmp_path / "calib.jsonl", ["boom"])
    model = _TwoLinears()
    model.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        get_act_scales(model, _StubTokenizer(64), calib, num_samples=1, seq_len=8)

    assert not model.first._forward_hooks


def test_resolve_calib_dataset_prefers_explicit_path(tmp_path, monkeypatch) -> None:
    calib = write_calib_jsonl(tmp_path / "calib.jsonl", ["x"])
    monkeypatch.setenv("SMOOTHQUANT_PILEVAL_PATH", "/nonexistent.jsonl")
    assert resolve_calib_dataset(str(calib)) == calib


def test_resolve_calib_dataset_falls_back_to_awq_env(tmp_path, monkeypatch) -> None:
    calib = write_calib_jsonl(tmp_path / "val.jsonl", ["x"])
    monkeypatch.delenv("SMOOTHQUANT_PILEVAL_PATH", raising=False)
    monkeypatch.setenv("AWQ_PILEVAL_PATH", str(calib))
    assert resolve_calib_dataset("pileval") == calib


def test_resolve_calib_dataset_rejects_non_jsonl(tmp_path, monkeypatch) -> None:
    other = tmp_path / "calib.txt"
    other.write_text("x")
    monkeypatch.delenv("SMOOTHQUANT_PILEVAL_PATH", raising=False)
    monkeypatch.delenv("AWQ_PILEVAL_PATH", raising=False)
    with pytest.raises(ValueError, match=".jsonl"):
        resolve_calib_dataset(str(other))
    with pytest.raises(ValueError, match="SMOOTHQUANT_PILEVAL_PATH"):
        resolve_calib_dataset("pileval")

# === REEX_SMOOTHQUANT END ===
