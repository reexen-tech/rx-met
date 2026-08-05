# === REEX_SMOOTHQUANT BEGIN: end-to-end export tests on a tiny Qwen2 checkpoint ===
"""End-to-end tests for exporting a tiny smoothed Qwen2 checkpoint."""

from __future__ import annotations

import json

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from llm_quant.smoothquant.scripts import export_smoothquant_hf
from .test_smoothquant_act_scales import _StubTokenizer, write_calib_jsonl


@pytest.fixture
def tiny_checkpoint(tmp_path, monkeypatch):
    """A saved tiny Qwen2 checkpoint plus a calibration file, tokenizer stubbed."""
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        use_cache=False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).eval()
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True)
    monkeypatch.setattr(
        export_smoothquant_hf.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _StubTokenizer(config.vocab_size),
    )
    calib = write_calib_jsonl(tmp_path / "calib.jsonl", ["hello world", "smoothquant"])
    return source, calib, model


def _export(tiny_checkpoint, tmp_path, **overrides):
    source, calib, _ = tiny_checkpoint
    kwargs = {
        "calib_data": str(calib),
        "n_samples": 2,
        "seqlen": 16,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return export_smoothquant_hf.export_smoothquant_hf(
        str(source), str(tmp_path / "output"), **kwargs
    )


def test_export_smooths_weights_and_writes_metadata(tiny_checkpoint, tmp_path) -> None:
    _, calib, original = tiny_checkpoint
    cache = tmp_path / "act_scales.pt"

    output = _export(tiny_checkpoint, tmp_path, act_scales_cache=str(cache))

    meta = json.loads((tmp_path / "output" / "smoothquant_export_meta.json").read_text())
    assert meta["alpha"] == 0.85  # qwen2 default from defaults.py
    assert meta["dtype"] == "float32"
    assert (meta["n_layers_smoothed"], meta["n_groups_smoothed"]) == (2, 4)
    assert meta["calib_data"] == str(calib.resolve())
    assert cache.is_file()

    smoothed = AutoModelForCausalLM.from_pretrained(output)
    before = original.model.layers[0]
    after = smoothed.model.layers[0]
    assert not torch.allclose(
        after.input_layernorm.weight, before.input_layernorm.weight
    )
    assert not torch.allclose(after.self_attn.q_proj.weight, before.self_attn.q_proj.weight)
    # untouched by the transform
    torch.testing.assert_close(
        after.self_attn.o_proj.weight, before.self_attn.o_proj.weight
    )


def test_export_reuses_cached_act_scales(tiny_checkpoint, tmp_path) -> None:
    cache = tmp_path / "act_scales.pt"
    _export(tiny_checkpoint, tmp_path, act_scales_cache=str(cache))

    _export(tiny_checkpoint, tmp_path, act_scales_cache=str(cache), reuse_act_scales=True)

    meta = json.loads((tmp_path / "output" / "smoothquant_export_meta.json").read_text())
    assert meta["act_scales_from_cache"] is True


def test_export_rejects_cache_from_a_different_run(tiny_checkpoint, tmp_path) -> None:
    cache = tmp_path / "act_scales.pt"
    _export(tiny_checkpoint, tmp_path, act_scales_cache=str(cache))

    with pytest.raises(ValueError, match="does not match this run"):
        _export(
            tiny_checkpoint,
            tmp_path,
            act_scales_cache=str(cache),
            reuse_act_scales=True,
            n_samples=1,
        )


def test_dtype_defaults_to_the_checkpoint_and_can_be_overridden() -> None:
    config = Qwen2Config(torch_dtype="bfloat16")
    assert export_smoothquant_hf.resolve_dtype(config, None) == torch.bfloat16
    assert export_smoothquant_hf.resolve_dtype(config, "float16") == torch.float16


def test_cli_defaults(tmp_path) -> None:
    args = export_smoothquant_hf.parse_args(
        ["--model_path", "model", "--output_dir", "output"]
    )
    assert args.alpha is None and args.dtype is None
    assert (args.n_samples, args.seqlen, args.device) == (512, 512, "auto")
    assert args.calib_data == "pileval"

# === REEX_SMOOTHQUANT END ===
