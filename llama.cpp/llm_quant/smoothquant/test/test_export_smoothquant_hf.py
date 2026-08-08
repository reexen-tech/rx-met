# === REEX_SMOOTHQUANT BEGIN: end-to-end export tests on a tiny Qwen2 checkpoint ===
"""End-to-end tests for exporting a tiny smoothed Qwen2 checkpoint."""

from __future__ import annotations

import json
from types import SimpleNamespace

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


def test_cli_exposes_smoke_formal_and_cache_resume(capsys) -> None:
    args = export_smoothquant_hf.parse_args(
        [
            "--model_path",
            "model",
            "--output_dir",
            "output",
            "--calibration-mode",
            "smoke",
            "--resume_act_scales",
        ]
    )
    assert args.calibration_mode == "smoke"
    assert args.resume_act_scales and not args.reuse_act_scales

    with pytest.raises(SystemExit) as exit_info:
        export_smoothquant_hf.parse_args(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--calibration-mode" in help_text
    assert "--resume_act_scales" in help_text
    assert "--reuse_act_scales" in help_text


def test_cli_rejects_resume_and_reuse_together() -> None:
    with pytest.raises(SystemExit):
        export_smoothquant_hf.parse_args(
            [
                "--model_path",
                "model",
                "--output_dir",
                "output",
                "--resume_act_scales",
                "--reuse_act_scales",
            ]
        )


def test_qwen35_export_does_not_restore_source_checkpoint_keys(
    tmp_path, monkeypatch
) -> None:
    text_config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        dtype=torch.bfloat16,
        use_cache=True,
    )
    outer_config = SimpleNamespace(
        model_type="qwen3_5_moe",
        text_config=text_config,
    )

    class SpyModel:
        config = text_config
        hf_device_map = {"": "cpu"}

        def __init__(self):
            self.save_kwargs = None

        def eval(self):
            return self

        def save_pretrained(self, _output_dir, **kwargs):
            self.save_kwargs = kwargs

    class SpyTokenizer:
        def save_pretrained(self, _output_dir):
            return None

    model = SpyModel()
    result = SimpleNamespace(
        alpha=0.8,
        n_samples=32,
        seqlen=128,
        calib_data="calib.jsonl",
        n_layers_smoothed=40,
        n_groups_smoothed=40,
        act_scales_cache="act_scales.pt",
        act_scales_from_cache=True,
        calibration_resumed=False,
        meta={},
        coverage={},
        group_stats=[],
        smoothing_policy={
            "scope": "moe",
            "scale_policy": "power_of_two",
            "norm_materialization": "fp32_offset",
        },
    )
    monkeypatch.setattr(
        export_smoothquant_hf.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: outer_config,
    )
    monkeypatch.setattr(
        export_smoothquant_hf.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: SpyTokenizer(),
    )
    monkeypatch.setattr(
        export_smoothquant_hf,
        "load_qwen35_text_model",
        lambda *_args, **_kwargs: (model, {"mapping_sha256": "test"}),
    )
    monkeypatch.setattr(
        export_smoothquant_hf,
        "run_smoothquant",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        export_smoothquant_hf,
        "validate_qwen35_smoothed_parameter_dtypes",
        lambda *_args, **_kwargs: {"fp32_parameter_count": 40},
    )

    export_smoothquant_hf.export_smoothquant_hf(
        "source", str(tmp_path / "output"), device="auto"
    )

    assert model.save_kwargs == {
        "safe_serialization": True,
        "save_original_format": False,
    }

# === REEX_SMOOTHQUANT END ===
