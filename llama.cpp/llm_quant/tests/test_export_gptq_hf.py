from __future__ import annotations

import json
from pathlib import Path

import torch

import export_gptq_hf
from llm_quant.gptq.layer_runner import GPTQRunResult


class _FakeModel:
    def eval(self):
        return self

    def cpu(self):
        return self

    def save_pretrained(self, output: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization
        (output / "model.safetensors").write_bytes(b"fake")
        (output / "config.json").write_text('{"model_type":"qwen2"}')


class _FakeTokenizer:
    def save_pretrained(self, output: Path) -> None:
        (output / "tokenizer.json").write_text("{}")


class _FakeConfig:
    model_type = "qwen2"


class _FakeQwen35Config:
    model_type = "qwen3_5_moe"


def test_export_writes_sidecar_and_metadata(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "source"
    output_path = tmp_path / "output"
    model_path.mkdir()
    monkeypatch.setattr(
        export_gptq_hf.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeConfig(),
    )
    monkeypatch.setattr(
        export_gptq_hf.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeModel(),
    )
    monkeypatch.setattr(
        export_gptq_hf.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTokenizer(),
    )
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": {
            "codes": torch.zeros((2, 64), dtype=torch.int8),
            "scales": torch.zeros((2, 1), dtype=torch.float16),
            "packed": torch.zeros((2, 34), dtype=torch.uint8),
        }
    }
    monkeypatch.setattr(
        export_gptq_hf,
        "run_gptq",
        lambda *_args, **_kwargs: GPTQRunResult(
            tensor_data=tensors,
            layer_stats=[{"layer": 0}],
            calibration_sequences=2,
            fixed_point_ok=False,
        ),
    )

    exported = export_gptq_hf.export_gptq_hf(
        str(model_path), str(output_path), device="cpu"
    )
    assert exported == str(output_path.resolve())
    sidecar = torch.load(
        output_path / "gptq_q4_0_64.pt", map_location="cpu", weights_only=True
    )
    assert sidecar["format"] == "Q4_0_64"
    assert set(sidecar["tensors"]) == set(tensors)
    metadata = json.loads((output_path / "gptq_export_meta.json").read_text())
    assert metadata["execution"] == "W4A8"
    assert metadata["group_size"] == 64
    assert metadata["gguf_export_path"] == "direct_q4_0_64_sidecar"


def test_qwen35_export_uses_bf16_model_and_sidecar_v2(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / "source"
    output_path = tmp_path / "output"
    model_path.mkdir()
    monkeypatch.setattr(
        export_gptq_hf.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeQwen35Config(),
    )
    monkeypatch.setattr(
        export_gptq_hf.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTokenizer(),
    )
    load_kwargs = {}

    def load_model(*_args, **kwargs):
        load_kwargs.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(
        export_gptq_hf.AutoModelForImageTextToText,
        "from_pretrained",
        load_model,
    )
    name = "model.language_model.layers.0.self_attn.q_proj.weight"
    monkeypatch.setattr(
        export_gptq_hf,
        "run_gptq",
        lambda *_args, **_kwargs: GPTQRunResult(
            tensor_data={
                name: {
                    "codes": torch.zeros((2, 64), dtype=torch.int8),
                    "scales": torch.zeros((2, 1), dtype=torch.float16),
                    "packed": torch.zeros((2, 34), dtype=torch.uint8),
                }
            },
            layer_stats=[{"layer": 0}],
            calibration_sequences=2,
            fixed_point_ok=True,
        ),
    )

    export_gptq_hf.export_gptq_hf(
        str(model_path), str(output_path), device="cpu"
    )
    assert load_kwargs["dtype"] == torch.bfloat16
    manifest = torch.load(
        output_path / "gptq_q4_0_64.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert manifest["version"] == 2
    assert manifest["layout"] == "packed-only-sharded"
    metadata = json.loads((output_path / "gptq_export_meta.json").read_text())
    assert metadata["model_family"] == "Qwen3.5-35B-A3B"
    assert metadata["fake_quant_dtype"] == "bfloat16"


def test_cli_only_exposes_paths_and_device() -> None:
    args = export_gptq_hf.parse_args(
        ["--model_path", "model", "--output_dir", "output"]
    )
    assert vars(args) == {
        "model_path": "model",
        "output_dir": "output",
        "device": "cuda",
        "resume": False,
        "max_layers": None,
    }
