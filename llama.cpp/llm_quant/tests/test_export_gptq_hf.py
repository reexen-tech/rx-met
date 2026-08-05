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
            "packed": torch.zeros((2, 66), dtype=torch.uint8),
            "shape": [2, 64],
        }
    }
    def fake_run(*_args, sidecar_writer, **_kwargs):
        stats = {"layer": 0, "linears": {}, "routed_experts": {}}
        sidecar_writer.write_layer(0, tensors, stats)
        return GPTQRunResult(
            tensor_data={},
            layer_stats=[stats],
            calibration_sequences=2,
            fixed_point_ok=False,
        )

    monkeypatch.setattr(
        export_gptq_hf,
        "run_gptq",
        fake_run,
    )

    exported = export_gptq_hf.export_gptq_hf(
        str(model_path), str(output_path), device="cpu", bits=8
    )
    assert exported == str(output_path.resolve())
    sidecar = torch.load(
        output_path / "gptq_q8_0_64.pt", map_location="cpu", weights_only=True
    )
    assert sidecar["format"] == "Q8_0_64"
    assert sidecar["version"] == 3
    assert len(sidecar["shards"]) == 1
    metadata = json.loads((output_path / "gptq_export_meta.json").read_text())
    assert metadata["execution"] == "W8A8"
    assert metadata["group_size"] == 64
    assert metadata["gguf_export_path"] == "direct_q8_0_64_sidecar"
    assert metadata["model_family"] == "source"


def test_qwen35_export_uses_bf16_model_and_sidecar_v3(
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
    tensors = {
        name: {
            "codes": torch.zeros((2, 64), dtype=torch.int8),
            "scales": torch.zeros((2, 1), dtype=torch.float16),
            "packed": torch.zeros((2, 34), dtype=torch.uint8),
            "shape": [2, 64],
        }
    }

    def fake_run(*_args, sidecar_writer, **_kwargs):
        stats = {"layer": 0, "linears": {}, "routed_experts": {}}
        sidecar_writer.write_layer(0, tensors, stats)
        return GPTQRunResult(
            tensor_data={},
            layer_stats=[stats],
            calibration_sequences=2,
            fixed_point_ok=True,
        )

    monkeypatch.setattr(export_gptq_hf, "run_gptq", fake_run)

    export_gptq_hf.export_gptq_hf(
        str(model_path), str(output_path), device="cpu"
    )
    assert load_kwargs["dtype"] == torch.bfloat16
    manifest = torch.load(
        output_path / "gptq_q4_0_64.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert manifest["version"] == 3
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
        "bits": 4,
        "format_name": None,
        "expert_hessian_weighting": "route_squared",
    }
