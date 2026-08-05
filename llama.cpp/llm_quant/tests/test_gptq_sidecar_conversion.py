from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import convert_hf_to_gguf as converter
import validate_gptq_gguf as validator
from llm_quant.gptq.formats import Q4_0_64, Q8_0_64
from llm_quant.gptq.gguf_adapter import GPTQGGUFAdapter
from llm_quant.gptq.q4_0_64 import pack_q4_0_64
from llm_quant.gptq.q8_0_64 import pack_q8_0_64


def _qwen35_model() -> converter.Qwen3_5MoeTextModel:
    model = converter.Qwen3_5MoeTextModel.__new__(converter.Qwen3_5MoeTextModel)
    model.hparams = {
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 128,
    }
    model._is_nvfp4 = False
    model.fuse_gate_up_exps = False
    model.map_tensor_name = lambda name: name
    return model


def test_lookup_distinguishes_split_gate_up_entries() -> None:
    source = "model.layers.0.mlp.experts.gate_up_proj"
    gate_name = "blk.0.ffn_gate_exps.weight"
    up_name = "blk.0.ffn_up_exps.weight"
    adapter = GPTQGGUFAdapter(
        {
            "gate": {"source_name": source, "gguf_name": gate_name},
            "up": {"source_name": source, "gguf_name": up_name},
        },
        Q4_0_64,
        {},
    )

    gate = adapter._lookup(source, gate_name)
    assert gate is not None and gate[0] == "gate"
    up = adapter._lookup(source, up_name)
    assert up is not None and up[0] == "up"


def test_qwen35_moe_experts_split_dim_minus_two_without_down_transpose() -> None:
    model = _qwen35_model()
    gate_up = torch.arange(2 * 6 * 64).reshape(2, 6, 64)
    outputs = list(model.modify_tensors(
        gate_up, "model.layers.0.mlp.experts.gate_up_proj", 0,
    ))
    assert [tensor.shape for _, tensor in outputs] == [(2, 3, 64), (2, 3, 64)]
    assert torch.equal(outputs[0][1], gate_up[:, :3, :])
    assert torch.equal(outputs[1][1], gate_up[:, 3:, :])

    down = torch.arange(2 * 64 * 3).reshape(2, 64, 3)
    [(_, converted_down)] = list(model.modify_tensors(
        down, "model.layers.0.mlp.experts.down_proj", 0,
    ))
    assert torch.equal(converted_down, down)


def test_linear_attention_sidecar_matches_v_head_reorder() -> None:
    model = _qwen35_model()
    adapter = GPTQGGUFAdapter({}, Q4_0_64, model.hparams)
    source = "model.layers.0.linear_attn.out_proj.weight"
    codes = (torch.arange(64 * 512).reshape(64, 512) % 16 - 8).to(torch.int8)
    scales = torch.arange(64 * 8, dtype=torch.float16).reshape(64, 8)

    converted_codes, converted_scales = adapter.transform(
        codes, scales, source,
    )
    expected_codes = model._reorder_v_heads(codes, 1, 2, 2, 128)
    expected_scales = model._reorder_v_heads(scales, 1, 2, 2, 2)
    assert torch.equal(converted_codes, expected_codes)
    assert torch.equal(converted_scales, expected_scales)
    assert pack_q4_0_64(converted_codes, converted_scales).shape == (64, 8 * 34)


def test_validator_accepts_gguf_names_and_3d_packed(
    tmp_path: Path, monkeypatch,
) -> None:
    codes = (torch.arange(2 * 3 * 64).reshape(2, 3, 64) % 16 - 8).to(torch.int8)
    scales = torch.ones((2, 3, 1), dtype=torch.float16)
    packed = pack_q4_0_64(codes, scales)
    gguf_name = "blk.0.ffn_gate_exps.weight"
    tensors = {
        "gate": {
            "source_name": "model.layers.0.mlp.experts.gate_up_proj",
            "gguf_name": gguf_name,
            "codes": codes,
            "scales": scales,
            "packed": packed,
        },
    }
    gguf_tensor = SimpleNamespace(
        name=gguf_name,
        tensor_type=validator.gguf.GGMLQuantizationType.Q4_0_64,
        data=np.asarray(packed),
    )
    monkeypatch.setattr(
        validator.gguf,
        "GGUFReader",
        lambda _: SimpleNamespace(tensors=[gguf_tensor]),
    )

    validator.validate_gguf(tensors, tmp_path / "test.gguf")


def test_validator_accepts_q8_3d_packed(
    tmp_path: Path, monkeypatch,
) -> None:
    codes = (torch.arange(2 * 3 * 64).reshape(2, 3, 64) % 255 - 127).to(
        torch.int8
    )
    scales = torch.ones((2, 3, 1), dtype=torch.float16)
    packed = pack_q8_0_64(codes, scales)
    gguf_name = "blk.0.ffn_gate_exps.weight"
    tensors = {
        "gate": {
            "source_name": "model.layers.0.mlp.experts.gate_up_proj",
            "gguf_name": gguf_name,
            "codes": codes,
            "scales": scales,
            "packed": packed,
        },
    }
    gguf_tensor = SimpleNamespace(
        name=gguf_name,
        tensor_type=validator.gguf.GGMLQuantizationType.Q8_0_64,
        data=np.asarray(packed),
    )
    monkeypatch.setattr(
        validator.gguf,
        "GGUFReader",
        lambda _: SimpleNamespace(tensors=[gguf_tensor]),
    )
    validator.validate_gguf(
        tensors, tmp_path / "test-q8.gguf", block_format=Q8_0_64
    )


def test_validator_keeps_qwen2_v1_name_mapping() -> None:
    entry = {
        "codes": torch.zeros((2, 64), dtype=torch.int8),
        "scales": torch.ones((2, 1), dtype=torch.float16),
    }
    assert validator.sidecar_gguf_name(
        "model.layers.3.self_attn.q_proj.weight", entry,
    ) == "blk.3.attn_q.weight"


def test_layout_hparams_prefers_qwen35_text_config() -> None:
    from llm_quant.gptq.gguf_adapter import layout_hparams

    root = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
        },
    }
    assert layout_hparams(root)["linear_num_key_heads"] == 16
    assert layout_hparams({"model_type": "qwen2"})["model_type"] == "qwen2"
