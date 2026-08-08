"""Strict text-only Qwen3.5 key mapping and save/reload integration tests."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from llm_quant.smoothquant.loading import (
    build_qwen35_text_key_mapping,
    checkpoint_keys,
    load_qwen35_text_model,
    validate_loading_info,
)


def _tiny_text_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        use_cache=False,
        mtp_num_hidden_layers=1,
    )


def _multimodal_style_checkpoint(tmp_path, *, sharded: bool = False):
    config = _tiny_text_config()
    torch.manual_seed(17)
    model = Qwen3_5MoeForCausalLM(config).eval()
    state = {}
    for key, value in model.state_dict().items():
        source_key = (
            "model.language_model." + key[len("model.") :]
            if key.startswith("model.")
            else key
        )
        state[source_key] = value.detach().contiguous()
    state["model.visual.fake.weight"] = torch.ones(1)
    state["mtp.fake.weight"] = torch.ones(1)
    source = tmp_path / "source"
    source.mkdir()
    if sharded:
        items = list(state.items())
        split_at = len(items) // 2
        weight_map = {}
        for shard_index, shard_items in enumerate(
            (items[:split_at], items[split_at:]), start=1
        ):
            filename = f"model-{shard_index:05d}-of-00002.safetensors"
            save_file(
                dict(shard_items),
                source / filename,
                metadata={"format": "pt"},
            )
            weight_map.update({key: filename for key, _ in shard_items})
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
    else:
        save_file(state, source / "model.safetensors", metadata={"format": "pt"})
    (source / "generation_config.json").write_text("{}\n")
    return source, config, model


def test_text_mapping_excludes_vision_and_mtp(tmp_path) -> None:
    source, _, original = _multimodal_style_checkpoint(tmp_path)
    mapping, summary = build_qwen35_text_key_mapping(source)

    assert len(mapping) == len(original.state_dict())
    assert mapping["model.language_model.layers.0.mlp.experts.gate_up_proj"] == (
        "model.layers.0.mlp.experts.gate_up_proj"
    )
    assert mapping["lm_head.weight"] == "lm_head.weight"
    assert not any(key.startswith(("model.visual.", "mtp.")) for key in mapping)
    assert summary["excluded_vision_keys"] == 1
    assert summary["excluded_mtp_keys"] == 1


def test_text_only_from_pretrained_mapping_and_save_reload(tmp_path) -> None:
    source, config, original = _multimodal_style_checkpoint(tmp_path)
    loaded, details = load_qwen35_text_model(
        source,
        config,
        dtype=torch.float32,
        device="cpu",
        require_production_shape=False,
    )

    assert type(loaded).__name__ == "Qwen3_5MoeForCausalLM"
    assert details["vision_loaded"] is False and details["mtp_loaded"] is False
    assert not details["loading_info"]["missing_keys"]
    assert not details["loading_info"]["unexpected_keys"]
    for key, expected in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected)

    output = tmp_path / "text-only"
    loaded.save_pretrained(
        output, safe_serialization=True, save_original_format=False
    )
    assert set(checkpoint_keys(output)) == set(loaded.state_dict())
    reloaded, loading_info = Qwen3_5MoeForCausalLM.from_pretrained(
        output,
        dtype=torch.float32,
        local_files_only=True,
        output_loading_info=True,
    )
    validate_loading_info(loading_info)
    assert type(reloaded).__name__ == "Qwen3_5MoeForCausalLM"
    assert set(reloaded.state_dict()) == set(loaded.state_dict())
    for key, expected in loaded.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], expected)


def test_sharded_text_mapping_device_map_save_reload(tmp_path) -> None:
    source, config, _ = _multimodal_style_checkpoint(tmp_path, sharded=True)
    loaded, _ = load_qwen35_text_model(
        source,
        config,
        dtype=torch.float32,
        device="auto",
        require_production_shape=False,
    )

    output = tmp_path / "text-only"
    loaded.save_pretrained(
        output,
        safe_serialization=True,
        save_original_format=False,
        max_shard_size="1KB",
    )
    assert (output / "model.safetensors.index.json").is_file()
    assert set(checkpoint_keys(output)) == set(loaded.state_dict())

    reloaded, loading_info = Qwen3_5MoeForCausalLM.from_pretrained(
        output,
        dtype=torch.float32,
        local_files_only=True,
        output_loading_info=True,
    )
    validate_loading_info(loading_info)
    for key, expected in loaded.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], expected.cpu())


def test_text_missing_key_fails_strict_validation() -> None:
    with pytest.raises(RuntimeError, match="loading was not exact"):
        validate_loading_info(
            {
                "missing_keys": ["model.layers.0.input_layernorm.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        )


def test_index_mapping_requires_classified_keys(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.embed_tokens.weight": "one.safetensors",
                    "unknown.weight": "one.safetensors",
                }
            }
        )
    )
    with pytest.raises(ValueError, match="unclassified"):
        build_qwen35_text_key_mapping(source)
