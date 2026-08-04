from __future__ import annotations

from pathlib import Path

import torch

from llm_quant.gptq.sidecar import (
    SidecarShardReader,
    iter_sidecar_v3,
    write_sidecar_v3,
)

SOURCE_SIGNATURE = {"path": "/models/test"}
QUANTIZATION_CONFIG = {"algorithm": "gptq", "format": "Q4_0_64"}


def test_v3_sidecar_is_packed_only_and_sharded_by_layer(tmp_path: Path) -> None:
    tensors = {}
    for layer in (0, 1):
        name = f"model.layers.{layer}.self_attn.q_proj.weight"
        tensors[name] = {
            "codes": torch.zeros((2, 64), dtype=torch.int8),
            "scales": torch.ones((2, 1), dtype=torch.float16),
            "packed": torch.zeros((2, 34), dtype=torch.uint8),
        }

    manifest_path = write_sidecar_v3(
        tmp_path,
        tensors,
        source_model="/models/test",
        source_signature=SOURCE_SIGNATURE,
        quantization_config=QUANTIZATION_CONFIG,
    )
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
    assert manifest["version"] == 3
    assert len(manifest["shards"]) == 2

    loaded = dict(iter_sidecar_v3(manifest_path))
    assert set(loaded) == set(tensors)
    for entry in loaded.values():
        assert set(entry) == {"packed", "shape", "source_name"}
        assert entry["shape"] == [2, 64]
    reader = SidecarShardReader(manifest_path)
    assert len(reader) == 2
    for name in tensors:
        assert reader.get(name)["shape"] == [2, 64]
    assert reader.get("missing") is None


def test_v3_sidecar_resume_reuses_valid_shards(tmp_path: Path) -> None:
    name = "model.layers.0.self_attn.q_proj.weight"
    tensors = {
        name: {
            "packed": torch.zeros((1, 34), dtype=torch.uint8),
            "shape": [1, 64],
        }
    }
    write_sidecar_v3(
        tmp_path,
        tensors,
        source_model="source",
        source_signature=SOURCE_SIGNATURE,
        quantization_config=QUANTIZATION_CONFIG,
    )
    shard_path = tmp_path / "gptq_q4_0_64.layer000.pt"
    before = shard_path.stat().st_mtime_ns
    write_sidecar_v3(
        tmp_path,
        tensors,
        source_model="source",
        source_signature=SOURCE_SIGNATURE,
        quantization_config=QUANTIZATION_CONFIG,
        resume=True,
    )
    assert shard_path.stat().st_mtime_ns == before


def test_v3_sidecar_splits_fused_expert_gate_up(tmp_path: Path) -> None:
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    tensors = {
        name: {
            "codes": torch.zeros((2, 4, 64), dtype=torch.int8),
            "scales": torch.ones((2, 4, 1), dtype=torch.float16),
            "packed": torch.zeros((2, 4, 34), dtype=torch.uint8),
        }
    }
    path = write_sidecar_v3(
        tmp_path,
        tensors,
        source_model="source",
        source_signature=SOURCE_SIGNATURE,
        quantization_config=QUANTIZATION_CONFIG,
    )
    loaded = dict(iter_sidecar_v3(path))
    assert set(loaded) == {
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
    }
    for key, entry in loaded.items():
        assert entry["gguf_name"] == key
        assert entry["source_name"] == name
        assert entry["shape"] == [2, 2, 64]


def test_v3_q8_sidecar_uses_66_byte_blocks(tmp_path: Path) -> None:
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    tensors = {
        name: {
            "codes": torch.zeros((2, 4, 64), dtype=torch.int8),
            "scales": torch.ones((2, 4, 1), dtype=torch.float16),
            "packed": torch.zeros((2, 4, 66), dtype=torch.uint8),
        }
    }
    path = write_sidecar_v3(
        tmp_path,
        tensors,
        source_model="source",
        source_signature=SOURCE_SIGNATURE,
        quantization_config={"algorithm": "gptq", "format": "Q8_0_64"},
        format_name="Q8_0_64",
    )
    assert path.name == "gptq_q8_0_64.pt"
    shard = tmp_path / "gptq_q8_0_64.layer000.pt"
    before = shard.stat().st_mtime_ns
    write_sidecar_v3(
        tmp_path,
        tensors,
        source_model="source",
        source_signature=SOURCE_SIGNATURE,
        quantization_config={"algorithm": "gptq", "format": "Q8_0_64"},
        format_name="Q8_0_64",
        resume=True,
    )
    assert shard.stat().st_mtime_ns == before
    reader = SidecarShardReader(path)
    assert reader.format_name == "Q8_0_64"
    for entry in dict(iter_sidecar_v3(path)).values():
        assert entry["shape"] == [2, 2, 64]
        assert entry["packed"].shape == (2, 2, 66)


def test_v3_resume_rejects_source_or_config_mismatch(tmp_path: Path) -> None:
    name = "model.layers.0.self_attn.q_proj.weight"
    tensors = {
        name: {
            "packed": torch.zeros((1, 34), dtype=torch.uint8),
            "shape": [1, 64],
        }
    }
    write_sidecar_v3(
        tmp_path,
        tensors,
        source_model="source",
        source_signature=SOURCE_SIGNATURE,
        quantization_config=QUANTIZATION_CONFIG,
    )
    for source_signature, quantization_config, message in (
        ({"path": "other"}, QUANTIZATION_CONFIG, "source mismatch"),
        (SOURCE_SIGNATURE, {"algorithm": "other"}, "configuration mismatch"),
    ):
        try:
            write_sidecar_v3(
                tmp_path,
                tensors,
                source_model="source",
                source_signature=source_signature,
                quantization_config=quantization_config,
                resume=True,
            )
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("mismatched resume sidecar was accepted")


def test_resume_rejects_legacy_manifest_without_shards(tmp_path: Path) -> None:
    torch.save(
        {"format": "Q4_0_64", "version": 1, "tensors": {}},
        tmp_path / "gptq_q4_0_64.pt",
    )
    try:
        write_sidecar_v3(
            tmp_path,
            {},
            source_model="source",
            source_signature=SOURCE_SIGNATURE,
            quantization_config=QUANTIZATION_CONFIG,
            resume=True,
        )
    except ValueError as exc:
        assert "expected version 3" in str(exc)
    else:
        raise AssertionError("legacy resume manifest was accepted")
