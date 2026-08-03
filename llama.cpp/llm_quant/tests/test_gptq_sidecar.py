from __future__ import annotations

from pathlib import Path

import torch

from llm_quant.gptq.sidecar import (
    SidecarShardReader,
    iter_sidecar_v2,
    write_sidecar_v2,
)


def test_v2_sidecar_is_packed_only_and_sharded_by_layer(tmp_path: Path) -> None:
    tensors = {}
    for layer in (0, 1):
        name = f"model.layers.{layer}.self_attn.q_proj.weight"
        tensors[name] = {
            "codes": torch.zeros((2, 64), dtype=torch.int8),
            "scales": torch.ones((2, 1), dtype=torch.float16),
            "packed": torch.zeros((2, 34), dtype=torch.uint8),
        }

    manifest_path = write_sidecar_v2(
        tmp_path,
        tensors,
        source_model="/models/test",
    )
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
    assert manifest["version"] == 2
    assert len(manifest["shards"]) == 2

    loaded = dict(iter_sidecar_v2(manifest_path))
    assert set(loaded) == set(tensors)
    for entry in loaded.values():
        assert set(entry) == {"packed", "shape"}
        assert entry["shape"] == [2, 64]
    reader = SidecarShardReader(manifest_path)
    assert len(reader) == 2
    for name in tensors:
        assert reader.get(name)["shape"] == [2, 64]
    assert reader.get("missing") is None


def test_v2_sidecar_resume_reuses_valid_shards(tmp_path: Path) -> None:
    name = "model.layers.0.self_attn.q_proj.weight"
    tensors = {
        name: {
            "packed": torch.zeros((1, 34), dtype=torch.uint8),
            "shape": [1, 64],
        }
    }
    write_sidecar_v2(tmp_path, tensors, source_model="source")
    shard_path = tmp_path / "gptq_q4_0_64.layer000.pt"
    before = shard_path.stat().st_mtime_ns
    write_sidecar_v2(tmp_path, tensors, source_model="source", resume=True)
    assert shard_path.stat().st_mtime_ns == before


def test_v2_sidecar_splits_fused_expert_gate_up(tmp_path: Path) -> None:
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    tensors = {
        name: {
            "codes": torch.zeros((2, 4, 64), dtype=torch.int8),
            "scales": torch.ones((2, 4, 1), dtype=torch.float16),
            "packed": torch.zeros((2, 4, 34), dtype=torch.uint8),
        }
    }
    path = write_sidecar_v2(tmp_path, tensors, source_model="source")
    loaded = dict(iter_sidecar_v2(path))
    assert set(loaded) == {
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
    }
    for key, entry in loaded.items():
        assert entry["gguf_name"] == key
        assert entry["source_name"] == name
        assert entry["shape"] == [2, 2, 64]
