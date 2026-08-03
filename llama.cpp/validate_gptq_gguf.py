#!/usr/bin/env python3
"""Validate exact GPTQ Q4_0_64 sidecar bytes in a converted GGUF."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402

from llm_quant.gptq.q4_0_64 import pack_q4_0_64, unpack_q4_0_64
from llm_quant.gptq.sidecar import SidecarShardReader


_QWEN2_NAMES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
_QWEN35_NAMES = {
    **_QWEN2_NAMES,
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.out_proj": "ssm_out",
    "mlp.shared_expert.gate_proj": "ffn_gate_shexp",
    "mlp.shared_expert.up_proj": "ffn_up_shexp",
    "mlp.shared_expert.down_proj": "ffn_down_shexp",
    "mlp.experts.down_proj": "ffn_down_exps",
}


def hf_to_gguf_name(name: str) -> str:
    is_qwen35 = name.startswith("model.language_model.")
    name = name.replace("model.language_model.", "model.", 1)
    if not name.endswith(".weight"):
        name = f"{name}.weight"
    match = re.fullmatch(
        r"model\.layers\.(\d+)\.(.+)\.weight",
        name,
    )
    names = _QWEN35_NAMES if is_qwen35 else _QWEN2_NAMES
    if match is None or match.group(2) not in names:
        raise ValueError(f"unsupported GPTQ tensor name: {name}")
    return f"blk.{match.group(1)}.{names[match.group(2)]}.weight"


def sidecar_gguf_name(key: str, entry: dict[str, object]) -> str:
    gguf_name = entry.get("gguf_name")
    if gguf_name is not None:
        if not isinstance(gguf_name, str):
            raise ValueError(f"invalid gguf_name for {key}")
        return gguf_name
    if key.startswith("blk."):
        return key
    source_name = entry.get("source_name", key)
    if not isinstance(source_name, str):
        raise ValueError(f"invalid source_name for {key}")
    return hf_to_gguf_name(source_name)


def validate_sidecar(sidecar_path: Path):
    payload = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "Q4_0_64"
    ):
        raise ValueError("not a Q4_0_64 GPTQ sidecar")
    if payload.get("version", 1) == 2:
        return SidecarShardReader(sidecar_path)
    if payload.get("version", 1) != 1:
        raise ValueError("unsupported Q4_0_64 GPTQ sidecar version")
    tensors = payload.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("sidecar contains no tensors")
    for name, entry in tensors.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("invalid sidecar tensor entry")
        sidecar_gguf_name(name, entry)
        repacked = pack_q4_0_64(entry["codes"], entry["scales"])
        if not torch.equal(repacked, entry["packed"]):
            raise ValueError(f"packed bytes do not match codes/scales for {name}")
    return tensors


def validate_gguf(tensors, gguf_path: Path) -> None:
    reader = gguf.GGUFReader(gguf_path)
    gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}
    transformer = None
    if isinstance(tensors, SidecarShardReader) and isinstance(
        tensors.source_model, str
    ):
        config_path = Path(tensors.source_model) / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("model_type") == "qwen3_5_moe":
                from convert_hf_to_gguf import Qwen3_5MoeTextModel

                transformer = Qwen3_5MoeTextModel.__new__(Qwen3_5MoeTextModel)
                transformer.hparams = config["text_config"]
    for sidecar_name in tensors:
        entry = tensors[sidecar_name]
        gguf_name = sidecar_gguf_name(sidecar_name, entry)
        if gguf_name not in gguf_tensors:
            raise ValueError(f"missing GGUF tensor {gguf_name}")
        tensor = gguf_tensors[gguf_name]
        if tensor.tensor_type != gguf.GGMLQuantizationType.Q4_0_64:
            raise ValueError(
                f"{gguf_name} has type {tensor.tensor_type.name}, expected Q4_0_64"
            )
        expected_packed = entry["packed"]
        source_name = entry.get("source_name", sidecar_name)
        if transformer is not None and "linear_attn." in source_name:
            codes = entry.get("codes")
            scales = entry.get("scales")
            if not isinstance(codes, torch.Tensor) or not isinstance(
                scales, torch.Tensor
            ):
                codes, scales = unpack_q4_0_64(
                    expected_packed,
                    logical_size=int(entry["shape"][-1]),
                )
            codes, scales = transformer._transform_gptq_sidecar(
                codes,
                scales,
                source_name,
                gguf_name,
                None,
            )
            expected_packed = pack_q4_0_64(codes, scales)
        expected = expected_packed.cpu().numpy().reshape(-1)
        actual = np.asarray(tensor.data).view(np.uint8).reshape(-1)
        if actual.size != expected.size:
            raise ValueError(
                f"{gguf_name} byte count {actual.size} does not match sidecar {expected.size}"
            )
        if not np.array_equal(actual, expected):
            differing = int(np.count_nonzero(actual != expected))
            raise ValueError(
                f"{gguf_name} differs from GPTQ sidecar in {differing} bytes"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tensors = validate_sidecar(args.sidecar)
    validate_gguf(tensors, args.gguf)
    print(f"Validated {len(tensors)} GPTQ Q4_0_64 tensors byte-for-byte.")


if __name__ == "__main__":
    main()
