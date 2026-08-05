#!/usr/bin/env python3
"""Validate exact GPTQ Q64 sidecar bytes in a converted GGUF."""

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

from llm_quant.gptq.formats import Q4_0_64, Q64Format
from llm_quant.gptq.gguf_adapter import GPTQGGUFAdapter, layout_hparams
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
    if not isinstance(payload, dict):
        raise ValueError("not a GPTQ sidecar")
    if payload.get("version") == 3:
        return SidecarShardReader(sidecar_path)
    raise ValueError(
        f"unsupported GPTQ sidecar version "
        f"{payload.get('version')!r}; expected version 3"
    )


def validate_gguf(
    tensors,
    gguf_path: Path,
    block_format: Q64Format | None = None,
) -> None:
    if block_format is None:
        block_format = (
            tensors.block_format
            if isinstance(tensors, SidecarShardReader)
            else Q4_0_64
        )
    expected_qtype = {
        "Q4_0_64": gguf.GGMLQuantizationType.Q4_0_64,
        "Q4_1_64": gguf.GGMLQuantizationType.Q4_1_64,
        "Q8_0_64": gguf.GGMLQuantizationType.Q8_0_64,
    }[block_format.name]
    reader = gguf.GGUFReader(gguf_path)
    gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}
    transformer: GPTQGGUFAdapter | None = None
    if isinstance(tensors, SidecarShardReader) and isinstance(
        tensors.source_model, str
    ):
        config_path = Path(tensors.source_model) / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if str(config.get("model_type", "")).startswith("qwen3_5"):
                transformer = GPTQGGUFAdapter(
                    tensors, block_format, layout_hparams(config)
                )
    for sidecar_name in tensors:
        entry = tensors[sidecar_name]
        gguf_name = sidecar_gguf_name(sidecar_name, entry)
        if gguf_name not in gguf_tensors:
            raise ValueError(f"missing GGUF tensor {gguf_name}")
        tensor = gguf_tensors[gguf_name]
        if tensor.tensor_type != expected_qtype:
            raise ValueError(
                f"{gguf_name} has type {tensor.tensor_type.name}, "
                f"expected {block_format.name}"
            )
        expected_packed = entry["packed"]
        source_name = entry.get("source_name", sidecar_name)
        if transformer is not None and "linear_attn." in source_name:
            codes = entry.get("codes")
            scales = entry.get("scales")
            if not isinstance(codes, torch.Tensor) or not isinstance(
                scales, torch.Tensor
            ):
                codes, scales = block_format.unpack(
                    expected_packed,
                    logical_size=int(entry["shape"][-1]),
                )
            codes, scales = transformer.transform(
                codes, scales, source_name
            )
            expected_packed = block_format.pack(codes, scales)
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
    print(
        f"Validated {len(tensors)} GPTQ "
        f"{tensors.block_format.name} tensors byte-for-byte."
    )


if __name__ == "__main__":
    main()
