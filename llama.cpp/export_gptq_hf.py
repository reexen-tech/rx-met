#!/usr/bin/env python3
"""Export Q4_0_64 GPTQ artifacts for supported Qwen models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from llm_quant.gptq.layer_runner import run_gptq
from llm_quant.gptq.metadata import fixed_gptq_config
from llm_quant.gptq.sidecar import SidecarShardWriter


def export_gptq_hf(
    model_path: str,
    output_dir: str,
    *,
    device: str = "cuda",
    resume: bool = False,
    max_layers: int | None = None,
) -> str:
    """Quantize in place and save fake HF plus exact Q4_0_64 sidecar."""

    source = Path(model_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if source == destination:
        raise ValueError("output_dir must differ from model_path")
    destination.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(source, trust_remote_code=False)
    is_qwen35_moe = config.model_type == "qwen3_5_moe"
    use_sharded_sidecar = config.model_type in {"qwen3", "qwen3_5_moe"}
    dtype = torch.bfloat16 if use_sharded_sidecar else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        source, use_fast=False, trust_remote_code=False
    )
    auto_model = AutoModelForImageTextToText if is_qwen35_moe else AutoModelForCausalLM
    model = auto_model.from_pretrained(
        source,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()

    if is_qwen35_moe:
        family = "Qwen3.5-35B-A3B"
    elif config.model_type == "qwen3":
        family = "Qwen3-8B"
    else:
        family = "Qwen2.5-0.5B"
    print(f"* Running fixed GPTQ configuration: {family} W4A8, Q4_0_64 G64")
    shard_writer = (
        SidecarShardWriter(
            destination,
            source_model=str(source),
            resume=resume,
        )
        if use_sharded_sidecar
        else None
    )
    result = run_gptq(
        model,
        tokenizer,
        device=device,
        packed_only=use_sharded_sidecar,
        layer_sink=(
            (
                lambda layer, tensors, _stats: shard_writer.write_layer(
                    layer, tensors
                )
            )
            if shard_writer is not None
            else None
        ),
        keep_tensor_data=not use_sharded_sidecar,
        max_layers=max_layers,
    )

    model.cpu()
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)

    if use_sharded_sidecar:
        assert shard_writer is not None
        sidecar_path = shard_writer.finish()
    else:
        sidecar = {
            "format": "Q4_0_64",
            "version": 1,
            "source_model": str(source),
            "tensors": result.tensor_data,
        }
        sidecar_path = destination / "gptq_q4_0_64.pt"
        torch.save(sidecar, sidecar_path)

    metadata = fixed_gptq_config(
        model_family=family,
        fake_quant_dtype="bfloat16" if use_sharded_sidecar else "float16",
    )
    metadata.update(
        {
            "source_model": str(source),
            "calibration_sequences": result.calibration_sequences,
            "fixed_point_ok": result.fixed_point_ok,
            "gguf_export_path": "direct_q4_0_64_sidecar",
            "sidecar": sidecar_path.name,
            "max_layers": max_layers,
            "layer_stats": result.layer_stats,
            "note": (
                "HF text weights contain dequantized GPTQ values for inspection. "
                "Exact production blocks are in the sidecar; "
                "convert_hf_to_gguf.py consumes it directly."
            ),
        }
    )
    with open(destination / "gptq_export_meta.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"* Saved fake-quant HF model to {destination}")
    print(f"* Saved exact Q4_0_64 sidecar to {sidecar_path}")
    return os.fspath(destination)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fixed Qwen GPTQ W4A8/G64 artifacts"
    )
    parser.add_argument("--model_path", required=True, help="Supported Qwen HF path")
    parser.add_argument("--output_dir", required=True, help="Output HF directory")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Quantization device (default: cuda)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed Qwen3.5 sidecar shards",
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=None,
        help="Quantize only the first N layers for a pilot run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = export_gptq_hf(
        args.model_path,
        args.output_dir,
        device=args.device,
        resume=args.resume,
        max_layers=args.max_layers,
    )
    print("Next:")
    print(
        f"  python convert_hf_to_gguf.py {output_dir} "
        "--outtype f16 --outfile model-gptq-Q4_0_64.gguf"
    )


if __name__ == "__main__":
    main()
