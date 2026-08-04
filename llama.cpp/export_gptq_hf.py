#!/usr/bin/env python3
"""Export Q4_0_64 or Q8_0_64 GPTQ artifacts for supported Qwen models."""

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

from llm_quant.artifacts import build_source_signature
from llm_quant.gptq.formats import format_for_bits
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
    expert_hessian_weighting: str = "route_squared",
    bits: int = 4,
) -> str:
    """Quantize in place and save fake HF plus an exact Q64 sidecar."""

    source = Path(model_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if source == destination:
        raise ValueError("output_dir must differ from model_path")
    if expert_hessian_weighting not in {"none", "route_squared"}:
        raise ValueError(
            f"unsupported expert Hessian weighting: "
            f"{expert_hessian_weighting}"
        )
    block_format = format_for_bits(bits)
    destination.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(source, trust_remote_code=False)
    is_qwen35_moe = config.model_type == "qwen3_5_moe"
    use_bfloat16 = config.model_type in {"qwen3", "qwen3_5_moe"}
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    if is_qwen35_moe:
        family = "Qwen3.5-35B-A3B"
    elif config.model_type == "qwen3":
        family = "Qwen3-8B"
    else:
        family = "Qwen2.5-0.5B"
    metadata = fixed_gptq_config(
        model_family=family,
        fake_quant_dtype="bfloat16" if use_bfloat16 else "float16",
        expert_hessian_weighting=expert_hessian_weighting,
        bits=bits,
    )
    quantization_config = {
        **metadata,
        "max_layers": max_layers,
    }
    source_signature = build_source_signature(source)
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
    shard_writer = SidecarShardWriter(
        destination,
        source_model=str(source),
        source_signature=source_signature,
        quantization_config=quantization_config,
        resume=resume,
        format_name=block_format.name,
    )

    print(
        f"* Running fixed GPTQ configuration: {family} "
        f"W{bits}A8, {block_format.name} G64"
    )
    result = run_gptq(
        model,
        tokenizer,
        device=device,
        packed_only=True,
        sidecar_writer=shard_writer,
        keep_tensor_data=False,
        max_layers=max_layers,
        expert_hessian_weighting=expert_hessian_weighting,
        bits=bits,
    )

    model.cpu()
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)

    sidecar_path = shard_writer.finish()
    metadata.update(
        {
            "source_model": str(source),
            "calibration_sequences": result.calibration_sequences,
            "fixed_point_ok": result.fixed_point_ok,
            "gguf_export_path": f"direct_{block_format.name.lower()}_sidecar",
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
    print(f"* Saved exact {block_format.name} sidecar to {sidecar_path}")
    return os.fspath(destination)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fixed Qwen GPTQ W4A8 or W8A8/G64 artifacts"
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
    parser.add_argument(
        "--bits",
        type=int,
        choices=(4, 8),
        default=4,
        help="Weight bit width (default: 4)",
    )
    parser.add_argument(
        "--expert-hessian-weighting",
        choices=("none", "route_squared"),
        default="route_squared",
        help="Routed-expert Hessian weighting (default: route_squared)",
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
        expert_hessian_weighting=args.expert_hessian_weighting,
        bits=args.bits,
    )
    print("Next:")
    print(
        f"  python convert_hf_to_gguf.py {output_dir} "
        f"--outtype f16 --outfile model-gptq-Q{args.bits}_0_64.gguf"
    )


if __name__ == "__main__":
    main()
