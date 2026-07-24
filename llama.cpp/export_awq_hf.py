#!/usr/bin/env python3
"""One-shot AWQ search → AWQ-scaled FP16 HuggingFace directory (no .pt / no INT4 pack).

Run from the llama.cpp repo root:

  python export_awq_hf.py \\
    --model_path /path/to/Qwen2.5-7B \\
    --output_dir /path/to/qwen2.5-7b-awq-g64-hf \\
    --w_bit 4 --q_group_size 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root is on sys.path when invoked as a script
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from llm_quant.awq import run_awq  # noqa: E402


def export_awq_hf(
    model_path: str,
    output_dir: str,
    *,
    dtype: str = "float16",
    w_bit: int = 4,
    q_group_size: int = 64,
    zero_point: bool = True,
    n_samples: int = 128,
    seqlen: int = 512,
    calib_data: str = "pileval",
    save_awq_cache: str | None = None,
) -> str:
    """Run AWQ search (in-place scale+clip) and save AWQ-scaled FP16 HF. Returns output_dir."""
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    q_config = {"zero_point": zero_point, "q_group_size": q_group_size}
    print("Quantization config:", q_config)
    print(f"* Loading {model_path}")

    tok = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    print("* Running AWQ search (applies scale+clip in-place)...")
    awq_results = run_awq(
        model,
        tok,
        w_bit=w_bit,
        q_config=q_config,
        n_samples=n_samples,
        seqlen=seqlen,
        calib_data=calib_data,
    )

    if save_awq_cache:
        parent = os.path.dirname(os.path.abspath(save_awq_cache))
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(awq_results, save_awq_cache)
        print(f"* Optional AWQ cache saved at {save_awq_cache}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"* Saving AWQ-scaled FP16 HF to {output_dir}")
    model.cpu()
    model.save_pretrained(output_dir, safe_serialization=True)
    tok.save_pretrained(output_dir)

    meta = {
        "source_model": os.path.abspath(model_path),
        "w_bit": w_bit,
        "q_group_size": q_group_size,
        "zero_point": zero_point,
        "n_samples": n_samples,
        "seqlen": seqlen,
        "calib_data": calib_data,
        "note": "FP16 weights with AWQ scale+clip fused; not INT4-packed. "
        "Next: convert_hf_to_gguf.py then llama-quantize.",
    }
    with open(os.path.join(output_dir, "awq_export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    return output_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="AWQ scale+clip search and export AWQ-scaled FP16 HF weights"
    )
    p.add_argument("--model_path", type=str, required=True, help="HF model directory")
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output HF directory (safetensors + tokenizer)",
    )
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--w_bit", type=int, default=4)
    p.add_argument("--q_group_size", type=int, default=64)
    p.add_argument("--no_zero_point", action="store_true")
    p.add_argument("--n_samples", type=int, default=128, help="AWQ calib samples")
    p.add_argument("--seqlen", type=int, default=512, help="AWQ calib block size")
    p.add_argument(
        "--calib_data",
        type=str,
        default="pileval",
        help="Calibration set name (default: pileval)",
    )
    p.add_argument(
        "--save_awq_cache",
        type=str,
        default=None,
        help="Optional: also dump search scales/clips as .pt (debug/reuse)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    export_awq_hf(
        args.model_path,
        args.output_dir,
        dtype=args.dtype,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
        zero_point=not args.no_zero_point,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
        calib_data=args.calib_data,
        save_awq_cache=args.save_awq_cache,
    )
    print("Next:")
    print("Conver to gguf")


if __name__ == "__main__":
    main()
