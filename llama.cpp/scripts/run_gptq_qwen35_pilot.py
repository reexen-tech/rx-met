#!/usr/bin/env python3
"""Run a bounded Qwen3.5 MoE GPTQ layer pilot without saving the full model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_quant.gptq.layer_runner import run_gptq
from llm_quant.gptq.sidecar import SidecarShardWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=False,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()
    writer = SidecarShardWriter(
        args.output_dir,
        source_model=str(args.model_path.resolve()),
    )
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    result = run_gptq(
        model,
        tokenizer,
        device=args.device,
        packed_only=True,
        layer_sink=lambda layer, tensors, _stats: writer.write_layer(
            layer, tensors
        ),
        keep_tensor_data=False,
        max_layers=args.layers,
    )
    sidecar = writer.finish()
    peak_memory = (
        torch.cuda.max_memory_allocated(torch.device(args.device))
        if torch.device(args.device).type == "cuda"
        else 0
    )
    report = {
        "model_path": str(args.model_path.resolve()),
        "layers": args.layers,
        "calibration_sequences": result.calibration_sequences,
        "fixed_point_ok": result.fixed_point_ok,
        "peak_gpu_bytes": peak_memory,
        "sidecar": sidecar.name,
        "layer_stats": result.layer_stats,
    }
    (args.output_dir / "pilot_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
