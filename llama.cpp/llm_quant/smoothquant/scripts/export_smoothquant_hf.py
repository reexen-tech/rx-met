#!/usr/bin/env python3
"""One-shot SmoothQuant Step 1 -> smoothed float HuggingFace directory.

The output is an ordinary FP16/BF16 checkpoint whose RMSNorm and Linear weights
have been rewritten by the SmoothQuant equivalent transform. Target Q64
quantization is left to ``llama-quantize``.

Run from the llama.cpp repo root:

  python llm_quant/smoothquant/scripts/export_smoothquant_hf.py \\
    --model_path /path/to/Qwen2.5-7B-Instruct \\
    --output_dir /path/to/qwen2.5-7b-smoothquant-hf \\
    --alpha 0.85 --n_samples 512 --seqlen 512 \\
    --calib_data /path/to/val.jsonl
"""
# === REEX_SMOOTHQUANT BEGIN: CLI exporting a SmoothQuant-smoothed HF checkpoint ===
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Ensure repo root is on sys.path when invoked as a script.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_quant.smoothquant import default_alpha, run_smoothquant  # noqa: E402

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def resolve_dtype(config, override: str | None) -> torch.dtype:
    """Match the checkpoint dtype unless the CLI overrides it."""
    if override:
        return _DTYPES[override]
    # transformers >= 5 renamed config.torch_dtype to config.dtype
    declared = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(declared, torch.dtype):
        return declared
    if isinstance(declared, str) and declared in _DTYPES:
        return _DTYPES[declared]
    return torch.float16


def export_smoothquant_hf(
    model_path: str,
    output_dir: str,
    *,
    dtype: str | None = None,
    alpha: float | None = None,
    n_samples: int = 512,
    seqlen: int = 512,
    calib_data: str = "pileval",
    device: str = "auto",
    act_scales_cache: str | None = None,
    reuse_act_scales: bool = False,
) -> str:
    """Calibrate, smooth in place and save the smoothed HF directory."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    torch_dtype = resolve_dtype(config, dtype)
    if alpha is None:
        alpha = default_alpha(getattr(config, "model_type", None))
        print(f"* alpha not given, using {alpha} for model_type="
              f"{getattr(config, 'model_type', None)}")

    print(f"* Loading {model_path} as {torch_dtype} (device={device})")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=None if device == "cpu" else device,
    )
    if device == "cpu":
        model.to("cpu")
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    print("* Running SmoothQuant calibration + smoothing (in-place)...")
    result = run_smoothquant(
        model,
        tok,
        alpha=alpha,
        n_samples=n_samples,
        seqlen=seqlen,
        calib_data=calib_data,
        model_path=model_path,
        act_scales_cache=act_scales_cache,
        reuse_act_scales=reuse_act_scales,
    )

    os.makedirs(output_dir, exist_ok=True)
    print(f"* Saving smoothed {torch_dtype} HF to {output_dir}")
    if getattr(model, "hf_device_map", None) is None:
        model.cpu()
    model.save_pretrained(output_dir, safe_serialization=True)
    tok.save_pretrained(output_dir)

    meta = {
        "source_model": os.path.abspath(model_path),
        "alpha": result.alpha,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "n_samples": result.n_samples,
        "seqlen": result.seqlen,
        "calib_data": result.calib_data,
        "n_layers_smoothed": result.n_layers_smoothed,
        "n_groups_smoothed": result.n_groups_smoothed,
        "act_scales_cache": result.act_scales_cache,
        "act_scales_from_cache": result.act_scales_from_cache,
        "calib_meta": result.meta,
        "note": "Float weights with the SmoothQuant transform fused; not quantized. "
        "Next: convert_hf_to_gguf.py then llama-quantize with the target Q64 type.",
    }
    with open(os.path.join(output_dir, "smoothquant_export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    return output_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SmoothQuant calibration + smoothing, exported as float HF weights"
    )
    p.add_argument("--model_path", type=str, required=True, help="HF model directory")
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output HF directory (safetensors + tokenizer)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=sorted(_DTYPES),
        help="Override the checkpoint dtype (default: match config.torch_dtype)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Smoothing strength (default: per model_type, see llm_quant/smoothquant/defaults.py)",
    )
    p.add_argument("--n_samples", type=int, default=512, help="Calibration samples")
    p.add_argument("--seqlen", type=int, default=512, help="Calibration truncation length")
    p.add_argument(
        "--calib_data",
        type=str,
        default="pileval",
        help="Local .jsonl path, or 'pileval' to use SMOOTHQUANT_PILEVAL_PATH",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device_map for calibration ('auto', 'cuda:0', 'cpu'; use cpu on OOM)",
    )
    p.add_argument(
        "--act_scales_cache",
        "--save_act_scales",
        dest="act_scales_cache",
        type=str,
        default=None,
        help="Path to read/write the calibrated act_scales .pt",
    )
    p.add_argument(
        "--reuse_act_scales",
        action="store_true",
        help="Load act_scales from --act_scales_cache instead of calibrating",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    export_smoothquant_hf(
        args.model_path,
        args.output_dir,
        dtype=args.dtype,
        alpha=args.alpha,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
        calib_data=args.calib_data,
        device=args.device,
        act_scales_cache=args.act_scales_cache,
        reuse_act_scales=args.reuse_act_scales,
    )
    print("Next:")
    print(f"  python convert_hf_to_gguf.py {args.output_dir} "
          "--outtype f16 --outfile model-smooth-f16.gguf")
    print("  ./build_cuda_q64/bin/llama-quantize "
          "--token-embedding-type f16 --output-tensor-type f16 --leave-output-tensor "
          "model-smooth-f16.gguf model-smooth-Q8_0_64.gguf Q8_0_64")


if __name__ == "__main__":
    main()
# === REEX_SMOOTHQUANT END ===
