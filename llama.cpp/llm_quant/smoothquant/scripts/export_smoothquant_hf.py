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
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Ensure repo root is on sys.path when invoked as a script.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_quant.smoothquant import default_alpha, run_smoothquant  # noqa: E402
from llm_quant.smoothquant.loading import (  # noqa: E402
    load_qwen35_text_model,
    validate_qwen35_smoothed_parameter_dtypes,
)

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
    resume_act_scales: bool = False,
    calibration_mode: str = "fixed",
    min_samples: int | None = None,
    max_samples: int | None = None,
) -> str:
    """Calibrate, smooth in place and save the smoothed HF directory."""
    outer_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    is_qwen35 = getattr(outer_config, "model_type", None) == "qwen3_5_moe"
    config = outer_config.text_config if is_qwen35 else outer_config
    torch_dtype = resolve_dtype(config, dtype)
    if is_qwen35 and torch_dtype != torch.bfloat16:
        raise ValueError(
            "Qwen3.5-35B-A3B SmoothQuant must load and save BF16; "
            f"resolved dtype is {torch_dtype}"
        )
    if alpha is None:
        alpha = default_alpha(getattr(config, "model_type", None))
        print(f"* alpha not given, using {alpha} for model_type="
              f"{getattr(config, 'model_type', None)}")

    print(f"* Loading {model_path} as {torch_dtype} (device={device})")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    loading_details = None
    if is_qwen35:
        model, loading_details = load_qwen35_text_model(
            model_path,
            config,
            dtype=torch_dtype,
            device=device,
            require_production_shape=True,
        )
    else:
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
        resume_act_scales=resume_act_scales,
        calibration_mode=calibration_mode,
        min_samples=min_samples,
        max_samples=max_samples,
        smoothing_scope="moe" if is_qwen35 else "all",
        scale_policy="power_of_two" if is_qwen35 else "requested",
        norm_materialization="fp32_offset" if is_qwen35 else "target_dtype",
    )

    parameter_dtype_policy = None
    if is_qwen35:
        if result.n_groups_smoothed != 40:
            raise RuntimeError(
                "Qwen3.5 MoE-only smoothing must transform exactly 40 groups; "
                f"got {result.n_groups_smoothed}"
            )
        parameter_dtype_policy = validate_qwen35_smoothed_parameter_dtypes(model)

    os.makedirs(output_dir, exist_ok=True)
    print(f"* Saving smoothed {torch_dtype} HF to {output_dir}")
    if getattr(model, "hf_device_map", None) is None:
        model.cpu()
    save_kwargs = {"safe_serialization": True}
    if is_qwen35:
        # Do not reverse the source multimodal key mapping recorded by
        # Transformers during text-only loading.
        save_kwargs["save_original_format"] = False
    model.save_pretrained(output_dir, **save_kwargs)
    tok.save_pretrained(output_dir)

    meta = {
        "source_model": os.path.abspath(model_path),
        "source_model_type": getattr(outer_config, "model_type", None),
        "saved_model_type": getattr(model.config, "model_type", None),
        "saved_model_class": type(model).__name__,
        "alpha": result.alpha,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "n_samples": result.n_samples,
        "seqlen": result.seqlen,
        "calib_data": result.calib_data,
        "n_layers_smoothed": result.n_layers_smoothed,
        "n_groups_smoothed": result.n_groups_smoothed,
        "act_scales_cache": result.act_scales_cache,
        "act_scales_from_cache": result.act_scales_from_cache,
        "calibration_resumed": result.calibration_resumed,
        "calib_meta": result.meta,
        "coverage": result.coverage,
        "smooth_groups": result.group_stats,
        "smoothing_policy": result.smoothing_policy,
        "parameter_dtype_policy": parameter_dtype_policy,
        "transformers_version": transformers.__version__,
        "text_only_loading": loading_details,
        "source_multimodal_checkpoint_unchanged": True if is_qwen35 else None,
        "vision_mmproj_in_smooth_hf": False if is_qwen35 else None,
        "mtp_in_smooth_hf": False if is_qwen35 else None,
        "transformers_save_original_format": False if is_qwen35 else True,
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
        "--calibration-mode",
        choices=("fixed", "smoke", "formal"),
        default="fixed",
        help="fixed uses --n_samples/--seqlen; smoke is 32x128; formal is min 512, max 2048 at 512 tokens with full expert coverage",
    )
    p.add_argument(
        "--min_samples",
        type=int,
        default=None,
        help="Minimum samples for fixed mode (default: --n_samples)",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples for fixed mode (default: --n_samples)",
    )
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
    cache_mode = p.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--reuse_act_scales",
        action="store_true",
        help="Reuse a matching complete cache; partial caches are rejected",
    )
    cache_mode.add_argument(
        "--resume_act_scales",
        action="store_true",
        help="Resume a matching partial cache (or reuse it if already complete)",
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
        resume_act_scales=args.resume_act_scales,
        calibration_mode=args.calibration_mode,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
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
