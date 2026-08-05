#!/usr/bin/env python3
"""Cross-validate llm_quant.smoothquant against the MIT reference implementation.

Runs calibration + smoothing twice on the same checkpoint and calibration file,
once through ``llm_quant.smoothquant`` and once through the upstream
https://github.com/mit-han-lab/smoothquant checkout, then reports the maximum
absolute difference of the activation scales and of the smoothed weights.

The reference ``smooth_lm`` dispatches on ``LlamaDecoderLayer``; for a
structurally identical architecture such as Qwen2 we rebind that name so the
reference code path is exercised unmodified.

  python llm_quant/smoothquant/scripts/compare_smoothquant_ref.py \\
    --model_path /mnt/data8t/share/models/Qwen/Qwen2.5-7B-Instruct \\
    --calib_data /path/to/val.jsonl --alpha 0.85 --n_samples 32
"""
# === REEX_SMOOTHQUANT BEGIN: MIT smoothquant cross-validation harness ===
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_quant.smoothquant.scripts.export_smoothquant_hf import resolve_dtype  # noqa: E402
from llm_quant.smoothquant import (  # noqa: E402
    get_act_scales,
    resolve_calib_dataset,
    smooth_lm,
)

DEFAULT_REF_REPO = os.environ.get("SMOOTHQUANT_REF_REPO", "/mnt/data8t/zcx/smoothquant")


def load_model(model_path: str, dtype: torch.dtype, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=None if device == "cpu" else device,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


def snapshot(model, layer_indices: list[int]) -> dict[str, torch.Tensor]:
    """Norm weights for every layer, plus Linear weights for sampled layers."""
    out: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(model.model.layers):
        for norm in ("input_layernorm", "post_attention_layernorm"):
            out[f"layers.{i}.{norm}.weight"] = (
                getattr(layer, norm).weight.detach().float().cpu()
            )
        if i in layer_indices:
            out[f"layers.{i}.self_attn.q_proj.weight"] = (
                layer.self_attn.q_proj.weight.detach().float().cpu()
            )
            out[f"layers.{i}.mlp.gate_proj.weight"] = (
                layer.mlp.gate_proj.weight.detach().float().cpu()
            )
    return out


def run_ours(args, dtype, calib_path, layer_indices):
    model, tokenizer = load_model(args.model_path, dtype, args.device)
    layer_indices = _resolve_layers(model, layer_indices)
    act_scales = get_act_scales(
        model, tokenizer, calib_path, num_samples=args.n_samples, seq_len=args.seqlen
    )
    smooth_lm(model, act_scales, args.alpha)
    weights = snapshot(model, layer_indices)
    _free(model)
    return act_scales, weights, layer_indices


def run_reference(args, dtype, calib_path, layer_indices):
    if args.ref_repo not in sys.path:
        sys.path.insert(0, args.ref_repo)
    from smoothquant.calibration import get_act_scales as ref_get_act_scales
    import smoothquant.smooth as ref_smooth

    model, tokenizer = load_model(args.model_path, dtype, args.device)
    note = _rebind_reference_dispatch(model, ref_smooth)
    print(f" * reference dispatch: {note}", flush=True)
    act_scales = ref_get_act_scales(
        model, tokenizer, str(calib_path), num_samples=args.n_samples, seq_len=args.seqlen
    )
    ref_smooth.smooth_lm(model, act_scales, args.alpha)
    weights = snapshot(model, layer_indices)
    _free(model)
    return act_scales, weights, note


def _rebind_reference_dispatch(model, ref_smooth) -> str:
    layer_cls = type(model.model.layers[0])
    norm_cls = type(model.model.layers[0].input_layernorm)
    if layer_cls.__name__ in ("LlamaDecoderLayer", "MistralDecoderLayer"):
        return f"native support for {layer_cls.__name__}"
    ref_smooth.LlamaDecoderLayer = layer_cls
    ref_smooth.LlamaRMSNorm = norm_cls
    return f"LlamaDecoderLayer->{layer_cls.__name__}, LlamaRMSNorm->{norm_cls.__name__}"


def _resolve_layers(model, requested: list[int] | None) -> list[int]:
    n = len(model.model.layers)
    if requested:
        return sorted({i % n for i in requested})
    return sorted({0, n // 2, n - 1})


def _free(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compare(ours: dict, reference: dict) -> tuple[float, list[str]]:
    problems = []
    missing = set(reference) - set(ours)
    extra = set(ours) - set(reference)
    if missing:
        problems.append(f"missing in ours: {sorted(missing)[:5]}")
    if extra:
        problems.append(f"only in ours: {sorted(extra)[:5]}")
    max_diff = 0.0
    for key in sorted(set(ours) & set(reference)):
        diff = (ours[key].float() - reference[key].float()).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff, problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--calib_data", default="pileval")
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"])
    p.add_argument("--ref_repo", default=DEFAULT_REF_REPO, help="MIT smoothquant checkout")
    p.add_argument(
        "--compare_layers",
        type=int,
        nargs="*",
        default=None,
        help="Layer indices whose Linear weights are compared (default: first/mid/last)",
    )
    p.add_argument("--atol_act_scales", type=float, default=1e-5)
    p.add_argument("--atol_weights", type=float, default=1e-4)
    p.add_argument("--report", default=None, help="Optional JSON summary path")
    args = p.parse_args(argv)

    if not os.path.isdir(args.ref_repo):
        raise SystemExit(
            f"reference checkout not found: {args.ref_repo} "
            "(set --ref_repo or SMOOTHQUANT_REF_REPO)"
        )
    calib_path = resolve_calib_dataset(args.calib_data)
    dtype = resolve_dtype(
        AutoConfig.from_pretrained(args.model_path, trust_remote_code=True), args.dtype
    )

    print("=== llm_quant.smoothquant ===", flush=True)
    our_scales, our_weights, layers = run_ours(args, dtype, calib_path, args.compare_layers)
    print("=== MIT smoothquant reference ===", flush=True)
    ref_scales, ref_weights, note = run_reference(args, dtype, calib_path, layers)

    scale_diff, scale_problems = compare(our_scales, ref_scales)
    weight_diff, weight_problems = compare(our_weights, ref_weights)
    scales_ok = not scale_problems and scale_diff <= args.atol_act_scales
    weights_ok = not weight_problems and weight_diff <= args.atol_weights

    summary = {
        "model_path": os.path.abspath(args.model_path),
        "ref_repo": os.path.abspath(args.ref_repo),
        "reference_dispatch": note,
        "alpha": args.alpha,
        "n_samples": args.n_samples,
        "seqlen": args.seqlen,
        "dtype": str(dtype).replace("torch.", ""),
        "calib_data": str(calib_path),
        "compared_layers": layers,
        "n_act_scale_keys": len(our_scales),
        "act_scales_max_diff": scale_diff,
        "act_scales_atol": args.atol_act_scales,
        "act_scales_ok": scales_ok,
        "n_weight_keys": len(our_weights),
        "weights_max_diff": weight_diff,
        "weights_atol": args.atol_weights,
        "weights_ok": weights_ok,
        "problems": scale_problems + weight_problems,
        "passed": scales_ok and weights_ok,
    }
    print(json.dumps(summary, indent=2))
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(summary, f, indent=2)
    print("PASS" if summary["passed"] else "FAIL")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
# === REEX_SMOOTHQUANT END ===
