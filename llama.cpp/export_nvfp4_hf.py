#!/usr/bin/env python3
"""ModelOpt NVFP4 PTQ → Unified HuggingFace checkpoint (via pip/whl API only).

Requires:
  pip install -U "nvidia-modelopt[all]" transformers accelerate

Run from the llama.cpp repo root (or any cwd):

  python export_nvfp4_hf.py \\
    --model_path /path/to/hf-model \\
    --output_dir /path/to/nvfp4-hf-out \\
    --qformat nvfp4_experts_only \\
    --calib_data /path/to/calib.jsonl

Then convert to GGUF (non-expert layers: use q8_0 to match MXFP4_MOE layout):

  python convert_hf_to_gguf.py /path/to/nvfp4-hf-out \\
    --outfile /path/to/model-nvfp4.gguf --outtype q8_0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch


QUANT_CFG_CHOICES = {
    "nvfp4": "NVFP4_DEFAULT_CFG",
    "nvfp4_experts_only": "NVFP4_EXPERTS_ONLY_CFG",
    "nvfp4_mlp_only": "NVFP4_MLP_ONLY_CFG",
    "nvfp4_omlp_only": "NVFP4_OMLP_ONLY_CFG",
    "nvfp4_awq": "NVFP4_AWQ_LITE_CFG",
}


def _resolve_quant_cfg(qformat: str):
    import modelopt.torch.quantization as mtq

    attr = QUANT_CFG_CHOICES.get(qformat)
    if attr is None:
        raise ValueError(
            f"Unknown --qformat {qformat!r}. Choose from: {sorted(QUANT_CFG_CHOICES)}"
        )
    if not hasattr(mtq, attr):
        # AWQ lite may be missing on older wheels; fall back to full AWQ cfg.
        if qformat == "nvfp4_awq" and hasattr(mtq, "NVFP4_AWQ_FULL_CFG"):
            return mtq.NVFP4_AWQ_FULL_CFG
        raise AttributeError(
            f"modelopt.torch.quantization has no {attr}. "
            f"Upgrade nvidia-modelopt or pick another --qformat."
        )
    return getattr(mtq, attr)


def _load_model_and_tokenizer(model_path: str, dtype: str, device_map: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    print(f"* Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # ModelOpt calib recommends left padding.
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"* Loading model ({dtype}, device_map={device_map})")
    load_kwargs = {
        "dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        print(f"* AutoModelForCausalLM failed ({e}); trying Qwen3_5MoeForConditionalGeneration")
        from transformers import Qwen3_5MoeForConditionalGeneration

        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(model_path, **load_kwargs)

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    _ensure_config_architectures(model, model_path)
    return model, tokenizer


def _infer_calib_device(model) -> str:
    """Device for calibration batches; must match embedding / first layer."""
    for path in ("model.embed_tokens.weight", "model.model.embed_tokens.weight"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "device"):
                return str(obj.device)
        except AttributeError:
            continue
    return str(next(model.parameters()).device)


def _ensure_config_architectures(model, model_path: str | None = None) -> None:
    """ModelOpt export calls ``is_multimodal_model`` which crashes when architectures is None."""
    cfg = model.config
    if getattr(cfg, "architectures", None):
        return

    if model_path:
        cfg_json = os.path.join(model_path, "config.json")
        if os.path.isfile(cfg_json):
            with open(cfg_json, encoding="utf-8") as f:
                data = json.load(f)
            arch = data.get("architectures")
            if not arch:
                arch = (data.get("text_config") or {}).get("architectures")
            if arch:
                cfg.architectures = arch
                print(f"* Patched config.architectures from {cfg_json}: {arch}")
                return

    cfg.architectures = [type(model).__name__]
    print(f"* Patched config.architectures -> {cfg.architectures}")


def _patch_modelopt_export_utils() -> None:
    """Patch ModelOpt helpers that crash on ``config.architectures is None``."""
    import modelopt.torch.export.model_utils as model_utils
    import modelopt.torch.export.unified_export_hf as unified_export_hf

    model_utils.is_multimodal_model = _is_multimodal_model
    unified_export_hf.is_multimodal_model = _is_multimodal_model


def _is_multimodal_model(model) -> bool:
    """Safe multimodal check.

    Upstream modelopt ``is_multimodal_model`` crashes when
    ``config.architectures is None`` (``getattr(..., [])`` still returns None).
    """
    config = model.config
    architectures = getattr(config, "architectures", None) or []
    is_nemotron_parse = any(
        isinstance(arch, str) and "nemotronparse" in arch.lower() for arch in architectures
    )
    return (
        hasattr(config, "vision_config")
        or hasattr(model, "language_model")
        or getattr(config, "model_type", "") == "phi4mm"
        or hasattr(config, "vision_lora")
        or hasattr(config, "audio_processor")
        or (
            hasattr(config, "embd_layer")
            and hasattr(config.embd_layer, "image_embd_layer")
        )
        or is_nemotron_parse
        # Qwen3.5 MoE HF wrapper: text under model.language_model / model.model.language_model
        or hasattr(config, "text_config")
        or (
            hasattr(model, "model")
            and hasattr(model.model, "language_model")
        )
    )


def _language_model_lineage(model):
    """Return [root, ..., language_model] or None."""
    from modelopt.torch.export.model_utils import get_language_model_from_vl

    lineage = get_language_model_from_vl(model)
    if lineage:
        return lineage
    # Fallback for wrappers modelopt does not recognize yet.
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return [model, model.model, model.model.language_model]
    if hasattr(model, "language_model"):
        return [model, model.language_model]
    return None


def _prepare_language_model_for_ptq(full_model):
    """For multimodal wrappers, disable quant on non-LM modules and return LM to quantize."""
    import modelopt.torch.quantization as mtq

    if not _is_multimodal_model(full_model):
        return full_model, full_model

    lineage = _language_model_lineage(full_model)
    if not lineage:
        print("* Multimodal detected but no language_model lineage; quantizing full model")
        return full_model, full_model

    language_model = lineage[-1]
    ancestors = lineage[:-1]
    disabled_quant_cfg = {
        "quant_cfg": [{"quantizer_name": "*", "enable": False}],
        "algorithm": "max",
    }
    memo = set(ancestors) | {language_model}
    for ancestor in ancestors:
        for _, module in ancestor.named_children():
            if module not in memo:
                mtq.quantize(module, disabled_quant_cfg, forward_loop=None)
                memo.add(module)

    print(f"* Multimodal model: PTQ target = {type(language_model).__name__}")
    return full_model, language_model


def _export_model_after_ptq(full_model, language_model):
    """Re-attach quantized LM into multimodal wrapper when needed."""
    if not _is_multimodal_model(full_model):
        return language_model

    lineage = _language_model_lineage(full_model)
    if lineage is not None and len(lineage) >= 2:
        print("* Updating full_model with quantized language_model...")
        setattr(lineage[-2], "language_model", language_model)
        return full_model
    return language_model



def export_nvfp4_hf(
    model_path: str,
    output_dir: str,
    *,
    qformat: str = "nvfp4_experts_only",
    dtype: str = "bfloat16",
    device_map: str = "auto",
    dataset_name: str,
    num_samples: int = 128,
    max_sample_length: int = 512,
    batch_size: int = 1,
) -> str:
    """Run ModelOpt NVFP4 PTQ and export Unified HF checkpoint. Returns output_dir."""
    try:
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
        from modelopt.torch.export import export_hf_checkpoint
        from modelopt.torch.utils.dataset_utils import create_forward_loop
    except ImportError as e:
        raise SystemExit(
            "nvidia-modelopt is not installed in this environment.\n"
            '  pip install -U "nvidia-modelopt[all]"\n'
            f"Original error: {e}"
        ) from e

    mto.enable_huggingface_checkpointing()
    quant_cfg = _resolve_quant_cfg(qformat)
    print(f"* Quant config: {qformat} -> {QUANT_CFG_CHOICES[qformat]}")

    full_model, tokenizer = _load_model_and_tokenizer(model_path, dtype, device_map)
    full_model, language_model = _prepare_language_model_for_ptq(full_model)

    # modelopt accepts either a registered HF dataset name or a local .jsonl path.
    if dataset_name.endswith(".jsonl"):
        calib_path = os.path.abspath(os.path.expanduser(dataset_name))
        if not os.path.isfile(calib_path):
            raise FileNotFoundError(
                f"Local calib JSONL not found: {calib_path}\n"
                "Pass --calib_data /path/to/val.jsonl (offline; no HuggingFace download)."
            )
        dataset_name = calib_path

    calib_device = _infer_calib_device(language_model)
    print(
        f"* Building calib forward_loop: dataset={dataset_name}, "
        f"num_samples={num_samples}, seqlen={max_sample_length}, bs={batch_size}, "
        f"device={calib_device}"
    )
    forward_loop = create_forward_loop(
        model=language_model,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=num_samples,
        max_sample_length=max_sample_length,
        device=calib_device,
    )

    print("* Running mtq.quantize (PTQ + calibration)...")
    with torch.inference_mode():
        language_model = mtq.quantize(language_model, quant_cfg, forward_loop=forward_loop)
        export_model = _export_model_after_ptq(full_model, language_model)

        os.makedirs(output_dir, exist_ok=True)
        print(f"* Exporting Unified HF checkpoint to {output_dir}")
        _ensure_config_architectures(export_model, model_path)
        _patch_modelopt_export_utils()
        export_hf_checkpoint(export_model, export_dir=output_dir)

    # Best-effort tokenizer / processor copy for convert_hf_to_gguf / chat.
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"* Warning: tokenizer.save_pretrained failed: {e}")

    meta = {
        "source_model": os.path.abspath(model_path),
        "qformat": qformat,
        "dtype": dtype,
        "dataset_name": dataset_name,
        "num_samples": num_samples,
        "max_sample_length": max_sample_length,
        "batch_size": batch_size,
        "note": "ModelOpt NVFP4 Unified HF. Next: convert_hf_to_gguf.py then llama-cli.",
    }
    with open(os.path.join(output_dir, "nvfp4_export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    return output_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ModelOpt NVFP4 PTQ and export Unified HF (nvidia-modelopt wheel API)"
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Input HuggingFace model directory",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output Unified HF directory (safetensors + hf_quant_config.json)",
    )
    p.add_argument(
        "--qformat",
        type=str,
        default="nvfp4_experts_only",
        choices=sorted(QUANT_CFG_CHOICES),
        help="NVFP4 recipe (MoE: prefer nvfp4_experts_only)",
    )
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='HF device_map (default: "auto")',
    )
    p.add_argument(
        "--calib_data",
        "--dataset_name",
        dest="dataset_name",
        type=str,
        required=True,
        help=(
            "Calibration source: local .jsonl path (offline, recommended) "
            "or a modelopt registered dataset name (may download from HuggingFace)"
        ),
    )
    p.add_argument("--num_samples", type=int, default=128, help="Calibration samples")
    p.add_argument("--max_sample_length", type=int, default=512, help="Calib sequence length")
    p.add_argument("--batch_size", type=int, default=1, help="Calib batch size")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    export_nvfp4_hf(
        args.model_path,
        args.output_dir,
        qformat=args.qformat,
        dtype=args.dtype,
        device_map=args.device_map,
        dataset_name=args.dataset_name,
        num_samples=args.num_samples,
        max_sample_length=args.max_sample_length,
        batch_size=args.batch_size,
    )
    print("Next:")
    print(
        f"  python convert_hf_to_gguf.py {args.output_dir} "
        f"--outfile {args.output_dir.rstrip('/')}.gguf --outtype q8_0"
    )


if __name__ == "__main__":
    main()
