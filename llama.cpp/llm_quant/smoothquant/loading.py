"""Strict text-only loading helpers for multimodal Qwen3.5 MoE checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


QWEN35_FP32_MOE_NORM_SUFFIX = "post_attention_layernorm.weight"


@contextmanager
def keep_qwen35_moe_norms_in_fp32():
    """Preserve mixed-precision SmoothQuant norms during HF reload."""
    from transformers import Qwen3_5MoeForCausalLM

    previous = Qwen3_5MoeForCausalLM._keep_in_fp32_modules_strict
    Qwen3_5MoeForCausalLM._keep_in_fp32_modules_strict = [
        "post_attention_layernorm"
    ]
    try:
        yield
    finally:
        Qwen3_5MoeForCausalLM._keep_in_fp32_modules_strict = previous


def validate_qwen35_smoothed_parameter_dtypes(
    model, *, expected_moe_norms: int = 40
) -> dict[str, Any]:
    """Require only MoE post-attention norm parameters to be FP32."""
    fp32 = []
    unexpected = []
    for name, parameter in model.named_parameters():
        if name.endswith(QWEN35_FP32_MOE_NORM_SUFFIX):
            if parameter.dtype != torch.float32:
                unexpected.append(f"{name}: expected float32, got {parameter.dtype}")
            else:
                fp32.append(name)
        elif parameter.dtype != torch.bfloat16:
            unexpected.append(f"{name}: expected bfloat16, got {parameter.dtype}")
    if len(fp32) != expected_moe_norms:
        unexpected.append(
            f"expected {expected_moe_norms} FP32 MoE norms, found {len(fp32)}"
        )
    if unexpected:
        raise RuntimeError(
            "invalid Qwen3.5 SmoothQuant dtype policy: "
            + "; ".join(unexpected[:20])
        )
    return {
        "bf16_default": True,
        "fp32_parameter_suffix": QWEN35_FP32_MOE_NORM_SUFFIX,
        "fp32_parameter_count": len(fp32),
        "fp32_scalar_count": sum(
            model.get_parameter(name).numel() for name in fp32
        ),
    }


def checkpoint_keys(model_path: str | Path) -> tuple[str, ...]:
    """Enumerate checkpoint keys without materializing tensor data."""
    root = Path(model_path).expanduser()
    index = root / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text())
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid safetensors index (missing weight_map): {index}")
        return tuple(weight_map)

    safetensors_files = sorted(root.glob("*.safetensors"))
    if len(safetensors_files) != 1:
        raise FileNotFoundError(
            f"expected model.safetensors.index.json or one safetensors file in {root}"
        )
    from safetensors import safe_open

    with safe_open(safetensors_files[0], framework="pt", device="cpu") as handle:
        return tuple(handle.keys())


def build_qwen35_text_key_mapping(
    model_path: str | Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build exact source-to-target mappings for the Qwen3.5 text tower."""
    keys = checkpoint_keys(model_path)
    prefix = "model.language_model."
    mapping = {
        key: "model." + key[len(prefix) :]
        for key in keys
        if key.startswith(prefix)
    }
    if "lm_head.weight" in keys:
        mapping["lm_head.weight"] = "lm_head.weight"
    if not mapping:
        raise ValueError(
            f"Qwen3.5 checkpoint {model_path} has no model.language_model.* weights"
        )
    targets = tuple(mapping.values())
    if len(set(targets)) != len(targets):
        raise ValueError("Qwen3.5 text key mapping contains duplicate target keys")

    classified = Counter()
    unknown = []
    for key in keys:
        if key.startswith(prefix) or key == "lm_head.weight":
            classified["text"] += 1
        elif key.startswith("model.visual."):
            classified["vision"] += 1
        elif key.startswith("mtp."):
            classified["mtp"] += 1
        else:
            classified["unknown"] += 1
            unknown.append(key)
    if unknown:
        raise ValueError(
            "unclassified keys in Qwen3.5 multimodal checkpoint: "
            f"{unknown[:20]}{' ...' if len(unknown) > 20 else ''}"
        )
    return mapping, {
        "checkpoint_keys": len(keys),
        "mapped_text_keys": classified["text"],
        "excluded_vision_keys": classified["vision"],
        "excluded_mtp_keys": classified["mtp"],
        "mapping_sha256": hashlib.sha256(
            json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def validate_loading_info(loading_info: dict[str, Any]) -> None:
    """Reject every missing, unexpected, mismatched, or loader error entry."""
    problems = {
        key: loading_info.get(key)
        for key in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
        if loading_info.get(key)
    }
    if problems:
        raise RuntimeError(
            "Qwen3.5 text-only checkpoint loading was not exact: "
            + json.dumps(problems, ensure_ascii=False, default=str)
        )


def validate_qwen35_text_structure(
    model,
    text_config,
    *,
    require_production_shape: bool = True,
) -> dict[str, Any]:
    """Validate real module paths, layer types, and expert parameter shapes."""
    text_model = getattr(model, "model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise TypeError("Qwen3_5MoeForCausalLM is missing model.layers")
    if len(layers) != text_config.num_hidden_layers:
        raise ValueError(
            f"layer count mismatch: model={len(layers)} config={text_config.num_hidden_layers}"
        )

    layer_types = []
    for index, layer in enumerate(layers):
        if type(layer).__name__ != "Qwen3_5MoeDecoderLayer":
            raise TypeError(
                f"model.layers.{index} is {type(layer).__name__}, expected "
                "Qwen3_5MoeDecoderLayer"
            )
        layer_type = getattr(layer, "layer_type", None)
        layer_types.append(layer_type)
        expected_type = text_config.layer_types[index]
        if layer_type != expected_type:
            raise ValueError(
                f"model.layers.{index}.layer_type={layer_type!r}, "
                f"expected {expected_type!r}"
            )
        attention_attr = (
            "self_attn" if layer_type == "full_attention" else "linear_attn"
        )
        if getattr(layer, attention_attr, None) is None:
            raise TypeError(f"model.layers.{index} is missing {attention_attr}")
        expert_weight = getattr(
            getattr(getattr(layer, "mlp", None), "experts", None),
            "gate_up_proj",
            None,
        )
        expected_shape = (
            text_config.num_experts,
            2 * text_config.moe_intermediate_size,
            text_config.hidden_size,
        )
        if not isinstance(expert_weight, torch.Tensor) or tuple(expert_weight.shape) != expected_shape:
            raise ValueError(
                f"model.layers.{index}.mlp.experts.gate_up_proj shape "
                f"{getattr(expert_weight, 'shape', None)} != {expected_shape}"
            )

    counts = Counter(layer_types)
    summary = {
        "num_layers": len(layers),
        "linear_attention_layers": counts["linear_attention"],
        "full_attention_layers": counts["full_attention"],
        "num_experts": text_config.num_experts,
        "num_experts_per_tok": text_config.num_experts_per_tok,
        "hidden_size": text_config.hidden_size,
    }
    if require_production_shape:
        expected = {
            "num_layers": 40,
            "linear_attention_layers": 30,
            "full_attention_layers": 10,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "hidden_size": 2048,
        }
        if summary != expected:
            raise ValueError(
                f"unexpected Qwen3.5-35B-A3B text structure: {summary}; expected {expected}"
            )
    return summary


def load_qwen35_text_model(
    model_path: str | Path,
    text_config,
    *,
    dtype: torch.dtype,
    device: str,
    require_production_shape: bool = True,
):
    """Load only ``Qwen3_5MoeForCausalLM`` from a multimodal checkpoint."""
    from transformers import Qwen3_5MoeForCausalLM

    mapping, mapping_summary = build_qwen35_text_key_mapping(model_path)
    model, loading_info = Qwen3_5MoeForCausalLM.from_pretrained(
        model_path,
        config=text_config,
        key_mapping=mapping,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map=None if device == "cpu" else device,
        output_loading_info=True,
    )
    if device == "cpu":
        model.to("cpu")
    validate_loading_info(loading_info)
    serializable_loading_info = {
        key: sorted(value) if isinstance(value, set) else value
        for key, value in loading_info.items()
    }
    structure = validate_qwen35_text_structure(
        model,
        text_config,
        require_production_shape=require_production_shape,
    )
    return model, {
        **mapping_summary,
        "structure": structure,
        "loading_info": serializable_loading_info,
        "model_class": type(model).__name__,
        "vision_loaded": False,
        "mtp_loaded": False,
    }
