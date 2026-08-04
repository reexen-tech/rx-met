"""Fixed configuration and export metadata for Q64 GPTQ."""

from __future__ import annotations

from typing import Any

from .formats import format_for_bits


def fixed_gptq_config(
    *,
    model_family: str = "Qwen2.5-0.5B",
    fake_quant_dtype: str = "float16",
    expert_hessian_weighting: str = "route_squared",
    bits: int = 4,
) -> dict[str, Any]:
    block_format = format_for_bits(bits)
    return {
        "algorithm": "gptq",
        "format_target": block_format.name,
        "execution": f"W{bits}A8",
        "activation_type": "Q8_0_64",
        "bits": bits,
        "group_size": block_format.group_size,
        "lazy_block_size": 128,
        "damp_percent": 0.01,
        "act_order": False,
        "static_groups": False,
        "true_sequential": False,
        "calib_data": "pileval",
        "n_samples": 128,
        "seqlen": 512,
        "fake_quant_dtype": fake_quant_dtype,
        "hessian_dtype": "float32",
        "expert_hessian_weighting": expert_hessian_weighting,
        "sidecar_version": 3,
        "model_family": model_family,
    }
