"""Fixed configuration and export metadata for Q4_0_64 GPTQ."""

from __future__ import annotations

from typing import Any


def fixed_gptq_config(
    *,
    model_family: str = "Qwen2.5-0.5B",
    fake_quant_dtype: str = "float16",
) -> dict[str, Any]:
    return {
        "algorithm": "gptq",
        "format_target": "Q4_0_64",
        "execution": "W4A8",
        "activation_type": "Q8_0_64",
        "bits": 4,
        "group_size": 64,
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
        "model_family": model_family,
    }
