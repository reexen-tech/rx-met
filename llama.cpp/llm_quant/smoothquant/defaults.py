"""Recommended SmoothQuant ``alpha`` per model family."""

# === REEX_SMOOTHQUANT BEGIN: default alpha lookup by HF model_type ===

from __future__ import annotations

DEFAULT_ALPHA = 0.85

# 0.85 for llama comes from the reference repo's ppl_eval.sh; the other values
# have no official reference and are conservative starting points.
_ALPHA_BY_MODEL_TYPE = {
    "llama": 0.85,
    "qwen2": 0.85,
    "qwen3": 0.85,
    "qwen3_5_moe": 0.85,
    "qwen3_5_moe_text": 0.85,
    "mistral": 0.80,
    "mixtral": 0.80,
    "falcon": 0.70,
}


def default_alpha(model_type: str | None) -> float:
    """Recommended alpha for an HF ``config.model_type``; CLI may override."""
    if not model_type:
        return DEFAULT_ALPHA
    return _ALPHA_BY_MODEL_TYPE.get(model_type.lower(), DEFAULT_ALPHA)

# === REEX_SMOOTHQUANT END ===
