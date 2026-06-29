"""aimet_llama: JSON-driven LLM quantization pipeline on top of llama.cpp.

Wraps the upstream `llama-quantize`, `llama-imatrix`, `llama-perplexity`
binaries with an AIMET v2 - style JSON configuration. The goal of this
P0 module is **experiment reproducibility / auditability / versioning**:
one JSON file replaces the ad-hoc shell scripts that drive llama.cpp.

Typical usage::

    from aimet_llama import LLMQuantPipeline

    pipe = LLMQuantPipeline.from_json("examples/qwen3_q4_0.json")
    pipe.run()

Or from CLI::

    python -m aimet_llama.pipeline examples/qwen3_q4_0.json
"""

from .schema import (
    DEFAULT_CONFIG,
    load_config,
    resolve_config,
    validate_config,
)
from .cli import (
    build_quantize_cmd,
    build_imatrix_cmd,
    build_perplexity_cmd,
)
from .pipeline import LLMQuantPipeline

__all__ = [
    "LLMQuantPipeline",
    "DEFAULT_CONFIG",
    "load_config",
    "resolve_config",
    "validate_config",
    "build_quantize_cmd",
    "build_imatrix_cmd",
    "build_perplexity_cmd",
]

__version__ = "0.1.0"
