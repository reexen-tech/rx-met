"""Model-specific accessors used by the GPTQ layer runner."""

from .qwen import QwenGPTQAdapter, get_model_adapter

__all__ = ["QwenGPTQAdapter", "get_model_adapter"]
