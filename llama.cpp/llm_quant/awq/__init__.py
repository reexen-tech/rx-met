"""AWQ search (scale+clip). Copied/adapted from llm-awq."""
from .pre_quant import run_awq, apply_awq
__all__ = ["run_awq", "apply_awq"]
