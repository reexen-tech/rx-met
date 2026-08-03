"""Duck-typed access to the supported Qwen text towers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class QwenGPTQAdapter:
    """The text-only pieces needed by layer-sequential GPTQ."""

    model: nn.Module
    text_model: nn.Module
    text_config: Any
    tensor_prefix: str

    @property
    def layers(self) -> Any:
        return self.text_model.layers

    def move_text_inputs(self, device: torch.device | str) -> None:
        self.text_model.embed_tokens = self.text_model.embed_tokens.to(device)
        rotary_emb = getattr(self.text_model, "rotary_emb", None)
        if rotary_emb is not None:
            self.text_model.rotary_emb = rotary_emb.to(device)

    def forward_text(self, input_ids: torch.Tensor) -> Any:
        return self.text_model(input_ids=input_ids)

    def named_linears(self, layer: nn.Module) -> dict[str, nn.Linear]:
        linears = {
            name: module
            for name, module in layer.named_modules()
            if isinstance(module, nn.Linear)
        }
        if type(self.model).__name__.startswith("Qwen3_5Moe"):
            linears.pop("mlp.shared_expert_gate", None)
        return linears

    def set_use_cache(self, value: bool) -> list[tuple[Any, Any]]:
        changed: list[tuple[Any, Any]] = []
        for config in (getattr(self.model, "config", None), self.text_config):
            if config is None or not hasattr(config, "use_cache"):
                continue
            if any(config is seen for seen, _ in changed):
                continue
            changed.append((config, config.use_cache))
            config.use_cache = value
        return changed


def get_model_adapter(model: nn.Module) -> QwenGPTQAdapter:
    """Resolve supported Qwen layouts without importing large model classes."""

    class_name = type(model).__name__
    if class_name == "Qwen2ForCausalLM":
        text_model = getattr(getattr(model, "model", None), "layers", None)
        if text_model is None:
            raise TypeError("Qwen2ForCausalLM is missing model.layers")
        return QwenGPTQAdapter(
            model=model,
            text_model=model.model,
            text_config=model.config,
            tensor_prefix="model.layers",
        )

    if class_name == "Qwen3ForCausalLM":
        text_model = getattr(model, "model", None)
        if text_model is None or not hasattr(text_model, "layers"):
            raise TypeError("Qwen3ForCausalLM is missing model.layers")
        return QwenGPTQAdapter(
            model=model,
            text_model=text_model,
            text_config=model.config,
            tensor_prefix="model.layers",
        )

    if class_name == "Qwen3_5MoeForConditionalGeneration":
        outer_model = getattr(model, "model", None)
        text_model = getattr(outer_model, "language_model", None)
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        if text_model is None or not hasattr(text_model, "layers"):
            raise TypeError(
                "Qwen3_5MoeForConditionalGeneration is missing "
                "model.language_model.layers"
            )
        if text_config is None:
            raise TypeError(
                "Qwen3_5MoeForConditionalGeneration config is missing text_config"
            )
        return QwenGPTQAdapter(
            model=model,
            text_model=text_model,
            text_config=text_config,
            tensor_prefix="model.language_model.layers",
        )

    if class_name == "Qwen3_5MoeForCausalLM":
        text_model = getattr(model, "model", None)
        if text_model is None or not hasattr(text_model, "layers"):
            raise TypeError("Qwen3_5MoeForCausalLM is missing model.layers")
        return QwenGPTQAdapter(
            model=model,
            text_model=text_model,
            text_config=model.config,
            tensor_prefix="model.layers",
        )

    raise NotImplementedError(f"unsupported GPTQ model: {class_name}")
