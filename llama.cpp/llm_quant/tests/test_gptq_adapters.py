from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from llm_quant.gptq.adapters import get_model_adapter


class _TextTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.rotary_emb = nn.Identity()
        self.layers = nn.ModuleList([nn.Identity()])


class Qwen3_5MoeForConditionalGeneration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        text_config = SimpleNamespace(use_cache=True, hidden_size=4)
        self.config = SimpleNamespace(text_config=text_config, use_cache=True)
        self.model = nn.Module()
        self.model.language_model = _TextTower()
        self.model.visual = nn.Linear(4, 4)


def test_qwen35_adapter_uses_nested_text_tower_and_config() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    visual = model.model.visual
    adapter = get_model_adapter(model)

    assert adapter.text_model is model.model.language_model
    assert adapter.text_config is model.config.text_config
    assert adapter.layers is model.model.language_model.layers
    assert adapter.tensor_prefix == "model.language_model.layers"

    changed = adapter.set_use_cache(False)
    assert model.config.use_cache is False
    assert model.config.text_config.use_cache is False
    adapter.move_text_inputs("cpu")
    assert model.model.visual is visual

    for config, original in changed:
        config.use_cache = original
    assert model.config.use_cache is True
    assert model.config.text_config.use_cache is True


def test_qwen35_adapter_keeps_shared_expert_router_gate_unquantized() -> None:
    model = Qwen3_5MoeForConditionalGeneration()
    adapter = get_model_adapter(model)
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(64, 64, bias=False)
    layer.mlp = nn.Module()
    layer.mlp.shared_expert_gate = nn.Linear(64, 1, bias=False)

    linears = adapter.named_linears(layer)
    assert "self_attn.q_proj" in linears
    assert "mlp.shared_expert_gate" not in linears
