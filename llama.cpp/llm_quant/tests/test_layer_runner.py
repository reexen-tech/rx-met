from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

import llm_quant.gptq.layer_runner as runner


class _TinyRoutedExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2, 128, 64) * 0.05)
        self.down_proj = nn.Parameter(torch.randn(2, 64, 64) * 0.05)
        self.act_fn = F.silu

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for expert_index in range(2):
            token_indices = torch.where(top_k_index[:, 0] == expert_index)[0]
            if token_indices.numel() == 0:
                continue
            selected = hidden_states[token_indices]
            gate, up = F.linear(
                selected, self.gate_up_proj[expert_index]
            ).chunk(2, dim=-1)
            expert_output = F.linear(
                self.act_fn(gate) * up, self.down_proj[expert_index]
            )
            output[token_indices] = (
                expert_output * top_k_weights[token_indices, :1]
            )
        return output


class _TinyMoeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(64, 64, bias=False)
        self.mlp = nn.Module()
        self.mlp.experts = _TinyRoutedExperts()

    def forward(self, hidden_states: torch.Tensor, **_kwargs) -> torch.Tensor:
        projected = self.proj(hidden_states)
        flat = projected.reshape(-1, 64)
        routes = torch.zeros(flat.shape[0], 1, dtype=torch.long)
        weights = torch.ones(flat.shape[0], 1)
        routed = self.mlp.experts(flat, routes, weights)
        return routed.reshape_as(projected)


class _TinyTextTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(128, 64)
        self.rotary_emb = nn.Identity()
        self.layers = nn.ModuleList([_TinyMoeLayer()])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Qwen3_5MoeForConditionalGeneration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        text_config = type("TextConfig", (), {"use_cache": True})()
        self.config = type(
            "Config", (), {"use_cache": True, "text_config": text_config}
        )()
        self.model = nn.Module()
        self.model.language_model = _TinyTextTower()
        self.model.visual = nn.Linear(64, 64)


def test_layer_runner_quantizes_and_propagates_quantized_output(
    monkeypatch,
) -> None:
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(config).eval()
    samples = [
        torch.arange(8, dtype=torch.long).reshape(1, 8),
        torch.arange(8, 16, dtype=torch.long).reshape(1, 8),
    ]
    monkeypatch.setattr(runner, "_validate_model", lambda _model: None)
    monkeypatch.setattr(
        runner,
        "get_calib_dataset",
        lambda **_kwargs: samples,
    )
    monkeypatch.setattr(runner, "CALIBRATION_SAMPLES", len(samples))

    result = runner.run_gptq(model, tokenizer=object(), device="cpu")
    suffixes = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    expected_names = {
        f"model.layers.{layer}.{suffix}"
        for layer in range(2)
        for suffix in suffixes
    }
    assert set(result.tensor_data) == expected_names
    assert result.calibration_sequences == 2
    assert len(result.layer_stats) == 2
    for tensors in result.tensor_data.values():
        assert tensors["codes"].dtype == torch.int8
        assert tensors["scales"].dtype == torch.float16
        assert tensors["packed"].dtype == torch.uint8


def test_mvp_model_validation_rejects_other_qwen_sizes() -> None:
    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = Qwen2ForCausalLM(config)
    try:
        runner._validate_model(model)
    except NotImplementedError as exc:
        assert "Qwen2.5-0.5B" in str(exc)
    else:
        raise AssertionError("non-MVP Qwen2 configuration was accepted")


def test_qwen35_conditional_quantizes_only_text_tower(monkeypatch) -> None:
    model = Qwen3_5MoeForConditionalGeneration().eval()
    visual_weight = model.model.visual.weight.detach().clone()
    samples = [torch.arange(8, dtype=torch.long).reshape(1, 8)]
    monkeypatch.setattr(runner, "get_calib_dataset", lambda **_kwargs: samples)
    monkeypatch.setattr(runner, "CALIBRATION_SAMPLES", len(samples))

    result = runner.run_gptq(model, tokenizer=object(), device="cpu")

    expected = {
        "model.language_model.layers.0.proj.weight",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    }
    assert set(result.tensor_data) == expected
    assert (
        result.layer_stats[0]["routed_experts"]["mlp.experts"]["experts"][1][
            "gate_up_proj"
        ]["fallback_reason"]
        == "zero_samples"
    )
    torch.testing.assert_close(model.model.visual.weight, visual_weight)
