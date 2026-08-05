# === REEX_SMOOTHQUANT BEGIN: equivalence tests for the smoothing transform ===
"""Tests for the mathematical equivalence of the smoothing transform."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from llm_quant.smoothquant.smooth import smooth_ln_fcs, smooth_lm


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def _act_scales(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(dim, generator=generator) * 1.5 + 0.5


def test_smooth_ln_fcs_preserves_rmsnorm_forward() -> None:
    torch.manual_seed(0)
    norm = _RMSNorm(32)
    norm.weight.data.uniform_(0.5, 1.5)
    fcs = [nn.Linear(32, 16, bias=False), nn.Linear(32, 16, bias=True)]
    x = torch.randn(4, 7, 32)
    expected = [fc(norm(x)) for fc in fcs]

    smooth_ln_fcs(norm, fcs, _act_scales(32, 1), alpha=0.85)

    for fc, want in zip(fcs, expected):
        torch.testing.assert_close(fc(norm(x)), want, rtol=1e-4, atol=1e-4)


def test_smooth_ln_fcs_divides_layernorm_bias() -> None:
    torch.manual_seed(0)
    norm = nn.LayerNorm(32)
    norm.bias.data.uniform_(-1, 1)
    fc = nn.Linear(32, 16)
    x = torch.randn(4, 7, 32)
    expected = fc(norm(x))
    bias_before = norm.bias.detach().clone()

    scales = smooth_ln_fcs(norm, fc, _act_scales(32, 2), alpha=0.85)

    torch.testing.assert_close(norm.bias, bias_before / scales)
    torch.testing.assert_close(fc(norm(x)), expected, rtol=1e-4, atol=1e-4)


def test_smooth_ln_fcs_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        smooth_ln_fcs(_RMSNorm(32), nn.Linear(32, 8), torch.ones(16))


def _tiny_qwen2(**overrides) -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        use_cache=False,
        **overrides,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config).eval()


def _qwen2_act_scales(model: Qwen2ForCausalLM) -> dict[str, torch.Tensor]:
    hidden = model.config.hidden_size
    return {
        f"model.layers.{i}.{suffix}": _act_scales(hidden, i * 10 + j)
        for i in range(model.config.num_hidden_layers)
        for j, suffix in enumerate(("self_attn.q_proj", "mlp.gate_proj"))
    }


def test_smooth_lm_keeps_tiny_qwen2_logits() -> None:
    model = _tiny_qwen2()
    input_ids = torch.arange(16, dtype=torch.long).reshape(1, 16)
    with torch.no_grad():
        expected = model(input_ids).logits
    norm_before = model.model.layers[0].input_layernorm.weight.detach().clone()

    n_layers, n_groups = smooth_lm(model, _qwen2_act_scales(model), alpha=0.85)

    assert (n_layers, n_groups) == (2, 4)
    assert not torch.allclose(model.model.layers[0].input_layernorm.weight, norm_before)
    with torch.no_grad():
        torch.testing.assert_close(
            model(input_ids).logits, expected, rtol=1e-3, atol=1e-3
        )


def test_smooth_lm_reports_missing_act_scale() -> None:
    model = _tiny_qwen2()
    scales = _qwen2_act_scales(model)
    del scales["model.layers.1.mlp.gate_proj"]
    with pytest.raises(KeyError, match="model.layers.1.mlp.gate_proj"):
        smooth_lm(model, scales, alpha=0.85)


def test_smooth_lm_rejects_model_without_decoder_layers() -> None:
    with pytest.raises(NotImplementedError, match="no supported decoder layer"):
        smooth_lm(nn.Sequential(nn.Linear(4, 4)), {}, alpha=0.85)

# === REEX_SMOOTHQUANT END ===
