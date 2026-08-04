from __future__ import annotations

import torch
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

import llm_quant.gptq.layer_runner as runner


def test_tiny_qwen35_moe_runs_text_gptq(monkeypatch) -> None:
    config = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention"],
        use_cache=False,
    )
    model = Qwen3_5MoeForCausalLM(config).eval()
    samples = [
        torch.arange(8, dtype=torch.long).reshape(1, 8),
        torch.arange(8, 16, dtype=torch.long).reshape(1, 8),
    ]
    monkeypatch.setattr(
        runner,
        "get_calib_dataset",
        lambda **_kwargs: samples,
    )
    monkeypatch.setattr(runner, "CALIBRATION_SAMPLES", len(samples))

    result = runner.run_gptq(
        model,
        tokenizer=object(),
        device="cpu",
        packed_only=True,
    )

    assert len(result.layer_stats) == 2
    assert any(name.endswith("mlp.experts.gate_up_proj") for name in result.tensor_data)
    assert all("mlp.shared_expert_gate" not in name for name in result.tensor_data)
    assert all(
        set(entry) == {"packed", "shape", "method"}
        for entry in result.tensor_data.values()
    )
