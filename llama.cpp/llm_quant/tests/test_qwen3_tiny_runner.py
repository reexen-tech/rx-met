from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import llm_quant.gptq.layer_runner as runner


def test_tiny_qwen3_runs_dense_text_gptq(monkeypatch) -> None:
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_cache=False,
    )
    model = Qwen3ForCausalLM(config).eval()
    samples = [
        torch.arange(8, dtype=torch.long).reshape(1, 8),
        torch.arange(8, 16, dtype=torch.long).reshape(1, 8),
    ]
    monkeypatch.setattr(runner, "get_calib_dataset", lambda **_kwargs: samples)
    monkeypatch.setattr(runner, "CALIBRATION_SAMPLES", len(samples))

    result = runner.run_gptq(
        model,
        tokenizer=object(),
        device="cpu",
        packed_only=True,
    )

    expected = {
        f"model.layers.0.{name}.weight"
        for name in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }
    assert set(result.tensor_data) == expected
    assert all(set(entry) == {"packed", "shape"} for entry in result.tensor_data.values())
