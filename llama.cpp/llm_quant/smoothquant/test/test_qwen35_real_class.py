"""Transformers 5.5.1 Qwen3.5 real-class smoothing integration test."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from safetensors import safe_open
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTopKRouter,
)

from llm_quant.smoothquant.adapters import qwen35_calibration_targets, smooth_groups
from llm_quant.smoothquant.smooth import smooth_lm, smooth_ln_fcs
from llm_quant.smoothquant.scripts.diagnose_qwen35_router import (
    _adjust_direct_gamma_scale,
    _direct_gamma_probe_config,
    _apply_bf16_scoped_smoothing,
    _apply_direct_gamma_moe_smoothing,
    _apply_runtime_fp32_dynamic_smoothing,
    _apply_runtime_fp32_router_only,
    _install_runtime_fp32_group,
    _install_runtime_fp32_norm,
    _install_runtime_fp32_router,
)

from .test_qwen35_text_loading import _tiny_text_config


def _run_router_boundary(dtype: torch.dtype):
    torch.manual_seed(20260805)
    hidden_size = 32
    norm = Qwen3_5MoeRMSNorm(hidden_size).to(dtype)
    router = Qwen3_5MoeTopKRouter(
        SimpleNamespace(
            num_experts_per_tok=8,
            num_experts=256,
            hidden_size=hidden_size,
        )
    ).to(dtype)
    router.weight.data.normal_(0, 0.02)
    shared = [
        nn.Linear(hidden_size, 8, bias=False, dtype=dtype) for _ in range(2)
    ]
    for linear in shared:
        linear.weight.data.fill_(1)
    expert = nn.Parameter(torch.ones(2, 16, hidden_size, dtype=dtype))
    hidden = torch.randn(256, hidden_size, dtype=dtype)

    with torch.no_grad():
        logits_before, _, topk_before = router(norm(hidden))
        desired_scale = torch.logspace(
            math.log10(2.0), math.log10(18.0), hidden_size
        )
        alpha = 0.8
        smooth_ln_fcs(
            norm,
            shared,
            desired_scale.pow(1.0 / alpha),
            alpha,
            offset_one_norm=True,
            expert_weight=expert,
            compensation_weights=(router.weight,),
        )
        logits_after, _, topk_after = router(norm(hidden))
    return logits_before, logits_after, topk_before, topk_after


@pytest.mark.xfail(
    strict=True,
    reason="offset-one RMSNorm and downstream scales cannot be reparameterized exactly in BF16",
)
def test_real_qwen35_router_topk_is_preserved_after_bf16_writeback() -> None:
    fp32_before, fp32_after, fp32_topk_before, fp32_topk_after = (
        _run_router_boundary(torch.float32)
    )
    torch.testing.assert_close(fp32_after, fp32_before, rtol=1e-5, atol=1e-7)
    assert torch.equal(fp32_topk_after, fp32_topk_before)

    _, _, bf16_topk_before, bf16_topk_after = _run_router_boundary(torch.bfloat16)
    assert torch.equal(bf16_topk_after, bf16_topk_before)


def test_runtime_fp32_scale_preserves_ideal_router_without_promoting_parameters(
) -> None:
    torch.manual_seed(20260806)
    hidden_size = 32
    norm = Qwen3_5MoeRMSNorm(hidden_size).to(torch.bfloat16)
    router = Qwen3_5MoeTopKRouter(
        SimpleNamespace(
            num_experts_per_tok=8,
            num_experts=256,
            hidden_size=hidden_size,
        )
    ).to(torch.bfloat16)
    router.weight.data.normal_(0, 0.02)
    hidden = torch.randn(256, hidden_size, dtype=torch.bfloat16)
    scale = torch.logspace(
        math.log10(2.0), math.log10(18.0), hidden_size
    )
    norm_before = norm.weight.detach().clone()
    router_before = router.weight.detach().clone()

    with torch.no_grad():
        ideal_norm = norm._norm(hidden.float()) * (1.0 + norm.weight.float())
        ideal_logits = F.linear(ideal_norm, router.weight.float()).softmax(dim=-1)
        ideal_topk = ideal_logits.topk(router.top_k, dim=-1).indices

        _install_runtime_fp32_norm(norm, scale)
        _install_runtime_fp32_router(router, scale)
        actual_logits, _, actual_topk = router(norm(hidden))

    torch.testing.assert_close(actual_logits, ideal_logits, rtol=1e-5, atol=1e-7)
    assert torch.equal(actual_topk, ideal_topk)
    assert norm.weight.dtype == torch.bfloat16
    assert router.weight.dtype == torch.bfloat16
    assert torch.equal(norm.weight, norm_before)
    assert torch.equal(router.weight, router_before)


def test_runtime_fp32_moe_group_runs_with_unchanged_bf16_parameters() -> None:
    torch.manual_seed(20260806)
    model = Qwen3_5MoeForCausalLM(_tiny_text_config()).to(torch.bfloat16).eval()
    layer = model.model.layers[0]
    group = smooth_groups(layer, "model.layers.0")[1]
    hidden = torch.randn(1, 4, model.config.hidden_size, dtype=torch.bfloat16)
    scale = torch.linspace(2.0, 8.0, model.config.hidden_size)
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in layer.mlp.named_parameters()
    }

    _install_runtime_fp32_group(layer, group, scale)
    with torch.no_grad():
        output = layer.mlp(group.norm(hidden))

    assert output.shape == hidden.shape
    assert output.dtype == torch.bfloat16
    for name, parameter in layer.mlp.named_parameters():
        assert parameter.dtype == torch.bfloat16
        assert torch.equal(parameter, parameters_before[name])


def _diagnostic_act_scales(model) -> dict[str, torch.Tensor]:
    act_scales = {}
    for index, layer in enumerate(model.model.layers):
        for group_index, group in enumerate(
            smooth_groups(layer, f"model.layers.{index}")
        ):
            act_scales[group.act_scales_key] = torch.linspace(
                0.4 + 0.1 * group_index,
                1.8 + 0.1 * group_index,
                model.config.hidden_size,
            )
    return act_scales


def test_bf16_moe_only_keeps_attention_groups_bitwise_unchanged() -> None:
    torch.manual_seed(20260806)
    model = Qwen3_5MoeForCausalLM(_tiny_text_config()).to(torch.bfloat16).eval()
    attention_before = {}
    moe_router_before = {}
    for index, layer in enumerate(model.model.layers):
        attention = smooth_groups(layer, f"model.layers.{index}")[0]
        attention_before[index] = {
            "norm": attention.norm.weight.detach().clone(),
            "fcs": [fc.weight.detach().clone() for fc in attention.fcs],
        }
        moe_router_before[index] = layer.mlp.gate.weight.detach().clone()

    applied = _apply_bf16_scoped_smoothing(
        model, _diagnostic_act_scales(model), alpha=0.65, scope="moe"
    )

    assert applied == {
        "layers": 2,
        "scope": "moe",
        "smoothed_groups": 2,
        "identity_groups": 2,
    }
    for index, layer in enumerate(model.model.layers):
        attention = smooth_groups(layer, f"model.layers.{index}")[0]
        assert torch.equal(attention.norm.weight, attention_before[index]["norm"])
        assert all(
            torch.equal(fc.weight, expected)
            for fc, expected in zip(attention.fcs, attention_before[index]["fcs"])
        )
        assert not torch.equal(layer.mlp.gate.weight, moe_router_before[index])


def test_runtime_fp32_moe_only_keeps_unselected_attention_at_identity() -> None:
    torch.manual_seed(20260806)
    model = Qwen3_5MoeForCausalLM(_tiny_text_config()).to(torch.bfloat16).eval()
    attention_before = {}
    for index, layer in enumerate(model.model.layers):
        attention = smooth_groups(layer, f"model.layers.{index}")[0]
        attention_before[index] = {
            "norm": attention.norm.weight.detach().clone(),
            "forwards": [fc.forward.__func__ for fc in attention.fcs],
            "fcs": [fc.weight.detach().clone() for fc in attention.fcs],
        }

    applied, moe_scales = _apply_runtime_fp32_dynamic_smoothing(
        model, _diagnostic_act_scales(model), alpha=0.65, scope="moe"
    )

    assert applied == {
        "layers": 2,
        "runtime_fp32_groups": 2,
        "identity_groups": 2,
    }
    assert sorted(moe_scales) == ["model.layers.0", "model.layers.1"]
    for index, layer in enumerate(model.model.layers):
        attention = smooth_groups(layer, f"model.layers.{index}")[0]
        assert torch.equal(attention.norm.weight, attention_before[index]["norm"])
        assert all(
            fc.forward.__func__ is expected
            for fc, expected in zip(attention.fcs, attention_before[index]["forwards"])
        )
        assert all(
            torch.equal(fc.weight, expected)
            for fc, expected in zip(attention.fcs, attention_before[index]["fcs"])
        )


def test_direct_gamma_moe_only_keeps_attention_and_norm_parameters_unchanged() -> None:
    torch.manual_seed(20260806)
    model = Qwen3_5MoeForCausalLM(_tiny_text_config()).to(torch.bfloat16).eval()
    attention_before = {}
    moe_norm_before = {}
    router_before = {}
    for index, layer in enumerate(model.model.layers):
        attention = smooth_groups(layer, f"model.layers.{index}")[0]
        attention_before[index] = {
            "norm": attention.norm.weight.detach().clone(),
            "fcs": [fc.weight.detach().clone() for fc in attention.fcs],
        }
        moe_norm_before[index] = (
            layer.post_attention_layernorm.weight.detach().clone()
        )
        router_before[index] = layer.mlp.gate.weight.detach().clone()

    applied, scales = _apply_direct_gamma_moe_smoothing(
        model,
        _diagnostic_act_scales(model),
        alpha=0.65,
        gamma_dtype=torch.bfloat16,
        weight_materialization="target_dtype_multiply",
        scale_policy="requested",
    )

    assert applied["layers"] == 2
    assert applied["direct_gamma_groups"] == 2
    assert applied["identity_groups"] == 2
    assert sorted(scales) == ["model.layers.0", "model.layers.1"]
    for index, layer in enumerate(model.model.layers):
        attention = smooth_groups(layer, f"model.layers.{index}")[0]
        assert torch.equal(attention.norm.weight, attention_before[index]["norm"])
        assert all(
            torch.equal(fc.weight, expected)
            for fc, expected in zip(attention.fcs, attention_before[index]["fcs"])
        )
        assert torch.equal(
            layer.post_attention_layernorm.weight, moe_norm_before[index]
        )
        assert layer.post_attention_layernorm._sq_direct_gamma.dtype == (
            torch.bfloat16
        )
        assert not torch.equal(layer.mlp.gate.weight, router_before[index])


def test_direct_gamma_can_materialize_only_router_in_fp32() -> None:
    torch.manual_seed(20260806)
    model = Qwen3_5MoeForCausalLM(_tiny_text_config()).to(torch.bfloat16).eval()
    router_before = [
        layer.mlp.gate.weight.detach().clone() for layer in model.model.layers
    ]

    _, scales = _apply_direct_gamma_moe_smoothing(
        model,
        _diagnostic_act_scales(model),
        alpha=0.65,
        gamma_dtype=torch.float32,
        weight_materialization="fp32_product",
        scale_policy="requested",
        router_materialization="fp32",
    )

    hidden = torch.randn(
        1, 4, model.config.hidden_size, dtype=torch.bfloat16
    )
    for index, layer in enumerate(model.model.layers):
        assert torch.equal(layer.mlp.gate.weight, router_before[index])
        materialized = layer.mlp.gate._sq_materialized_fp32_weight
        assert materialized.dtype == torch.float32
        torch.testing.assert_close(
            materialized,
            router_before[index].float()
            * scales[f"model.layers.{index}"].view(1, -1),
            rtol=0,
            atol=0,
        )
        with torch.no_grad():
            norm_output = layer.post_attention_layernorm(hidden)
            router_output = layer.mlp.gate(norm_output)
            expected_probabilities = F.softmax(
                F.linear(norm_output.float(), materialized),
                dtype=torch.float,
                dim=-1,
            ).reshape(-1, layer.mlp.gate.num_experts)
        torch.testing.assert_close(
            router_output[0], expected_probabilities, rtol=0, atol=0
        )
        assert router_output[2].shape == (4, layer.mlp.gate.top_k)
        assert scales[f"model.layers.{index}"].shape == (
            model.config.hidden_size,
        )

    source_control = Qwen3_5MoeForCausalLM(
        _tiny_text_config()
    ).to(torch.bfloat16).eval()
    applied = _apply_runtime_fp32_router_only(source_control)
    assert applied == {"layers": 2, "routers": 2}
    assert all(
        layer.mlp.gate._sq_materialized_fp32_weight.dtype == torch.float32
        for layer in source_control.model.layers
    )
    assert all(
        torch.equal(
            layer.mlp.gate._sq_materialized_fp32_weight,
            layer.mlp.gate.weight.float(),
        )
        for layer in source_control.model.layers
    )



def test_direct_gamma_can_materialize_moe_norm_as_fp32_offset() -> None:
    torch.manual_seed(20260806)
    model = Qwen3_5MoeForCausalLM(_tiny_text_config()).to(torch.bfloat16).eval()
    norm_before = [
        layer.post_attention_layernorm.weight.detach().clone()
        for layer in model.model.layers
    ]

    _, scales = _apply_direct_gamma_moe_smoothing(
        model,
        _diagnostic_act_scales(model),
        alpha=0.65,
        gamma_dtype=torch.float32,
        weight_materialization="target_dtype_multiply",
        scale_policy="power_of_two",
        norm_materialization="fp32_offset",
    )

    hidden = torch.randn(1, 4, model.config.hidden_size, dtype=torch.bfloat16)
    for index, layer in enumerate(model.model.layers):
        norm = layer.post_attention_layernorm
        assert norm.weight.dtype == torch.float32
        assert not hasattr(norm, "_sq_direct_gamma")
        expected_offset = (
            (1.0 + norm_before[index].float())
            / scales[f"model.layers.{index}"]
            - 1.0
        )
        torch.testing.assert_close(norm.weight, expected_offset, rtol=0, atol=0)
        attention = smooth_groups(layer, f"model.layers.{index}")[0]
        assert attention.norm.weight.dtype == torch.bfloat16
        with torch.no_grad():
            output = norm(hidden)
        assert output.dtype == torch.bfloat16



def test_fp32_offset_norm_round_trips_through_hf_safetensors(tmp_path) -> None:
    torch.manual_seed(20260806)
    model = Qwen3_5MoeForCausalLM(_tiny_text_config()).to(torch.bfloat16).eval()
    _apply_direct_gamma_moe_smoothing(
        model,
        _diagnostic_act_scales(model),
        alpha=0.65,
        gamma_dtype=torch.float32,
        weight_materialization="target_dtype_multiply",
        scale_policy="power_of_two",
        norm_materialization="fp32_offset",
    )
    input_ids = torch.randint(
        0, model.config.vocab_size, (1, 4)
    )
    with torch.inference_mode():
        logits_before = model(input_ids=input_ids, use_cache=False).logits

    with torch.inference_mode():
        logits_repeated = model(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(
        logits_repeated, logits_before, rtol=0, atol=0
    )

    model.save_pretrained(tmp_path, safe_serialization=True)
    with safe_open(tmp_path / "model.safetensors", framework="pt") as checkpoint:
        stored_norms = [
            checkpoint.get_tensor(key)
            for key in checkpoint.keys()
            if key.endswith("post_attention_layernorm.weight")
        ]
    assert len(stored_norms) == len(model.model.layers)
    assert all(weight.dtype == torch.float32 for weight in stored_norms)

    default_reloaded = Qwen3_5MoeForCausalLM.from_pretrained(
        tmp_path,
        local_files_only=True,
    ).eval()
    assert all(
        layer.post_attention_layernorm.weight.dtype == torch.bfloat16
        for layer in default_reloaded.model.layers
    )

    previous_policy = Qwen3_5MoeForCausalLM._keep_in_fp32_modules_strict
    Qwen3_5MoeForCausalLM._keep_in_fp32_modules_strict = ["post_attention_layernorm"]
    try:
        reloaded = Qwen3_5MoeForCausalLM.from_pretrained(
            tmp_path,
            local_files_only=True,
            dtype=torch.bfloat16,
        ).eval()
    finally:
        Qwen3_5MoeForCausalLM._keep_in_fp32_modules_strict = previous_policy
    with torch.inference_mode():
        logits_after = reloaded(input_ids=input_ids, use_cache=False).logits

    reloaded_state = reloaded.state_dict()
    for name, expected in model.state_dict().items():
        assert reloaded_state[name].dtype == expected.dtype
        torch.testing.assert_close(
            reloaded_state[name], expected, rtol=0, atol=0
        )

    for index, layer in enumerate(reloaded.model.layers):
        assert layer.post_attention_layernorm.weight.dtype == torch.float32
        attention = smooth_groups(layer, f"model.layers.{index}")[0]
        assert attention.norm.weight.dtype == torch.bfloat16
        assert layer.mlp.gate.weight.dtype == torch.bfloat16
    torch.testing.assert_close(
        logits_after, logits_before, rtol=0, atol=0
    )



def test_direct_gamma_scale_policies_are_explicit_and_deterministic() -> None:
    requested = torch.tensor([0.3, 0.9, 1.1, 3.7], dtype=torch.float32)

    unchanged = _adjust_direct_gamma_scale(requested, "requested", None)
    power_of_two = _adjust_direct_gamma_scale(requested, "power_of_two", None)
    shrunk = _adjust_direct_gamma_scale(requested, "log_shrink", 0.25)

    assert torch.equal(unchanged, requested)
    torch.testing.assert_close(
        power_of_two.log2(), power_of_two.log2().round(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        shrunk, requested.pow(0.25), rtol=1e-6, atol=1e-7
    )
    config = _direct_gamma_probe_config(
        "power-of-two-fp32-gamma"
    )
    assert config == {
        "gamma_dtype": torch.float32,
        "weight_materialization": "target_dtype_multiply",
        "scale_policy": "power_of_two",
        "scale_strength": None,
    }

    offset_config = _direct_gamma_probe_config(
        "power-of-two-fp32-offset"
    )
    assert offset_config["norm_materialization"] == "fp32_offset"


def test_real_qwen35_layers_router_and_3d_experts_are_equivalent() -> None:
    torch.manual_seed(23)
    model = Qwen3_5MoeForCausalLM(_tiny_text_config()).float().eval()
    targets = qwen35_calibration_targets(model)
    assert targets is not None
    assert len(targets.act_module_names) == 4
    assert len(targets.routers) == 2
    assert all(target.num_experts == 4 for target in targets.routers)

    hidden = torch.randn(1, 4, model.config.hidden_size)
    boundary_outputs = []
    router_topk = []
    untouched = []
    act_scales = {}
    with torch.no_grad():
        for index, layer in enumerate(model.model.layers):
            groups = smooth_groups(layer, f"model.layers.{index}")
            assert groups[1].expert_weight.ndim == 3
            assert groups[1].expert_weight.shape[-1] == model.config.hidden_size
            boundary_outputs.append(
                [fc(groups[0].norm(hidden)) for fc in groups[0].fcs]
            )
            moe_input = groups[1].norm(hidden)
            boundary_outputs.append(layer.mlp(moe_input))
            router_topk.append(layer.mlp.gate(moe_input)[2])
            untouched.append(
                {
                    "expert_down": layer.mlp.experts.down_proj.detach().clone(),
                    "shared_down": layer.mlp.shared_expert.down_proj.weight.detach().clone(),
                    "attn_out": (
                        layer.self_attn.o_proj.weight.detach().clone()
                        if layer.layer_type == "full_attention"
                        else layer.linear_attn.out_proj.weight.detach().clone()
                    ),
                }
            )
            for group_index, group in enumerate(groups):
                act_scales[group.act_scales_key] = torch.linspace(
                    0.4 + 0.1 * group_index,
                    1.8 + 0.1 * group_index,
                    model.config.hidden_size,
                )

    assert smooth_lm(model, act_scales, alpha=0.85) == (2, 4)

    with torch.no_grad():
        output_index = 0
        for index, layer in enumerate(model.model.layers):
            groups = smooth_groups(layer, f"model.layers.{index}")
            for actual, expected in zip(
                [fc(groups[0].norm(hidden)) for fc in groups[0].fcs],
                boundary_outputs[output_index],
            ):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            output_index += 1
            moe_input = groups[1].norm(hidden)
            torch.testing.assert_close(
                layer.mlp(moe_input),
                boundary_outputs[output_index],
                rtol=3e-5,
                atol=3e-6,
            )
            output_index += 1
            assert torch.equal(layer.mlp.gate(moe_input)[2], router_topk[index])
            torch.testing.assert_close(
                layer.mlp.experts.down_proj, untouched[index]["expert_down"]
            )
            torch.testing.assert_close(
                layer.mlp.shared_expert.down_proj.weight,
                untouched[index]["shared_down"],
            )
            actual_attn_out = (
                layer.self_attn.o_proj.weight
                if layer.layer_type == "full_attention"
                else layer.linear_attn.out_proj.weight
            )
            torch.testing.assert_close(actual_attn_out, untouched[index]["attn_out"])


def test_formal_qwen35_policy_is_moe_only_power2_with_fp32_offsets() -> None:
    torch.manual_seed(20260806)
    model = Qwen3_5MoeForCausalLM(_tiny_text_config()).to(torch.bfloat16).eval()
    attention_before = [
        layer.input_layernorm.weight.detach().clone()
        for layer in model.model.layers
    ]

    layers, groups, stats = smooth_lm(
        model,
        _diagnostic_act_scales(model),
        alpha=0.65,
        return_group_stats=True,
        scope="moe",
        scale_policy="power_of_two",
        norm_materialization="fp32_offset",
    )

    assert (layers, groups) == (2, 2)
    assert all(item["kind"] == "moe" for item in stats)
    assert all(item["scale_log2_rounding_error_max"] == 0 for item in stats)
    for layer, expected in zip(model.model.layers, attention_before):
        assert torch.equal(layer.input_layernorm.weight, expected)
        assert layer.post_attention_layernorm.weight.dtype == torch.float32
        assert layer.mlp.gate.weight.dtype == torch.bfloat16
