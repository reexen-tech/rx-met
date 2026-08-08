#!/usr/bin/env python3
"""Diagnose Qwen3.5 SmoothQuant router drift at every MoE boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_quant.smoothquant.adapters import is_decoder_layer, smooth_groups  # noqa: E402
from llm_quant.smoothquant.loading import load_qwen35_text_model  # noqa: E402
from llm_quant.smoothquant.runner import (  # noqa: E402
    _file_fingerprint,
    _source_model_fingerprint,
)
from llm_quant.smoothquant.scripts.compare_qwen35_bf16 import (  # noqa: E402
    _assert_no_calibration_overlap,
    _load_holdout,
    _load_smoothed,
    _model_input_device,
    _release_cuda_memory,
    _render_holdout,
)
from llm_quant.smoothquant.smooth import (  # noqa: E402
    chunked_absmax_last_dim,
    smooth_lm,
    smooth_ln_fcs,
)


class _BoundaryMetrics:
    def __init__(self) -> None:
        self.norm_abs_sum = 0.0
        self.norm_abs_max = 0.0
        self.norm_count = 0
        self.router_abs_sum = 0.0
        self.router_abs_max = 0.0
        self.router_count = 0
        self.projection_abs_sum = 0.0
        self.projection_abs_max = 0.0
        self.projection_count = 0
        self.projection_exact = 0
        self.position_same = 0
        self.position_count = 0
        self.set_exact = 0
        self.token_count = 0
        self.member_overlap = 0
        self.member_count = 0
        self.margins: list[torch.Tensor] = []
        self.changed_margins: list[torch.Tensor] = []

    def update(
        self,
        reference: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        scale: torch.Tensor,
    ) -> None:
        aligned_norm = target["norm_output"].float() * scale.view(1, 1, -1)
        norm_error = (aligned_norm - reference["norm_output"].float()).abs()
        self.norm_abs_sum += float(norm_error.sum().item())
        self.norm_abs_max = max(self.norm_abs_max, float(norm_error.max().item()))
        self.norm_count += norm_error.numel()

        router_error = (
            target["router_probabilities"].float()
            - reference["router_probabilities"].float()
        ).abs()
        self.router_abs_sum += float(router_error.sum().item())
        self.router_abs_max = max(
            self.router_abs_max, float(router_error.max().item())
        )
        self.router_count += router_error.numel()

        if "input_projection" in reference and "input_projection" in target:
            projection_reference = reference["input_projection"]
            projection_target = target["input_projection"]
            projection_error = (
                projection_target.float() - projection_reference.float()
            ).abs()
            self.projection_abs_sum += float(projection_error.sum().item())
            self.projection_abs_max = max(
                self.projection_abs_max, float(projection_error.max().item())
            )
            self.projection_count += projection_error.numel()
            self.projection_exact += int(
                torch.eq(projection_target, projection_reference).sum().item()
            )

        base_indices = reference["router_indices"]
        target_indices = target["router_indices"]
        self.position_same += int((base_indices == target_indices).sum().item())
        self.position_count += base_indices.numel()
        base_sorted = base_indices.sort(dim=-1).values
        target_sorted = target_indices.sort(dim=-1).values
        exact = (base_sorted == target_sorted).all(dim=-1)
        self.set_exact += int(exact.sum().item())
        self.token_count += exact.numel()
        overlap = (
            base_indices.unsqueeze(-1) == target_indices.unsqueeze(-2)
        ).any(dim=-1)
        self.member_overlap += int(overlap.sum().item())
        self.member_count += overlap.numel()

        top_k = base_indices.shape[-1]
        top_values = reference["router_probabilities"].topk(top_k + 1, dim=-1).values
        margin = (top_values[:, top_k - 1] - top_values[:, top_k]).float()
        self.margins.append(margin)
        if (~exact).any():
            self.changed_margins.append(margin[~exact])

    def result(self) -> dict[str, Any]:
        margins = torch.cat(self.margins)
        changed = (
            torch.cat(self.changed_margins)
            if self.changed_margins
            else torch.empty(0)
        )
        return {
            "aligned_rmsnorm_output_abs": {
                "mean": self.norm_abs_sum / self.norm_count,
                "max": self.norm_abs_max,
            },
            "router_probability_abs": {
                "mean": self.router_abs_sum / self.router_count,
                "max": self.router_abs_max,
            },
            "input_projection_abs": (
                {
                    "mean": self.projection_abs_sum / self.projection_count,
                    "max": self.projection_abs_max,
                    "exact_percent": (
                        100.0 * self.projection_exact / self.projection_count
                    ),
                }
                if self.projection_count
                else None
            ),
            "topk_position_percent": 100.0
            * self.position_same
            / self.position_count,
            "topk_set_exact_percent": 100.0 * self.set_exact / self.token_count,
            "topk_member_overlap_percent": 100.0
            * self.member_overlap
            / self.member_count,
            "tokens": self.token_count,
            "changed_set_tokens": self.token_count - self.set_exact,
            "top8_9_probability_margin": {
                "median": float(margins.median().item()),
                "p10": float(torch.quantile(margins, 0.1).item()),
                "changed_median": (
                    float(changed.median().item()) if changed.numel() else None
                ),
                "changed_max": float(changed.max().item()) if changed.numel() else None,
            },
        }


class _LogitMetrics:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.absolute_sum = 0.0
        self.absolute_max = 0.0
        self.logit_count = 0
        self.absolute_samples: list[torch.Tensor] = []
        self.relative_samples: list[torch.Tensor] = []
        self.kld_values: list[torch.Tensor] = []
        self.same_top = 0
        self.token_count = 0
        self.all_finite = True

    def update(self, reference: torch.Tensor, target: torch.Tensor) -> None:
        base_logits = reference.to(self.device)
        target_logits = target.to(self.device)
        self.all_finite = self.all_finite and bool(
            torch.isfinite(target_logits).all().item()
        )
        absolute = (target_logits - base_logits).abs()
        self.absolute_sum += float(absolute.sum().item())
        self.absolute_max = max(self.absolute_max, float(absolute.max().item()))
        self.logit_count += absolute.numel()
        denominator = torch.maximum(
            target_logits.abs(), base_logits.abs()
        ).clamp_min(1e-6)
        relative = absolute / denominator
        stride = max(1, absolute.numel() // 250_000)
        self.absolute_samples.append(absolute.reshape(-1)[::stride].cpu())
        self.relative_samples.append(relative.reshape(-1)[::stride].cpu())

        base_logp = torch.log_softmax(base_logits, dim=-1)
        target_logp = torch.log_softmax(target_logits, dim=-1)
        self.kld_values.append(
            (base_logp.exp() * (base_logp - target_logp)).sum(dim=-1).cpu()
        )
        self.same_top += int(
            (base_logits.argmax(dim=-1) == target_logits.argmax(dim=-1))
            .sum()
            .item()
        )
        self.token_count += target_logits.shape[0]

    def result(self) -> dict[str, Any]:
        absolute = torch.cat(self.absolute_samples)
        relative = torch.cat(self.relative_samples)
        kld = torch.cat(self.kld_values)
        quantiles = torch.tensor([0.50, 0.95, 0.99])
        absolute_q = torch.quantile(absolute, quantiles)
        relative_q = torch.quantile(relative, quantiles)
        kld_q = torch.quantile(kld, quantiles)
        return {
            "logits_absolute": {
                "max": self.absolute_max,
                "mean": self.absolute_sum / self.logit_count,
                "p50_sampled": float(absolute_q[0].item()),
                "p95_sampled": float(absolute_q[1].item()),
                "p99_sampled": float(absolute_q[2].item()),
                "sample_count": absolute.numel(),
            },
            "logits_symmetric_relative": {
                "mean_sampled": float(relative.mean().item()),
                "p50_sampled": float(relative_q[0].item()),
                "p95_sampled": float(relative_q[1].item()),
                "p99_sampled": float(relative_q[2].item()),
                "max_sampled": float(relative.max().item()),
                "sample_count": relative.numel(),
            },
            "kld_base_to_target": {
                "mean": float(kld.mean().item()),
                "max": float(kld.max().item()),
                "p50": float(kld_q[0].item()),
                "p95": float(kld_q[1].item()),
                "p99": float(kld_q[2].item()),
                "token_count": kld.numel(),
            },
            "same_top_p": {
                "percent": 100.0 * self.same_top / self.token_count,
                "same_tokens": self.same_top,
                "token_count": self.token_count,
            },
            "all_target_logits_finite": self.all_finite,
        }


def _capture_boundaries(model, input_ids: torch.Tensor) -> dict[str, Any]:
    post_norm_inputs: dict[str, torch.Tensor] = {}
    boundaries: dict[str, dict[str, torch.Tensor]] = {}
    hooks = []
    for layer_index, layer in enumerate(model.model.layers):
        name = f"model.layers.{layer_index}"
        attention_projection = smooth_groups(layer, name)[0].fcs[0]
        hooks.append(
            attention_projection.register_forward_hook(
                lambda _module, _inputs, output, name=name: boundaries.setdefault(
                    name, {}
                ).__setitem__("input_projection", output.detach().cpu())
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                lambda _module, inputs, name=name: post_norm_inputs.__setitem__(
                    name, inputs[0].detach().cpu()
                )
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                lambda _module, _inputs, output, name=name: boundaries.setdefault(
                    name, {}
                ).__setitem__("norm_output", output.detach().cpu())
            )
        )
        hooks.append(
            layer.mlp.gate.register_forward_hook(
                lambda _module, _inputs, output, name=name: boundaries.setdefault(
                    name, {}
                ).update(
                    {
                        "router_probabilities": output[0].detach().cpu(),
                        "router_indices": output[2].detach().cpu(),
                    }
                )
            )
        )
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=False)
        logits = output.logits.squeeze(0).detach().float().cpu()
        del output
    finally:
        for hook in hooks:
            hook.remove()
    expected = [f"model.layers.{index}" for index in range(len(model.model.layers))]
    if list(boundaries) != expected or list(post_norm_inputs) != expected:
        raise RuntimeError("Qwen3.5 boundary hook order or coverage mismatch")
    return {
        "post_norm_inputs": post_norm_inputs,
        "boundaries": boundaries,
        "logits": logits,
    }


def _capture_local_boundary(
    layer, source_input: torch.Tensor
) -> dict[str, torch.Tensor]:
    device = layer.post_attention_layernorm.weight.device
    with torch.inference_mode():
        norm_output = layer.post_attention_layernorm(source_input.to(device))
        router_output = layer.mlp.gate(norm_output)
    return {
        "norm_output": norm_output.detach().cpu(),
        "router_probabilities": router_output[0].detach().cpu(),
        "router_indices": router_output[2].detach().cpu(),
    }


def _estimate_scale(
    source_router_weight: torch.Tensor, target_router_weight: torch.Tensor
) -> torch.Tensor:
    source = source_router_weight.float()
    target = target_router_weight.float()
    denominator = source.square().sum(dim=0).clamp_min(1e-20)
    return (source * target).sum(dim=0) / denominator


def _parameter_diagnostics(
    source_norm: torch.Tensor,
    source_router: torch.Tensor,
    target_norm: torch.Tensor,
    target_router: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, Any]:
    source_gamma = 1.0 + source_norm.float()
    target_gamma_roundtrip = (1.0 + target_norm.float()) * scale
    gamma_error = (target_gamma_roundtrip - source_gamma).abs()
    router_error = (target_router.float() - source_router.float() * scale).abs()
    return {
        "estimated_scale": {
            "min": float(scale.min().item()),
            "max": float(scale.max().item()),
            "mean": float(scale.mean().item()),
        },
        "effective_gamma_roundtrip_abs": {
            "mean": float(gamma_error.mean().item()),
            "max": float(gamma_error.max().item()),
        },
        "router_weight_writeback_abs": {
            "mean": float(router_error.mean().item()),
            "max": float(router_error.max().item()),
        },
    }


def _scope_selects(scope: str, group) -> bool:
    return scope == "all" or (scope == "moe") == group.is_moe


def _apply_bf16_scoped_smoothing(
    model,
    act_scales: dict[str, torch.Tensor],
    alpha: float,
    scope: str,
) -> dict[str, int]:
    if scope not in {"attention", "moe", "all"}:
        raise ValueError(f"unsupported BF16 smoothing scope: {scope}")
    layers = 0
    smoothed_groups = 0
    identity_groups = 0
    for name, module in model.named_modules():
        if not is_decoder_layer(module):
            continue
        layers += 1
        for group in smooth_groups(module, name):
            if not _scope_selects(scope, group):
                identity_groups += 1
                continue
            smooth_ln_fcs(
                group.norm,
                group.fcs,
                act_scales[group.act_scales_key],
                alpha,
                offset_one_norm=group.offset_one_norm,
                expert_weight=group.expert_weight,
                compensation_weights=group.compensation_weights,
            )
            smoothed_groups += 1
    expected_layers = len(model.model.layers)
    expected_smoothed = expected_layers * (2 if scope == "all" else 1)
    expected_identity = 0 if scope == "all" else expected_layers
    if (
        layers != expected_layers
        or smoothed_groups != expected_smoothed
        or identity_groups != expected_identity
    ):
        raise RuntimeError(
            f"{scope} BF16 control expected {expected_layers} layers and "
            f"{expected_smoothed}/{expected_identity} smoothed/identity groups, "
            f"got {layers}/{smoothed_groups}/{identity_groups}"
        )
    return {
        "layers": layers,
        "scope": scope,
        "smoothed_groups": smoothed_groups,
        "identity_groups": identity_groups,
    }


@torch.no_grad()
def _smooth_group_with_realized_scale(group, act_scales, alpha: float) -> None:
    if not group.offset_one_norm:
        raise TypeError("realized-scale diagnostic requires offset-one RMSNorm")
    if group.expert_weight is not None:
        device = group.expert_weight.device
        weight_scales = chunked_absmax_last_dim(
            group.expert_weight, chunk_size=8
        )
    else:
        device = group.fcs[0].weight.device
        weight_scales = torch.zeros(
            act_scales.numel(), device=device, dtype=torch.float32
        )
    for linear in group.fcs:
        current = linear.weight.detach().float().abs().amax(dim=0)
        torch.maximum(weight_scales, current, out=weight_scales)
    weight_scales.clamp_(min=1e-5)
    requested_scale = (
        act_scales.detach().to(device=device, dtype=torch.float32).pow(alpha)
        / weight_scales.pow(1.0 - alpha)
    ).clamp(min=1e-5)

    norm = group.norm
    original_gamma = 1.0 + norm.weight.float()
    norm.weight.copy_(
        (original_gamma / requested_scale - 1.0).to(norm.weight.dtype)
    )
    stored_gamma = 1.0 + norm.weight.float()
    if (stored_gamma == 0).any():
        raise RuntimeError("realized-scale diagnostic produced zero effective gamma")
    realized_scale = original_gamma / stored_gamma

    for linear in group.fcs:
        linear.weight.mul_(
            realized_scale.to(linear.weight.dtype).view(1, -1)
        )
    for weight in group.compensation_weights:
        weight.mul_(realized_scale.to(weight.dtype).view(1, -1))
    if group.expert_weight is not None:
        group.expert_weight.mul_(
            realized_scale.to(group.expert_weight.dtype).view(1, 1, -1)
        )


def _apply_realized_scale_smoothing(
    model, act_scales: dict[str, torch.Tensor], alpha: float
) -> dict[str, int]:
    layers = 0
    groups = 0
    for name, module in model.named_modules():
        if not is_decoder_layer(module):
            continue
        layers += 1
        for group in smooth_groups(module, name):
            _smooth_group_with_realized_scale(
                group, act_scales[group.act_scales_key], alpha
            )
            groups += 1
    if layers != 40 or groups != 80:
        raise RuntimeError(
            f"realized-scale control expected 40 layers/80 groups, got {layers}/{groups}"
        )
    return {"layers": layers, "groups": groups}


def _requested_group_scale(
    group, act_scales: torch.Tensor, alpha: float
) -> torch.Tensor:
    if group.expert_weight is not None:
        device = group.expert_weight.device
        weight_scales = chunked_absmax_last_dim(
            group.expert_weight, chunk_size=8
        )
    else:
        device = group.fcs[0].weight.device
        weight_scales = torch.zeros(
            act_scales.numel(), device=device, dtype=torch.float32
        )
    for linear in group.fcs:
        current = linear.weight.detach().float().abs().amax(dim=0)
        torch.maximum(weight_scales, current, out=weight_scales)
    weight_scales.clamp_(min=1e-5)
    return (
        act_scales.detach().to(device=device, dtype=torch.float32).pow(alpha)
        / weight_scales.pow(1.0 - alpha)
    ).clamp(min=1e-5)


def _adjust_direct_gamma_scale(
    requested_scale: torch.Tensor,
    policy: str,
    strength: float | None,
) -> torch.Tensor:
    if policy == "requested":
        if strength is not None:
            raise ValueError("requested scale policy does not accept strength")
        return requested_scale
    if policy == "power_of_two":
        if strength is not None:
            raise ValueError("power-of-two scale policy does not accept strength")
        return torch.pow(2.0, requested_scale.log2().round())
    if policy == "log_shrink":
        if strength is None or not 0.0 <= strength <= 1.0:
            raise ValueError("log-shrink strength must be in [0, 1]")
        return requested_scale.pow(strength)
    raise ValueError(f"unsupported direct-gamma scale policy: {policy}")


def _install_direct_gamma_norm(
    norm: nn.Module,
    direct_gamma: torch.Tensor,
    gamma_dtype: torch.dtype,
) -> None:
    if hasattr(norm, "_sq_direct_gamma"):
        raise RuntimeError("direct-gamma norm probe was installed twice")
    norm._sq_direct_gamma = direct_gamma.detach().to(
        device=norm.weight.device, dtype=gamma_dtype
    )

    def forward(module, hidden_states):
        output = module._norm(hidden_states.float())
        output = output * module._sq_direct_gamma.float()
        return output.type_as(hidden_states)

    norm.forward = MethodType(forward, norm)


def _install_fp32_offset_norm(
    norm: nn.Module, direct_gamma: torch.Tensor
) -> None:
    direct_offset = (
        direct_gamma.detach().to(device=norm.weight.device, dtype=torch.float32) - 1.0
    )
    norm.weight = nn.Parameter(
        direct_offset,
        requires_grad=norm.weight.requires_grad,
    )


def _install_materialized_fp32_router(
    router: nn.Module, materialized_weight: torch.Tensor
) -> None:
    if hasattr(router, "_sq_materialized_fp32_weight"):
        raise RuntimeError("materialized FP32 router probe was installed twice")
    if materialized_weight.shape != router.weight.shape:
        raise ValueError(
            "materialized FP32 router weight shape mismatch: "
            f"{tuple(materialized_weight.shape)} != {tuple(router.weight.shape)}"
        )
    router._sq_materialized_fp32_weight = materialized_weight.detach().to(
        device=router.weight.device, dtype=torch.float32
    )

    def forward(module, hidden_states):
        hidden_states = hidden_states.reshape(-1, module.hidden_dim)
        router_probabilities = F.softmax(
            F.linear(hidden_states.float(), module._sq_materialized_fp32_weight),
            dtype=torch.float,
            dim=-1,
        )
        router_scores, router_indices = torch.topk(
            router_probabilities, module.top_k, dim=-1
        )
        router_scores /= router_scores.sum(dim=-1, keepdim=True)
        return router_probabilities, router_scores, router_indices

    router.forward = MethodType(forward, router)


@torch.no_grad()
def _multiply_weight_for_storage(
    weight: torch.Tensor,
    scale: torch.Tensor,
    materialization: str,
) -> None:
    view_shape = [1] * weight.ndim
    view_shape[-1] = scale.numel()
    if materialization == "target_dtype_multiply":
        weight.mul_(
            scale.to(device=weight.device, dtype=weight.dtype).view(view_shape)
        )
        return
    if materialization == "fp32_product":
        product = weight.float() * scale.to(
            device=weight.device, dtype=torch.float32
        ).view(view_shape)
        weight.copy_(product.to(weight.dtype))
        return
    raise ValueError(
        f"unsupported direct-gamma weight materialization: {materialization}"
    )


@torch.no_grad()
def _apply_direct_gamma_moe_smoothing(
    model,
    act_scales: dict[str, torch.Tensor],
    alpha: float,
    *,
    gamma_dtype: torch.dtype,
    weight_materialization: str,
    scale_policy: str,
    scale_strength: float | None = None,
    router_materialization: str = "target_dtype",
    norm_materialization: str = "direct_gamma",
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if gamma_dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported direct-gamma dtype: {gamma_dtype}")
    if router_materialization not in {"target_dtype", "fp32"}:
        raise ValueError(
            "unsupported direct-gamma router materialization: "
            f"{router_materialization}"
        )
    if norm_materialization not in {"direct_gamma", "fp32_offset"}:
        raise ValueError(
            "unsupported direct-gamma norm materialization: "
            f"{norm_materialization}"
        )
    if norm_materialization == "fp32_offset" and gamma_dtype is not torch.float32:
        raise ValueError("FP32 offset norm materialization requires FP32 gamma")
    layers = 0
    direct_gamma_groups = 0
    identity_groups = 0
    applied_scales = {}
    for name, module in model.named_modules():
        if not is_decoder_layer(module):
            continue
        layers += 1
        groups = smooth_groups(module, name)
        for group in groups:
            if not group.is_moe:
                identity_groups += 1
                continue
            requested_scale = _requested_group_scale(
                group, act_scales[group.act_scales_key], alpha
            )
            scale = _adjust_direct_gamma_scale(
                requested_scale, scale_policy, scale_strength
            )
            original_gamma = 1.0 + group.norm.weight.float()
            direct_gamma = original_gamma / scale
            if norm_materialization == "direct_gamma":
                _install_direct_gamma_norm(
                    group.norm, direct_gamma, gamma_dtype
                )
            else:
                _install_fp32_offset_norm(group.norm, direct_gamma)
            for linear in group.fcs:
                _multiply_weight_for_storage(
                    linear.weight, scale, weight_materialization
                )
            router_weight = module.mlp.gate.weight
            for weight in group.compensation_weights:
                if weight is router_weight and router_materialization == "fp32":
                    materialized_weight = weight.float() * scale.to(
                        device=weight.device, dtype=torch.float32
                    ).view(1, -1)
                    _install_materialized_fp32_router(
                        module.mlp.gate, materialized_weight
                    )
                else:
                    _multiply_weight_for_storage(
                        weight, scale, weight_materialization
                    )
            if group.expert_weight is not None:
                _multiply_weight_for_storage(
                    group.expert_weight, scale, weight_materialization
                )
            applied_scales[name] = scale.detach().cpu()
            direct_gamma_groups += 1

    expected_layers = len(model.model.layers)
    if (
        layers != expected_layers
        or direct_gamma_groups != expected_layers
        or identity_groups != expected_layers
    ):
        raise RuntimeError(
            "direct-gamma MoE probe expected one selected and one identity "
            f"group per layer, got {layers}/{direct_gamma_groups}/{identity_groups}"
        )
    return (
        {
            "layers": layers,
            "direct_gamma_groups": direct_gamma_groups,
            "identity_groups": identity_groups,
            "gamma_dtype": str(gamma_dtype),
            "weight_materialization": weight_materialization,
            "router_materialization": router_materialization,
            "norm_materialization": norm_materialization,
            "scale_policy": scale_policy,
            "scale_strength": scale_strength,
        },
        applied_scales,
    )



def _install_runtime_fp32_norm(norm: nn.Module, scale: torch.Tensor | None) -> None:
    if hasattr(norm, "_sq_runtime_fp32_scale"):
        raise RuntimeError("runtime FP32 norm probe was installed twice")
    norm._sq_runtime_fp32_scale = scale

    def forward(module, hidden_states):
        output = module._norm(hidden_states.float())
        gamma = 1.0 + module.weight.float()
        if module._sq_runtime_fp32_scale is not None:
            gamma = gamma / module._sq_runtime_fp32_scale
        return output * gamma

    norm.forward = MethodType(forward, norm)


def _install_runtime_fp32_linear(
    linear: nn.Linear, scale: torch.Tensor | None
) -> None:
    if hasattr(linear, "_sq_runtime_fp32_scale"):
        raise RuntimeError("runtime FP32 linear probe was installed twice")
    linear._sq_runtime_fp32_scale = scale

    def forward(module, hidden_states):
        weight = module.weight.float()
        if module._sq_runtime_fp32_scale is not None:
            weight = weight * module._sq_runtime_fp32_scale.view(1, -1)
        bias = module.bias.float() if module.bias is not None else None
        return F.linear(hidden_states.float(), weight, bias).to(module.weight.dtype)

    linear.forward = MethodType(forward, linear)


def _install_runtime_fp32_router(router: nn.Module, scale: torch.Tensor | None) -> None:
    if hasattr(router, "_sq_runtime_fp32_scale"):
        raise RuntimeError("runtime FP32 router probe was installed twice")
    router._sq_runtime_fp32_scale = scale

    def forward(module, hidden_states):
        hidden_states = hidden_states.reshape(-1, module.hidden_dim)
        weight = module.weight.float()
        if module._sq_runtime_fp32_scale is not None:
            weight = weight * module._sq_runtime_fp32_scale.view(1, -1)
        router_probabilities = F.softmax(
            F.linear(hidden_states.float(), weight), dtype=torch.float, dim=-1
        )
        router_scores, router_indices = torch.topk(
            router_probabilities, module.top_k, dim=-1
        )
        router_scores /= router_scores.sum(dim=-1, keepdim=True)
        return router_probabilities, router_scores, router_indices

    router.forward = MethodType(forward, router)


def _apply_runtime_fp32_router_only(model) -> dict[str, int]:
    layers = 0
    routers = 0
    for module in model.modules():
        if not is_decoder_layer(module):
            continue
        layers += 1
        _install_materialized_fp32_router(
            module.mlp.gate, module.mlp.gate.weight.float()
        )
        routers += 1
    expected_layers = len(model.model.layers)
    if layers != expected_layers or routers != expected_layers:
        raise RuntimeError(
            "FP32-router-only probe expected one router per layer, got "
            f"{layers}/{routers} for {expected_layers} layers"
        )
    return {"layers": layers, "routers": routers}


def _install_runtime_fp32_experts(
    experts: nn.Module, scale: torch.Tensor | None
) -> None:
    if hasattr(experts, "_sq_runtime_fp32_scale"):
        raise RuntimeError("runtime FP32 expert probe was installed twice")
    experts._sq_runtime_fp32_scale = scale

    def forward(module, hidden_states, top_k_index, top_k_weights):
        output_dtype = module.gate_up_proj.dtype
        final_hidden_states = torch.zeros(
            hidden_states.shape, device=hidden_states.device, dtype=output_dtype
        )
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=module.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_index_tensor in expert_hit:
            expert_index = expert_index_tensor[0]
            top_k_position, token_index = torch.where(expert_mask[expert_index])
            current_state = hidden_states[token_index]
            weight = module.gate_up_proj[expert_index].float()
            if module._sq_runtime_fp32_scale is not None:
                weight = weight * module._sq_runtime_fp32_scale.view(1, -1)
            gate, up = F.linear(current_state.float(), weight).chunk(2, dim=-1)
            gate = gate.to(output_dtype)
            up = up.to(output_dtype)
            current_hidden_states = module.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states, module.down_proj[expert_index]
            )
            current_hidden_states = current_hidden_states * top_k_weights[
                token_index, top_k_position, None
            ]
            final_hidden_states.index_add_(
                0, token_index, current_hidden_states.to(output_dtype)
            )
        return final_hidden_states

    experts.forward = MethodType(forward, experts)


def _install_runtime_fp32_group(layer, group, scale: torch.Tensor | None) -> None:
    _install_runtime_fp32_norm(group.norm, scale)
    for linear in group.fcs:
        _install_runtime_fp32_linear(linear, scale)
    if group.is_moe:
        _install_runtime_fp32_router(layer.mlp.gate, scale)
        _install_runtime_fp32_linear(layer.mlp.shared_expert_gate, scale)
        _install_runtime_fp32_experts(layer.mlp.experts, scale)


def _apply_runtime_fp32_compute(model, scope: str) -> dict[str, int]:
    layers = 0
    selected_groups = 0
    for name, module in model.named_modules():
        if not is_decoder_layer(module):
            continue
        layers += 1
        for group in smooth_groups(module, name):
            if _scope_selects(scope, group):
                _install_runtime_fp32_group(module, group, None)
                selected_groups += 1
    expected_groups = 80 if scope == "all" else 40
    if layers != 40 or selected_groups != expected_groups:
        raise RuntimeError(
            "runtime FP32 compute probe expected 40 layers and "
            f"{expected_groups} selected groups, got {layers}/{selected_groups}"
        )
    return {"layers": layers, "selected_groups": selected_groups}


@torch.no_grad()
def _apply_runtime_fp32_dynamic_smoothing(
    model,
    act_scales: dict[str, torch.Tensor],
    alpha: float,
    scope: str,
) -> tuple[dict[str, int], dict[str, torch.Tensor]]:
    layers = 0
    selected_groups = 0
    identity_groups = 0
    runtime_moe_scales = {}
    for name, module in model.named_modules():
        if not is_decoder_layer(module):
            continue
        layers += 1
        for group in smooth_groups(module, name):
            group_act_scales = act_scales[group.act_scales_key]
            if _scope_selects(scope, group):
                scale = _requested_group_scale(group, group_act_scales, alpha)
                _install_runtime_fp32_group(module, group, scale)
                selected_groups += 1
                if group.is_moe:
                    runtime_moe_scales[name] = scale.detach().cpu()
            else:
                identity_groups += 1
    expected_layers = len(model.model.layers)
    expected_selected = expected_layers * (2 if scope == "all" else 1)
    expected_identity = 0 if scope == "all" else expected_layers
    if (
        layers != expected_layers
        or selected_groups != expected_selected
        or identity_groups != expected_identity
    ):
        raise RuntimeError(
            f"dynamic FP32 probe expected {expected_layers} layers and "
            f"{expected_selected}/{expected_identity} FP32/identity groups, got "
            f"{layers}/{selected_groups}/{identity_groups}"
        )
    return (
        {
            "layers": layers,
            "runtime_fp32_groups": selected_groups,
            "identity_groups": identity_groups,
        },
        runtime_moe_scales,
    )


def _parse_scoped_alpha(value: str) -> tuple[str, float]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "runtime FP32 alpha must be SCOPE:ALPHA (scope: attention, moe, all)"
        )
    scope, alpha_text = value.split(":", 1)
    if scope not in {"attention", "moe", "all"}:
        raise argparse.ArgumentTypeError(
            f"invalid runtime FP32 scope {scope!r}; expected attention, moe, or all"
        )
    try:
        alpha = float(alpha_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid runtime FP32 alpha {alpha_text!r}"
        ) from error
    return scope, alpha


def _parse_direct_gamma_probe(value: str) -> tuple[str, float]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "direct-gamma probe must be MODE:ALPHA"
        )
    mode, alpha_text = value.split(":", 1)
    allowed = {
        "bf16",
        "late-bf16",
        "fp32-gamma",
        "fp32-router",
        "fp32-gamma-router",
        "power-of-two",
        "power-of-two-fp32-offset",
        "power-of-two-fp32-gamma",
        "power-of-two-fp32-router",
        "power-of-two-fp32-gamma-router",
        "shrink50",
        "shrink25",
    }
    if mode not in allowed:
        raise argparse.ArgumentTypeError(
            f"invalid direct-gamma mode {mode!r}; expected one of "
            + ", ".join(sorted(allowed))
        )
    try:
        alpha = float(alpha_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid direct-gamma alpha {alpha_text!r}"
        ) from error
    return mode, alpha


def _direct_gamma_probe_config(mode: str) -> dict[str, Any]:
    configs = {
        "bf16": {
            "gamma_dtype": torch.bfloat16,
            "weight_materialization": "target_dtype_multiply",
            "scale_policy": "requested",
            "scale_strength": None,
        },
        "late-bf16": {
            "gamma_dtype": torch.bfloat16,
            "weight_materialization": "fp32_product",
            "scale_policy": "requested",
            "scale_strength": None,
        },
        "fp32-gamma": {
            "gamma_dtype": torch.float32,
            "weight_materialization": "fp32_product",
            "scale_policy": "requested",
            "scale_strength": None,
        },
        "fp32-router": {
            "gamma_dtype": torch.bfloat16,
            "weight_materialization": "fp32_product",
            "router_materialization": "fp32",
            "scale_policy": "requested",
            "scale_strength": None,
        },
        "fp32-gamma-router": {
            "gamma_dtype": torch.float32,
            "weight_materialization": "fp32_product",
            "router_materialization": "fp32",
            "scale_policy": "requested",
            "scale_strength": None,
        },
        "power-of-two": {
            "gamma_dtype": torch.bfloat16,
            "weight_materialization": "target_dtype_multiply",
            "scale_policy": "power_of_two",
            "scale_strength": None,
        },
        "power-of-two-fp32-gamma": {
            "gamma_dtype": torch.float32,
            "weight_materialization": "target_dtype_multiply",
            "scale_policy": "power_of_two",
            "scale_strength": None,
        },
        "power-of-two-fp32-offset": {
            "gamma_dtype": torch.float32,
            "weight_materialization": "target_dtype_multiply",
            "norm_materialization": "fp32_offset",
            "scale_policy": "power_of_two",
            "scale_strength": None,
        },
        "power-of-two-fp32-router": {
            "gamma_dtype": torch.bfloat16,
            "weight_materialization": "target_dtype_multiply",
            "router_materialization": "fp32",
            "scale_policy": "power_of_two",
            "scale_strength": None,
        },
        "power-of-two-fp32-gamma-router": {
            "gamma_dtype": torch.float32,
            "weight_materialization": "target_dtype_multiply",
            "router_materialization": "fp32",
            "scale_policy": "power_of_two",
            "scale_strength": None,
        },
        "shrink50": {
            "gamma_dtype": torch.bfloat16,
            "weight_materialization": "fp32_product",
            "scale_policy": "log_shrink",
            "scale_strength": 0.5,
        },
        "shrink25": {
            "gamma_dtype": torch.bfloat16,
            "weight_materialization": "fp32_product",
            "scale_policy": "log_shrink",
            "scale_strength": 0.25,
        },
    }
    return configs[mode]


def _parse_target(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("target must be LABEL=/path/to/model")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("target must be LABEL=/path/to/model")
    return label, Path(path).expanduser()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target", action="append", type=_parse_target, default=[])
    parser.add_argument(
        "--fp32-compute-target", action="append", type=_parse_target, default=[]
    )
    parser.add_argument(
        "--fp32-source-scope",
        action="append",
        choices=("attention", "moe", "all"),
        default=[],
    )
    parser.add_argument(
        "--reference-fp32-scope",
        choices=("attention", "moe", "all"),
    )
    parser.add_argument("--reference-fp32-router-only", action="store_true")
    parser.add_argument("--act-scales-cache")
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--fp32-router-source", action="store_true")
    parser.add_argument("--bf16-alpha", action="append", type=float, default=[])
    parser.add_argument(
        "--attention-only-alpha", action="append", type=float, default=[]
    )
    parser.add_argument(
        "--moe-only-alpha", action="append", type=float, default=[]
    )
    parser.add_argument(
        "--realized-scale-alpha", action="append", type=float, default=[]
    )
    parser.add_argument(
        "--runtime-fp32-alpha",
        action="append",
        type=_parse_scoped_alpha,
        default=[],
        metavar="SCOPE:ALPHA",
    )
    parser.add_argument(
        "--direct-gamma-probe",
        action="append",
        type=_parse_direct_gamma_probe,
        default=[],
        metavar="MODE:ALPHA",
    )
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--calibration-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    source_path = Path(args.source_model).expanduser().resolve()
    holdout_path = Path(args.holdout).expanduser().resolve()
    calibration_path = Path(args.calibration_data).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    control_alphas = (
        args.bf16_alpha
        + args.attention_only_alpha
        + args.moe_only_alpha
        + args.realized_scale_alpha
        + [alpha for _scope, alpha in args.runtime_fp32_alpha]
        + [alpha for _mode, alpha in args.direct_gamma_probe]
    )
    if (
        not args.target
        and not args.fp32_compute_target
        and not args.fp32_source_scope
        and not args.identity
        and not args.fp32_router_source
        and not control_alphas
    ):
        raise ValueError("at least one target or diagnostic alpha is required")
    if control_alphas and not args.act_scales_cache:
        raise ValueError("diagnostic alpha modes require --act-scales-cache")
    holdout = _load_holdout(holdout_path)
    _assert_no_calibration_overlap(calibration_path, holdout)
    tokenizer = AutoTokenizer.from_pretrained(source_path, local_files_only=True)
    rendered = _render_holdout(tokenizer, holdout)
    rendered_bytes = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in rendered
    ).encode()

    outer_config = AutoConfig.from_pretrained(source_path, local_files_only=True)
    source, source_loading = load_qwen35_text_model(
        source_path,
        outer_config.text_config,
        dtype=torch.bfloat16,
        device=args.device,
        require_production_shape=True,
    )
    source.eval()
    source.config.use_cache = False
    source_parameters = {}
    for layer_index, layer in enumerate(source.model.layers):
        name = f"model.layers.{layer_index}"
        source_parameters[name] = {
            "norm": layer.post_attention_layernorm.weight.detach().cpu().clone(),
            "router": layer.mlp.gate.weight.detach().cpu().clone(),
        }

    if (
        args.reference_fp32_scope is not None
        and args.reference_fp32_router_only
    ):
        raise ValueError(
            "--reference-fp32-scope and --reference-fp32-router-only are "
            "mutually exclusive"
        )
    reference_variant = {
        "kind": "source_bf16_compute",
        "stored_parameters": "unchanged source BF16 checkpoint values",
    }
    if args.reference_fp32_scope is not None:
        reference_variant = {
            "kind": "source_runtime_fp32_compute",
            "scope": args.reference_fp32_scope,
            "applied": _apply_runtime_fp32_compute(
                source, args.reference_fp32_scope
            ),
            "stored_parameters": "unchanged source BF16 checkpoint values",
        }
    elif args.reference_fp32_router_only:
        reference_variant = {
            "kind": "source_fp32_router_only",
            "applied": _apply_runtime_fp32_router_only(source),
            "stored_parameters": "unchanged source BF16 checkpoint values",
        }

    references = []
    for record in rendered:
        input_ids = tokenizer(
            record["rendered"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).input_ids
        captured = _capture_boundaries(
            source, input_ids.to(_model_input_device(source))
        )
        references.append({"input_ids": input_ids, **captured})
    del source
    _release_cuda_memory()

    control_cache = None
    cache_path = None
    if control_alphas:
        cache_path = Path(args.act_scales_cache).expanduser().resolve()
        control_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if control_cache.get("complete") is not True:
            raise ValueError(
                f"diagnostic control requires a complete cache: {cache_path}"
            )

    evaluation_specs = []
    if args.identity:
        evaluation_specs.append(("identity", "identity", None, None, None))
    if args.fp32_router_source:
        evaluation_specs.append(
            ("fp32_router_source", "runtime-fp32-router-source", None, None, None)
        )
    evaluation_specs.extend(
        ("hf", label, path.resolve(), None, None) for label, path in args.target
    )
    evaluation_specs.extend(
        ("fp32_compute", label, path.resolve(), None, "all")
        for label, path in args.fp32_compute_target
    )
    evaluation_specs.extend(
        (
            "fp32_source",
            f"runtime-fp32-source-{scope}",
            None,
            None,
            scope,
        )
        for scope in args.fp32_source_scope
    )
    for alpha in args.bf16_alpha:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"BF16 alpha must be in [0, 1], got {alpha}")
        evaluation_specs.append(
            ("bf16_smoothing", f"bf16-alpha{alpha:.2f}", None, alpha, None)
        )
    for alpha in args.attention_only_alpha:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"attention-only alpha must be in [0, 1], got {alpha}")
        evaluation_specs.append(
            ("attention_only", f"attention-only-alpha{alpha:.2f}", None, alpha, None)
        )
    for alpha in args.moe_only_alpha:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"MoE-only alpha must be in [0, 1], got {alpha}")
        evaluation_specs.append(
            ("moe_only", f"moe-only-alpha{alpha:.2f}", None, alpha, None)
        )
    for alpha in args.realized_scale_alpha:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"realized-scale alpha must be in [0, 1], got {alpha}")
        evaluation_specs.append(
            ("realized_scale", f"realized-scale-alpha{alpha:.2f}", None, alpha, None)
        )
    for scope, alpha in args.runtime_fp32_alpha:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"runtime FP32 alpha must be in [0, 1], got {alpha}")
        evaluation_specs.append(
            (
                "runtime_fp32",
                f"runtime-fp32-{scope}-alpha{alpha:.2f}",
                None,
                alpha,
                scope,
            )
        )

    for mode, alpha in args.direct_gamma_probe:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"direct-gamma alpha must be in [0, 1], got {alpha}"
            )
        evaluation_specs.append(
            (
                "direct_gamma",
                f"direct-gamma-{mode}-alpha{alpha:.2f}",
                None,
                alpha,
                mode,
            )
        )


    targets = {}
    for kind, label, target_path, alpha, scope in evaluation_specs:
        runtime_moe_scales = {}
        if kind in {"hf", "fp32_compute"}:
            model = _load_smoothed(target_path, device=args.device)
            fingerprint = _source_model_fingerprint(str(target_path))
            if kind == "fp32_compute":
                applied = _apply_runtime_fp32_compute(model, scope)
                variant = {
                    "kind": "smoothed_hf_runtime_fp32_compute_control",
                    "scope": scope,
                    "applied": applied,
                    "stored_parameters": "unchanged BF16 smoothed checkpoint values",
                }
            else:
                variant = {"kind": "smoothed_hf"}
        else:
            model, control_loading = load_qwen35_text_model(
                source_path,
                outer_config.text_config,
                dtype=torch.bfloat16,
                device=args.device,
                require_production_shape=True,
            )
            model.eval()
            model.config.use_cache = False
            if kind == "identity":
                applied = {
                    "layers": len(model.model.layers),
                    "smoothed_groups": 0,
                    "identity_groups": 2 * len(model.model.layers),
                }
                variant_details = {
                    "scale_policy": "identity; no smoothing or runtime override",
                    "stored_parameters": "unchanged source BF16 checkpoint values",
                }
            elif kind == "fp32_router_source":
                applied = _apply_runtime_fp32_router_only(model)
                variant_details = {
                    "scope": "router_only",
                    "scale_policy": "identity; FP32 router compute control",
                    "stored_parameters": "unchanged source BF16 checkpoint values",
                }
            elif kind == "fp32_source":
                applied = _apply_runtime_fp32_compute(model, scope)
                variant_details = {
                    "scope": scope,
                    "scale_policy": "no smoothing; FP32 compute control",
                    "stored_parameters": "unchanged source BF16 checkpoint values",
                }
            elif kind == "bf16_smoothing":
                layers, groups, group_stats = smooth_lm(
                    model,
                    control_cache["act_scales"],
                    alpha,
                    return_group_stats=True,
                )
                if layers != 40 or groups != 80:
                    raise RuntimeError(
                        "BF16 sweep expected 40 layers/80 groups, got "
                        f"{layers}/{groups}"
                    )
                applied = {
                    "layers": layers,
                    "groups": groups,
                    "group_stats": group_stats,
                }
                variant_details = {
                    "scale_policy": "formal BF16 SmoothQuant writeback path",
                    "stored_parameters": "transformed values written to BF16",
                }
            elif kind in {"attention_only", "moe_only"}:
                selected_scope = "attention" if kind == "attention_only" else "moe"
                applied = _apply_bf16_scoped_smoothing(
                    model,
                    control_cache["act_scales"],
                    alpha,
                    selected_scope,
                )
                variant_details = {
                    "scope": selected_scope,
                    "scale_policy": "formal BF16 SmoothQuant writeback for selected groups",
                    "unselected_groups": "identity",
                }
            elif kind == "realized_scale":
                applied = _apply_realized_scale_smoothing(
                    model, control_cache["act_scales"], alpha
                )
                variant_details = {
                    "scale_policy": "derive downstream scale from stored BF16 effective gamma"
                }
            elif kind == "direct_gamma":
                config = _direct_gamma_probe_config(scope)
                applied, runtime_moe_scales = (
                    _apply_direct_gamma_moe_smoothing(
                        model,
                        control_cache["act_scales"],
                        alpha,
                        **config,
                    )
                )
                variant_details = {
                    "scope": "moe",
                    "direct_gamma_mode": scope,
                    "unselected_groups": "identity",
                    "stored_parameters": (
                        "source norm parameter unchanged; diagnostic direct "
                        "gamma stored separately"
                    ),
                }
            else:
                applied, runtime_moe_scales = _apply_runtime_fp32_dynamic_smoothing(
                    model, control_cache["act_scales"], alpha, scope
                )
                variant_details = {
                    "scope": scope,
                    "scale_policy": (
                        "apply FP32 scale at runtime to unchanged source BF16 "
                        "parameters"
                    ),
                    "stored_parameters": (
                        "original BF16 checkpoint values; not promoted or "
                        "rewritten for selected groups"
                    ),
                }
            fingerprint = {
                "source_model": _source_model_fingerprint(str(source_path)),
            }
            if kind not in {"fp32_source", "fp32_router_source", "identity"}:
                fingerprint["cache"] = _file_fingerprint(cache_path)
            variant = {
                "kind": f"{kind}_control",
                "alpha": alpha,
                "applied": applied,
                "loading": control_loading,
                **variant_details,
            }
        local_metrics = {
            name: _BoundaryMetrics() for name in source_parameters
        }
        end_to_end_metrics = {
            name: _BoundaryMetrics() for name in source_parameters
        }
        local_total = _BoundaryMetrics()
        end_to_end_total = _BoundaryMetrics()
        logit_metrics = _LogitMetrics(_model_input_device(model))
        scales = {}
        parameter_metrics = {}
        for layer_index, layer in enumerate(model.model.layers):
            name = f"model.layers.{layer_index}"
            source_param = source_parameters[name]
            target_norm = layer.post_attention_layernorm.weight.detach().cpu()
            stored_target_router = layer.mlp.gate.weight.detach().cpu()
            target_router = getattr(
                layer.mlp.gate,
                "_sq_materialized_fp32_weight",
                layer.mlp.gate.weight,
            ).detach().cpu()
            scale = runtime_moe_scales.get(name)
            if scale is None:
                if torch.equal(source_param["router"], target_router):
                    scale = torch.ones(
                        target_router.shape[-1], dtype=torch.float32
                    )
                else:
                    scale = _estimate_scale(source_param["router"], target_router)
            scales[name] = scale
            if name in runtime_moe_scales:
                parameter_metrics[name] = {
                    "runtime_reparameterization": True,
                    "stored_norm_unchanged": bool(
                        torch.equal(source_param["norm"], target_norm)
                    ),
                    "stored_router_unchanged": bool(
                        torch.equal(source_param["router"], stored_target_router)
                    ),
                    "effective_router_dtype": str(target_router.dtype),
                    "runtime_scale": {
                        "min": float(scale.min().item()),
                        "max": float(scale.max().item()),
                        "mean": float(scale.mean().item()),
                    },
                }
            else:
                parameter_metrics[name] = _parameter_diagnostics(
                    source_param["norm"],
                    source_param["router"],
                    target_norm,
                    target_router,
                    scale,
                )

        for reference in references:
            actual = None
            if not args.local_only:
                actual = _capture_boundaries(
                    model, reference["input_ids"].to(_model_input_device(model))
                )
                logit_metrics.update(reference["logits"], actual["logits"])
            for layer_index, layer in enumerate(model.model.layers):
                name = f"model.layers.{layer_index}"
                base = reference["boundaries"][name]
                local = _capture_local_boundary(
                    layer, reference["post_norm_inputs"][name]
                )
                local_metrics[name].update(base, local, scales[name])
                local_total.update(base, local, scales[name])
                if actual is not None:
                    end_to_end_metrics[name].update(
                        base, actual["boundaries"][name], scales[name]
                    )
                    end_to_end_total.update(
                        base, actual["boundaries"][name], scales[name]
                    )

        summary = {"local_same_input": local_total.result()}
        layer_results = {
            name: {
                "parameters": parameter_metrics[name],
                "local_same_input": local_metrics[name].result(),
            }
            for name in source_parameters
        }
        if not args.local_only:
            summary.update(
                {
                    "end_to_end": end_to_end_total.result(),
                    "logits": logit_metrics.result(),
                }
            )
            for name in source_parameters:
                layer_results[name]["end_to_end"] = (
                    end_to_end_metrics[name].result()
                )

        targets[label] = {
            "fingerprint": fingerprint,
            "variant": variant,
            "summary": summary,
            "layers": layer_results,
        }
        del model
        _release_cuda_memory()

    git_commit = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = {
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": "bfloat16",
            "git_commit": git_commit,
        },
        "inputs": {
            "source_model": _source_model_fingerprint(str(source_path)),
            "holdout": _file_fingerprint(holdout_path),
            "rendered_holdout_sha256": hashlib.sha256(rendered_bytes).hexdigest(),
            "calibration_data": _file_fingerprint(calibration_path),
            "holdout_calibration_exact_text_overlap": False,
        },
        "source_loading": source_loading,
        "reference_variant": reference_variant,
        "targets": targets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    print(json.dumps({label: value["summary"] for label, value in targets.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
