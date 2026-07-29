import torch
import torch.nn as nn
from .quantizer import pseudo_quantize_tensor
import gc

__all__ = ["auto_clip_block"]

QWEN3_5_MOE_GATE_UP_CLIP = "__qwen3_5_moe_gate_up_proj__"
QWEN3_5_MOE_DOWN_CLIP = "__qwen3_5_moe_down_proj__"


# weight quantization
@torch.no_grad()
def auto_clip_layer(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = (
        q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    )
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: max(1, input_feat.shape[1] // n_sample_token)]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = min(256, w.shape[0])  # prevent OOM
    w_all = w
    best_max_val_all = []

    for i_b in range(0, w.shape[0], oc_batch_size):
        w = w_all[i_b : i_b + oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1)


@torch.no_grad()
def auto_clip_expert_layer(w, input_feat, n_bit, q_config):
    """Find clip thresholds for a [experts, out_features, in_features] tensor."""
    return torch.stack(
        [
            auto_clip_layer(
                expert_weight,
                input_feat,
                n_bit=n_bit,
                q_config=q_config,
            )
            for expert_weight in w
        ]
    )


@torch.no_grad()
def qwen3_5_moe_down_inputs(expert_gate_up_proj, input_feat):
    """Approximate each expert's down-projection input using calibration tokens."""
    tokens = input_feat.reshape(-1, input_feat.shape[-1]).to(expert_gate_up_proj.device)
    max_tokens = 512
    if tokens.shape[0] > max_tokens:
        tokens = tokens[:: max(1, tokens.shape[0] // max_tokens)][:max_tokens]
    inputs = []
    for expert_weight in expert_gate_up_proj:
        gate, up = torch.nn.functional.linear(tokens, expert_weight).chunk(2, dim=-1)
        inputs.append(torch.nn.functional.silu(gate) * up)
    return inputs


@torch.no_grad()
def auto_clip_block(module, w_bit, q_config, input_feat):
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }

    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue
        named_linears[name].cuda()
        max_val = auto_clip_layer(
            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config
        )
        clip_list.append((name, max_val))
        named_linears[name].cpu()

    if module.__class__.__name__ == "Qwen3_5MoeDecoderLayer":
        experts = module.mlp.experts
        expert_input = input_feat["mlp.shared_expert.gate_proj"]
        experts.cuda()
        gate_up_max_val = auto_clip_expert_layer(
            experts.gate_up_proj,
            expert_input,
            n_bit=w_bit,
            q_config=q_config,
        )
        down_inputs = qwen3_5_moe_down_inputs(experts.gate_up_proj, expert_input)
        down_max_val = torch.stack(
            [
                auto_clip_layer(
                    expert_weight,
                    expert_input,
                    n_bit=w_bit,
                    q_config=q_config,
                )
                for expert_weight, expert_input in zip(experts.down_proj, down_inputs)
            ]
        )
        clip_list.extend(
            [
                (QWEN3_5_MOE_GATE_UP_CLIP, gate_up_max_val.cpu()),
                (QWEN3_5_MOE_DOWN_CLIP, down_max_val.cpu()),
            ]
        )
        experts.cpu()
    return clip_list


@torch.no_grad()
def apply_clip(module, clip_list):
    from .utils.module import get_op_by_name

    for name, max_val in clip_list:
        if name.endswith(QWEN3_5_MOE_GATE_UP_CLIP) or name.endswith(
            QWEN3_5_MOE_DOWN_CLIP
        ):
            marker = (
                QWEN3_5_MOE_GATE_UP_CLIP
                if name.endswith(QWEN3_5_MOE_GATE_UP_CLIP)
                else QWEN3_5_MOE_DOWN_CLIP
            )
            block_path = name[: -len(marker)].rstrip(".")
            block = get_op_by_name(module, block_path) if block_path else module
            experts = block.mlp.experts
            experts.cuda()
            weight = (
                experts.gate_up_proj
                if marker == QWEN3_5_MOE_GATE_UP_CLIP
                else experts.down_proj
            )
            max_val = max_val.to(weight.device).to(weight.dtype)
            weight.data = torch.clamp(
                weight.data.reshape(*max_val.shape, -1),
                -max_val.unsqueeze(-1),
                max_val.unsqueeze(-1),
            ).reshape_as(weight)
            experts.cpu()
            continue
        layer = get_op_by_name(module, name)
        layer.cuda()
        max_val = max_val.to(layer.weight.device).to(layer.weight.dtype)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()
