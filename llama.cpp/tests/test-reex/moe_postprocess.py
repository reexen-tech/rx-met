#!/usr/bin/env python3
"""MoE 采数后处理: 把 dump_moe_prefill 产生的 raw .bin (+ .json sidecar) 转换成
归一化的 .npy 以及变长 expert->token 映射 / 专家负载占比.

输入 (由 C++ 工具产生):
    <root>/run_metadata.json
    <root>/case_XX/input_token_ids.bin (+ .json)
    <root>/case_XX/raw/<tensor>.bin (+ .json)

输出:
    <root>/case_XX/input_token_ids.npy                  [n_tokens]            int32
    <root>/case_XX/layer_YY/router_expert_ids.npy       [n_tokens, topk]      int32
    <root>/case_XX/layer_YY/router_expert_weights.npy   [n_tokens, topk]      runtime dtype
    <root>/case_XX/layer_YY/expert_outputs_unweighted.npy [n_tokens, topk, hidden]
    <root>/case_XX/layer_YY/expert_outputs_weighted.npy   [n_tokens, topk, hidden]
    <root>/case_XX/layer_YY/moe_out.npy                 [n_tokens, hidden]    (sanity check)
    <root>/case_XX/layer_YY/expert_token_map.json       expert -> [{token_id, topk_slot}]
    <root>/case_XX/layer_YY/expert_load_ratio.json      per-expert count / ratio

ggml 原始 layout: ne = [ne0, ne1, ne2, ne3], ne0 为最内 (变化最快); raw .bin 已按
逻辑顺序连续写出 (i0 最快). 因此 numpy reshape 到 reversed(ne) 即得到归一化 layout.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

# ggml dtype 名称 -> numpy dtype
GGML_TO_NP = {
    "f32": np.float32,
    "f16": np.float16,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8":  np.int8,
}

ROUTER_WEIGHT_PREFERENCE = ["ffn_moe_weights_norm", "ffn_moe_weights"]


def load_sidecar(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def load_tensor(bin_path, meta):
    """读取 raw .bin -> numpy, 形状归一化为 reversed(ne) 并去掉 size-1 维度后按语义返回."""
    dtype_name = meta["ggml_type"]
    if dtype_name not in GGML_TO_NP:
        if dtype_name == "bf16":
            # bf16: 以 uint16 读出原始 bits, 上层用 ml_dtypes 解释; 这里直接转 float32.
            raw = np.fromfile(bin_path, dtype=np.uint16)
            f32 = (raw.astype(np.uint32) << 16).view(np.float32)
            return f32, meta["ne"]
        raise ValueError("unsupported ggml dtype: %s" % dtype_name)
    np_dtype = GGML_TO_NP[dtype_name]
    arr = np.fromfile(bin_path, dtype=np_dtype)
    ne = meta["ne"]  # [ne0, ne1, ne2, ne3]
    shape = tuple(reversed(ne))  # (ne3, ne2, ne1, ne0)
    arr = arr.reshape(shape)
    return arr, ne


def normalize_topk(arr):
    # ne = [topk, token, 1, 1] -> reversed (1,1,token,topk) -> (token, topk)
    a = arr.reshape(arr.shape[-2], arr.shape[-1])  # (token, topk)
    return np.ascontiguousarray(a.astype(np.int32))


def normalize_weights(arr):
    # ne = [1, topk, token, 1] -> reversed (1, token, topk, 1) -> (token, topk)
    a = np.squeeze(arr)  # -> (token, topk)
    if a.ndim == 1:
        a = a.reshape(a.shape[0], 1)
    return np.ascontiguousarray(a)


def normalize_experts(arr):
    # ne = [hidden, topk, token, 1] -> reversed (1, token, topk, hidden) -> (token, topk, hidden)
    a = arr.reshape(arr.shape[-3], arr.shape[-2], arr.shape[-1])
    return np.ascontiguousarray(a)


def normalize_moe_out(arr):
    # ne = [hidden, token, 1, 1] -> reversed (1,1,token,hidden) -> (token, hidden)
    a = arr.reshape(arr.shape[-2], arr.shape[-1])
    return np.ascontiguousarray(a)


def collect_raw(raw_dir):
    """raw_dir 内所有 tensor 按 layer 分组: {layer: {base: (bin_path, meta)}}."""
    by_layer = {}
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".json"):
            continue
        meta = load_sidecar(os.path.join(raw_dir, fn))
        base = meta.get("base")
        layer = meta.get("layer")
        if base is None or layer is None:
            continue
        bin_path = os.path.join(raw_dir, fn[:-5] + ".bin")
        if not os.path.exists(bin_path):
            print("  WARN: missing bin for %s" % fn, file=sys.stderr)
            continue
        by_layer.setdefault(layer, {})[base] = (bin_path, meta)
    return by_layer


def build_expert_maps(ids, n_expert):
    """ids: [token, topk] int32 -> expert_token_map, expert_load_ratio."""
    n_tokens, topk = ids.shape
    token_map = {e: [] for e in range(n_expert)}
    for t in range(n_tokens):
        for s in range(topk):
            e = int(ids[t, s])
            token_map.setdefault(e, []).append({"token_id": t, "topk_slot": s})
    total = n_tokens * topk
    load = {}
    for e in range(n_expert):
        c = len(token_map.get(e, []))
        load[e] = {"count": c, "ratio": c / total if total else 0.0}
    return token_map, load, total


def process_layer(out_dir, tensors, n_expert):
    os.makedirs(out_dir, exist_ok=True)
    produced = []

    # router_expert_ids (required)
    if "ffn_moe_topk" not in tensors:
        raise RuntimeError("missing ffn_moe_topk in %s" % out_dir)
    arr, _ = load_tensor(*tensors["ffn_moe_topk"])
    ids = normalize_topk(arr)
    np.save(os.path.join(out_dir, "router_expert_ids.npy"), ids)
    produced.append("router_expert_ids.npy")

    # router_expert_weights (prefer normalized)
    wkey = next((k for k in ROUTER_WEIGHT_PREFERENCE if k in tensors), None)
    if wkey is not None:
        warr, _ = load_tensor(*tensors[wkey])
        weights = normalize_weights(warr)
        np.save(os.path.join(out_dir, "router_expert_weights.npy"), weights)
        produced.append("router_expert_weights.npy (%s)" % wkey)

    # expert_outputs_unweighted
    if "ffn_moe_down" in tensors:
        darr, _ = load_tensor(*tensors["ffn_moe_down"])
        np.save(os.path.join(out_dir, "expert_outputs_unweighted.npy"), normalize_experts(darr))
        produced.append("expert_outputs_unweighted.npy")

    # expert_outputs_weighted
    if "ffn_moe_weighted" in tensors:
        warr, _ = load_tensor(*tensors["ffn_moe_weighted"])
        np.save(os.path.join(out_dir, "expert_outputs_weighted.npy"), normalize_experts(warr))
        produced.append("expert_outputs_weighted.npy")

    # moe_out (sanity)
    if "ffn_moe_out" in tensors:
        oarr, _ = load_tensor(*tensors["ffn_moe_out"])
        np.save(os.path.join(out_dir, "moe_out.npy"), normalize_moe_out(oarr))
        produced.append("moe_out.npy")

    # expert maps
    token_map, load, total = build_expert_maps(ids, n_expert)
    with open(os.path.join(out_dir, "expert_token_map.json"), "w") as f:
        json.dump({str(e): token_map.get(e, []) for e in range(n_expert)}, f)
    with open(os.path.join(out_dir, "expert_load_ratio.json"), "w") as f:
        json.dump({
            "n_expert": n_expert,
            "n_tokens": int(ids.shape[0]),
            "topk": int(ids.shape[1]),
            "total_assignments": total,
            "experts": {str(e): load[e] for e in range(n_expert)},
        }, f, indent=2)
    produced.append("expert_token_map.json")
    produced.append("expert_load_ratio.json")
    return produced


def process_case(case_dir, n_expert, keep_raw):
    raw_dir = os.path.join(case_dir, "raw")
    if not os.path.isdir(raw_dir):
        print("  skip (no raw/): %s" % case_dir, file=sys.stderr)
        return

    # input_token_ids.bin -> .npy
    in_bin = os.path.join(case_dir, "input_token_ids.bin")
    if os.path.exists(in_bin):
        ids = np.fromfile(in_bin, dtype=np.int32)
        np.save(os.path.join(case_dir, "input_token_ids.npy"), ids)

    by_layer = collect_raw(raw_dir)
    if not by_layer:
        print("  WARN: no MoE tensors found in %s" % raw_dir, file=sys.stderr)
        return

    for layer in sorted(by_layer):
        out_dir = os.path.join(case_dir, "layer_%02d" % layer)
        produced = process_layer(out_dir, by_layer[layer], n_expert)
        print("  layer %02d: %s" % (layer, ", ".join(produced)))

    if not keep_raw:
        shutil.rmtree(raw_dir)
        print("  removed raw dir: %s" % raw_dir)


def main():
    ap = argparse.ArgumentParser(description="MoE dump post-process (raw .bin -> .npy)")
    ap.add_argument("--root", default="/mnt/data8t/zcx/aimet_rx/llama.cpp/output",
                    help="output root containing case_XX dirs and run_metadata.json")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep raw .bin intermediates (default: delete after conversion)")
    ap.add_argument("--n-expert", type=int, default=None,
                    help="override n_expert (default: read run_metadata.json)")
    args = ap.parse_args()

    n_expert = args.n_expert
    meta_path = os.path.join(args.root, "run_metadata.json")
    if n_expert is None and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_expert = int(meta.get("n_expert", 0))
    if not n_expert or n_expert <= 0:
        print("ERROR: n_expert unknown; pass --n-expert", file=sys.stderr)
        return 1

    cases = sorted(d for d in os.listdir(args.root) if d.startswith("case_"))
    if not cases:
        print("ERROR: no case_XX dirs under %s" % args.root, file=sys.stderr)
        return 1

    for c in cases:
        case_dir = os.path.join(args.root, c)
        print("[%s] processing (n_expert=%d)" % (c, n_expert))
        process_case(case_dir, n_expert, args.keep_raw)

    print("post-process done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
