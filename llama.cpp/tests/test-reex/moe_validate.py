#!/usr/bin/env python3
"""MoE 采数严格校验脚本.

针对每个 case/layer 校验:
  1. 每个 token 恰好 topk (默认 8) 个 expert, 且 top-k 内 expert 互不相同.
  2. 每层 expert_token_map 总项数 == n_tokens * topk (如 8192*8 = 65536).
  3. 重放 expert_token_map 能完整还原 router_expert_ids 的每个 (token, topk_slot).
  4. expert_outputs_unweighted / expert_outputs_weighted 的 token/topk 维度与
     router_expert_ids 一致.
  5. sum(expert_outputs_weighted over topk) 与 moe_out 在按 dtype 自动选择的容差内一致,
     报告 max/mean error, 并把阈值写入 validation_report.json.

非零退出码表示存在校验失败.
"""
import argparse
import json
import os
import sys

import numpy as np

# 按 dtype 自动选择的绝对容差 (weighted-sum vs moe_out).
DTYPE_ATOL = {
    np.dtype(np.float32): 1e-2,
    np.dtype(np.float16): 2e-1,
}


def fail(errors, msg):
    errors.append(msg)
    print("    FAIL: %s" % msg, file=sys.stderr)


def validate_layer(layer_dir, n_expert):
    errors = []
    report = {"layer_dir": layer_dir}

    ids_path = os.path.join(layer_dir, "router_expert_ids.npy")
    if not os.path.exists(ids_path):
        fail(errors, "missing router_expert_ids.npy")
        return errors, report
    ids = np.load(ids_path)
    n_tokens, topk = ids.shape
    report["n_tokens"] = int(n_tokens)
    report["topk"] = int(topk)

    # 1. 每个 token 恰好 topk 个 expert, top-k 内不重复.
    if ids.shape[1] != topk:
        fail(errors, "router_expert_ids second dim != topk")
    dup_tokens = 0
    for t in range(n_tokens):
        if len(np.unique(ids[t])) != topk:
            dup_tokens += 1
    if dup_tokens:
        fail(errors, "%d tokens have duplicate experts in top-k" % dup_tokens)
    if ids.min() < 0 or ids.max() >= n_expert:
        fail(errors, "expert id out of range [0, %d)" % n_expert)

    # 2 + 3. expert_token_map 计数 + 重放.
    map_path = os.path.join(layer_dir, "expert_token_map.json")
    if not os.path.exists(map_path):
        fail(errors, "missing expert_token_map.json")
    else:
        with open(map_path) as f:
            token_map = json.load(f)
        total = sum(len(v) for v in token_map.values())
        report["expert_map_total"] = total
        report["expert_map_expected"] = int(n_tokens * topk)
        if total != n_tokens * topk:
            fail(errors, "expert_token_map total %d != %d" % (total, n_tokens * topk))
        # replay
        recon = np.full((n_tokens, topk), -1, dtype=np.int64)
        for e_str, entries in token_map.items():
            e = int(e_str)
            for ent in entries:
                recon[ent["token_id"], ent["topk_slot"]] = e
        if not np.array_equal(recon, ids.astype(np.int64)):
            mismatches = int(np.sum(recon != ids.astype(np.int64)))
            fail(errors, "expert_token_map replay mismatch in %d slots" % mismatches)

    # 4. expert_outputs_* 维度一致.
    unw_path = os.path.join(layer_dir, "expert_outputs_unweighted.npy")
    w_path = os.path.join(layer_dir, "expert_outputs_weighted.npy")
    out_path = os.path.join(layer_dir, "moe_out.npy")
    weighted = None
    for nm, p in (("unweighted", unw_path), ("weighted", w_path)):
        if not os.path.exists(p):
            fail(errors, "missing expert_outputs_%s.npy" % nm)
            continue
        a = np.load(p)
        if a.shape[0] != n_tokens or a.shape[1] != topk:
            fail(errors, "expert_outputs_%s token/topk dims %s != (%d,%d)"
                 % (nm, a.shape[:2], n_tokens, topk))
        if nm == "weighted":
            weighted = a

    # 5. sum(weighted over topk) vs moe_out.
    if weighted is not None and os.path.exists(out_path):
        moe_out = np.load(out_path)
        wsum = weighted.astype(np.float64).sum(axis=1)  # [token, hidden]
        if wsum.shape != moe_out.shape:
            fail(errors, "weighted-sum shape %s != moe_out %s" % (wsum.shape, moe_out.shape))
        else:
            diff = np.abs(wsum - moe_out.astype(np.float64))
            max_err = float(diff.max())
            mean_err = float(diff.mean())
            atol = DTYPE_ATOL.get(np.dtype(moe_out.dtype), 1e-2)
            report["weighted_sum_vs_moe_out"] = {
                "dtype": str(moe_out.dtype),
                "atol": atol,
                "max_abs_err": max_err,
                "mean_abs_err": mean_err,
                "passed": bool(max_err <= atol),
            }
            print("    weighted-sum vs moe_out: max=%.3e mean=%.3e (atol=%.1e, dtype=%s)"
                  % (max_err, mean_err, atol, moe_out.dtype))
            if max_err > atol:
                fail(errors, "weighted-sum vs moe_out max_err %.3e > atol %.1e" % (max_err, atol))
    elif weighted is not None:
        report["weighted_sum_vs_moe_out"] = {"skipped": "moe_out.npy missing"}

    report["errors"] = errors
    return errors, report


def main():
    ap = argparse.ArgumentParser(description="MoE dump strict validation")
    ap.add_argument("--root", default="/mnt/data8t/zcx/aimet_rx/llama.cpp/output")
    ap.add_argument("--n-expert", type=int, default=None)
    args = ap.parse_args()

    n_expert = args.n_expert
    meta_path = os.path.join(args.root, "run_metadata.json")
    if n_expert is None and os.path.exists(meta_path):
        with open(meta_path) as f:
            n_expert = int(json.load(f).get("n_expert", 0))
    if not n_expert or n_expert <= 0:
        print("ERROR: n_expert unknown; pass --n-expert", file=sys.stderr)
        return 2

    cases = sorted(d for d in os.listdir(args.root) if d.startswith("case_"))
    all_errors = 0
    full_report = {"n_expert": n_expert, "cases": {}}
    for c in cases:
        case_dir = os.path.join(args.root, c)
        layers = sorted(d for d in os.listdir(case_dir) if d.startswith("layer_"))
        print("[%s] validating %d layers" % (c, len(layers)))
        case_report = {}
        for ld in layers:
            layer_dir = os.path.join(case_dir, ld)
            print("  %s" % ld)
            errors, report = validate_layer(layer_dir, n_expert)
            case_report[ld] = report
            all_errors += len(errors)
        full_report["cases"][c] = case_report

    report_path = os.path.join(args.root, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print("validation report -> %s" % report_path)

    if all_errors:
        print("VALIDATION FAILED: %d errors" % all_errors, file=sys.stderr)
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
