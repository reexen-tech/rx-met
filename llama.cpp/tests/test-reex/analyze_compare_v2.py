#!/usr/bin/env python3
"""Group per-tensor pass/cos statistics from compare_layer_dumps_v2.py output.

Usage:
  analyze_compare_v2.py <compare.log> [--cos-thr 0.9999]
"""
import argparse
import re
import sys
from collections import defaultdict
from statistics import mean, median


_NUM = r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?"
ROW = re.compile(
    rf"^\s+(?P<name>[A-Za-z_][\w]*)-(?P<layer>\d+)\s+"
    rf"(?P<cos>-?\d\.\d+)[*~]?\s+"
    rf"(?P<mae>{_NUM})[+\^!]?\s+"
    rf"(?P<maxv>{_NUM})[+\^!]?\s+"
    rf"(?P<rmse>{_NUM})[+\^!]?\s+"
    rf"(?P<lsb>{_NUM})[+\^!]?\s*$"
)
IDX_ROW = re.compile(
    r"^\s+(?P<name>[A-Za-z_][\w]*)-(?P<layer>\d+)\s+\[idx\]\s+(?P<m>[\d.]+)%"
)


GROUPS = [
    ("attn",     re.compile(r"^(attn_|Qcur|Kcur|Vcur|kqv|attn_output|attn_norm|attn_post_norm|attn_q|attn_k|attn_v|attn_residual)")),
    ("ssm",      re.compile(r"^(ssm_|conv_|state_|alpha|beta|gate|qkv_mixed|linear_attn|new_state|last_conv|__fgdn|q_conv|k_conv|v_conv|z$|z-)")),
    ("ffn_moe",  re.compile(r"^ffn_moe_")),
    ("ffn_se",   re.compile(r"^(ffn_gate$|ffn_up$|ffn_down$|ffn_swiglu$|ffn_shexp|shared_expert|ffn_out$|ffn_norm)")),
    ("norm",     re.compile(r"^(norm$|ffn_norm|post_attention_norm)")),
    ("misc",     re.compile(r".*")),
]


def classify(name: str) -> str:
    for grp, pat in GROUPS:
        if pat.match(name):
            return grp
    return "misc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--cos-thr", type=float, default=0.9999)
    ap.add_argument("--top-bad", type=int, default=0,
                    help="print top-N tensors with lowest cosine across all logs")
    ap.add_argument("--summary-only", action="store_true",
                    help="print compact one-line summary per log "
                         "(label cos_min cos_avg pass l_out-39 final_logits topk-39)")
    args = ap.parse_args()

    if args.summary_only:
        print(f"{'label':<35} {'cos_min':>9} {'cos_avg':>9} {'pass':>10} "
              f"{'l_out-39':>9} {'final_lo':>9} {'topk-39':>9}")
        print("-" * 95)
    else:
        print(f"{'group':<10} {'n':>5} {'pass':>5} {'pass%':>7} "
              f"{'cos_min':>9} {'cos_med':>9} {'cos_avg':>9}  layers")
        print("-" * 90)

    all_rows = []
    for log_path in args.logs:
        rows = []
        idx_rows = []
        with open(log_path) as f:
            for line in f:
                m = ROW.match(line)
                if m:
                    rows.append((m["name"], int(m["layer"]),
                                 float(m["cos"]), float(m["mae"]),
                                 float(m["maxv"]), float(m["rmse"]),
                                 float(m["lsb"])))
                    continue
                im = IDX_ROW.match(line)
                if im:
                    idx_rows.append((im["name"], int(im["layer"]),
                                     float(im["m"])))

        if not rows:
            print(f"  (no rows parsed from {log_path})")
            continue

        all_rows.extend((log_path, *r) for r in rows)

        if args.summary_only:
            cs = [c for _, _, c, *_ in rows]
            passes = sum(1 for c in cs if c >= args.cos_thr)
            d = {f"{n}-{l}": c for n, l, c, *_ in rows}
            label = log_path.rsplit("/", 1)[-1]
            print(f"{label:<35} {min(cs):>9.4f} {mean(cs):>9.4f} "
                  f"{passes:>4}/{len(cs):<5} "
                  f"{d.get('l_out-39', float('nan')):>9.4f} "
                  f"{d.get('final_logits-0', d.get('final_logits-9999', float('nan'))):>9.4f} "
                  f"{next((m for n,l,m in idx_rows if n=='ffn_moe_topk' and l==39), float('nan')):>9.2f}")
            continue

        print(f"\n=== {log_path} ===")
        groups: dict[str, list] = defaultdict(list)
        for name, layer, cos, mae, mx, rmse, lsb in rows:
            groups[classify(name)].append((name, layer, cos))

        total_pass = sum(1 for _, _, c in
                         ((n, l, c) for g in groups.values() for n, l, c in g)
                         if c >= args.cos_thr)
        for grp in [g for g, _ in GROUPS]:
            items = groups.get(grp, [])
            if not items:
                continue
            cs = [c for _, _, c in items]
            passes = sum(1 for c in cs if c >= args.cos_thr)
            layers = sorted({l for _, l, _ in items})
            lr = f"[{layers[0]}..{layers[-1]}]" if layers else "[]"
            print(f"{grp:<10} {len(cs):>5} {passes:>5} "
                  f"{100*passes/len(cs):>6.2f}% "
                  f"{min(cs):>9.4f} {median(cs):>9.4f} {mean(cs):>9.4f}  {lr}")
        print(f"{'TOTAL':<10} {len(rows):>5} {total_pass:>5} "
              f"{100*total_pass/len(rows):>6.2f}%")

        if idx_rows:
            avg = mean(m for _, _, m in idx_rows)
            mn = min(m for _, _, m in idx_rows)
            print(f"\nindex tensors: n={len(idx_rows)} avg_match={avg:.2f}% min={mn:.2f}%")
            for name in sorted({n for n, _, _ in idx_rows}):
                vals = [m for nm, _, m in idx_rows if nm == name]
                print(f"  {name:<24} n={len(vals)} avg={mean(vals):.2f}% min={min(vals):.2f}%")

    if args.top_bad:
        bad = sorted(all_rows, key=lambda r: r[3])[: args.top_bad]
        print(f"\n=== top {args.top_bad} worst-cos tensors across {len(args.logs)} log(s) ===")
        print(f"{'log':<40} {'tensor':<28} {'cos':>9} {'mae':>11} {'max':>11} {'rmse':>11} {'mae/LSB':>9}")
        for lp, name, layer, cos, mae, mx, rmse, lsb in bad:
            tag = lp.rsplit("/", 1)[-1]
            print(f"{tag:<40} {f'{name}-{layer}':<28} "
                  f"{cos:>9.4f} {mae:>11.3e} {mx:>11.3e} {rmse:>11.3e} {lsb:>9.2f}")


if __name__ == "__main__":
    sys.exit(main())
