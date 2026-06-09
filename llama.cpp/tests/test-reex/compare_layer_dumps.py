#!/usr/bin/env python3
"""Compare multiple sets of Reex layer tensor dumps against a gold reference.

Evaluation criteria (multi-path mode):
  Part 1 — Gold (F16) vs each REEX path:
           Shows total error from weight quantization (Q4_K) + activation format.
           Metrics: cosine similarity, absolute MAE, max absolute error.
  Part 2 — Cross-comparison between REEX paths (baseline = first --test group).
           Isolates activation-format-specific error.
           PASS requires:  cosine >= threshold  AND  MAE < threshold × 1 LSB.
           1 LSB = max(|baseline_tensor|) / (2^(bits-1) - 1), where bits is the
           activation quantization width (8 for Q8, 16 for Q16).

Usage:
    python compare_layer_dumps.py --gold dump_f32f32 \\
        --test dump_w4f32 "W4×float" \\
        --test dump_w4q16 "W4×INT16" \\
        --test dump_w4q8  "W4×INT8"

    python compare_layer_dumps.py <ref_dir> <test_dir>   # legacy 2-dir mode
"""

import argparse
import glob
import os
import re
import sys

import numpy as np


# ─── Helpers ───────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    dot = np.dot(a64, b64)
    na = np.sqrt(np.dot(a64, a64))
    nb = np.sqrt(np.dot(b64, b64))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(dot / (na * nb))


def compare_arrays(ref: np.ndarray, tst: np.ndarray) -> dict:
    cos = cosine_similarity(ref, tst)
    diff = np.abs(ref.astype(np.float64) - tst.astype(np.float64))
    mae = float(np.mean(diff))
    max_ae = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ref64 = ref.astype(np.float64)
    absmax = float(np.max(np.abs(ref64)))
    mean_abs = float(np.mean(np.abs(ref64)))
    rel_mae = mae / mean_abs if mean_abs > 1e-30 else float("inf")
    return dict(cos=cos, mae=mae, max_ae=max_ae, rmse=rmse,
                rel_mae=rel_mae, absmax=absmax, mean_abs=mean_abs)


def load_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def infer_activation_bits(label: str):
    """Infer activation quantization bits from label. Returns None for float."""
    lo = label.lower()
    if re.search(r'int8|q_?8|×8', lo):
        return 8
    if re.search(r'int16|q_?16|×16', lo):
        return 16
    return None


def estimate_lsb(absmax: float, bits: int) -> float:
    """1 LSB for symmetric quantization: absmax / (2^(bits-1) - 1)."""
    if absmax < 1e-30:
        return 1e-30
    return absmax / (2 ** (bits - 1) - 1)


def subscript(n: int) -> str:
    """Convert integer to Unicode subscript string."""
    table = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(table)


_INDEX_TENSOR_PATTERNS = re.compile(r'argsort|topk', re.IGNORECASE)

def is_index_tensor(name: str) -> bool:
    """Detect discrete integer index tensors (argsort, topk) by name."""
    return bool(_INDEX_TENSOR_PATTERNS.search(name))


def compare_index_arrays(ref: np.ndarray, tst: np.ndarray) -> dict:
    """Metrics for integer index tensors: exact match rate and max difference."""
    ri = np.rint(ref).astype(np.int64)
    ti = np.rint(tst).astype(np.int64)
    exact = float(np.mean(ri == ti))
    max_diff = int(np.max(np.abs(ri - ti))) if len(ri) > 0 else 0
    return dict(exact=exact, max_diff=max_diff)


# ─── Legacy 2-dir mode ────────────────────────────────────────────────────────

def run_legacy(args):
    ref_files = sorted(glob.glob(os.path.join(args.ref_dir, "*.bin")))
    if not ref_files:
        print(f"ERROR: no .bin files found in {args.ref_dir}")
        sys.exit(1)

    all_ok = True
    results = []
    for rf in ref_files:
        name = os.path.basename(rf)
        tf = os.path.join(args.test_dir, name)
        if not os.path.exists(tf):
            print(f"  SKIP  {name} (not found in test dir)")
            continue
        ref = load_bin(rf)
        tst = load_bin(tf)
        assert ref.shape == tst.shape, f"shape mismatch: {ref.shape} vs {tst.shape}"
        r = compare_arrays(ref, tst)
        r["name"] = name
        r["n"] = len(ref)
        r["cos_ok"] = r["cos"] >= args.cos_threshold
        r["mae_ok"] = r["rel_mae"] < args.lsb_q16
        r["ok"] = r["cos_ok"] and r["mae_ok"]
        results.append(r)
        if not r["ok"]:
            all_ok = False

    print("=" * 100)
    print(f"  Comparing: {args.ref_dir}  vs  {args.test_dir}")
    print(f"  Thresholds: cosine >= {args.cos_threshold},  rel_mae < {args.lsb_q16:.2e}")
    print("=" * 100)
    fmt = ("  {name:<28s}  n={n:>8d}  cos={cos:.6f}  mae={mae:.3e}  "
           "max_ae={max_ae:.3e}  rel_mae={rel_mae:.3e}  {status}")
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        flags = []
        if not r["cos_ok"]:
            flags.append("cos")
        if not r["mae_ok"]:
            flags.append("mae")
        if flags:
            status += " (" + ",".join(flags) + ")"
        print(fmt.format(status=status, **r))
    print("=" * 100)
    print(f"  RESULT: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print()
    sys.exit(0 if all_ok else 1)


# ─── Multi-path mode ──────────────────────────────────────────────────────────

def run_multipath(gold_dir: str, tests: list, cos_thr: float, lsb_thr: float):
    gold_files = sorted(glob.glob(os.path.join(gold_dir, "*.bin")))
    if not gold_files:
        print(f"ERROR: no .bin files in gold dir {gold_dir}")
        sys.exit(1)

    tensor_names = [os.path.basename(f) for f in gold_files]
    n_tests = len(tests)
    labels = [t[1] for t in tests]
    bits_list = [infer_activation_bits(l) for l in labels]

    gold = {tn: load_bin(os.path.join(gold_dir, tn)) for tn in tensor_names}
    tdata = []
    for tdir, _ in tests:
        d = {}
        for tn in tensor_names:
            p = os.path.join(tdir, tn)
            if os.path.exists(p):
                d[tn] = load_bin(p)
        tdata.append(d)

    # ════════ Part 1: Gold vs Each ════════

    # Use a uniform LSB reference (Q8_K = 8-bit) so all paths are directly comparable.
    REF_BITS = 8
    ref_lsb_tag = f"LSB{subscript(REF_BITS)}"

    COL1 = 34
    W1 = 32 + COL1 * n_tests
    SEP = "=" * W1
    THIN = "-" * W1

    print()
    print(SEP)
    print(f"  Gold: {gold_dir}")
    print(f"  Tests: {', '.join(labels)}")
    print(f"  Thresholds: cos >= {cos_thr},  MAE/LSB < {lsb_thr}")
    print(SEP)
    print()
    print("  ▸ Part 1: Gold vs Each REEX Path")
    print(f"    mae/{ref_lsb_tag} uses uniform {REF_BITS}-bit LSB "
          f"(= max|gold| / {2**(REF_BITS-1)-1}) for all paths — directly comparable")
    print(f"  {THIN}")

    h1 = f"  {'Tensor':<30s}"
    h2 = f"  {'':<30s}"
    for lb in labels:
        h1 += f"{lb:^{COL1}s}"
        h2 += f"  {'cos':>8s} {'mae':>11s} {'mae/' + ref_lsb_tag:>10s} "
    print(h1)
    print(h2)
    print(f"  {THIN}")

    p1 = [dict(cv=[], max_lsb=0.0, npas=0, ntot=0) for _ in range(n_tests)]
    n_idx_tensors = 0

    for tn in tensor_names:
        g = gold[tn]
        name = tn.replace(".bin", "")
        idx_t = is_index_tensor(name)
        if idx_t:
            n_idx_tensors += 1
        line = f"  {name:<30s}"
        g_absmax = float(np.max(np.abs(g.astype(np.float64))))
        ref_lsb = estimate_lsb(g_absmax, REF_BITS) if g_absmax > 1e-30 else 1e-30

        for i in range(n_tests):
            t = tdata[i].get(tn)
            if t is None:
                line += f"  {'---':^{COL1 - 2}s} "
                continue
            if g.shape != t.shape:
                line += f"  {'SHAPE!':^{COL1 - 2}s} "
                if not idx_t:
                    p1[i]["ntot"] += 1
                continue

            if idx_t:
                ir = compare_index_arrays(g, t)
                pct = ir["exact"] * 100
                md = ir["max_diff"]
                line += f"  {'[idx]':>8s} {pct:>7.1f}%match  Δmax={md:<4d}  "
            else:
                r = compare_arrays(g, t)
                c = r["cos"]
                p1[i]["ntot"] += 1
                p1[i]["cv"].append(c)
                flg = " " if c >= cos_thr else "*"
                if c >= cos_thr:
                    p1[i]["npas"] += 1
                ml = r["mae"] / ref_lsb
                p1[i]["max_lsb"] = max(p1[i]["max_lsb"], ml)
                line += f"  {c:>8.6f}{flg}{r['mae']:>11.4e} {ml:>9.4f}  "
        print(line)

    print(f"  {THIN}")
    if n_idx_tensors:
        print(f"    [idx] = integer index tensor ({n_idx_tensors} total):"
              " evaluated by exact match %, excluded from cos/MAE statistics")
    for i, lb in enumerate(labels):
        s = p1[i]
        mn = min(s["cv"]) if s["cv"] else 0
        av = float(np.mean(s["cv"])) if s["cv"] else 0
        info = (f"    {lb}: cos_min={mn:.6f}  cos_avg={av:.6f}"
                f"  max_mae/{ref_lsb_tag}={s['max_lsb']:.4f}"
                f"  pass(cos>={cos_thr})={s['npas']}/{s['ntot']}")
        print(info)
    print()

    # ════════ Part 2: Cross-Comparison ════════

    if n_tests < 2:
        print("  (Single test group — skipping cross-comparison)")
        print()
        sys.exit(0)

    base_i = 0
    base_label = labels[base_i]
    cross = [(i, labels[i], bits_list[i]) for i in range(n_tests) if i != base_i]
    nc = len(cross)

    COL2 = 32
    W2 = 32 + COL2 * nc
    WM = max(W1, W2)
    SEP2 = "=" * WM
    THIN2 = "-" * WM

    print(f"  ▸ Part 2: Cross-Comparison (baseline: {base_label})")
    bits_with_quant = [b for _, _, b in cross if b]
    if bits_with_quant:
        print(f"    Isolates activation-format error.  1 LSB = max|base| / (2^(bits-1)−1)")
    print(f"  {THIN2}")

    h1 = f"  {'Tensor':<30s}"
    h2 = f"  {'':<30s}"
    for _, lb, bits in cross:
        tag = f"{lb} vs base"
        h1 += f"  {tag:^{COL2 - 2}s}"
        if bits:
            h2 += f"  {'cos':>8s} {'mae':>10s} {'mae/LSB' + subscript(bits):>10s} "
        else:
            h2 += f"  {'cos':>8s} {'mae':>10s} {'':>10s} "
    print(h1)
    print(h2)
    print(f"  {THIN2}")

    p2 = [dict(cv=[], max_lsb=0.0, npas=0, ntot=0) for _ in cross]
    all_cross_pass = True

    for tn in tensor_names:
        base = tdata[base_i].get(tn)
        if base is None:
            continue
        name = tn.replace(".bin", "")
        idx_t = is_index_tensor(name)
        line = f"  {name:<30s}"

        for j, (ci, lb, bits) in enumerate(cross):
            t = tdata[ci].get(tn)
            if t is None:
                line += f"  {'---':^{COL2 - 2}s}"
                continue
            if base.shape != t.shape:
                line += f"  {'SHAPE!':^{COL2 - 2}s}"
                if not idx_t:
                    p2[j]["ntot"] += 1
                    all_cross_pass = False
                continue

            if idx_t:
                ir = compare_index_arrays(base, t)
                pct = ir["exact"] * 100
                md = ir["max_diff"]
                line += f"  {'[idx]':>8s} {pct:>7.1f}%match  Δmax={md:<4d} "
            else:
                r = compare_arrays(base, t)
                c = r["cos"]
                p2[j]["ntot"] += 1
                p2[j]["cv"].append(c)
                fc = " " if c >= cos_thr else "*"

                if bits and r["absmax"] > 1e-30:
                    lsb_val = estimate_lsb(r["absmax"], bits)
                    ml = r["mae"] / lsb_val
                    p2[j]["max_lsb"] = max(p2[j]["max_lsb"], ml)
                    fl = " " if ml < lsb_thr else "!"
                    lsb_s = f"{ml:>9.4f}{fl}"
                    ok = (c >= cos_thr) and (ml < lsb_thr)
                else:
                    lsb_s = f"{'N/A':>10s}"
                    ok = (c >= cos_thr)

                if ok:
                    p2[j]["npas"] += 1
                else:
                    all_cross_pass = False

                line += f"  {c:>8.6f}{fc}{r['mae']:>10.2e} {lsb_s} "

        print(line)

    print(f"  {THIN2}")
    for j, (ci, lb, bits) in enumerate(cross):
        s = p2[j]
        mn = min(s["cv"]) if s["cv"] else 0.0
        av = float(np.mean(s["cv"])) if s["cv"] else 0.0
        info = f"    {lb}: cos_min={mn:.6f}  cos_avg={av:.6f}"
        if bits:
            info += f"  max_mae/LSB{subscript(bits)}={s['max_lsb']:.4f}"
        info += f"  pass={s['npas']}/{s['ntot']}"
        print(info)

    print()
    print(SEP2)
    if all_cross_pass:
        print(f"  VERDICT: ALL PASS  (cos >= {cos_thr}  AND  MAE < {lsb_thr} LSB)")
    else:
        print(f"  VERDICT: SOME FAILED  (* = cos < {cos_thr}   ! = MAE >= {lsb_thr} LSB)")
    print(SEP2)
    print()
    sys.exit(0 if all_cross_pass else 1)


# ─── Entry ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ref_dir", nargs="?", default=None,
                        help="(Legacy) Reference dump directory")
    parser.add_argument("test_dir", nargs="?", default=None,
                        help="(Legacy) Test dump directory")
    parser.add_argument("--gold", type=str, default=None,
                        help="Gold reference directory (multi-path mode)")
    parser.add_argument("--test", nargs=2, action="append", metavar=("DIR", "LABEL"),
                        help="Test directory and label (can repeat)")
    parser.add_argument("--cos-threshold", type=float, default=0.9999,
                        help="Cosine similarity threshold (default: 0.9999)")
    parser.add_argument("--mae-lsb-threshold", type=float, default=1.0,
                        help="MAE/LSB threshold for cross-comparison (default: 1.0)")
    parser.add_argument("--lsb-q16", type=float, default=1.0 / 32768.0,
                        help="Relative MAE threshold for legacy mode")

    args = parser.parse_args()

    if args.gold and args.test:
        run_multipath(args.gold, args.test, args.cos_threshold, args.mae_lsb_threshold)
    elif args.ref_dir and args.test_dir:
        run_legacy(args)
    else:
        parser.print_help()
        print("\nError: provide either --gold + --test or positional ref_dir test_dir")
        sys.exit(2)


if __name__ == "__main__":
    main()
