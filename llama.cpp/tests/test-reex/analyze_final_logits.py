#!/usr/bin/env python3
"""Direct numerical KL/JS/top-k analysis on a pair of dump dirs.

Usage:
  analyze_final_logits.py <gold_dir> <test_dir> [<test_dir2> ...] \
      [--tensors final_logits l_out-39 l_out-0]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x.astype(np.float64))
    return e / e.sum()


def kl_js(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    kl = float(np.sum(p * np.log(p / q)))
    m = 0.5 * (p + q)
    js = 0.5 * float(np.sum(p * np.log(p / m))) + 0.5 * float(np.sum(q * np.log(q / m)))
    return kl, js


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    abs_diff = np.abs(diff)
    return dict(
        n=int(a.size),
        cos=cosine(a, b),
        mae=float(abs_diff.mean()),
        max=float(abs_diff.max()),
        rmse=float(math.sqrt(float((diff ** 2).mean()))),
        gold_max=float(np.abs(a).max()),
        gold_std=float(a.std()),
    )


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    ia = np.argpartition(a, -k)[-k:]
    ib = np.argpartition(b, -k)[-k:]
    inter = len(set(ia.tolist()) & set(ib.tolist()))
    return {f"top{k}_overlap": inter, f"top{k}_n": k, f"top{k}_ratio": inter / k}


def load_bin(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", type=Path)
    ap.add_argument("tests", nargs="+", type=Path)
    ap.add_argument("--tensors", nargs="+",
                    default=["final_logits", "l_out-0", "l_out-9",
                             "l_out-19", "l_out-29", "l_out-39"])
    args = ap.parse_args()

    for tname in args.tensors:
        gold_path = args.gold / f"{tname}.bin"
        if not gold_path.exists():
            print(f"--- {tname}: NO GOLD ---")
            continue
        gold = load_bin(gold_path)
        print(f"\n=== {tname}  (n={gold.size}) ===")
        for tdir in args.tests:
            test_path = tdir / f"{tname}.bin"
            if not test_path.exists():
                print(f"  {tdir.name:<40} MISSING")
                continue
            test = load_bin(test_path)
            if test.size != gold.size:
                print(f"  {tdir.name:<40} SHAPE MISMATCH {test.size} vs {gold.size}")
                continue
            s = stats(gold, test)
            row = (f"  {tdir.name:<40} cos={s['cos']:.6f} "
                   f"mae={s['mae']:.3e} max={s['max']:.3e} rmse={s['rmse']:.3e}")
            if tname == "final_logits":
                pg = softmax(gold)
                pt = softmax(test)
                kl, js = kl_js(pg, pt)
                tops = {}
                for k in (1, 5, 10):
                    tops.update(topk_overlap(gold, test, k))
                row += (f"  KL={kl:.3e} JS={js:.3e}"
                        f"  top1={tops['top1_overlap']}/1"
                        f" top5={tops['top5_overlap']}/5"
                        f" top10={tops['top10_overlap']}/10")
            print(row)


if __name__ == "__main__":
    sys.exit(main())
