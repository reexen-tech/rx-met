#!/usr/bin/env python3
"""Compare Reex layer tensor dumps against a gold reference (v2 metrics).

规范文档：`tests/test-reex/数据精度评价说明_v2.md`
v1 脚本（保留 rel_mae）：`compare_layer_dumps.py`

指标分层
---------
算子 / 层 / block（Part 1 & 2）：
  cosine, MAE, max, RMSE, mae/LSB

全模型 / 分布（Part 3，仅概率语义张量）：
  KL(P_ref || P_test), JS(P_ref, P_test)

三部分对比逻辑
--------------
Part 1 — Gold vs 各 REEX 路径
  MAE = mean(|test - gold|)
  mae/LSB_8 = MAE / (max|gold| / 127)     # 所有路径统一 8-bit 分母，便于横向比较
  pass：仅统计 cosine >= 阈值；mae/LSB 为辅助列

Part 2 — REEX 路径交叉对比（baseline = 第一个 --test）
  MAE = mean(|test - baseline|)
  mae/LSB_b = MAE / (max|baseline| / (2^(b-1)-1))   # b 取待比路径激活位宽
  pass：cosine >= 阈值 且 mae/LSB_b < 阈值

Part 3 — 分布指标（Gold vs 各路径，默认开启，--no-distribution 可跳过）
  final_logits / result_output：先 softmax 再算 KL / JS
  ffn_moe_probs / *_probs / softmax：clip + 重归一化后直接算 KL / JS
  不对 Qcur/Kcur/Vcur 等有符号激活计算 KL（见文档 §4.3）

输出标记（见 §8.1 / MarkerThresholds / print_marker_legend）
--------
  cos 列尾：  (空) 达标  |  ~ 提示关注  |  * 强告警/不达标
  mae/LSB 列尾：  (空) 达标  |  ! 超阈值
  max 列尾：  + 单点 outlier 偏大（max > ratio×MAE）
  rmse 列尾：  ^ RMSE 显著大于 MAE（outlier 主导）
  [idx]：  match% 尾 * 未达 match 阈值；Δmax 尾 ! 非零偏差
  异常：  --- 缺文件  |  #SHP  shape 不一致
  Part 3 KL/JS 尾：  ~ 超关注阈值  |  * 超告警阈值

用法
----
    # multipath（推荐）
    python compare_layer_dumps_v2.py --gold dump_f32f32 \\
        --test dump_w4f32 "W4×float" \\
        --test dump_w4q16 "W4×INT16" \\
        --test dump_w4q8  "W4×INT8"

    # legacy 双目录（pass = cos 且 mae/LSB_8）
    python compare_layer_dumps_v2.py dump_ref dump_test

    # 跳过 Part 3
    python compare_layer_dumps_v2.py --gold ... --test ... --no-distribution
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class MarkerThresholds:
    """各指标不达标 / 异常判定阈值（与文档 §3.1、§3.4、§3.6 对齐）。"""

    cos: float = 0.9999          # pass 阈值
    cos_warn: float = 0.999      # [cos_warn, cos) → ~ 提示关注
    lsb: float = 1.0             # mae/LSB 上限
    rmse_mae_ratio: float = 3.0  # RMSE > ratio×MAE → ^ outlier 主导
    max_mae_ratio: float = 10.0  # max > ratio×MAE → + 单点异常
    idx_match: float = 1.0       # 索引张量 exact match 下限（1.0 = 100%）
    kl_warn: float = 0.01        # Part 3 KL 关注线
    kl_alert: float = 0.1        # Part 3 KL 强告警
    js_warn: float = 0.005
    js_alert: float = 0.05


def cos_marker(cos: float, thr: MarkerThresholds) -> str:
    if cos >= thr.cos:
        return " "
    if cos >= thr.cos_warn:
        return "~"
    return "*"


def lsb_marker(mae_lsb: float, thr: MarkerThresholds) -> str:
    return " " if mae_lsb < thr.lsb else "!"


def rmse_marker(rmse: float, mae: float, thr: MarkerThresholds) -> str:
    if mae < 1e-30:
        return " "
    return "^" if rmse > mae * thr.rmse_mae_ratio else " "


def max_marker(max_ae: float, mae: float, thr: MarkerThresholds) -> str:
    if mae < 1e-30:
        return " "
    return "+" if max_ae > mae * thr.max_mae_ratio else " "


def dist_marker(kl: float, js: float, thr: MarkerThresholds) -> tuple[str, str]:
    if kl >= thr.kl_alert:
        km = "*"
    elif kl >= thr.kl_warn:
        km = "~"
    else:
        km = " "
    if js >= thr.js_alert:
        jm = "*"
    elif js >= thr.js_warn:
        jm = "~"
    else:
        jm = " "
    return km, jm


def print_marker_legend(thr: MarkerThresholds, *, part2_lsb: bool = True) -> None:
    print("    标记: cos (空)=达标  ~=提示关注(cos∈["
          f"{thr.cos_warn},{thr.cos}))  *=强告警(cos<{thr.cos_warn})")
    lsb_note = f"mae/LSB!=超{thr.lsb}" if part2_lsb else f"mae/LSB!=超{thr.lsb}(辅助)"
    print(f"          {lsb_note}  max+=outlier  rmse^=outlier主导  "
          f"idx:*=match<{thr.idx_match:.0%} !=Δmax>0")
    print("          异常:---缺文件 #SHP=shape  KL/JS:~关注 *=告警")


# ─── 算子级指标（文档 §3.1–§3.4、§3.6）────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度（文档 §3.1）。

    cos(r,t) = (r·t) / (||r||_2 * ||t||_2)

    零向量边界（与 common_embd_similarity_cos 一致）：
      - r=t=0 → 1.0
      - 仅一侧为零 → 0.0
    """
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    na = np.sqrt(np.dot(a64, a64))
    nb = np.sqrt(np.dot(b64, b64))
    if na < 1e-30 and nb < 1e-30:
        return 1.0
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


def compare_arrays(ref: np.ndarray, tst: np.ndarray) -> dict:
    """逐元素对比 ref（Gold / baseline）与 tst，返回 v2 算子级指标。

    Returns:
        cos, mae, max_ae, rmse, absmax
        absmax = max_i |ref_i|，供 estimate_lsb() 使用
    """
    cos = cosine_similarity(ref, tst)
    diff = np.abs(ref.astype(np.float64) - tst.astype(np.float64))
    mae = float(np.mean(diff))
    max_ae = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ref64 = ref.astype(np.float64)
    absmax = float(np.max(np.abs(ref64)))
    return dict(
        cos=cos,
        mae=mae,
        max_ae=max_ae,
        rmse=rmse,
        absmax=absmax,
    )


# ─── 分布指标（文档 §4.1–§4.2）────────────────────────────────────────────────

def normalize_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """clip 到 [eps, +inf) 后归一化为概率分布；全零时退化为均匀分布。"""
    p64 = np.clip(p.astype(np.float64), eps, None)
    s = np.sum(p64)
    if s < eps:
        n = len(p64)
        return np.full(n, 1.0 / n, dtype=np.float64)
    return p64 / s


def softmax_probs(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """数值稳定 softmax：P_i = exp(x_i - max(x)) / sum(...)，再 clip + 重归一化。"""
    x64 = x.astype(np.float64)
    e = np.exp(x64 - np.max(x64))
    return normalize_prob(e, eps)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(P||Q) = sum_i P_i * (log P_i - log Q_i)；非对称，P 为参考侧。"""
    p_n = normalize_prob(p, eps)
    q_n = normalize_prob(q, eps)
    return float(np.sum(p_n * (np.log(p_n) - np.log(q_n))))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """JS(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)，M = (P+Q)/2；对称且更稳定。"""
    p_n = normalize_prob(p, eps)
    q_n = normalize_prob(q, eps)
    m = 0.5 * (p_n + q_n)
    return 0.5 * kl_divergence(p_n, m, eps) + 0.5 * kl_divergence(q_n, m, eps)


def kl_logits(ref: np.ndarray, test: np.ndarray, eps: float = 1e-12) -> float:
    """对 logits 先 softmax 再算 KL(P_ref || P_test)（文档 §4.1.2）。"""
    return kl_divergence(softmax_probs(ref, eps), softmax_probs(test, eps), eps)


def js_logits(ref: np.ndarray, test: np.ndarray, eps: float = 1e-12) -> float:
    """对 logits 先 softmax 再算 JS(P_ref, P_test)。"""
    return js_divergence(softmax_probs(ref, eps), softmax_probs(test, eps), eps)


def distribution_metrics(ref: np.ndarray, tst: np.ndarray, kind: str,
                         eps: float = 1e-12) -> dict:
    """按张量语义选择 logits（softmax）或 probs（直接归一化）路径。"""
    if kind == "logits":
        kl = kl_logits(ref, tst, eps)
        js = js_logits(ref, tst, eps)
    else:
        kl = kl_divergence(ref, tst, eps)
        js = js_divergence(ref, tst, eps)
    return dict(kl=kl, js=js)


# ─── I/O 与辅助 ───────────────────────────────────────────────────────────────

def load_bin(path: str) -> np.ndarray:
    """读取 dump 的 float32 平坦数组（文档 §2 数据格式）。"""
    return np.fromfile(path, dtype=np.float32)


def infer_activation_bits(label: str):
    """从 --test 标签推断激活量化位宽，供 Part 2 的 mae/LSB_b 使用。

    匹配 INT8 / Q8 / ×8 → 8；INT16 / Q16 / ×16 → 16；否则 None（float 路径）。
    """
    lo = label.lower()
    if re.search(r"int8|q_?8|×8", lo):
        return 8
    if re.search(r"int16|q_?16|×16", lo):
        return 16
    return None


def estimate_lsb(absmax: float, bits: int) -> float:
    """对称有符号量化步长：LSB_b = absmax / (2^(b-1) - 1)（文档 §3.6）。"""
    if absmax < 1e-30:
        return 1e-30
    return absmax / (2 ** (bits - 1) - 1)


def subscript(n: int) -> str:
    """将位宽数字转为下标字符，用于表头 mae/LSB₈、mae/LSB₁₆。"""
    table = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(table)


# 张量名匹配规则（Part 3 与索引张量特殊处理）
_INDEX_TENSOR_PATTERNS = re.compile(r"argsort|topk", re.IGNORECASE)
_DIST_LOGITS_PATTERNS = re.compile(r"final_logits|result_output", re.IGNORECASE)
_DIST_PROBS_PATTERNS = re.compile(r"ffn_moe_probs|_probs|softmax", re.IGNORECASE)


def is_index_tensor(name: str) -> bool:
    """离散索引张量不参与 cos/MAE 汇总，改用 exact match %。"""
    return bool(_INDEX_TENSOR_PATTERNS.search(name))


def distribution_kind(name: str) -> str | None:
    """判断张量是否适合 Part 3 分布指标；None 表示跳过 KL/JS。"""
    base = name.replace(".bin", "")
    if _DIST_LOGITS_PATTERNS.search(base):
        return "logits"
    if _DIST_PROBS_PATTERNS.search(base):
        return "probs"
    return None


def compare_index_arrays(ref: np.ndarray, tst: np.ndarray) -> dict:
    """索引张量：exact match 比例与最大整数偏差 Δmax。"""
    ri = np.rint(ref).astype(np.int64)
    ti = np.rint(tst).astype(np.int64)
    exact = float(np.mean(ri == ti))
    max_diff = int(np.max(np.abs(ri - ti))) if len(ri) > 0 else 0
    return dict(exact=exact, max_diff=max_diff)


def format_metric_block(r: dict, thr: MarkerThresholds, ref_lsb: float,
                        check_lsb: bool) -> str:
    """格式化 cos / mae / max / rmse / mae/LSB 列块，末尾附不达标标记。"""
    cm = cos_marker(r["cos"], thr)
    ml = r["mae"] / ref_lsb
    lm = lsb_marker(ml, thr) if check_lsb else " "
    xm = max_marker(r["max_ae"], r["mae"], thr)
    rm = rmse_marker(r["rmse"], r["mae"], thr)
    return (
        f"  {r['cos']:>8.6f}{cm}"
        f"{r['mae']:>11.3e}"
        f"{r['max_ae']:>11.3e}{xm}"
        f"{r['rmse']:>11.3e}{rm}"
        f" {ml:>8.4f}{lm} "
    )


def format_index_block(ir: dict, thr: MarkerThresholds) -> str:
    """索引张量列块：match% 与 Δmax 附不达标标记。"""
    pct = ir["exact"] * 100
    match_mk = "*" if ir["exact"] < thr.idx_match else " "
    delta_mk = "!" if ir["max_diff"] > 0 else " "
    return (
        f"  {'[idx]':>8s} {pct:>7.1f}%{match_mk}match"
        f"  Δmax={ir['max_diff']:<4d}{delta_mk}  "
    )


def format_dist_block(kl: float, js: float, thr: MarkerThresholds) -> str:
    km, jm = dist_marker(kl, js, thr)
    return f"  {kl:>12.6e}{km} {js:>12.6e}{jm} "


# ─── Legacy 2-dir mode ────────────────────────────────────────────────────────

def run_legacy(args, thr: MarkerThresholds):
    """双目录模式：ref_dir vs test_dir，逐 .bin 文件对比。

    与 v1 差异：主输出含 max/rmse；pass 用 mae/LSB_8（Gold absmax）而非 rel_mae。
    若存在概率语义张量，末尾附加 Distribution metrics 小节。
    """
    ref_files = sorted(glob.glob(os.path.join(args.ref_dir, "*.bin")))
    if not ref_files:
        print(f"ERROR: no .bin files found in {args.ref_dir}")
        sys.exit(1)

    ref_lsb_tag = f"LSB{subscript(8)}"
    all_ok = True
    results = []
    dist_rows = []

    for rf in ref_files:
        name = os.path.basename(rf)
        tf = os.path.join(args.test_dir, name)
        if not os.path.exists(tf):
            print(f"  SKIP  {name} (not found in test dir)")
            continue
        ref = load_bin(rf)
        tst = load_bin(tf)
        assert ref.shape == tst.shape, f"shape mismatch: {ref.shape} vs {tst.shape}"

        base = name.replace(".bin", "")
        if is_index_tensor(base):
            ir = compare_index_arrays(ref, tst)
            idx_ok = ir["exact"] >= thr.idx_match and ir["max_diff"] == 0
            results.append(dict(
                name=name, kind="idx", exact=ir["exact"], max_diff=ir["max_diff"],
                ok=idx_ok,
            ))
            if not idx_ok:
                all_ok = False
            continue

        r = compare_arrays(ref, tst)
        # legacy 与 Part 1 相同：LSB 分母固定 8-bit，absmax 来自 Gold（ref）
        ref_lsb = estimate_lsb(r["absmax"], 8)
        mae_lsb = r["mae"] / ref_lsb
        r.update(
            name=name,
            kind="float",
            n=len(ref),
            mae_lsb=mae_lsb,
            cos_ok=r["cos"] >= thr.cos,
            mae_lsb_ok=mae_lsb < thr.lsb,
        )
        r["ok"] = r["cos_ok"] and r["mae_lsb_ok"]
        results.append(r)
        if not r["ok"]:
            all_ok = False

        kind = distribution_kind(name)
        if kind and not args.no_distribution:
            dm = distribution_metrics(ref, tst, kind, args.dist_eps)
            dist_rows.append(dict(name=base, kind=kind, **dm))

    print("=" * 110)
    print(f"  Comparing: {args.ref_dir}  vs  {args.test_dir}")
    print(f"  Thresholds: cosine >= {thr.cos},  "
          f"mae/{ref_lsb_tag} < {thr.lsb}")
    print_marker_legend(thr, part2_lsb=True)
    print("=" * 110)
    for r in results:
        if r["kind"] == "idx":
            ir = dict(exact=r["exact"], max_diff=r["max_diff"])
            blk = format_index_block(ir, thr).strip()
            status = "PASS" if r["ok"] else "FAIL"
            print(f"  {r['name']:<28s}  {blk}  {status}")
            continue
        cm = cos_marker(r["cos"], thr)
        lm = lsb_marker(r["mae_lsb"], thr)
        xm = max_marker(r["max_ae"], r["mae"], thr)
        rm = rmse_marker(r["rmse"], r["mae"], thr)
        status = "PASS" if r["ok"] else "FAIL"
        flags = []
        if not r["cos_ok"]:
            flags.append("cos")
        if not r["mae_lsb_ok"]:
            flags.append("mae/LSB")
        if flags:
            status += " (" + ",".join(flags) + ")"
        print(
            f"  {r['name']:<28s}  n={r['n']:>8d}  "
            f"cos={r['cos']:.6f}{cm}  mae={r['mae']:.3e}  "
            f"max={r['max_ae']:.3e}{xm}  rmse={r['rmse']:.3e}{rm}  "
            f"mae/{ref_lsb_tag}={r['mae_lsb']:.4f}{lm}  {status}"
        )

    if dist_rows:
        print()
        print("  ▸ Distribution metrics (KL / JS)")
        print("  " + "-" * 80)
        for row in dist_rows:
            km, jm = dist_marker(row["kl"], row["js"], thr)
            print(f"  {row['name']:<28s}  kind={row['kind']:<6s}  "
                  f"KL={row['kl']:.6e}{km}  JS={row['js']:.6e}{jm}")

    print("=" * 110)
    print(f"  RESULT: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print()
    sys.exit(0 if all_ok else 1)


# ─── Multi-path mode ──────────────────────────────────────────────────────────

def run_multipath(gold_dir: str, tests: list, thr: MarkerThresholds,
                  no_distribution: bool, dist_eps: float):
    """多路径对比：Part 1（Gold vs 各路径）→ Part 3（分布）→ Part 2（交叉）。

    tests: [(dir_path, label), ...]；第一个 --test 为 Part 2 baseline。
    exit code：Part 2 全部 pass 为 0，否则 1（Part 1/3 不参与 exit code）。
    """
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

    # Part 1 统一 8-bit LSB，使 W4×float / W4×INT8 / W4×INT16 的 mae/LSB 可直接比较
    REF_BITS = 8
    ref_lsb_tag = f"LSB{subscript(REF_BITS)}"
    COL1 = 62
    W1 = 32 + COL1 * n_tests
    SEP = "=" * W1
    THIN = "-" * W1

    print()
    print(SEP)
    print(f"  Gold: {gold_dir}")
    print(f"  Tests: {', '.join(labels)}")
    print(f"  Thresholds: cos >= {thr.cos},  MAE/LSB < {thr.lsb}")
    print(SEP)
    print()
    print("  ▸ Part 1: Gold vs Each REEX Path")
    print(f"    mae/{ref_lsb_tag} uses uniform {REF_BITS}-bit LSB "
          f"(= max|gold| / {2 ** (REF_BITS - 1) - 1}) for all paths")
    print(f"    Per-tensor columns: cos  mae  max  rmse  mae/{ref_lsb_tag}")
    print_marker_legend(thr, part2_lsb=False)
    print(f"  {THIN}")

    h1 = f"  {'Tensor':<30s}"
    h2 = f"  {'':<30s}"
    for lb in labels:
        h1 += f"{lb:^{COL1}s}"
        h2 += (f"  {'cos':>8s} {'mae':>11s} {'max':>11s} {'rmse':>11s} "
               f"{'mae/' + ref_lsb_tag:>9s} ")
    print(h1)
    print(h2)
    print(f"  {THIN}")

    p1 = [dict(cv=[], max_lsb=0.0, max_mae=0.0, max_max=0.0, max_rmse=0.0,
               npas=0, ntot=0) for _ in range(n_tests)]
    n_idx_tensors = 0
    dist_tensors = []

    for tn in tensor_names:
        g = gold[tn]
        name = tn.replace(".bin", "")
        idx_t = is_index_tensor(name)
        if idx_t:
            n_idx_tensors += 1
        line = f"  {name:<30s}"
        g_absmax = float(np.max(np.abs(g.astype(np.float64))))
        ref_lsb = estimate_lsb(g_absmax, REF_BITS) if g_absmax > 1e-30 else 1e-30

        dkind = distribution_kind(name)
        if dkind and not no_distribution:
            dist_tensors.append((tn, name, dkind))

        for i in range(n_tests):
            t = tdata[i].get(tn)
            if t is None:
                line += f"  {'---':^{COL1 - 2}s} "
                continue
            if g.shape != t.shape:
                line += f"  {'#SHP':^{COL1 - 2}s} "
                if not idx_t:
                    p1[i]["ntot"] += 1
                continue

            if idx_t:
                ir = compare_index_arrays(g, t)
                line += format_index_block(ir, thr)
            else:
                r = compare_arrays(g, t)
                c = r["cos"]
                p1[i]["ntot"] += 1
                p1[i]["cv"].append(c)
                if c >= thr.cos:
                    p1[i]["npas"] += 1
                p1[i]["max_mae"] = max(p1[i]["max_mae"], r["mae"])
                p1[i]["max_max"] = max(p1[i]["max_max"], r["max_ae"])
                p1[i]["max_rmse"] = max(p1[i]["max_rmse"], r["rmse"])
                ml = r["mae"] / ref_lsb
                p1[i]["max_lsb"] = max(p1[i]["max_lsb"], ml)
                line += format_metric_block(r, thr, ref_lsb, check_lsb=True)
        print(line)

    print(f"  {THIN}")
    if n_idx_tensors:
        print(f"    [idx] = integer index tensor ({n_idx_tensors} total): "
              "exact match %, excluded from cos/MAE statistics")
    for i, lb in enumerate(labels):
        s = p1[i]
        mn = min(s["cv"]) if s["cv"] else 0
        av = float(np.mean(s["cv"])) if s["cv"] else 0
        print(
            f"    {lb}: cos_min={mn:.6f}  cos_avg={av:.6f}  "
            f"max_mae={s['max_mae']:.4e}  max={s['max_max']:.4e}  "
            f"max_rmse={s['max_rmse']:.4e}  max_mae/{ref_lsb_tag}={s['max_lsb']:.4f}  "
            f"pass(cos>={thr.cos})={s['npas']}/{s['ntot']}"
        )
    print()

    if not no_distribution and dist_tensors:
        print("  ▸ Part 3: Distribution Metrics (Gold vs Each REEX Path)")
        print("    KL(P_ref || P_test), JS(P_ref, P_test); logits use softmax first")
        print(f"    KL: ~>={thr.kl_warn} *>= {thr.kl_alert}  "
              f"JS: ~>={thr.js_warn} *>= {thr.js_alert}")
        COL3 = 28
        W3 = max(W1, 32 + COL3 * n_tests)
        THIN3 = "-" * W3
        print(f"  {THIN3}")
        h1 = f"  {'Tensor':<30s}"
        h2 = f"  {'':<30s}"
        for lb in labels:
            h1 += f"{lb:^{COL3}s}"
            h2 += f"  {'KL':>12s} {'JS':>12s} "
        print(h1)
        print(h2)
        print(f"  {THIN3}")

        for tn, name, dkind in dist_tensors:
            g = gold[tn]
            line = f"  {name:<30s}"
            for i in range(n_tests):
                t = tdata[i].get(tn)
                if t is None or g.shape != t.shape:
                    line += f"  {'---':^{COL3 - 2}s} "
                    continue
                dm = distribution_metrics(g, t, dkind, dist_eps)
                line += format_dist_block(dm["kl"], dm["js"], thr)
            print(line)
        print(f"  {THIN3}")
        print()

    if n_tests < 2:
        print("  (Single test group — skipping Part 2 cross-comparison)")
        print()
        sys.exit(0)

    base_i = 0
    base_label = labels[base_i]
    cross = [(i, labels[i], bits_list[i]) for i in range(n_tests) if i != base_i]
    nc = len(cross)

    COL2 = 62
    W2 = 32 + COL2 * nc
    WM = max(W1, W2)
    SEP2 = "=" * WM
    THIN2 = "-" * WM

    print(f"  ▸ Part 2: Cross-Comparison (baseline: {base_label})")
    print("    Isolates activation-format error; pass = cos AND mae/LSB")
    print_marker_legend(thr, part2_lsb=True)
    print(f"  {THIN2}")

    h1 = f"  {'Tensor':<30s}"
    h2 = f"  {'':<30s}"
    for _, lb, bits in cross:
        tag = f"{lb} vs base"
        lsb_hdr = f"mae/LSB{subscript(bits)}" if bits else "mae/LSB"
        h1 += f"  {tag:^{COL2 - 2}s}"
        h2 += (f"  {'cos':>8s} {'mae':>11s} {'max':>11s} {'rmse':>11s} "
               f"{lsb_hdr:>9s} ")
    print(h1)
    print(h2)
    print(f"  {THIN2}")

    p2 = [dict(cv=[], max_lsb=0.0, max_mae=0.0, max_max=0.0, max_rmse=0.0,
               npas=0, ntot=0) for _ in cross]
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
                line += f"  {'#SHP':^{COL2 - 2}s}"
                if not idx_t:
                    p2[j]["ntot"] += 1
                    all_cross_pass = False
                continue

            if idx_t:
                ir = compare_index_arrays(base, t)
                line += format_index_block(ir, thr).rstrip()
            else:
                r = compare_arrays(base, t)
                c = r["cos"]
                p2[j]["ntot"] += 1
                p2[j]["cv"].append(c)
                p2[j]["max_mae"] = max(p2[j]["max_mae"], r["mae"])
                p2[j]["max_max"] = max(p2[j]["max_max"], r["max_ae"])
                p2[j]["max_rmse"] = max(p2[j]["max_rmse"], r["rmse"])

                # Part 2：absmax 来自 baseline，位宽 b 来自待比路径标签
                if bits and r["absmax"] > 1e-30:
                    lsb_val = estimate_lsb(r["absmax"], bits)
                    ok = (c >= thr.cos) and (r["mae"] / lsb_val < thr.lsb)
                    p2[j]["max_lsb"] = max(p2[j]["max_lsb"], r["mae"] / lsb_val)
                else:
                    # float baseline 或无 absmax：仅检查 cosine
                    ok = (c >= thr.cos)

                if ok:
                    p2[j]["npas"] += 1
                else:
                    all_cross_pass = False

                ref_lsb = estimate_lsb(r["absmax"], bits) if bits and r["absmax"] > 1e-30 else 1e-30
                line += format_metric_block(r, thr, ref_lsb, check_lsb=bool(bits))

        print(line)

    print(f"  {THIN2}")
    for j, (ci, lb, bits) in enumerate(cross):
        s = p2[j]
        mn = min(s["cv"]) if s["cv"] else 0.0
        av = float(np.mean(s["cv"])) if s["cv"] else 0.0
        info = (
            f"    {lb}: cos_min={mn:.6f}  cos_avg={av:.6f}  "
            f"max_mae={s['max_mae']:.4e}  max={s['max_max']:.4e}  "
            f"max_rmse={s['max_rmse']:.4e}"
        )
        if bits:
            info += f"  max_mae/LSB{subscript(bits)}={s['max_lsb']:.4f}"
        info += f"  pass={s['npas']}/{s['ntot']}"
        print(info)

    print()
    print(SEP2)
    if all_cross_pass:
        print(f"  VERDICT: ALL PASS  (cos >= {thr.cos}  AND  mae/LSB < {thr.lsb})")
    else:
        print(f"  VERDICT: SOME FAILED  (~/*=cos  !=mae/LSB  +=max ^=rmse  #SHP=shape)")
    print(SEP2)
    print()
    sys.exit(0 if all_cross_pass else 1)


# ─── Entry ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ref_dir", nargs="?", default=None,
                        help="(Legacy) Gold / 参考 dump 目录")
    parser.add_argument("test_dir", nargs="?", default=None,
                        help="(Legacy) 待测 dump 目录")
    parser.add_argument("--gold", type=str, default=None,
                        help="Gold 参考目录（multipath 模式，对应 Part 1/3 的 ref）")
    parser.add_argument("--test", nargs=2, action="append", metavar=("DIR", "LABEL"),
                        help="待测目录与标签，可重复；第一个为 Part 2 baseline")
    parser.add_argument("--cos-threshold", type=float, default=0.9999,
                        help="余弦 pass 阈值（默认 0.9999）")
    parser.add_argument("--cos-warn", type=float, default=0.999,
                        help="余弦 ~ 提示关注下界（默认 0.999，见文档 §3.1）")
    parser.add_argument("--mae-lsb-threshold", type=float, default=1.0,
                        help="mae/LSB pass 上限（默认 1.0）")
    parser.add_argument("--rmse-mae-ratio", type=float, default=3.0,
                        help="RMSE > ratio×MAE 时标 ^（默认 3.0）")
    parser.add_argument("--max-mae-ratio", type=float, default=10.0,
                        help="max > ratio×MAE 时标 +（默认 10.0）")
    parser.add_argument("--idx-match-threshold", type=float, default=1.0,
                        help="索引张量 match 下限，1.0=100%%（默认 1.0）")
    parser.add_argument("--kl-warn", type=float, default=0.01,
                        help="Part 3 KL ~ 关注线（默认 0.01）")
    parser.add_argument("--kl-alert", type=float, default=0.1,
                        help="Part 3 KL * 告警线（默认 0.1）")
    parser.add_argument("--js-warn", type=float, default=0.005,
                        help="Part 3 JS ~ 关注线（默认 0.005）")
    parser.add_argument("--js-alert", type=float, default=0.05,
                        help="Part 3 JS * 告警线（默认 0.05）")
    parser.add_argument("--no-distribution", action="store_true",
                        help="跳过 Part 3（final_logits / ffn_moe_probs 等的 KL/JS）")
    parser.add_argument("--dist-eps", type=float, default=1e-12,
                        help="KL/JS 的 clip 平滑项，防止 log(0)（默认 1e-12）")

    args = parser.parse_args()
    thr = MarkerThresholds(
        cos=args.cos_threshold,
        cos_warn=args.cos_warn,
        lsb=args.mae_lsb_threshold,
        rmse_mae_ratio=args.rmse_mae_ratio,
        max_mae_ratio=args.max_mae_ratio,
        idx_match=args.idx_match_threshold,
        kl_warn=args.kl_warn,
        kl_alert=args.kl_alert,
        js_warn=args.js_warn,
        js_alert=args.js_alert,
    )

    if args.gold and args.test:
        run_multipath(
            args.gold, args.test, thr,
            args.no_distribution, args.dist_eps,
        )
    elif args.ref_dir and args.test_dir:
        run_legacy(args, thr)
    else:
        parser.print_help()
        print("\nError: provide either --gold + --test or positional ref_dir test_dir")
        sys.exit(2)


if __name__ == "__main__":
    main()
