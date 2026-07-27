#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验「上下层 scale 是否一致」的工具。

背景
----
本工具用于排查/验证量化流水线中相邻算子（上层输出 ↔ 下层输入）的 scale 是否对齐。
典型的不一致来源：历史上 ``QuantBitWidth::qmax_auto_scale()`` 对 int8 返回 128，使
``num_steps = 256``，scale = range/256；而 AIMET 用 ``num_steps = 2^bits-1 = 255``，
scale = range/255。两者比值恰为 256/255 ≈ 1.003922，本工具会自动识别并提示该特征比值。

支持的输入格式（自动识别）
--------------------------
1. quant-gru 导出格式（``QuantGRU.export_quant_params`` 生成）：顶层含 ``operators``
   （以及双向时的 ``operators_reverse``）。每个算子形如：
       {"dtype": "INT8", "symmetric": false, "scale": <float|list>, "zero_point": ...,
        "real_min": ..., "real_max": ..., "enc_type": "PER_TENSOR"}
2. AIMET encodings 格式（``QuantGRU.export_quant_params_to_aimet_format`` 生成）：顶层含
   ``activation_encodings``，其中每个 module 形如：
       {"is_GRU": true, "input": [enc], "output": [enc],
        "internal_ops": {"<op>": {"output": [enc]}}, ...}
   以及 ``param_encodings``。enc 含 ``scale``（标量或 list）。

子命令
------
- chain  : 跨两个文件比较「上层输出 scale」与「下层输入 scale」。
           适用于多层各自单独导出的场景（每层一个 json）。
- order  : 在单个 AIMET encodings 文件内，按给定的 module 顺序，逐对比较
           module[i].output 与 module[i+1].input 的 scale。
- dump   : 打印某个文件里能解析到的所有 tensor → scale（用于查名字）。

退出码：0 = 全部一致；1 = 存在不一致；2 = 用法/解析错误。

示例
----
    # 两个独立导出的层（quant-gru 格式）
    python verify_scale_consistency.py chain --upper layer0.json --lower layer1.json

    # 单个 AIMET 文件内多层按顺序检查
    python verify_scale_consistency.py order --file model_encodings.json \
        --order gru0,gru1,gru2

    # 查看某文件所有 scale
    python verify_scale_consistency.py dump --file layer0.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Dict, List, Optional, Tuple


# 256/255 等特征比值：命中说明大概率是 qmax_auto_scale 用 128 的旧 bug。
_BUG_RATIO = 256.0 / 255.0


# ----------------------------------------------------------------------------
# 解析：把任意支持的格式拍平成 {tensor_name: {"scale": [..], "dtype": str, ...}}
# ----------------------------------------------------------------------------
def _as_scale_list(scale) -> List[float]:
    """把 scale 字段（标量或 list）统一成 float 列表。"""
    if scale is None:
        return []
    if isinstance(scale, (list, tuple)):
        return [float(v) for v in scale]
    return [float(scale)]


def _enc_to_record(enc) -> Optional[dict]:
    """从一个 encoding 节点（dict 或 [dict]）提取 scale 记录。"""
    if isinstance(enc, list):
        if not enc:
            return None
        enc = enc[0]
    if not isinstance(enc, dict):
        return None
    if "scale" not in enc:
        return None
    return {
        "scale": _as_scale_list(enc.get("scale")),
        "dtype": enc.get("dtype"),
        "is_symmetric": enc.get("is_symmetric", enc.get("symmetric")),
        "enc_type": enc.get("enc_type"),
    }


def _flatten_quant_gru_export(data: dict) -> Dict[str, dict]:
    """解析 quant-gru export_quant_params 格式。"""
    out: Dict[str, dict] = {}
    for top_key, prefix in (("operators", ""), ("operators_reverse", "reverse/")):
        ops = data.get(top_key)
        if not isinstance(ops, dict):
            continue
        for op_name, op_data in ops.items():
            rec = _enc_to_record(op_data)
            if rec is not None:
                out[f"{prefix}{op_name}"] = rec
    return out


def _flatten_aimet_encodings(data: dict) -> Dict[str, dict]:
    """解析 AIMET encodings 格式（含 GRU 嵌套 internal_ops）。"""
    out: Dict[str, dict] = {}
    act = data.get("activation_encodings")
    if isinstance(act, dict):
        for module_name, mod in act.items():
            if not isinstance(mod, dict):
                # 也可能是 {tensor: enc} 的扁平结构
                rec = _enc_to_record(mod)
                if rec is not None:
                    out[str(module_name)] = rec
                continue
            # GRU 风格：input / output / internal_ops
            for field in ("input", "output"):
                if field in mod:
                    rec = _enc_to_record(mod[field])
                    if rec is not None:
                        out[f"{module_name}/{field}"] = rec
            for io_key in ("internal_ops", "internal_ops_reverse"):
                io = mod.get(io_key)
                if isinstance(io, dict):
                    for op_name, op_node in io.items():
                        node = op_node.get("output") if isinstance(op_node, dict) else op_node
                        rec = _enc_to_record(node)
                        if rec is not None:
                            out[f"{module_name}/{io_key}/{op_name}"] = rec
            # 非 GRU module 也可能直接挂 scale
            rec = _enc_to_record(mod)
            if rec is not None and f"{module_name}/output" not in out:
                out[str(module_name)] = rec

    params = data.get("param_encodings")
    if isinstance(params, dict):
        for key, enc in params.items():
            rec = _enc_to_record(enc)
            if rec is not None:
                out[f"param/{key}"] = rec
    return out


def load_scales(path: str) -> Tuple[Dict[str, dict], str]:
    """加载文件并拍平为 {name: record}，返回 (records, 格式名)。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"[错误] 无法读取/解析 JSON: {path}\n  {e}")

    if not isinstance(data, dict):
        raise SystemExit(f"[错误] 顶层不是 JSON 对象: {path}")

    if "operators" in data or "operators_reverse" in data:
        return _flatten_quant_gru_export(data), "quant-gru-export"
    if "activation_encodings" in data or "param_encodings" in data:
        return _flatten_aimet_encodings(data), "aimet-encodings"

    raise SystemExit(
        f"[错误] 无法识别的格式（既无 'operators' 也无 'activation_encodings'）: {path}"
    )


# ----------------------------------------------------------------------------
# 比较
# ----------------------------------------------------------------------------
def _pick(records: Dict[str, dict], module: Optional[str], field: str,
          path: str) -> Tuple[str, dict]:
    """
    在 records 中定位某个算子的 scale 记录。

    优先精确匹配候选 key；失败则在所有 key 中做后缀/包含匹配。
    """
    candidates = []
    if module:
        candidates += [f"{module}/{field}", f"{module}/{field}/output"]
    candidates += [field, f"operators/{field}"]
    for c in candidates:
        if c in records:
            return c, records[c]

    # 模糊匹配：key 以 /field 结尾（可叠加 module 过滤）
    suffix = "/" + field
    fuzzy = [
        k for k in records
        if (k == field or k.endswith(suffix))
        and (module is None or module in k)
    ]
    if len(fuzzy) == 1:
        return fuzzy[0], records[fuzzy[0]]
    if len(fuzzy) > 1:
        raise SystemExit(
            f"[错误] 在 {path} 中 '{field}'"
            f"{f'（module={module}）' if module else ''} 匹配到多个键，请用 --*-module 指定：\n"
            + "\n".join(f"    - {k}" for k in sorted(fuzzy))
        )
    raise SystemExit(
        f"[错误] 在 {path} 中找不到算子 '{field}'"
        f"{f'（module={module}）' if module else ''}。\n"
        f"可用 `dump` 子命令查看所有可用键。"
    )


def _compare_scales(name: str, sa: List[float], sb: List[float],
                    rtol: float, atol: float) -> Tuple[bool, List[str]]:
    """逐元素比较两个 scale 列表，返回 (是否一致, 明细行列表)。"""
    lines: List[str] = []
    if len(sa) != len(sb):
        lines.append(
            f"  [{name}] scale 长度不一致: 上={len(sa)} vs 下={len(sb)}"
        )
        return False, lines

    ok_all = True
    for i, (a, b) in enumerate(zip(sa, sb)):
        idx = f"[{i}]" if len(sa) > 1 else ""
        if b == 0.0:
            same = (a == 0.0)
            ratio = float("inf") if a != 0.0 else 1.0
        else:
            ratio = a / b
            same = abs(a - b) <= (atol + rtol * abs(b))
        flag = "OK " if same else "FAIL"
        hint = ""
        if not same:
            ok_all = False
            # 命中 256/255（或其倒数）特征：旧 qmax_auto_scale=128 bug
            for r, tag in ((_BUG_RATIO, "≈256/255"), (1.0 / _BUG_RATIO, "≈255/256")):
                if abs(ratio - r) <= 1e-4:
                    hint = f"  <== 比值{tag}，疑似 qmax_auto_scale=128 旧 bug"
                    break
        lines.append(
            f"  [{flag}] {name}{idx}: 上={a:.10g}  下={b:.10g}  "
            f"比值={ratio:.8g}{hint}"
        )
    return ok_all, lines


# ----------------------------------------------------------------------------
# 子命令
# ----------------------------------------------------------------------------
def cmd_chain(args) -> int:
    up_recs, up_fmt = load_scales(args.upper)
    lo_recs, lo_fmt = load_scales(args.lower)

    up_key, up_rec = _pick(up_recs, args.upper_module, args.upper_op, args.upper)
    lo_key, lo_rec = _pick(lo_recs, args.lower_module, args.lower_op, args.lower)

    print("=" * 72)
    print("上下层 scale 一致性检查 (chain)")
    print(f"  上层: {args.upper}  [{up_fmt}]")
    print(f"        {up_key}  dtype={up_rec.get('dtype')}")
    print(f"  下层: {args.lower}  [{lo_fmt}]")
    print(f"        {lo_key}  dtype={lo_rec.get('dtype')}")
    print(f"  容差: rtol={args.rtol}  atol={args.atol}")
    print("-" * 72)

    ok, lines = _compare_scales(
        f"{args.upper_op}->{args.lower_op}",
        up_rec["scale"], lo_rec["scale"], args.rtol, args.atol,
    )
    print("\n".join(lines))
    print("-" * 72)
    print("结果: " + ("✅ 一致" if ok else "❌ 不一致"))
    print("=" * 72)
    return 0 if ok else 1


def cmd_order(args) -> int:
    recs, fmt = load_scales(args.file)
    modules = [m.strip() for m in args.order.split(",") if m.strip()]
    if len(modules) < 2:
        raise SystemExit("[错误] --order 至少需要 2 个 module 名（逗号分隔）")

    print("=" * 72)
    print("上下层 scale 一致性检查 (order)")
    print(f"  文件: {args.file}  [{fmt}]")
    print(f"  顺序: {' -> '.join(modules)}")
    print(f"  容差: rtol={args.rtol}  atol={args.atol}")
    print("-" * 72)

    all_ok = True
    for upper, lower in zip(modules[:-1], modules[1:]):
        up_key, up_rec = _pick(recs, upper, args.upper_op, args.file)
        lo_key, lo_rec = _pick(recs, lower, args.lower_op, args.file)
        ok, lines = _compare_scales(
            f"{up_key}  ↔  {lo_key}",
            up_rec["scale"], lo_rec["scale"], args.rtol, args.atol,
        )
        print("\n".join(lines))
        all_ok = all_ok and ok
    print("-" * 72)
    print("结果: " + ("✅ 全部一致" if all_ok else "❌ 存在不一致"))
    print("=" * 72)
    return 0 if all_ok else 1


def cmd_dump(args) -> int:
    recs, fmt = load_scales(args.file)
    print(f"# {args.file}  [{fmt}]  共 {len(recs)} 项")
    for name in sorted(recs):
        rec = recs[name]
        sc = rec["scale"]
        if len(sc) == 1:
            sc_str = f"{sc[0]:.10g}"
        elif len(sc) <= 4:
            sc_str = "[" + ", ".join(f"{v:.6g}" for v in sc) + "]"
        else:
            sc_str = f"[{sc[0]:.6g}, ..., {sc[-1]:.6g}] (n={len(sc)})"
        print(f"  {name:60s} dtype={str(rec.get('dtype')):8s} scale={sc_str}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="校验上下层（上层输出 ↔ 下层输入）量化 scale 是否一致。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--rtol", type=float, default=1e-6,
                        help="相对容差（默认 1e-6；一致时通常应完全相等）")
    common.add_argument("--atol", type=float, default=0.0,
                        help="绝对容差（默认 0）")
    common.add_argument("--upper-op", default="output",
                        help="上层用于比较的算子名（默认 output）")
    common.add_argument("--lower-op", default="input",
                        help="下层用于比较的算子名（默认 input）")

    pc = sub.add_parser("chain", parents=[common],
                        help="跨两个文件比较 上层.output 与 下层.input")
    pc.add_argument("--upper", required=True, help="上层导出 json")
    pc.add_argument("--lower", required=True, help="下层导出 json")
    pc.add_argument("--upper-module", default=None,
                    help="上层 module 名（AIMET 多模块文件时用于定位）")
    pc.add_argument("--lower-module", default=None,
                    help="下层 module 名（AIMET 多模块文件时用于定位）")
    pc.set_defaults(func=cmd_chain)

    po = sub.add_parser("order", parents=[common],
                        help="单个 AIMET 文件内按 module 顺序逐对检查")
    po.add_argument("--file", required=True, help="AIMET encodings json")
    po.add_argument("--order", required=True,
                    help="按数据流顺序排列的 module 名，逗号分隔")
    po.set_defaults(func=cmd_order)

    pd = sub.add_parser("dump", help="打印文件中所有可解析的 tensor scale")
    pd.add_argument("--file", required=True, help="待解析 json")
    pd.set_defaults(func=cmd_dump)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
