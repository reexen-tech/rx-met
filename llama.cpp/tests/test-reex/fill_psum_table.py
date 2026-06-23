#!/usr/bin/env python3
"""Fill the §6.2 PSUM sweep table in the block64 report.

Scans `prefill8192_B${B}_compare.log` files plus the numerical
`analyze_final_logits.py` output for each B and emits a markdown table
row.  Designed to be re-run idempotently as B-sweep completes.

Usage:
  fill_psum_table.py <result_root> [--bvals off 16 12 8] [--update <report.md>]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean


_NUM = r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?"
ROW = re.compile(
    rf"^\s+(?P<name>[A-Za-z_][\w]*)-(?P<layer>\d+)\s+"
    rf"(?P<cos>-?\d\.\d+)[*~]?\s+"
    rf"(?P<mae>{_NUM})[+\^!]?\s+"
    rf"(?P<maxv>{_NUM})[+\^!]?\s+"
    rf"(?P<rmse>{_NUM})[+\^!]?\s+"
    rf"(?P<lsb>{_NUM})[+\^!]?\s*$"
)


def parse_compare(log: Path) -> dict:
    if not log.exists():
        return {}
    cs, named = [], {}
    for line in log.read_text().splitlines():
        m = ROW.match(line)
        if not m:
            continue
        cs.append(float(m["cos"]))
        named[f"{m['name']}-{m['layer']}"] = float(m["cos"])
    if not cs:
        return {}
    return dict(
        n=len(cs),
        cos_min=min(cs),
        cos_avg=mean(cs),
        pass_cnt=sum(1 for c in cs if c >= 0.9999),
        l_out_39=named.get("l_out-39", float("nan")),
        final=named.get("final_logits-0",
                        next((v for k, v in named.items()
                              if k.startswith("final_logits-")), float("nan"))),
    )


def numerical_kl_top1(gold: Path, test: Path) -> dict:
    """Call analyze_final_logits.py for KL/top1 if test exists."""
    if not test.exists():
        return {}
    here = Path(__file__).parent
    script = here / "analyze_final_logits.py"
    try:
        out = subprocess.check_output(
            ["python3", str(script), str(gold), str(test),
             "--tensors", "final_logits"], text=True, timeout=60)
    except Exception as e:
        return {"err": str(e)}
    res = {}
    for line in out.splitlines():
        if "KL=" in line and "top1=" in line:
            m = re.search(r"cos=([\d.eE+-]+).*KL=([\d.eE+-]+).*top1=(\d+)/", line)
            if m:
                res["final_cos"] = float(m.group(1))
                res["kl"] = float(m.group(2))
                res["top1"] = int(m.group(3))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path,
                    help="results/qwen35b_a3b_block64_tensor")
    ap.add_argument("--bvals", nargs="+", default=["off", "16", "12", "8"])
    ap.add_argument("--update", type=Path, default=None,
                    help="overwrite §6.2 table in this report markdown")
    args = ap.parse_args()

    gold = args.root / "dump_f16_prefill8192"
    rows = []
    print(f"{'B':>4} {'pass':>10} {'cos_avg':>9} {'l_out-39':>9} "
          f"{'final':>9} {'KL':>10} {'top1':>5}  status")
    print("-" * 80)
    for B in args.bvals:
        log = args.root / f"prefill8192_B{B}_compare.log"
        suf = "Boff" if B == "off" else f"B{B}"
        test = args.root / f"dump_Q4_0_64_{suf}_prefill8192"
        # off uses original compare log path
        if B == "off" and not log.exists():
            log = args.root / "prefill8192_compare.log"
        c = parse_compare(log)
        n = numerical_kl_top1(gold, test) if c else {}
        status = "OK" if c else "PENDING"
        if c:
            fc = n.get('final_cos', c['final'])
            line = (f"{B:>4} {c['pass_cnt']:>4}/{c['n']:<5} "
                    f"{c['cos_avg']:>9.4f} {c['l_out_39']:>9.4f} "
                    f"{fc:>9.4f} "
                    f"{n.get('kl', float('nan')):>10.2e} "
                    f"{n.get('top1', '?'):>5}  {status}")
            rows.append((B, c, n))
        else:
            line = f"{B:>4} {'-':>10} {'-':>9} {'-':>9} {'-':>9} {'-':>10} {'-':>5}  {status}"
        print(line)

    if args.update and rows:
        md = args.update.read_text()
        block_lines = [
            "| B | pass | cos_avg | l_out-39 cos | final_logits cos | KL | top-1 | 报告路径 |",
            "|---|---:|---:|---:|---:|---:|:---:|---|",
        ]
        for B, c, n in rows:
            suf = "Boff" if B == "off" else f"B{B}"
            fc = n.get('final_cos', c['final'])
            block_lines.append(
                f"| {B} | {c['pass_cnt']}/{c['n']} | "
                f"{c['cos_avg']:.4f} | {c['l_out_39']:.4f} | "
                f"{fc:.4f} | {n.get('kl', float('nan')):.2e} | "
                f"{n.get('top1', '?')}/1 | `dump_*_{suf}_prefill8192` |"
            )
        new_block = "\n".join(block_lines)
        pat = re.compile(
            r"(\| B \| pass \|[^\n]*\n\|---[^\n]*\n)(?:\|[^\n]*\n)+",
            re.MULTILINE)
        new_md, count = pat.subn(new_block + "\n", md, count=1)
        if count:
            args.update.write_text(new_md)
            print(f"\n[updated §6.2 table in {args.update} ({len(rows)} rows)]")
        else:
            print(f"\n[WARN] could not locate §6.2 table block in {args.update}")


if __name__ == "__main__":
    sys.exit(main())
