#!/usr/bin/env python3
"""Fill §7.2 Block-64 PPL table from perplexity JSON files."""
import argparse
import json
import re
from pathlib import Path

F16_PPL = 7.0066
Q4_0_PPL = 7.2103

TABLE_START = "### 7.2 Block-64 PPL 回填表"
TABLE_END = "## 8."


def load_ppl(jf: Path) -> dict:
    if not jf.is_file():
        return {}
    try:
        d = json.loads(jf.read_text())
        r = d.get("result", {})
        if "ppl" in r:
            return {"ppl": r["ppl"], "unc": r.get("ppl_uncertainty")}
    except Exception:
        pass
    return {}


def row(t: str, b: str, jf: Path, lf: Path) -> str:
    d = load_ppl(jf)
    if not d:
        return (
            f"| {t} | `validation.txt` | {b} | — | — | — | — | "
            f"`{jf.name}` | **PENDING** |"
        )
    ppl = d["ppl"]
    unc = d.get("unc", "—")
    d_f16 = ppl - F16_PPL
    d_q4 = ppl - Q4_0_PPL
    log_ref = lf.name if lf.is_file() else jf.name
    return (
        f"| {t} | `validation.txt` | {b} | {ppl:.4f} | {unc} | "
        f"{d_f16:+.4f} | {d_q4:+.4f} | `{jf.name}` / `{log_ref}` | **DONE** |"
    )


def build_table(ppl_dir: Path, log_dir: Path, types: list[str], b_main: str) -> str:
    lines = [
        TABLE_START,
        "",
        "| 量化类型 | 数据集 | B | PPL | 不确定度 | 相对 F16 ΔPPL | 相对原生 Q4_0 ΔPPL | 文件/日志 | 状态 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for t in types:
        lines.append(row(t, b_main, ppl_dir / f"perplexity_Qwen3.5-35B-A3B-{t}_B{b_main}_validation.json",
                         log_dir / f"ppl_{t}_B{b_main}.log"))
    lines.append(row("Q4_0_64", "off",
                     ppl_dir / "perplexity_Qwen3.5-35B-A3B-Q4_0_64_Boff_validation.json",
                     log_dir / "ppl_Q4_0_64_Boff.log"))
    lines.append("")
    return "\n".join(lines)


def update_report(report: Path, table: str) -> None:
    text = report.read_text(encoding="utf-8")
    pat = re.compile(
        rf"({re.escape(TABLE_START)}.*?)(?=\n{re.escape(TABLE_END)})",
        re.DOTALL,
    )
    if not pat.search(text):
        raise SystemExit(f"§7.2 anchor not found in {report}")
    report.write_text(pat.sub(table + "\n", text, count=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ppl_dir", type=Path)
    ap.add_argument("--log-dir", type=Path, default=Path("/tmp/qwen35_block64_ppl_logs"))
    ap.add_argument("--types", nargs="+",
                    default=["Q4_0_64", "Q8_0_64", "Q2_K_64", "Q3_K_64",
                             "Q4_K_64", "Q5_K_64", "Q6_K_64"])
    ap.add_argument("--b", default="8")
    ap.add_argument("--update", type=Path)
    args = ap.parse_args()
    table = build_table(args.ppl_dir, args.log_dir, args.types, args.b)
    print(table)
    if args.update:
        update_report(args.update, table)
        print(f"updated {args.update}")


if __name__ == "__main__":
    main()
