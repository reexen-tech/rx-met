#!/usr/bin/env python3
"""Batch-validate all Phase 1/2 hardware dump cases under a result root.

Usage:
  hw_validate_all.py <result_root> [--roots hw_vectors_smoke hw_vectors_prefill8192 ...]
"""
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOTS = [
    "hw_vectors_smoke",
    "hw_vectors_prefill8192",
    "hw_vectors_phase2_down",
]


def find_cases(root: Path) -> list[Path]:
    cases = []
    for meta in sorted(root.rglob("meta.json")):
        if meta.parent == root:
            continue
        if (meta.parent / "output_f32.bin").is_file():
            cases.append(meta.parent)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_root", type=Path)
    ap.add_argument(
        "--roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help="subdirs under result_root to scan (default: common hw_vectors_*)",
    )
    args = ap.parse_args()

    script = Path(__file__).with_name("hw_dump_validate_sw.py")
    if not script.is_file():
        print(f"MISSING {script}", file=sys.stderr)
        return 2

    ok = fail = skip = 0
    for sub in args.roots:
        base = args.result_root / sub
        if not base.is_dir():
            print(f"SKIP root (missing): {base}")
            continue
        print(f"=== {base} ===")
        for case in find_cases(base):
            proc = subprocess.run(
                [sys.executable, str(script), str(case)],
                text=True,
                capture_output=True,
            )
            line = proc.stdout.strip() or proc.stderr.strip()
            if "SKIP:" in line:
                print(f"  SKIP {case.name}: {line}")
                skip += 1
            elif proc.returncode == 0 and "exact=True" in line:
                print(f"  OK   {line}")
                ok += 1
            else:
                print(f"  FAIL {case.name}: {line}", file=sys.stderr)
                if proc.stderr:
                    print(proc.stderr, file=sys.stderr)
                fail += 1

    print(f"summary: ok={ok} fail={fail} skip={skip}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
