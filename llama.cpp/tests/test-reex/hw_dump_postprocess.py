#!/usr/bin/env python3
"""Validate hardware dump directory layout (Phase 1 skeleton).

Usage:
  hw_dump_postprocess.py <hw_vectors_dir> [--expect-psum]
"""
import argparse
import json
import sys
from pathlib import Path

REQUIRED = ["meta.json"]
OPTIONAL = [
    "weight_block.bin",
    "act_q8_block.bin",
    "psum_pre_trunc_i32.bin",
    "psum_post_trunc_i32.bin",
    "output_f32.bin",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--expect-psum", action="store_true",
                    help="fail if psum bins missing (Phase 1+ only)")
    args = ap.parse_args()
    root = args.root
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    meta_path = root / "meta.json"
    if not meta_path.exists():
        print(f"MISSING {meta_path}")
        return 1

    meta = json.loads(meta_path.read_text())
    print(f"phase={meta.get('phase')} status={meta.get('status')} "
          f"records={meta.get('record_count')} psum_bits={meta.get('psum_bits')}")

    present = [p.name for p in root.iterdir() if p.is_file()]
    for name in REQUIRED:
        ok = name in present
        print(f"  {'OK' if ok else 'MISSING':7} {name}")

    for name in OPTIONAL:
        ok = name in present
        tag = "OK" if ok else ("MISSING" if args.expect_psum else "pending")
        print(f"  {tag:7} {name}")

    if args.expect_psum:
        missing = [n for n in OPTIONAL if n not in present]
        if missing:
            print(f"FAIL: missing {missing}", file=sys.stderr)
            return 1
    print("PASS (skeleton)" if meta.get("status") == "skeleton" else "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
