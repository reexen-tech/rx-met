#!/usr/bin/env python3
"""Compare hw dump output_f32.bin against sw float from the same eval run."""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) < 2:
        print("usage: hw_dump_validate_sw.py <case_dir> [sw_bin_path]", file=sys.stderr)
        return 2

    case = Path(sys.argv[1])
    meta_path = case / "meta.json"
    if not meta_path.exists():
        print(f"MISSING {meta_path}", file=sys.stderr)
        return 1

    meta = json.loads(meta_path.read_text())
    tensor = meta.get("tensor", case.name)
    layer = meta.get("layer", -1)

    hw_path = case / "output_f32.bin"
    if len(sys.argv) >= 3:
        sw_path = Path(sys.argv[2])
    else:
        candidates = [
            case.parent / f"sw_float_L{layer}" / f"{tensor}.bin",
            case.parent / "sw_float" / f"{tensor}.bin",
        ]
        sw_path = next((p for p in candidates if p.is_file()), candidates[0])

    if not hw_path.is_file():
        print(f"MISSING {hw_path}", file=sys.stderr)
        return 1
    if not sw_path.is_file():
        print(f"SKIP: no sw float at {sw_path}")
        return 0

    hw = np.fromfile(hw_path, dtype=np.float32)
    sw = np.fromfile(sw_path, dtype=np.float32)
    n = min(hw.size, sw.size)
    if n == 0:
        print("ERROR: empty arrays", file=sys.stderr)
        return 1

    hw = hw[:n]
    sw = sw[:n]
    diff = np.abs(hw - sw)
    cos = float(np.dot(hw, sw) / (np.linalg.norm(hw) * np.linalg.norm(sw) + 1e-12))
    exact = bool(np.all(diff == 0))

    print(f"{tensor}: n={n} cos={cos:.6f} max_diff={diff.max():.6g} exact={exact}")
    if not exact and cos < 0.9999:
        print("WARN: hw vs sw mismatch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
