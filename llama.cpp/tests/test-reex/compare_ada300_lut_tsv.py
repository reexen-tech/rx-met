#!/usr/bin/env python3
"""Compare CPU/CUDA ADA300 result TSVs with the Python golden."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


HEADER_PATTERN = re.compile(
    r"# reex-lut-bit-v1 segments=(16|31|63) precision=mixed_fp16"
)


def ordered(bits: int) -> int:
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else (bits | 0x80000000)


def load(path: Path) -> tuple[str, list[tuple[str, str, int, int, str]]]:
    rows = []
    with path.open() as source:
        header = source.readline().rstrip("\n")
        if HEADER_PATTERN.fullmatch(header) is None:
            raise ValueError(f"{path}: invalid header: {header!r}")
        for line_number, line in enumerate(source, 2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected five TSV fields")
            op, case_id, input_hex, output_hex, cls = fields
            rows.append((op, case_id, int(input_hex, 16), int(output_hex, 16), cls))
    return header, rows


def compare(golden_path: Path, actual_path: Path) -> dict:
    golden_header, golden = load(golden_path)
    actual_header, actual = load(actual_path)
    if golden_header != actual_header:
        raise ValueError(
            f"TSV contract differs: golden={golden_header!r} actual={actual_header!r}")
    if len(golden) != len(actual):
        raise ValueError(f"row count differs: golden={len(golden)} actual={len(actual)}")

    stats = defaultdict(lambda: {"total": 0, "exact": 0, "max_ulp": 0, "failures": 0})
    first_failure = None
    for index, (expected, got) in enumerate(zip(golden, actual), 1):
        e_op, e_id, e_input, e_output, e_class = expected
        a_op, a_id, a_input, a_output, a_class = got
        if (e_op, e_id, e_input) != (a_op, a_id, a_input):
            raise ValueError(f"row identity differs at row {index}: {expected[:3]} != {got[:3]}")

        row = stats[e_op]
        row["total"] += 1
        exact = e_output == a_output or (e_class == "nan" and a_class == "nan")
        if exact:
            row["exact"] += 1

        if e_class == "finite" and a_class == "finite":
            ulp = abs(ordered(e_output) - ordered(a_output))
        elif e_class == a_class:
            ulp = 0
        else:
            ulp = 0xFFFFFFFF
        row["max_ulp"] = max(row["max_ulp"], ulp)
        if ulp > 1:
            row["failures"] += 1
            if first_failure is None:
                first_failure = {
                    "row": index,
                    "op": e_op,
                    "case_id": e_id,
                    "input_bits": f"{e_input:08x}",
                    "expected_bits": f"{e_output:08x}",
                    "expected_class": e_class,
                    "actual_bits": f"{a_output:08x}",
                    "actual_class": a_class,
                    "ulp": ulp,
                }

    for row in stats.values():
        row["exact_rate"] = row["exact"] / row["total"] if row["total"] else 1.0
    return {
        "contract": golden_header,
        "golden": str(golden_path.resolve()),
        "actual": str(actual_path.resolve()),
        "passed": first_failure is None,
        "operators": dict(stats),
        "first_failure": first_failure,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu is None and args.cuda is None:
        parser.error("at least one of --cpu/--cuda is required")

    summary = {"schema": "reex-lut-compare-v1"}
    if args.cpu is not None:
        summary["cpu"] = compare(args.golden, args.cpu)
    if args.cuda is not None:
        summary["cuda"] = compare(args.golden, args.cuda)
    summary["passed"] = all(value["passed"] for key, value in summary.items() if key in ("cpu", "cuda"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
