#!/usr/bin/env python3
"""Strictly validate a saved ONNX/schema-v4 encoding pair without modifying it."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from aimet_common.compiler_artifacts import validate_compiler_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--encodings", required=True, type=Path)
    parser.add_argument("--allow-affine", action="store_true", help="Permit affine scales with rounded n")
    parser.add_argument("--allow-partial", action="store_true", help="Check declared grids without requiring all compute inputs")
    parser.add_argument("--report", type=Path, help="Write the validation result as JSON")
    args = parser.parse_args(argv)
    if args.report and args.report.resolve() in (args.onnx.resolve(), args.encodings.resolve()):
        parser.error("The report must not overwrite an input artifact")
    try:
        report = validate_compiler_artifacts(args.onnx, args.encodings, require_po2=not args.allow_affine,
                                             require_complete=not args.allow_partial)
    except Exception as error:
        report = {"status": "failed", "onnx": str(args.onnx.resolve()),
                  "encodings": str(args.encodings.resolve()), "error": str(error)}
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
