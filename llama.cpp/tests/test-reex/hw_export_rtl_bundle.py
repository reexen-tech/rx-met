#!/usr/bin/env python3
"""Export hardware dump cases into RTL/testbench-friendly bundle layout.

Input layout (from eval_single_token hw dump):
  <root>/hw_vectors_*/<tensor>-<layer>/{weight_block,act_q8,...}.bin

Output layout:
  <out>/manifest.json
  <out>/cases/<case_id>/{meta.json, *.bin}

Usage:
  hw_export_rtl_bundle.py <result_root> [--out <dir>] [--validate]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SRC_ROOTS = [
    "hw_vectors_smoke",
    "hw_vectors_prefill8192",
    "hw_vectors_phase2_down",
]

BIN_NAMES = [
    "weight_block.bin",
    "act_q8_block.bin",
    "psum_pre_trunc_i32.bin",
    "psum_post_trunc_i32.bin",
    "output_f32.bin",
]


def find_cases(root: Path) -> list[Path]:
    cases = []
    for meta in sorted(root.rglob("meta.json")):
        if meta.parent == root:
            continue
        if (meta.parent / "output_f32.bin").is_file():
            cases.append(meta.parent)
    return cases


def case_id(case_dir: Path, src_root_name: str) -> str:
    return f"{src_root_name}__{case_dir.name}"


def file_entry(path: Path) -> dict:
    st = path.stat()
    return {"path": path.name, "bytes": st.st_size}


def validate_case(case_dir: Path, script: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(script), str(case_dir)],
        text=True,
        capture_output=True,
    )
    line = (proc.stdout or proc.stderr).strip().splitlines()
    last = line[-1] if line else ""
    exact = "exact=True" in last
    return {
        "ok": proc.returncode == 0 and exact,
        "line": last,
    }


def export_bundle(result_root: Path, out_dir: Path, src_roots: list[str], validate: bool) -> int:
    validate_script = Path(__file__).with_name("hw_dump_validate_sw.py")
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_out = out_dir / "cases"
    if cases_out.exists():
        shutil.rmtree(cases_out)
    cases_out.mkdir()

    manifest_cases = []
    n_ok = 0

    for src_name in src_roots:
        src = result_root / src_name
        if not src.is_dir():
            print(f"SKIP missing: {src}")
            continue
        print(f"=== scan {src} ===")
        for case_dir in find_cases(src):
            cid = case_id(case_dir, src_name)
            dest = cases_out / cid
            dest.mkdir()
            files = []
            for name in BIN_NAMES:
                src_file = case_dir / name
                if src_file.is_file():
                    shutil.copy2(src_file, dest / name)
                    files.append(file_entry(dest / name))
            meta = json.loads((case_dir / "meta.json").read_text())
            entry = {
                "case_id": cid,
                "source_root": src_name,
                "source_dir": str(case_dir.relative_to(result_root)),
                "tensor": meta.get("tensor", case_dir.name),
                "layer": meta.get("layer"),
                "psum_bits": meta.get("psum_bits"),
                "record_count": meta.get("record_count"),
                "output_floats": meta.get("output_floats"),
                "files": files,
            }
            if validate and validate_script.is_file():
                v = validate_case(case_dir, validate_script)
                entry["sw_validate"] = v
                if v.get("ok"):
                    n_ok += 1
            manifest_cases.append(entry)
            print(f"  exported {cid} ({len(files)} bins)")

    manifest = {
        "format": "reex_q64_hw_vectors_v1",
        "timestamp": int(time.time()),
        "result_root": str(result_root),
        "quant_type": "Q4_0_64",
        "model": "Qwen3.5-35B-A3B",
        "prefill_tokens": 8192,
        "recommended_psum_bits": 8,
        "case_count": len(manifest_cases),
        "cases": manifest_cases,
        "notes": [
            "psum_pre/post are int32 arrays length=record_count",
            "output_f32 is float32 length=output_floats",
            "L19/L23 record_count may hit REEX_Q64_HW_MAX_RECORDS=65536",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out_dir / 'manifest.json'} ({len(manifest_cases)} cases)")
    if validate:
        print(f"sw_validate ok={n_ok}/{len(manifest_cases)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_root", type=Path)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <result_root>/hw_vectors_rtl_bundle",
    )
    ap.add_argument("--roots", nargs="*", default=DEFAULT_SRC_ROOTS)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    out = args.out or (args.result_root / "hw_vectors_rtl_bundle")
    return export_bundle(args.result_root, out, args.roots, args.validate)


if __name__ == "__main__":
    sys.exit(main())
