"""LLM quantization quick start - the llama.cpp counterpart of
``aimet_rx/examples/quick_start.py``.

This script intentionally has the **same shape** as AIMET's KWS demo:

    1. import the toolkit
    2. point at a JSON config
    3. one call kicks off the whole pipeline

For LLM workloads the heavy lifting (model loading, quantization,
calibration, inference) happens inside ``llama.cpp`` binaries; this
Python file is purely the orchestrator.

Usage (run from the ``aimet_rx`` directory)::

    python examples/llm_quick_start.py examples/qwen3_30b_q4_0_minimal.json
    python examples/llm_quick_start.py examples/qwen3_30b_mixed_precision.json
    python examples/llm_quick_start.py CONFIG.json --dry-run    # print plan only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
# aimet_rx/examples/ -> aimet_rx/ (which contains the aimet_llama source package)
sys.path.insert(0, str(HERE.parent))

from rx_met_llm import LLMQuantPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm_quick_start.py",
        description="JSON-driven LLM quantization (AIMET-style quick start)",
    )
    parser.add_argument("config", type=Path, help="Path to JSON config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved CLI plan and exit (no binaries invoked)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AIMET-llama quick start")
    print("=" * 70)
    print(f"config: {args.config}")
    print()

    pipe = LLMQuantPipeline.from_json(args.config)

    if args.dry_run:
        print("[dry-run] planned CLI commands:\n")
        pipe.print_plan()
        return

    summary = pipe.run()

    print("\n" + "=" * 70)
    print("Done. Summary:")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
