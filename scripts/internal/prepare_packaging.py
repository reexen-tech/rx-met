#!/usr/bin/env python3
"""在产品 builder 中准备指定 CUDA 变体的 rx-met wheel 源码树。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _strip_pyproject(text: str) -> str:
    text = re.sub(
        r"^description = \".*\"\n",
        'description = "Offline PyTorch / AIMET / QuantGRU quantization toolkit"\n',
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r"\n\[project\.scripts\]\nrx-met = \"rx_met_llm\.pipeline:_cli\"\n",
        "\n",
        text,
        count=1,
    )
    text = text.replace('    "rx_met_llm",\n    "rx_met_llm.*",\n', "")
    text = text.replace('rx_met_llm = ["*.md"]\n', "")
    # Newer pip/tomli rejects `aimet_common = [...]` plus `aimet_common.quantsim_config`.
    text = text.replace(
        'aimet_common = ["*.json", "*.so", "*.pyd", "*.dll", "bin/*"]\n'
        'aimet_common.quantsim_config = ["*.json"]\n',
        'aimet_common = ["*.json", "*.so", "*.pyd", "*.dll", "bin/*", "quantsim_config/*.json"]\n',
    )
    if "rx_met_llm" in text:
        raise SystemExit("failed to strip rx_met_llm from pyproject.toml")
    if "rx-met =" in text:
        raise SystemExit("failed to strip rx-met console script from pyproject.toml")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--cuda-variant", required=True)
    args = parser.parse_args()

    root: Path = args.root
    variants = json.loads(
        (root / "docker" / "variants.json").read_text(encoding="utf-8")
    )["variants"]
    if args.cuda_variant not in variants:
        raise SystemExit(f"unsupported CUDA variant: {args.cuda_variant}")
    version = args.version.strip()
    # Wheel / PEP 440 cannot start with "v". Use the component MAJOR.MINOR.PATCH.
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise SystemExit(f"invalid wheel version, expected MAJOR.MINOR.PATCH: {version}")

    variant = args.cuda_variant
    wheel_version = f"{version}+{variant}"
    pyproject = root / "pyproject.toml"
    pyproject.write_text(_strip_pyproject(pyproject.read_text(encoding="utf-8")), encoding="utf-8")
    (root / "VERSION").write_text(f"{wheel_version}\n", encoding="utf-8")

    quant_version = root / "operators" / "quant-gru" / "pytorch" / "_version.py"
    quant_text = quant_version.read_text(encoding="utf-8")
    quant_text, count = re.subn(
        r'^__version__ = "([0-9]+\.[0-9]+\.[0-9]+)(?:\+[^\"]+)?"$',
        rf'__version__ = "\1+{variant}"',
        quant_text,
        count=1,
        flags=re.M,
    )
    if count != 1:
        raise SystemExit(f"failed to set QuantGRU CUDA variant in {quant_version}")
    quant_version.write_text(quant_text, encoding="utf-8")
    print(f"prepared rx-met packaging at {root} version={wheel_version}")


if __name__ == "__main__":
    main()
