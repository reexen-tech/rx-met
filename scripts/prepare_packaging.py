#!/usr/bin/env python3
"""Prepare the rx-met wheel tree inside the builder image.

Writes VERSION and ada200-aligned requirements.txt. Also rejects any leftover
LLM packaging if a fullstack pyproject is copied in by mistake.
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    root: Path = args.root
    version = args.version.strip()
    # Wheel / PEP 440 cannot start with "v". Use the component MAJOR.MINOR.PATCH.
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise SystemExit(f"invalid wheel version, expected MAJOR.MINOR.PATCH: {version}")

    pyproject = root / "pyproject.toml"
    pyproject.write_text(_strip_pyproject(pyproject.read_text(encoding="utf-8")), encoding="utf-8")
    (root / "VERSION").write_text(f"{version}\n", encoding="utf-8")
    # Align declared deps with ada200_docker. Extra wheels are installed separately.
    (root / "requirements.txt").write_text(
        "\n".join(
            [
                "numpy>=1.20.0,<3",
                "scipy>=1.7.0",
                "onnx>=1.11.0",
                "onnxruntime>=1.23.1",
                "torch>=2.8.0",
                "torchvision>=0.23.0",
                "pillow>=8.0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"prepared rx-met packaging at {root} version={version}")


if __name__ == "__main__":
    main()
