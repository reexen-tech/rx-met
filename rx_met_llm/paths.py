"""Resolve llama.cpp binary paths at **usage** time (after install).

Install layout (``$RX_MET_HOME``, default ``/opt/rx-met``)::

    $RX_MET_HOME/bin/    llama-quantize, llama-imatrix, ...
    $RX_MET_HOME/lib/     libggml*.so

Release packaging uses a separate ``native/*.tar.gz`` artifact; that tarball
expands directly into ``bin/`` + ``lib/`` under ``$PREFIX`` — there is no
``native/`` directory on the installed system.

Development fallback: ``aimet_rx/llama.cpp/build_cuda/bin/``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict


class NativeBinaryError(RuntimeError):
    """Raised when prebuilt llama.cpp binaries cannot be located."""


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LLAMA_CPP_ROOT = _REPO_ROOT / "llama.cpp"


def _bin_dir_ready(bin_dir: Path) -> bool:
    required = ("llama-quantize", "llama-imatrix", "llama-perplexity")
    return all((bin_dir / name).is_file() for name in required)


def resolve_install_root() -> Path:
    """Return rx-met install prefix (contains ``bin/`` and ``lib/``)."""
    candidates: list[Path] = []

    rx_home = os.environ.get("RX_MET_HOME")
    if rx_home:
        candidates.append(Path(rx_home))

    candidates.append(Path("/opt/rx-met"))

    for build_name in ("build_cuda", "build_cuda_q64", "build_cpu"):
        candidates.append(_LLAMA_CPP_ROOT / build_name)

    for root in candidates:
        bin_dir = root / "bin"
        if _bin_dir_ready(bin_dir):
            return root

    raise NativeBinaryError(
        "llama.cpp binaries not found. Install rx-met (see install.sh), set "
        "RX_MET_HOME, or build llama.cpp under "
        f"{_LLAMA_CPP_ROOT} (see REEX_Q64_USAGE.md)."
    )


def _tool_path(bin_dir: Path, name: str) -> str:
    path = bin_dir / name
    if path.is_file():
        return str(path.resolve())
    found = shutil.which(name)
    if found:
        return found
    raise NativeBinaryError(f"Required tool not found: {bin_dir / name}")


def resolve_convert_hf_script(bin_dir: Path) -> str:
    for cand in (
        bin_dir / "convert_hf_to_gguf.py",
        _LLAMA_CPP_ROOT / "convert_hf_to_gguf.py",
    ):
        if cand.is_file():
            return str(cand.resolve())
    raise NativeBinaryError(
        "convert_hf_to_gguf.py not found; expected under install bin/ "
        "or llama.cpp source root."
    )


def resolve_hw_export_binary(bin_dir: Path) -> str:
    for cand in (
        bin_dir / "reex-hw-convert",
        _LLAMA_CPP_ROOT / "build_cpu_wconvert" / "bin" / "reex-hw-convert",
    ):
        if cand.is_file():
            return str(cand.resolve())
    return _tool_path(bin_dir, "reex-hw-convert")


def resolve_binaries() -> Dict[str, str]:
    """Paths injected into resolved pipeline config (not user JSON)."""
    root = resolve_install_root()
    bin_dir = root / "bin"
    lib_dir = root / "lib"

    ld_parts = [str(bin_dir.resolve())]
    if lib_dir.is_dir():
        ld_parts.insert(0, str(lib_dir.resolve()))

    return {
        "llama_quantize": _tool_path(bin_dir, "llama-quantize"),
        "llama_imatrix": _tool_path(bin_dir, "llama-imatrix"),
        "llama_perplexity": _tool_path(bin_dir, "llama-perplexity"),
        "convert_hf_to_gguf": resolve_convert_hf_script(bin_dir),
        "ld_library_path": ":".join(ld_parts),
        "hw_export": resolve_hw_export_binary(bin_dir),
    }
