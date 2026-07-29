"""JSON config -> CLI argv translators for llama.cpp binaries.

Each ``build_*_cmd`` function returns a ``list[str]`` ready for
``subprocess.run``. The translators are **pure functions** of the
resolved config dict; they do not touch the filesystem and have no
side effects. This makes them trivially unit-testable and lets the
caller dry-run a config to inspect the exact commands that will be
executed (audit trail).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import (
    hf_gguf_intermediate_path,
    imatrix_path,
    quantized_path,
    hw_export_input,
    hw_export_output_gguf,
)


def _str(value: Any) -> str:
    return str(value)


def _input_gguf_path(cfg: Dict[str, Any]) -> Path:
    """Return an existing GGUF input or the planned HF conversion output."""
    gguf = cfg["model"].get("gguf_fp16_path")
    return Path(gguf) if gguf else hf_gguf_intermediate_path(cfg)


def build_imatrix_cmd(cfg: Dict[str, Any]) -> Optional[List[str]]:
    """Build the ``llama-imatrix`` invocation, or ``None`` if calibration is
    disabled or an existing imatrix is being reused."""
    cal = cfg["calibration"]
    if not cal["enabled"] or cal.get("reuse_imatrix"):
        return None

    out_path = imatrix_path(cfg)
    assert out_path is not None, "imatrix_path should be non-None when imatrix runs"

    argv: List[str] = [
        cfg["binaries"]["llama_imatrix"],
        "-m", _str(_input_gguf_path(cfg)),
        "-f", _str(cal["dataset_file"]),
        "-o", _str(out_path),
        "--output-format", cal["output_format"],
        "-c", _str(cal["context_length"]),
        "-ngl", _str(cal["n_gpu_layers"]),
        "--output-frequency", _str(cal["output_frequency"]),
    ]

    if cal.get("n_chunks", -1) != -1:
        argv += ["--chunks", _str(cal["n_chunks"])]
    if cal.get("from_chunk", 0) != 0:
        argv += ["--chunk", _str(cal["from_chunk"])]
    if cal.get("save_frequency", 0) != 0:
        argv += ["--save-frequency", _str(cal["save_frequency"])]
    if cal.get("process_output", False):
        argv += ["--process-output"]
    if cal.get("no_ppl", True):
        argv += ["--no-ppl"]
    if cal.get("parse_special", False):
        argv += ["--parse-special"]
    for in_file in cal.get("in_files", []) or []:
        argv += ["--in-file", _str(in_file)]

    return argv


def build_quantize_cmd(cfg: Dict[str, Any]) -> List[str]:
    """Build the ``llama-quantize`` invocation."""
    q = cfg["quantization"]
    out_path = quantized_path(cfg)
    argv: List[str] = [cfg["binaries"]["llama_quantize"]]

    # Optional flags (must come BEFORE positional args)
    if q.get("allow_requantize", False):
        argv += ["--allow-requantize"]
    if q.get("leave_output_tensor", False):
        argv += ["--leave-output-tensor"]
    if q.get("pure", False):
        argv += ["--pure"]
    if q.get("keep_split", False):
        argv += ["--keep-split"]
    if q.get("dry_run", False):
        argv += ["--dry-run"]

    imat = imatrix_path(cfg)
    if q.get("use_imatrix", True) and imat is not None:
        argv += ["--imatrix", _str(imat)]
        inc = q.get("imatrix_include_weights") or []
        exc = q.get("imatrix_exclude_weights") or []
        for w in inc:
            argv += ["--include-weights", _str(w)]
        for w in exc:
            argv += ["--exclude-weights", _str(w)]

    if q.get("output_tensor_type"):
        argv += ["--output-tensor-type", q["output_tensor_type"]]
    if q.get("token_embedding_type"):
        argv += ["--token-embedding-type", q["token_embedding_type"]]

    for override in q.get("tensor_type_overrides", []):
        argv += ["--tensor-type", f"{override['pattern']}={override['type']}"]

    if q.get("prune_layers"):
        argv += ["--prune-layers", ",".join(_str(x) for x in q["prune_layers"])]

    for kv in q.get("override_kv", []) or []:
        argv += ["--override-kv", _str(kv)]

    # Positional args last: input, output, type, [threads]
    argv += [
        _str(_input_gguf_path(cfg)),
        _str(out_path),
        q["default_type"],
        _str(q.get("n_threads", 8)),
    ]
    return argv


def build_perplexity_cmd(cfg: Dict[str, Any]) -> Optional[List[str]]:
    """Build the ``llama-perplexity`` invocation, or ``None`` if disabled."""
    ev = cfg["evaluation"]
    if not ev["enabled"]:
        return None
    ppl = ev["perplexity"]
    out_path = quantized_path(cfg)

    argv: List[str] = [
        cfg["binaries"]["llama_perplexity"],
        "-m", _str(out_path),
        "-f", _str(ppl["dataset_file"]),
        "-c", _str(ppl["context_length"]),
        "-ngl", _str(ppl["n_gpu_layers"]),
        "-np", _str(ppl.get("parallel", 1)),
    ]
    if ppl.get("flash_attn", True):
        argv += ["--flash-attn", "on"]
    if ppl.get("chunks", -1) != -1:
        argv += ["--chunks", _str(ppl["chunks"])]
    if ppl.get("cache_type_k"):
        argv += ["-ctk", _str(ppl["cache_type_k"]).lower()]
    if ppl.get("cache_type_v"):
        argv += ["-ctv", _str(ppl["cache_type_v"]).lower()]
    argv += [_str(x) for x in ppl.get("extra_args", []) or []]
    return argv


def build_hw_export_cmd(cfg: Dict[str, Any]) -> Optional[List[str]]:
    """Build the ``reex-hw-convert --in-gguf ... --out ...`` invocation that
    produces the HW-tiled GGUF, or ``None`` if the hw_export stage is disabled."""
    he = cfg.get("hw_export", {})
    if not he.get("enabled"):
        return None

    argv: List[str] = [
        he["binary"],
        "--in-gguf", _str(hw_export_input(cfg)),
        "--out", _str(hw_export_output_gguf(cfg)),
    ]
    for pat in he.get("patterns", []) or []:
        argv += ["--pattern", _str(pat)]
    if he.get("only_tensor"):
        argv += ["--tensor", _str(he["only_tensor"])]
    if he.get("dump_dir"):
        argv += ["--dump-dir", _str(he["dump_dir"])]
    return argv


def format_cmd(argv: List[str]) -> str:
    """Pretty-print a CLI invocation for audit logs."""
    out = [argv[0]]
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-") and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            out.append(f"    {tok} {argv[i + 1]}")
            i += 2
        else:
            out.append(f"    {tok}")
            i += 1
    return " \\\n".join(out)
