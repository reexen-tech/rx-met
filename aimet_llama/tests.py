"""Minimal self-checks for aimet_llama.

No pytest dependency; runs via ``python -m aimet_llama.tests`` or directly.

Covers:

1. Defaults are applied via deep-merge.
2. Schema validators reject malformed configs.
3. ``plan()`` is deterministic given the same JSON (reproducibility hook).
4. CLI translator produces the exact argv shape expected by upstream
   ``llama-quantize`` / ``llama-imatrix`` / ``llama-perplexity``.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

# Allow running this file both as ``python -m aimet_llama.tests`` and
# directly via ``python aimet_llama/tests.py``.
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))

from aimet_llama import LLMQuantPipeline
from aimet_llama.schema import (
    ConfigError,
    DEFAULT_CONFIG,
    SCHEMA_VERSION,
    resolve_config,
    validate_config,
    _deep_merge,
)
from aimet_llama.cli import (
    build_imatrix_cmd,
    build_perplexity_cmd,
    build_quantize_cmd,
)


def _tmp_file(suffix: str = ".gguf") -> Path:
    fh = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fh.write(b"\x00")
    fh.close()
    return Path(fh.name)


def _base_config(model_path: Path, ppl_path: Path) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "name": "unit_test",
            "output_dir": str(_here.parent / "runs" / "unit_test"),
        },
        "model": {"gguf_fp16_path": str(model_path)},
        "quantization": {
            "default_type": "Q4_0",
            "output_quantized": "out.gguf",
            "use_imatrix": False,
        },
        "evaluation": {
            "enabled": True,
            "perplexity": {"dataset_file": str(ppl_path)},
        },
    }


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_deep_merge_overrides_leaves():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = _deep_merge(base, {"a": {"y": 99}, "c": 4})
    expect(merged == {"a": {"x": 1, "y": 99}, "b": 3, "c": 4}, "deep merge")


def test_defaults_applied():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    resolved = resolve_config(cfg)
    expect(
        resolved["quantization"]["n_threads"]
        == DEFAULT_CONFIG["quantization"]["n_threads"],
        "n_threads default",
    )
    expect(resolved["calibration"]["enabled"] is False, "calibration off by default")
    expect(resolved["report"]["write_markdown"] is True, "markdown report on")


def test_unknown_ggml_type_rejected():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["quantization"]["default_type"] = "Q42_K"
    try:
        resolve_config(cfg)
    except ConfigError as exc:
        expect("Q42_K" in str(exc), "error mentions bad type")
        return
    raise AssertionError("invalid ggml type should have been rejected")


def test_bad_regex_rejected():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["quantization"]["tensor_type_overrides"] = [
        {"pattern": "(unclosed", "type": "Q5_K"},
    ]
    try:
        resolve_config(cfg)
    except ConfigError as exc:
        expect("regex" in str(exc).lower(), "error mentions regex")
        return
    raise AssertionError("bad regex should have been rejected")


def test_include_exclude_mutually_exclusive():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["quantization"]["imatrix_include_weights"] = ["attn_v"]
    cfg["quantization"]["imatrix_exclude_weights"] = ["ffn_down"]
    try:
        resolve_config(cfg)
    except ConfigError:
        return
    raise AssertionError("include + exclude should be rejected")


def test_calibration_requires_dataset_or_reuse():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["calibration"] = {"enabled": True}  # no dataset, no reuse
    try:
        resolve_config(cfg)
    except ConfigError:
        return
    raise AssertionError("calibration without dataset/reuse should be rejected")


def test_plan_is_deterministic():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    p1 = LLMQuantPipeline(copy.deepcopy(cfg)).plan()
    p2 = LLMQuantPipeline(copy.deepcopy(cfg)).plan()
    expect(p1 == p2, "plan should be deterministic for identical config")


def test_quantize_cmd_shape():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["quantization"]["output_tensor_type"] = "Q6_K"
    cfg["quantization"]["tensor_type_overrides"] = [
        {"pattern": ".*\\.attn_v\\.weight$", "type": "Q5_K"},
    ]
    resolved = resolve_config(cfg)
    argv = build_quantize_cmd(resolved)
    expect(argv[0].endswith("llama-quantize"), "binary first")
    expect("--output-tensor-type" in argv, "output tensor type forwarded")
    expect("Q6_K" in argv, "Q6_K value forwarded")
    # tensor-type pattern=type is one combined token
    joined = " ".join(argv)
    expect(".*\\.attn_v\\.weight$=Q5_K" in joined, "regex override forwarded")
    # positional args order: input, output, type, threads
    expect(argv[-3].endswith("out.gguf"), "output GGUF before type")
    expect(argv[-2] == "Q4_0", "default type as positional")
    expect(argv[-1] == "8", "default thread count appears last")


def test_imatrix_cmd_shape_and_skip():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    expect(build_imatrix_cmd(resolve_config(cfg)) is None, "no imatrix by default")

    calib_ds = _tmp_file(".txt")
    cfg["calibration"] = {
        "enabled": True,
        "dataset_file": str(calib_ds),
        "n_chunks": 50,
    }
    cfg["quantization"]["use_imatrix"] = True
    resolved = resolve_config(cfg)
    argv = build_imatrix_cmd(resolved)
    expect(argv[0].endswith("llama-imatrix"), "imatrix binary")
    expect("-f" in argv, "-f flag present")
    expect(str(calib_ds) in argv, "dataset path present")
    expect("--chunks" in argv and "50" in argv, "--chunks forwarded")
    # Quantize should reference the imatrix file we just planned
    qargv = build_quantize_cmd(resolved)
    expect("--imatrix" in qargv, "quantize picks up imatrix")


def test_perplexity_skip_when_disabled():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["evaluation"] = {"enabled": False}
    resolved = resolve_config(cfg)
    expect(build_perplexity_cmd(resolved) is None, "perplexity skipped")


def test_perplexity_kv_cache_types_forwarded():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["evaluation"]["perplexity"]["cache_type_k"] = "q4_0"
    cfg["evaluation"]["perplexity"]["cache_type_v"] = "q8_0"
    cfg["evaluation"]["perplexity"]["chunks"] = 10
    resolved = resolve_config(cfg)
    argv = build_perplexity_cmd(resolved)
    expect("-ctk" in argv, "-ctk forwarded")
    expect("q4_0" in argv, "cache_type_k value present")
    expect("-ctv" in argv, "-ctv forwarded")
    expect("q8_0" in argv, "cache_type_v value present")
    expect("--chunks" in argv and "10" in argv, "chunks forwarded")


def test_perplexity_kv_cache_invalid_rejected():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["evaluation"]["perplexity"]["cache_type_k"] = "not_a_type"
    try:
        resolve_config(cfg)
    except ConfigError as exc:
        expect("cache_type_k" in str(exc), "error mentions the bad field")
        return
    raise AssertionError("invalid KV cache type should have been rejected")


def test_reex_q64_type_accepted():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["quantization"]["default_type"] = "Q4_K_64"
    cfg["quantization"]["tensor_type_overrides"] = [
        {"pattern": ".*\\.ffn_down\\..*", "type": "Q6_K_64"},
    ]
    resolved = resolve_config(cfg)
    argv = build_quantize_cmd(resolved)
    expect(argv[-2] == "Q4_K_64", "REEX block-64 default type forwarded")
    expect(".*\\.ffn_down\\..*=Q6_K_64" in " ".join(argv), "REEX override forwarded")


def test_reex_psum_bits_sets_env():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["evaluation"]["perplexity"]["reex_psum_bits"] = 8
    resolved = resolve_config(cfg)
    expect(
        resolved["evaluation"]["perplexity"]["reex_psum_bits"] == 8,
        "reex_psum_bits preserved through resolve",
    )
    pipe = LLMQuantPipeline(copy.deepcopy(cfg))
    # plan() is argv-only; psum bits must NOT leak into argv (it is an env var)
    expect(
        "REEX_Q64_PSUM_BITS" not in " ".join(pipe.plan()["perplexity"]),
        "psum bits stays out of argv",
    )


def test_reex_psum_bits_invalid_rejected():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["evaluation"]["perplexity"]["reex_psum_bits"] = "eight"
    try:
        resolve_config(cfg)
    except ConfigError as exc:
        expect("reex_psum_bits" in str(exc), "error mentions the bad field")
        return
    raise AssertionError("non-integer reex_psum_bits should have been rejected")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as exc:
            failed.append((t.__name__, exc))
            print(f"  FAIL {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append((t.__name__, exc))
            print(f"  ERR  {t.__name__}: {type(exc).__name__}: {exc}")
    print()
    if failed:
        print(f"{len(failed)} / {len(tests)} test(s) failed")
        sys.exit(1)
    print(f"All {len(tests)} tests passed")


if __name__ == "__main__":
    main()
