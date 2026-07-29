"""Minimal self-checks for rx_met_llm. Run: ``python -m rx_met_llm.tests``"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))

import rx_met_llm.paths as paths_mod
from rx_met_llm import LLMQuantPipeline
from rx_met_llm.schema import (
    ConfigError,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    resolve_config,
    _deep_merge,
)
from rx_met_llm.cli import (
    build_hw_export_cmd,
    build_imatrix_cmd,
    build_perplexity_cmd,
    build_quantize_cmd,
)
from rx_met_llm.schema import hw_export_output_gguf, quantized_path

_FAKE_BIN = None


def _patch_binaries() -> None:
    global _FAKE_BIN
    _FAKE_BIN = _tmp_file()

    def _fake_resolve() -> Dict[str, str]:
        p = str(_FAKE_BIN)
        return {
            "llama_quantize": p,
            "llama_imatrix": p,
            "llama_perplexity": p,
            "convert_hf_to_gguf": p,
            "ld_library_path": "/tmp",
            "hw_export": p,
        }

    paths_mod.resolve_binaries = _fake_resolve  # type: ignore[assignment]


def _tmp_file(suffix: str = ".gguf") -> Path:
    fh = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fh.write(b"\x00")
    fh.close()
    return Path(fh.name)


def _base_config(model_path: Path, ppl_path: Path) -> dict:
    return {
        "model": str(model_path),
        "quant": "Q4_0",
        "eval": {"dataset": str(ppl_path)},
    }


def _legacy_config(model_path: Path, ppl_path: Path) -> dict:
    return {
        "schema_version": LEGACY_SCHEMA_VERSION,
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


def test_defaults_applied():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    resolved = resolve_config(_base_config(model, ppl))
    expect(resolved["quantization"]["n_threads"] == (os.cpu_count() or 8), "n_threads")
    expect(resolved["schema_version"] == SCHEMA_VERSION, "schema v2")


def test_user_v2_output_name():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    resolved = resolve_config(_base_config(model, ppl))
    expect(resolved["quantization"]["output_quantized"].endswith("-Q4_0.gguf"), "output name")


def test_unknown_ggml_type_rejected():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["quant"] = "Q42_K"
    try:
        resolve_config(cfg)
    except ConfigError:
        return
    raise AssertionError("bad quant type")


def test_plan_is_deterministic():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    expect(LLMQuantPipeline(copy.deepcopy(cfg)).plan() == LLMQuantPipeline(copy.deepcopy(cfg)).plan(), "plan")


def test_quantize_cmd_shape():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["mixed_precision"] = [{"pattern": ".*\\.attn_v\\.weight$", "type": "Q5_K"}]
    argv = build_quantize_cmd(resolve_config(cfg))
    expect(argv[-2] == "Q4_0", "quant type positional")


def test_hf_plan_uses_conversion_output():
    with tempfile.TemporaryDirectory() as model_dir:
        (Path(model_dir) / "config.json").write_text("{}", encoding="utf-8")
        ppl = _tmp_file(".txt")
        plan = LLMQuantPipeline(_base_config(Path(model_dir), ppl)).plan()
        expected = str(
            Path.cwd()
            / "runs"
            / f"{Path(model_dir).name}-Q4_0"
            / f"{Path(model_dir).name}-f16.gguf"
        )
        expect(plan["quantize"][-4] == expected, "HF quantize input")
        expect("None" not in plan["quantize"], "HF plan must not contain None")


def test_imatrix_from_calib_dataset():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    calib_ds = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["calib"] = {"dataset": str(calib_ds), "chunks": 50}
    resolved = resolve_config(cfg)
    argv = build_imatrix_cmd(resolved)
    expect(argv is not None and "--chunks" in argv, "imatrix when calib set")


def test_hw_export_bool():
    model = _tmp_file()
    ppl = _tmp_file(".txt")
    cfg = _base_config(model, ppl)
    cfg["hw_export"] = True
    resolved = resolve_config(cfg)
    expect(resolved["hw_export"]["enabled"] is True, "hw_export true")
    expect(build_hw_export_cmd(resolved) is not None, "hw cmd")


def main():
    _patch_binaries()
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append((t.__name__, exc))
            print(f"  FAIL {t.__name__}: {exc}")
    if failed:
        sys.exit(1)
    print(f"All {len(tests)} tests passed")


if __name__ == "__main__":
    main()
