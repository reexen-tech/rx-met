# === REEX_SMOOTHQUANT BEGIN: generic quantization experiment CLI tests ===
"""Tests for SmoothQuant experiment command construction and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_quant.smoothquant.scripts import run_smoothquant_experiment


def _args(tmp_path: Path, *extra: str):
    return run_smoothquant_experiment.parse_args(
        [
            "--model_path",
            "model",
            "--output_dir",
            str(tmp_path),
            "--ppl_dataset",
            "wiki.test.raw",
            *extra,
        ]
    )


def test_quant_type_defaults_to_q8_0_64(tmp_path: Path) -> None:
    args = _args(tmp_path)
    experiment = run_smoothquant_experiment.SmoothQuantExperiment(args)

    assert args.quant_type == "Q8_0_64"
    assert experiment.baseline_quant.name == "model-baseline-Q8_0_64.gguf"
    assert experiment.smooth_quant.name == "model-smooth-Q8_0_64.gguf"


def test_quant_type_controls_commands_and_output_names(tmp_path: Path) -> None:
    args = _args(tmp_path, "--quant-type", "q4_k_64")
    experiment = run_smoothquant_experiment.SmoothQuantExperiment(args)
    plan = dict(experiment.plan())

    assert args.quant_type == "Q4_K_64"
    assert experiment.baseline_quant.name == "model-baseline-Q4_K_64.gguf"
    assert experiment.smooth_quant.name == "model-smooth-Q4_K_64.gguf"

    for stage in ("quantize_smooth", "quantize_baseline"):
        cmd = plan[stage]
        assert cmd[-1] == "Q4_K_64"
        assert "--pure" not in cmd
        assert cmd[1:6] == [
            "--token-embedding-type",
            "f16",
            "--output-tensor-type",
            "f16",
            "--leave-output-tensor",
        ]

    assert "ppl_baseline_quant" in plan
    assert "ppl_smooth_quant" in plan


def test_quantize_full_uses_quantizer_defaults(tmp_path: Path) -> None:
    args = _args(tmp_path, "--quant-type", "Q4_0_64", "--quantize_full")
    experiment = run_smoothquant_experiment.SmoothQuantExperiment(args)

    assert experiment._cmd_quantize(Path("in.gguf"), Path("out.gguf")) == [
        args.llama_quantize,
        "in.gguf",
        "out.gguf",
        "Q4_0_64",
    ]


def test_ref_compare_uses_colocated_smoothquant_script(tmp_path: Path) -> None:
    args = _args(tmp_path, "--ref_repo", "/tmp/smoothquant-ref")
    experiment = run_smoothquant_experiment.SmoothQuantExperiment(args)

    script = Path(experiment._cmd_ref_compare()[1])
    assert script == (
        Path(run_smoothquant_experiment.__file__).resolve().parent
        / "compare_smoothquant_ref.py"
    )
    assert script.is_file()


def test_quant_type_rejects_unsafe_filename_characters(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _args(tmp_path, "--quant-type", "../Q4_K_64")


# === REEX_SMOOTHQUANT END ===
