# === REEX_SMOOTHQUANT BEGIN: generic quantization experiment CLI tests ===
"""Tests for SmoothQuant experiment command construction and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_quant.smoothquant.scripts import run_smoothquant_experiment
from llm_quant.smoothquant.runner import _source_model_fingerprint


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


def _external_baseline_args(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen3_5_moe"}\n')
    binary = tmp_path / "llama-quantize"
    binary.write_text("quantizer\n")
    fp16 = tmp_path / "baseline" / "model-f16.gguf"
    fp16.parent.mkdir()
    fp16.write_bytes(b"fp16-artifact")
    quant = fp16.parent / "model-direct-Q8_0_64.gguf"
    quant.write_bytes(b"quant-artifact")
    provenance = fp16.parent / "provenance.json"
    args = run_smoothquant_experiment.parse_args(
        [
            "--model_path",
            str(source),
            "--output_dir",
            str(tmp_path / "run"),
            "--ppl_dataset",
            str(tmp_path / "wiki.raw"),
            "--llama_quantize",
            str(binary),
            "--fp16-gguf",
            str(fp16),
            "--baseline-quant",
            str(quant),
            "--baseline-provenance",
            str(provenance),
        ]
    )
    payload = {
        "schema_version": run_smoothquant_experiment.BASELINE_PROVENANCE_SCHEMA_VERSION,
        "source_model_fingerprint": _source_model_fingerprint(str(source)),
        "llama_cpp_commit": run_smoothquant_experiment._capture(
            [
                "git",
                "-C",
                str(run_smoothquant_experiment._ROOT),
                "rev-parse",
                "HEAD",
            ]
        ),
        "fp16_gguf": run_smoothquant_experiment._fingerprint_path(str(fp16)),
        "baseline_quant": run_smoothquant_experiment._fingerprint_path(str(quant)),
        "quantize": {
            "command": [
                str(binary),
                *run_smoothquant_experiment.QUANTIZE_FLAGS,
                str(fp16.resolve()),
                str(quant.resolve()),
                "Q8_0_64",
            ],
            "flags": run_smoothquant_experiment.QUANTIZE_FLAGS,
            "quant_type": "Q8_0_64",
            "binary": run_smoothquant_experiment._fingerprint_path(str(binary)),
        },
    }
    provenance.write_text(json.dumps(payload))
    return args, provenance


def test_external_baseline_reused_only_with_complete_provenance(tmp_path: Path) -> None:
    args, _ = _external_baseline_args(tmp_path)
    experiment = run_smoothquant_experiment.SmoothQuantExperiment(args)
    stages = dict(experiment.plan())

    assert experiment.baseline_reuse["fp16_gguf"]["reused"]
    assert experiment.baseline_reuse["baseline_quant"]["reused"]
    assert "convert_fp16" not in stages
    assert "quantize_baseline" not in stages
    assert stages["ppl_fp16"][2] == str(experiment.fp16_gguf)
    assert stages["ppl_baseline_quant"][2] == str(experiment.baseline_quant)


def test_external_baseline_command_mismatch_falls_back_to_rebuild(
    tmp_path: Path,
) -> None:
    args, provenance_path = _external_baseline_args(tmp_path)
    provenance = json.loads(provenance_path.read_text())
    provenance["quantize"]["command"].insert(1, "--pure")
    provenance_path.write_text(json.dumps(provenance))

    with pytest.warns(RuntimeWarning, match="command/flags mismatch"):
        experiment = run_smoothquant_experiment.SmoothQuantExperiment(args)
    stages = dict(experiment.plan())

    assert experiment.baseline_reuse["fp16_gguf"]["reused"]
    assert not experiment.baseline_reuse["baseline_quant"]["reused"]
    assert "convert_fp16" not in stages
    assert "quantize_baseline" in stages
    assert "--pure" not in stages["quantize_baseline"]


def test_cli_help_lists_external_artifact_options(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        run_smoothquant_experiment.parse_args(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--fp16-gguf" in help_text
    assert "--baseline-quant" in help_text
    assert "--baseline-provenance" in help_text
    assert "--act-scales-cache" in help_text


def test_export_command_passes_shared_cache_and_formal_resume(tmp_path: Path) -> None:
    shared_cache = tmp_path / "calibration" / "act_scales.pt"
    args = _args(
        tmp_path / "alpha0.80",
        "--act-scales-cache",
        str(shared_cache),
        "--calibration-mode",
        "formal",
        "--resume_act_scales",
    )
    command = dict(run_smoothquant_experiment.SmoothQuantExperiment(args).plan())[
        "smoothquant_export"
    ]

    assert command[command.index("--act_scales_cache") + 1] == str(shared_cache.resolve())
    assert command[command.index("--calibration-mode") + 1] == "formal"
    assert "--resume_act_scales" in command
    assert "--reuse_act_scales" not in command


# === REEX_SMOOTHQUANT END ===
