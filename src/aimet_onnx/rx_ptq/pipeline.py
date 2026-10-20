"""Reusable ONNX PTQ steps shared by the CLI and examples/quick_start_onnx_ptq.py."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import onnx

from aimet_common.compiler_encodings import normalize_compiler_encodings
from aimet_common.evaluation import classification_accuracy, compare_outputs
from aimet_common.onnx_shapes import save_onnx_with_shapes, validate_onnx_shapes
from .calibration import inspect_model_input_dtypes, inspect_model_inputs, iter_calibration_feeds
from .compiler_encodings import CompilerEncodingConverter
from .evaluation import evaluate_onnx_dataset, load_onnx_validation_inputs
from .mixed_precision import apply_onnx_mixed_precision_bitwidth
from .po2 import apply_power_of_2_workflow, disable_onnx_bias_quantizers
from .validation import validate_quantizer_state

_EXAMPLES_CONFIG = Path(__file__).resolve().parent / "config"
if not _EXAMPLES_CONFIG.is_dir():
    _EXAMPLES_CONFIG = Path(__file__).resolve().parents[3] / "examples" / "config"
DEFAULT_QUANTSIM_CONFIG = (
    _EXAMPLES_CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"
)
DEFAULT_BITWIDTH_CONFIG = _EXAMPLES_CONFIG / "quick_start_full_quant.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def force_model_io_quantizers(sim: Any, model: onnx.ModelProto) -> None:
    """Ensure every graph input/output has an enabled activation quantizer."""
    input_names = {value_info.name for value_info in model.graph.input}
    output_names = {value_info.name for value_info in model.graph.output}
    missing = []
    added = False
    for name in sorted(input_names | output_names):
        quantizer = sim.qc_quantize_op_dict.get(name)
        if quantizer is None:
            if name not in sim.connected_graph.get_all_products():
                missing.append(name)
                continue
            sim._insert_quantizer(name, is_param=False)
            sim.activation_names.append(name)
            added = True
            quantizer = sim.qc_quantize_op_dict[name]
        quantizer.enabled = True

        if name in output_names:
            updated_name = name + "_updated"
            for value_info in sim.model.graph().output:
                if value_info.name == name:
                    value_info.name = updated_name

    if missing:
        raise RuntimeError(
            "配置没有为模型输入/输出创建 quantizer: " + ", ".join(missing)
        )
    if added:
        from aimet_onnx.utils import OrtInferenceSession

        sim.session = OrtInferenceSession(
            sim.model.model,
            sim.providers,
            session_options=sim._ort_session_options,
            path=sim._path,
        )


def load_onnx_ptq_inputs(
    model_path: str | Path,
    calibration_dir: str | Path,
    calibration_limit: int | None = None,
) -> tuple[onnx.ModelProto, Dict[str, List[int]], List[Dict[str, np.ndarray]]]:
    """Load the original ONNX graph and real calibration feeds."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    original_model = onnx.load(str(model_path))
    input_shapes = inspect_model_inputs(original_model)
    feeds = list(
        iter_calibration_feeds(
            calibration_dir,
            input_shapes,
            limit=calibration_limit,
            input_dtypes=inspect_model_input_dtypes(original_model),
        )
    )
    if not feeds:
        raise RuntimeError("没有可用的真实校准样本")
    return original_model, input_shapes, feeds


def create_onnx_ptq_sim(
    original_model: onnx.ModelProto,
    dummy_input: Dict[str, np.ndarray],
    *,
    config_file: str | None,
    param_type: str = "int8",
    activation_type: str = "int8",
    quant_scheme: str = "percentile",
    percentile_value: float = 99.99,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    work_dir: str | Path | None = None,
) -> Any:
    """Create QuantSim, force IO quantizers, then set RX percentile."""
    from aimet_onnx import QuantizationSimModel

    sim = QuantizationSimModel(
        copy.deepcopy(original_model),
        param_type=param_type,
        activation_type=activation_type,
        quant_scheme=quant_scheme,
        config_file=config_file,
        providers=list(providers),
        path=None if work_dir is None else str(work_dir),
        dummy_input=dummy_input,
    )
    force_model_io_quantizers(sim, original_model)
    if quant_scheme == "percentile":
        setter = getattr(sim, "set_percentile_value", None)
        if setter is None:
            raise RuntimeError(
                "当前 aimet_onnx QuantizationSimModel 缺少 set_percentile_value"
            )
        setter(percentile_value)
    return sim


def run_onnx_feed(
    source: Any,
    feed: Dict[str, np.ndarray],
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> List[np.ndarray]:
    """Run one feed on FP32 ONNX, a QuantSim, or an InferenceSession."""
    if hasattr(source, "session"):
        return source.session.run(None, feed)
    if hasattr(source, "run"):
        return source.run(None, feed)
    import onnxruntime as ort

    session = ort.InferenceSession(str(source), providers=list(providers))
    return session.run(None, feed)


def evaluate_onnx_outputs(
    source: Any,
    feed: Dict[str, np.ndarray],
    reference: Sequence[np.ndarray],
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> List[Dict[str, Any]]:
    return compare_outputs(reference, run_onnx_feed(source, feed, providers))


def export_onnx_json(
    sim: Any,
    original_model: onnx.ModelProto,
    output_dir: str | Path,
    filename_prefix: str,
) -> Dict[str, Any]:
    """Export clean ONNX, AIMET raw encodings, and compiler-native encodings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, coverage = CompilerEncodingConverter(sim, original_model).convert()
    raw_encodings_path = output_dir / f"{filename_prefix}.aimet.encodings"
    sim._export_encodings(str(raw_encodings_path))
    sim.export(str(output_dir), filename_prefix, export_model=True)

    onnx_output_path = output_dir / f"{filename_prefix}.onnx"
    shape_report = save_onnx_with_shapes(onnx_output_path)
    compiler_encodings_path = output_dir / f"{filename_prefix}.encodings"
    compiler_encodings_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "onnx": onnx_output_path,
        "encodings": compiler_encodings_path,
        "aimet_encodings": raw_encodings_path,
        "coverage": coverage,
        "onnx_shapes": shape_report,
    }


def verify_onnx_ptq_artifacts(
    artifacts: Dict[str, Any],
    feed: Dict[str, np.ndarray],
    expected_outputs: Sequence[np.ndarray],
    *,
    config_file: str | None,
    param_type: str = "int8",
    activation_type: str = "int8",
    quant_scheme: str = "percentile",
    percentile_value: float = 99.99,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    atol: float = 1e-5,
    rtol: float = 1e-4,
    validation_feeds: Sequence[Dict[str, np.ndarray]] | None = None,
    expected_validation_outputs: Sequence[Sequence[np.ndarray]] | None = None,
    validation_labels: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    """Restore saved quantization, verify compiler grids, and compare outputs."""
    from aimet_onnx.quantsim import load_encodings_to_sim

    if (validation_feeds is None) != (expected_validation_outputs is None):
        raise ValueError("Validation feeds and expected outputs must be supplied together")
    model = onnx.load(str(artifacts["onnx"]))
    validate_onnx_shapes(model)
    onnx.checker.check_model(str(artifacts["onnx"]), full_check=True)
    with tempfile.TemporaryDirectory(prefix="rx-met-onnx-reload-") as work_dir:
        restored = create_onnx_ptq_sim(
            model, feed, config_file=config_file, param_type=param_type,
            activation_type=activation_type, quant_scheme=quant_scheme,
            percentile_value=percentile_value, providers=providers, work_dir=work_dir,
        )
        restored._rx_expected_quantizers = frozenset(
            artifacts["coverage"]["enabled_quantizer_names"]
        )
        load_encodings_to_sim(
            restored, str(artifacts["aimet_encodings"]), strict=False,
            allow_overwrite=False, disable_missing_quantizers=True,
        )
        validate_quantizer_state(restored)
        loaded_payload, _ = CompilerEncodingConverter(restored, model).convert()
        saved_payload = normalize_compiler_encodings(
            json.loads(Path(artifacts["encodings"]).read_text(encoding="utf-8")), require_po2=True
        )
        if loaded_payload != saved_payload:
            raise RuntimeError("Restored quantizers do not match the saved compiler encodings")
        outputs = run_onnx_feed(restored, feed, providers)
        report = compare_outputs(expected_outputs, outputs)
        for index, (expected, actual) in enumerate(zip(expected_outputs, outputs)):
            if not np.allclose(actual, expected, atol=atol, rtol=rtol):
                raise RuntimeError(
                    f"Restored quantized output {index} differs from its pre-export value: {report[index]}"
                )
        if validation_feeds is not None:
            restored_outputs, dataset_report = evaluate_onnx_dataset(
                restored, validation_feeds, reference=expected_validation_outputs,
                labels=validation_labels, providers=providers,
            )
            for batch_index, (expected_batch, actual_batch) in enumerate(
                zip(expected_validation_outputs, restored_outputs)
            ):
                for index, (expected, actual) in enumerate(zip(expected_batch, actual_batch)):
                    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
                        raise RuntimeError(f"Restored quantized output {index} differs in batch {batch_index}")
            report = dataset_report["comparison"]
            if validation_labels is not None:
                expected_accuracy = classification_accuracy(expected_validation_outputs, validation_labels)
                correct_gap = abs(dataset_report["correct_count"] - expected_accuracy["correct_count"])
                if correct_gap * 200 >= expected_accuracy["sample_count"]:
                    raise RuntimeError("Restored classification accuracy differs by at least 0.5 percentage points")
                report[0].update({key: dataset_report[key]
                                  for key in ("sample_count", "correct_count", "top1_accuracy")})
                report[0]["reference_top1_accuracy"] = expected_accuracy["top1_accuracy"]
    return report


def export_onnx_compiler_artifacts(
    sim: Any,
    original_model: onnx.ModelProto,
    output_dir: str | Path,
    filename_prefix: str,
) -> Dict[str, Any]:
    """Compatibility entry point; preserves the original artifact dictionary."""
    return export_onnx_json(sim, original_model, output_dir, filename_prefix)


def write_onnx_ptq_metadata(
    output_dir: str | Path,
    filename_prefix: str,
    payload: Dict[str, Any],
) -> Path:
    metadata_path = Path(output_dir) / f"{filename_prefix}.meta.json"
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata_path


# Backward-compatible aliases used by earlier call sites.
_force_model_io_quantizers = force_model_io_quantizers
_compare_outputs = compare_outputs


def run_onnx_ptq(
    *,
    model_path: str | Path,
    calibration_dir: str | Path,
    output_dir: str | Path,
    filename_prefix: str,
    config_file: str | None = None,
    bitwidth_config_file: str | None = None,
    param_type: str = "int8",
    activation_type: str = "int8",
    quant_scheme: str = "percentile",
    percentile_value: float = 99.99,
    po2_method: str = "cover_range",
    po2_tolerance: float = 0.02,
    align_bias_scale: bool = True,
    bias_bitwidth: int = 32,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    calibration_limit: int | None = None,
    validation_dir: str | Path | None = None,
    validation_labels: Sequence[int] | None = None,
) -> Dict[str, Path]:
    """Run PTQ and emit clean ONNX plus compiler-native encodings."""
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_model, input_shapes, feeds = load_onnx_ptq_inputs(
        model_path, calibration_dir, calibration_limit
    )
    evaluation_feeds = feeds
    if validation_dir is not None:
        evaluation_feeds, validation_labels = load_onnx_validation_inputs(
            original_model, validation_dir, labels=validation_labels
        )
    reference_batches, fp32_evaluation = evaluate_onnx_dataset(
        model_path, evaluation_feeds, labels=validation_labels, providers=providers
    )
    if config_file is None:
        config_file = str(DEFAULT_QUANTSIM_CONFIG)
    sim = create_onnx_ptq_sim(
        original_model,
        feeds[0],
        config_file=config_file,
        param_type=param_type,
        activation_type=activation_type,
        quant_scheme=quant_scheme,
        percentile_value=percentile_value,
        providers=providers,
        work_dir=output_dir / "_aimet_tmp",
    )
    if bitwidth_config_file is None:
        bitwidth_config_file = str(DEFAULT_BITWIDTH_CONFIG)
    if bitwidth_config_file:
        apply_onnx_mixed_precision_bitwidth(
            sim, bitwidth_config_file, verbose=False
        )
        force_model_io_quantizers(sim, original_model)
    if align_bias_scale:
        disable_onnx_bias_quantizers(sim)
    sim.compute_encodings(feeds)
    _, ptq_evaluation = evaluate_onnx_dataset(
        sim, evaluation_feeds, reference=reference_batches, labels=validation_labels, providers=providers
    )
    po2_report = apply_power_of_2_workflow(
        sim,
        method=po2_method,
        tolerance=po2_tolerance,
        align_bias_scale=align_bias_scale,
        verbose=False,
        bias_bitwidth=bias_bitwidth,
    )
    quantized_batches, po2_evaluation = evaluate_onnx_dataset(
        sim, evaluation_feeds, reference=reference_batches, labels=validation_labels, providers=providers
    )
    sim_report = po2_evaluation["comparison"]
    artifacts = export_onnx_json(
        sim, original_model, output_dir, filename_prefix
    )
    _, clean_evaluation = evaluate_onnx_dataset(
        artifacts["onnx"], evaluation_feeds, reference=reference_batches, providers=providers
    )
    clean_report = clean_evaluation["comparison"]
    if any(item["max_abs"] > 0 for item in clean_report):
        raise RuntimeError("Exported clean ONNX differs from the original floating-point model")
    reload_report = verify_onnx_ptq_artifacts(
        artifacts, evaluation_feeds[0], quantized_batches[0], config_file=config_file,
        param_type=param_type, activation_type=activation_type, quant_scheme=quant_scheme,
        percentile_value=percentile_value, providers=providers,
        validation_feeds=evaluation_feeds, expected_validation_outputs=quantized_batches,
        validation_labels=validation_labels,
    )
    metadata_path = write_onnx_ptq_metadata(
        output_dir,
        filename_prefix,
        {
            "model": str(model_path),
            "model_sha256": _sha256(model_path),
            "input_shapes": input_shapes,
            "calibration_dir": str(Path(calibration_dir).resolve()),
            "calibration_count": len(feeds),
            "evaluation_source": "validation" if validation_dir is not None else "calibration",
            "validation_dir": str(Path(validation_dir).resolve()) if validation_dir is not None else None,
            "evaluation": {"fp32": fp32_evaluation, "ptq": ptq_evaluation, "po2": po2_evaluation},
            "config_file": str(config_file),
            "bitwidth_config_file": str(bitwidth_config_file) if bitwidth_config_file else None,
            "param_type": param_type,
            "activation_type": activation_type,
            "quant_scheme": quant_scheme,
            "percentile_value": percentile_value if quant_scheme == "percentile" else None,
            "providers": list(providers),
            "po2_method": po2_method,
            "po2_tolerance": po2_tolerance,
            "align_bias_scale": align_bias_scale,
            "bias_bitwidth": bias_bitwidth,
            "po2_changes": po2_report["po2_changes"],
            "bias_align": po2_report["bias_align"],
            "all_power_of_2": po2_report["all_power_of_2"],
            "not_power_of_2": po2_report["not_power_of_2"],
            "coverage": artifacts["coverage"],
            "onnx_shapes": artifacts["onnx_shapes"],
            "sim_vs_fp32": sim_report,
            "clean_onnx_vs_fp32": clean_report,
            "restored_quantized_vs_pre_export": reload_report,
            "raw_aimet_encodings": str(artifacts["aimet_encodings"]),
            "compiler_encodings": str(artifacts["encodings"]),
            "onnx": str(artifacts["onnx"]),
        },
    )
    return {
        "onnx": artifacts["onnx"],
        "encodings": artifacts["encodings"],
        "aimet_encodings": artifacts["aimet_encodings"],
        "metadata": metadata_path,
    }
