"""Reusable ONNX PTQ steps shared by the CLI and examples/quick_start_onnx_ptq.py."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import onnx

from aimet_common.compiler_artifacts import load_compiler_encodings
from aimet_common.compiler_graph_validation import validate_compiler_encoding_graph
from aimet_common.evaluation import classification_accuracy, compare_output_batches, compare_outputs
from aimet_common.run_context import (
    get_run, logged_export, logged_operation, logged_workflow, metadata_output, note_output,
)
from aimet_common.onnx_shapes import save_onnx_with_shapes, validate_onnx_shapes
from .calibration import inspect_model_input_dtypes, inspect_model_inputs, iter_calibration_feeds
from .compiler_encodings import CompilerEncodingConverter
from .conv_bn import apply_conv_bn_boundaries
from .evaluation import evaluate_onnx_dataset, load_onnx_validation_inputs
from .mixed_precision import apply_onnx_mixed_precision_bitwidth, validate_linear_quantizer_config
from .linear import linear_groups, normalize_linear_graph, validate_legacy_linear_boundaries
from .po2 import align_onnx_bias_scale, apply_power_of_2_workflow, disable_onnx_bias_quantizers
from .execution import ptq_execution_providers
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


def _check_linear_float_equivalence(reference, actual, *, atol=1e-6, rtol=1e-5):
    report = compare_output_batches(reference, actual)
    for batch_index, (expected, values) in enumerate(zip(reference, actual)):
        for output_index, (left, right) in enumerate(zip(expected, values)):
            if not np.allclose(right, left, atol=atol, rtol=rtol):
                raise RuntimeError(
                    f"Linear normalization changed FP32 output {output_index} in batch {batch_index}: "
                    f"{report[output_index]} (atol={atol}, rtol={rtol})"
                )
    for output_index, item in enumerate(report):
        item['max_rel'] = max(
            float(np.max(np.abs(np.asarray(values[output_index], dtype=np.float64)
                                - np.asarray(expected[output_index], dtype=np.float64))
                         / np.maximum(np.abs(expected[output_index]), 1e-12)))
            for expected, values in zip(reference, actual)
        )
    return report


def _float_session(model, providers):
    import onnxruntime as ort
    return ort.InferenceSession(model.SerializeToString(), providers=ptq_execution_providers(providers))


def _enforce_linear_boundaries(sim, model):
    groups = getattr(sim, '_rx_linear_groups', {})
    initialized = getattr(sim, '_rx_linear_boundary_owners', set())
    graph_outputs = {value.name for value in model.graph.output}
    added = False
    for name, group in groups.items():
        if group['gemm_output'] == group['output']:
            continue
        terminal = sim.qc_quantize_op_dict.get(group['output'])
        template = sim.qc_quantize_op_dict.get(group['gemm_output'])
        if terminal is None:
            sim._insert_quantizer(group['output'], is_param=False)
            sim.activation_names.append(group['output'])
            terminal = sim.qc_quantize_op_dict[group['output']]
            added = True
        if name not in initialized and template is not None:
            terminal.set_bitwidth(template.bitwidth)
            terminal.use_strict_symmetric = False
            terminal.signed = template.signed
            terminal.use_symmetric_encodings = template.use_symmetric_encodings
            terminal.use_strict_symmetric = template.use_strict_symmetric
            terminal.use_unsigned_symmetric = template.use_unsigned_symmetric
            terminal.data_type = template.data_type
            terminal.enabled = template.enabled or group['output'] in graph_outputs
        initialized.add(name)
        # Reshapes carry the existing input/output grids; they add no rounding.
        for tensor in (group['gemm_input'], group['gemm_output']):
            if tensor not in (group['input'], group['output']):
                quantizer = sim.qc_quantize_op_dict.get(tensor)
                if quantizer is not None:
                    quantizer.enabled = False
    sim._rx_linear_boundary_owners = initialized
    return added


def force_model_io_quantizers(sim: Any, model: onnx.ModelProto) -> None:
    """Establish default model I/O grids without overriding explicit exclusions."""
    model = getattr(sim, '_rx_normalized_model', model)
    input_names = {value_info.name for value_info in model.graph.input}
    output_names = {value_info.name for value_info in model.graph.output}
    conv_bn_outputs = {group['bn_output']
                       for group in getattr(sim, '_rx_conv_bn_groups', {}).values()}
    missing = []
    added = False
    for name in sorted(input_names | output_names):
        if not sim._is_quantizable_dtype(name):
            continue
        if name in sim._input_roles.control_only:
            continue
        if (name in getattr(sim, "_rx_native_io", ())
                or name in getattr(sim, "_rx_input_owned_io", ())):
            continue
        if name in conv_bn_outputs and getattr(sim, '_rx_conv_bn_initialized', False):
            continue
        quantizer = sim.qc_quantize_op_dict.get(name)
        if quantizer is not None and id(quantizer) in getattr(sim, '_rx_explicit_disabled_quantizers', ()):
            continue
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
    if _enforce_linear_boundaries(sim, model) or added:
        sim._rebuild_session()


@logged_operation
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


@logged_operation(capture_config=False)
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
    from .input_quantization import InputQuantizationSim

    normalized_model = normalize_linear_graph(original_model)
    groups = linear_groups(normalized_model)
    validate_linear_quantizer_config(normalized_model, config_file)
    normalization_report = []
    if groups:
        reference = _float_session(original_model, providers).run(None, dummy_input)
        actual = _float_session(normalized_model, providers).run(None, dummy_input)
        normalization_report = _check_linear_float_equivalence([reference], [actual])
    sim_type = InputQuantizationSim
    if any(node.op_type in ("GRU", "LSTM") and not node.domain for node in normalized_model.graph.node):
        from .recurrent import NativeRecurrentSim
        sim_type = NativeRecurrentSim
    sim = sim_type(
        copy.deepcopy(normalized_model),
        param_type=param_type,
        activation_type=activation_type,
        quant_scheme=quant_scheme,
        config_file=config_file,
        providers=ptq_execution_providers(providers),
        path=None if work_dir is None else str(work_dir),
        dummy_input=dummy_input,
    )
    sim._rx_normalized_model = normalized_model
    sim._rx_linear_groups = groups
    sim._rx_linear_normalization_report = normalization_report
    force_model_io_quantizers(sim, normalized_model)
    if apply_conv_bn_boundaries(sim, normalized_model, config_file=config_file):
        sim._rebuild_session()
    if quant_scheme == "percentile":
        setter = getattr(sim, "set_percentile_value", None)
        if setter is None:
            raise RuntimeError(
                "当前 aimet_onnx QuantizationSimModel 缺少 set_percentile_value"
            )
        setter(percentile_value)
    return sim


@logged_operation
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

    session = ort.InferenceSession(str(source), providers=ptq_execution_providers(providers))
    return session.run(None, feed)


@logged_operation
def evaluate_onnx_outputs(
    source: Any,
    feed: Dict[str, np.ndarray],
    reference: Sequence[np.ndarray],
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> List[Dict[str, Any]]:
    return compare_outputs(reference, run_onnx_feed(source, feed, providers))


@logged_export('output_dir', task='onnx_export')
def export_onnx_json(
    sim: Any,
    original_model: onnx.ModelProto,
    output_dir: str | Path,
    filename_prefix: str,
) -> Dict[str, Any]:
    """Export clean ONNX, raw AIMET encodings, and compiler encodings as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_model = getattr(sim, '_rx_normalized_model', original_model)
    validate_legacy_linear_boundaries(original_model)
    payload, coverage = CompilerEncodingConverter(sim, original_model).convert()
    raw_encodings_path = output_dir / f"{filename_prefix}.aimet.encodings"
    sim._export_encodings(str(raw_encodings_path))
    sim.export(str(output_dir), filename_prefix, export_model=True)

    onnx_output_path = output_dir / f"{filename_prefix}.onnx"
    shape_report = save_onnx_with_shapes(onnx_output_path)
    compiler_encodings_path = output_dir / f"{filename_prefix}.json"
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


@logged_operation(capture_config=False)
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
    try:
        saved_payload = load_compiler_encodings(artifacts["encodings"])
        validate_compiler_encoding_graph(model, saved_payload)
    except ValueError as error:
        raise RuntimeError(f"Saved compiler encodings failed strict validation: {error}") from error
    with tempfile.TemporaryDirectory(prefix="rx-met-onnx-reload-") as work_dir:
        restored = create_onnx_ptq_sim(
            model, feed, config_file=config_file, param_type=param_type,
            activation_type=activation_type, quant_scheme=quant_scheme,
            percentile_value=percentile_value, providers=providers, work_dir=work_dir,
        )
        restored._rx_expected_quantizers = frozenset(
            artifacts["coverage"]["enabled_quantizer_names"]
        )
        restored._rx_expected_native_recurrent = frozenset(
            artifacts["coverage"].get("enabled_native_recurrent_nodes", ())
        )
        load_encodings_to_sim(
            restored, str(artifacts["aimet_encodings"]), strict=False,
            allow_overwrite=False, disable_missing_quantizers=True,
        )
        validate_quantizer_state(restored)
        loaded_payload, _ = CompilerEncodingConverter(restored, model).convert()
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
    with metadata_output(output_dir) as directory:
        metadata_path = directory / f"{filename_prefix}.meta.json"
        metadata_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        note_output(metadata_path)
    return metadata_path


# Backward-compatible aliases used by earlier call sites.
_force_model_io_quantizers = force_model_io_quantizers
_compare_outputs = compare_outputs


@logged_workflow('output_dir', task='onnx_ptq', capture_config=False)
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
    enable_po2: bool = True,
    align_bias_scale: bool = True,
    bias_bitwidth: int = 32,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    calibration_limit: int | None = None,
    validation_dir: str | Path | None = None,
    validation_labels: Sequence[int] | None = None,
    on_evaluation: Callable[[str, list[list[np.ndarray]]], None] | None = None,
) -> Dict[str, Path]:
    """Run PTQ with accumulator bias alignment and optional Po2 conversion.

    ``align_bias_scale`` remains for call compatibility; the standard workflow
    requires it to be true. Use ``enable_po2=False`` to export affine encodings.

    ``on_evaluation(stage, batches)`` receives evaluation outputs after each
    executed ``fp32``, ``ptq`` and ``po2`` stage. Each batch contains the model
    outputs in ONNX order. Callbacks should treat the arrays as read-only.
    Callback exceptions propagate to the caller.
    """
    if not align_bias_scale:
        raise ValueError("run_onnx_ptq requires align_bias_scale=True for accumulator bias alignment")
    providers = ptq_execution_providers(providers)
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
    if on_evaluation is not None:
        on_evaluation("fp32", reference_batches)
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
        work_dir=get_run().work / "aimet",
    )
    normalized_model = sim._rx_normalized_model
    groups = linear_groups(normalized_model)
    normalized_reference_batches = reference_batches
    normalization = {'groups': list(groups.values()), 'atol': 1e-6, 'rtol': 1e-5}
    if groups:
        normalized_session = _float_session(normalized_model, providers)
        calibration_reference = reference_batches
        if evaluation_feeds is not feeds:
            calibration_reference, _ = evaluate_onnx_dataset(model_path, feeds, providers=providers)
        normalized_calibration, _ = evaluate_onnx_dataset(normalized_session, feeds, providers=providers)
        normalization['calibration'] = _check_linear_float_equivalence(
            calibration_reference, normalized_calibration)
        if evaluation_feeds is feeds:
            normalized_reference_batches = normalized_calibration
        else:
            normalized_reference_batches, _ = evaluate_onnx_dataset(
                normalized_session, evaluation_feeds, providers=providers)
        normalization['evaluation'] = _check_linear_float_equivalence(
            reference_batches, normalized_reference_batches)
    if bitwidth_config_file is None:
        bitwidth_config_file = str(DEFAULT_BITWIDTH_CONFIG)
    if bitwidth_config_file:
        apply_onnx_mixed_precision_bitwidth(
            sim, bitwidth_config_file, verbose=False
        )
        force_model_io_quantizers(sim, normalized_model)
    first_provider = providers[0] if providers else None
    if isinstance(first_provider, tuple):
        first_provider = first_provider[0]
    if first_provider == "CUDAExecutionProvider":
        active = sim.session.get_providers()
        if not active or active[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"ONNX PTQ requested CUDAExecutionProvider but initialized {active}")
    disable_onnx_bias_quantizers(sim, preserve_shared_percentile=True)
    sim.compute_encodings(feeds)
    bias_report = align_onnx_bias_scale(sim, bias_bitwidth=bias_bitwidth)
    ptq_batches, ptq_evaluation = evaluate_onnx_dataset(
        sim, evaluation_feeds, reference=reference_batches, labels=validation_labels, providers=providers
    )
    if on_evaluation is not None:
        on_evaluation("ptq", ptq_batches)
    del ptq_batches
    evaluations = {"fp32": fp32_evaluation, "ptq": ptq_evaluation}
    po2_report = None
    po2_metadata = {}
    if enable_po2:
        po2_report = apply_power_of_2_workflow(
            sim,
            method=po2_method,
            tolerance=po2_tolerance,
            align_bias_scale=False,
            verbose=False,
        )
        po2_bias_report = align_onnx_bias_scale(sim, bias_bitwidth=bias_bitwidth)
        po2_batches, po2_evaluation = evaluate_onnx_dataset(
            sim, evaluation_feeds, reference=reference_batches, labels=validation_labels, providers=providers
        )
        if on_evaluation is not None:
            on_evaluation("po2", po2_batches)
        del po2_batches
        evaluations["po2"] = po2_evaluation
        po2_metadata = {
            "po2_method": po2_method,
            "po2_tolerance": po2_tolerance,
            "po2_changes": po2_report["po2_changes"],
            "po2_bias_align": po2_bias_report,
            "all_power_of_2": po2_report["all_power_of_2"],
            "not_power_of_2": po2_report["not_power_of_2"],
        }
    quantized_batches, final_evaluation = evaluate_onnx_dataset(
        sim, evaluation_feeds, reference=reference_batches, labels=validation_labels, providers=providers
    )
    sim_report = final_evaluation["comparison"]
    artifacts = export_onnx_json(
        sim, original_model, output_dir, filename_prefix
    )
    _, clean_evaluation = evaluate_onnx_dataset(
        artifacts["onnx"], evaluation_feeds, reference=reference_batches, providers=providers
    )
    clean_report = clean_evaluation["comparison"]
    clean_normalized_report = clean_report
    if groups:
        _, clean_normalized_evaluation = evaluate_onnx_dataset(
            artifacts['onnx'], evaluation_feeds, reference=normalized_reference_batches, providers=providers)
        clean_normalized_report = clean_normalized_evaluation['comparison']
    if any(item["max_abs"] > 0 for item in clean_normalized_report):
        raise RuntimeError("Exported clean ONNX differs from the normalized floating-point model")
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
            "evaluation": evaluations,
            "config_file": str(config_file),
            "bitwidth_config_file": str(bitwidth_config_file) if bitwidth_config_file else None,
            "param_type": param_type,
            "activation_type": activation_type,
            "quant_scheme": quant_scheme,
            "percentile_value": percentile_value if quant_scheme == "percentile" else None,
            "providers": list(providers),
            "enable_po2": enable_po2,
            "align_bias_scale": align_bias_scale,
            "bias_bitwidth": bias_bitwidth,
            "bias_align": bias_report,
            **po2_metadata,
            "coverage": artifacts["coverage"],
            "onnx_shapes": artifacts["onnx_shapes"],
            "sim_vs_fp32": sim_report,
            "clean_onnx_vs_fp32": clean_report,
            "linear_normalization": normalization,
            "clean_onnx_vs_normalized_fp32": clean_normalized_report,
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
