"""Reusable ONNX PTQ steps shared by the CLI and examples/onnx_ptq_quick_start.py."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import onnx

from .calibration import inspect_model_inputs, iter_calibration_feeds
from .compiler_encodings import CompilerEncodingConverter
from .mixed_precision import apply_onnx_mixed_precision_bitwidth
from .po2 import apply_power_of_2_workflow, disable_onnx_bias_quantizers

_EXAMPLES_CONFIG = Path(__file__).resolve().parents[2] / "examples" / "config"
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


def compare_outputs(
    reference: Sequence[np.ndarray], actual: Sequence[np.ndarray]
) -> List[Dict[str, Any]]:
    if len(reference) != len(actual):
        raise RuntimeError(
            f"输出数量不一致: reference={len(reference)}, actual={len(actual)}"
        )
    report: List[Dict[str, Any]] = []
    for index, (ref, value) in enumerate(zip(reference, actual)):
        if ref.shape != value.shape:
            raise RuntimeError(
                f"输出 {index} shape 不一致: {ref.shape} vs {value.shape}"
            )
        diff = np.asarray(value, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
        report.append(
            {
                "index": index,
                "shape": list(ref.shape),
                "max_abs": float(np.max(np.abs(diff))),
                "mean_abs": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(np.square(diff)))),
                "reference_max_abs": float(np.max(np.abs(ref))),
            }
        )
    return report


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


def export_onnx_compiler_artifacts(
    sim: Any,
    original_model: onnx.ModelProto,
    output_dir: str | Path,
    filename_prefix: str,
) -> Dict[str, Any]:
    """Export clean ONNX, AIMET raw encodings, and compiler-native encodings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_encodings_path = output_dir / f"{filename_prefix}.aimet.encodings"
    sim._export_encodings(str(raw_encodings_path))
    sim.export(str(output_dir), filename_prefix, export_model=True)

    onnx_output_path = output_dir / f"{filename_prefix}.onnx"
    compiler_encodings_path = output_dir / f"{filename_prefix}.encodings"
    payload, coverage = CompilerEncodingConverter(sim, original_model).convert()
    compiler_encodings_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    onnx.checker.check_model(str(onnx_output_path))
    return {
        "onnx": onnx_output_path,
        "encodings": compiler_encodings_path,
        "aimet_encodings": raw_encodings_path,
        "coverage": coverage,
    }


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
) -> Dict[str, Path]:
    """Run PTQ and emit clean ONNX plus compiler-native encodings."""
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_model, input_shapes, feeds = load_onnx_ptq_inputs(
        model_path, calibration_dir, calibration_limit
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
    disable_onnx_bias_quantizers(sim)
    sim.compute_encodings(feeds)
    po2_report = apply_power_of_2_workflow(
        sim,
        method=po2_method,
        tolerance=po2_tolerance,
        align_bias_scale=align_bias_scale,
        verbose=False,
        bias_bitwidth=bias_bitwidth,
    )
    reference_outputs = run_onnx_feed(model_path, feeds[0], providers)
    sim_report = evaluate_onnx_outputs(sim, feeds[0], reference_outputs, providers)
    artifacts = export_onnx_compiler_artifacts(
        sim, original_model, output_dir, filename_prefix
    )
    clean_report = evaluate_onnx_outputs(
        artifacts["onnx"], feeds[0], reference_outputs, providers
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
            "sim_vs_fp32": sim_report,
            "clean_onnx_vs_fp32": clean_report,
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
