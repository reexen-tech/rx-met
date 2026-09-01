"""RX Power-of-2 and bias-scale alignment for ONNX AIMET quantizers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from aimet_common import libpymo
from aimet_common.defs import QuantizationDataType
from aimet_common.power_of_2 import (
    compute_aligned_bias_range,
    find_closest_power_of_2_scale,
    recompute_min_max_for_new_scale,
    verify_power_of_2_scale,
)


_ANALYTIC_BIAS_OPS = frozenset({"Conv", "ConvTranspose", "Gemm", "MatMul"})


def disable_onnx_bias_quantizers(sim: Any) -> List[str]:
    """Turn off Conv/Gemm/MatMul bias fake-quant before activation calibration.

    AIMET ONNX per-channel INT8/INT32 bias encodings use ``scale = max/qmax``.
    A near-zero bias channel gets ``scale=0``, and fake-quant then injects Inf
    into the rest of the graph.  RX bias encodings are not statistically
    calibrated; ``align_onnx_bias_scale`` writes ``Sb = Sx * Sw`` afterwards.
    """
    disabled: List[str] = []
    getter = getattr(sim, "_get_weight_and_bias", None)
    if getter is None:
        raise RuntimeError("当前 aimet_onnx QuantizationSimModel 缺少 _get_weight_and_bias")
    for op in sim.connected_graph.get_all_ops().values():
        if op.type not in _ANALYTIC_BIAS_OPS:
            continue
        _weight, bias = getter(op)
        if bias is None:
            continue
        quantizer = sim.qc_quantize_op_dict.get(bias.name)
        if quantizer is None or not quantizer.enabled:
            continue
        quantizer.enabled = False
        disabled.append(bias.name)
    return disabled


def align_encoding_values(
    *,
    scale: float,
    offset: int,
    real_min: float,
    real_max: float,
    bitwidth: int,
    symmetric: bool,
    method: str = "cover_range",
    tolerance: float = 0.02,
) -> Dict[str, Any]:
    """Align one TfEncoding-equivalent payload with the Torch Po2 contract."""
    # TfEncoding 约定 qmin=0、qmax=2^bw-1；选 scale / 重算范围都走 RX 原文。
    qmin = 0
    qmax = (1 << int(bitwidth)) - 1
    new_scale, exponent = find_closest_power_of_2_scale(
        float(scale),
        method=method,
        qmin=qmin,
        qmax=qmax,
        rmin=float(real_min),
        rmax=float(real_max),
        tolerance=tolerance,
    )
    new_offset, new_min, new_max, _, _ = recompute_min_max_for_new_scale(
        new_scale,
        qmin,
        qmax,
        float(real_min),
        float(offset),
        symmetric,
    )
    return {
        "bitwidth": int(bitwidth),
        "scale": float(new_scale),
        "n": int(exponent),
        "offset": int(round(new_offset)) if new_offset is not None else 0,
        "real_min": float(new_min),
        "real_max": float(new_max),
    }


def compute_aligned_bias_encoding(
    input_scale: float, weight_scale: float, bitwidth: int = 32
) -> Dict[str, Any]:
    """Derive ``Sb = Sx * Sw`` with signed symmetric integer bounds.

    RX production mixed-precision uses 32-bit bias so ``Qb = bias / (Sx*Sw)``
    still fits after the scale is tightened to the accumulator.
    """
    if input_scale <= 0 or weight_scale <= 0:
        raise ValueError(
            f"bias scale 需要对正的 Sx/Sw，得到 Sx={input_scale}, Sw={weight_scale}"
        )
    scale, qmin, _qmax, real_min, real_max = compute_aligned_bias_range(
        input_scale, weight_scale, bitwidth
    )
    return {
        "bitwidth": int(bitwidth),
        "scale": scale,
        "offset": int(qmin),
        "real_min": float(real_min),
        "real_max": float(real_max),
    }


def _encoding_from_values(values: Dict[str, Any]) -> Any:
    encoding = libpymo.TfEncoding()
    encoding.bw = int(values["bitwidth"])
    encoding.delta = float(values["scale"])
    encoding.offset = int(values["offset"])
    encoding.min = float(values["real_min"])
    encoding.max = float(values["real_max"])
    return encoding


def _align_quantizer(
    quantizer: Any,
    *,
    method: str,
    tolerance: float,
) -> List[float]:
    encodings = quantizer.get_encodings()
    if not encodings:
        raise RuntimeError("enabled quantizer 尚未初始化")
    symmetric = bool(getattr(quantizer, "use_symmetric_encodings", False))
    aligned = [
        align_encoding_values(
            scale=float(encoding.delta),
            offset=int(round(encoding.offset)),
            real_min=float(encoding.min),
            real_max=float(encoding.max),
            bitwidth=int(encoding.bw),
            symmetric=symmetric,
            method=method,
            tolerance=tolerance,
        )
        for encoding in encodings
    ]
    quantizer.load_encodings([_encoding_from_values(item) for item in aligned])
    return [item["scale"] for item in aligned]


def _scalar_scale(encodings: Sequence[Any]) -> float:
    scales = [float(encoding.delta) for encoding in encodings]
    if not scales:
        raise RuntimeError("quantizer encodings 为空")
    return scales[0]


def align_onnx_bias_scale(sim: Any, *, bias_bitwidth: int = 32) -> Dict[str, Any]:
    """Set Conv/Gemm/MatMul bias encodings to ``Sb = Sx * Sw``.

    This mirrors ``aimet_torch.utils_rx._align_conv_bias_scale`` plus the RX
    mixed-precision default ``bias_bitwidth=32``.  Official INT32 concretize is
    not used; the scale still comes from the live Po2 input/weight encodings.

    Bias quantizers may be disabled / uninitialized after calibration; this
    function re-enables them and writes derived encodings.
    """
    report: Dict[str, Any] = {
        "total": 0,
        "modified": 0,
        "skipped": [],
        "scale_changes": [],
    }
    get_weight_and_bias = getattr(sim, "_get_weight_and_bias", None)
    get_input_quantizer = getattr(sim, "_get_enabled_quantizer", None)
    if get_weight_and_bias is None or get_input_quantizer is None:
        raise RuntimeError("当前 aimet_onnx QuantizationSimModel 缺少 bias 对齐接口")

    for op in sim.connected_graph.get_all_ops().values():
        if op.type not in _ANALYTIC_BIAS_OPS:
            continue
        weight, bias = get_weight_and_bias(op)
        if bias is None:
            continue
        report["total"] += 1
        bias_quantizer = sim.qc_quantize_op_dict.get(bias.name)
        if bias_quantizer is None or bias_quantizer.data_type != QuantizationDataType.int:
            report["skipped"].append((bias.name, "bias quantizer missing or not integer"))
            continue
        if weight is None:
            report["skipped"].append((bias.name, "op has no weight"))
            continue
        weight_quantizer = sim.qc_quantize_op_dict.get(weight.name)
        if (
            weight_quantizer is None
            or not weight_quantizer.enabled
            or not weight_quantizer.is_initialized()
        ):
            report["skipped"].append((bias.name, "weight quantizer missing or uninitialized"))
            continue
        try:
            input_quantizer = get_input_quantizer(op.inputs[0].name)
        except KeyError:
            input_quantizer = None
        if (
            input_quantizer is None
            or not input_quantizer.enabled
            or not input_quantizer.is_initialized()
        ):
            report["skipped"].append((bias.name, "input quantizer missing or uninitialized"))
            continue

        input_scale = _scalar_scale(input_quantizer.get_encodings())
        weight_encodings = weight_quantizer.get_encodings()
        if not weight_encodings:
            report["skipped"].append((bias.name, "weight encodings empty"))
            continue

        try:
            if hasattr(bias_quantizer, "enable_per_channel_quantization"):
                if len(weight_encodings) > 1:
                    if not bias_quantizer.quant_info.usePerChannelMode:
                        bias_quantizer.enable_per_channel_quantization()
                elif bias_quantizer.quant_info.usePerChannelMode:
                    bias_quantizer.enable_per_channel_quantization(False)
        except Exception as exc:  # pylint: disable=broad-except
            report["skipped"].append((bias.name, f"cannot match weight granularity: {exc}"))
            continue

        old_scales = [float(encoding.delta) for encoding in bias_quantizer.get_encodings() or []]
        aligned = [
            compute_aligned_bias_encoding(
                input_scale,
                float(weight_encoding.delta),
                bias_bitwidth,
            )
            for weight_encoding in weight_encodings
        ]
        if hasattr(bias_quantizer, "set_bitwidth"):
            bias_quantizer.set_bitwidth(bias_bitwidth)
        bias_quantizer.load_encodings([_encoding_from_values(item) for item in aligned])
        bias_quantizer.enabled = True
        new_scales = [item["scale"] for item in aligned]
        report["modified"] += 1
        report["scale_changes"].append(
            {
                "name": bias.name,
                "op": getattr(op, "name", op.type),
                "old_scale": old_scales,
                "new_scale": new_scales,
                "bitwidth": int(bias_bitwidth),
            }
        )
    return report


def align_sim_encodings_to_po2(
    sim: Any,
    *,
    method: str = "cover_range",
    tolerance: float = 0.02,
    align_bias_scale: bool = True,
    bias_bitwidth: int = 32,
) -> Dict[str, Any]:
    """Apply the RX Po2 workflow to live ONNX quantizers.

    1. Align every enabled integer quantizer with ``cover_range`` / ``round``.
    2. Optionally overwrite Conv/Gemm/MatMul bias with ``Sb = Sx * Sw``.
    """
    changes: List[Dict[str, Any]] = []
    skipped_non_positive: List[str] = []
    for name, quantizer in sim.qc_quantize_op_dict.items():
        if not quantizer.enabled:
            continue
        if quantizer.data_type != QuantizationDataType.int:
            continue
        encodings = quantizer.get_encodings()
        if not encodings:
            raise RuntimeError(f"enabled quantizer 尚未初始化: {name}")
        old_scales = [float(encoding.delta) for encoding in encodings]
        if any(scale <= 0 or not math.isfinite(scale) for scale in old_scales):
            quantizer.enabled = False
            skipped_non_positive.append(name)
            continue
        new_scales = _align_quantizer(quantizer, method=method, tolerance=tolerance)
        if old_scales != new_scales:
            changes.append(
                {
                    "name": name,
                    "old_scale": old_scales,
                    "new_scale": new_scales,
                }
            )

    bias_report: Dict[str, Any] = {
        "total": 0,
        "modified": 0,
        "skipped": [],
        "scale_changes": [],
    }
    if align_bias_scale:
        bias_report = align_onnx_bias_scale(sim, bias_bitwidth=bias_bitwidth)

    not_po2 = []
    for name, quantizer in sim.qc_quantize_op_dict.items():
        if not quantizer.enabled or quantizer.data_type != QuantizationDataType.int:
            continue
        encodings = quantizer.get_encodings() or []
        if any(not verify_power_of_2_scale(float(encoding.delta))[0] for encoding in encodings):
            not_po2.append(name)

    return {
        "method": method,
        "tolerance": tolerance,
        "align_bias_scale": align_bias_scale,
        "bias_bitwidth": bias_bitwidth,
        "po2_changes": changes,
        "bias_align": bias_report,
        "skipped_non_positive": skipped_non_positive,
        "all_power_of_2": not not_po2,
        "not_power_of_2": not_po2,
    }


def apply_power_of_2_workflow(
    sim: Any,
    method: str = "cover_range",
    tolerance: float = 0.02,
    align_bias_scale: bool = True,
    verbose: bool = True,
    bias_bitwidth: int = 32,
) -> Dict[str, Any]:
    """ONNX 侧与 ``aimet_torch.utils_rx.apply_power_of_2_workflow`` 对齐的入口。"""
    report = align_sim_encodings_to_po2(
        sim,
        method=method,
        tolerance=tolerance,
        align_bias_scale=align_bias_scale,
        bias_bitwidth=bias_bitwidth,
    )
    if verbose:
        print(
            f"Po2 method={report['method']} tolerance={report['tolerance']}: "
            f"修改 {len(report['po2_changes'])} 个 quantizer"
        )
        bias = report["bias_align"]
        print(
            f"bias 对齐 Sb=Sx*Sw bitwidth={report['bias_bitwidth']}: "
            f"{bias['modified']}/{bias['total']}"
        )
        if bias["skipped"]:
            print(f"  skipped: {bias['skipped']}")
        if report.get("skipped_non_positive"):
            print(f"  关闭 scale<=0 的 quantizer: {report['skipped_non_positive']}")
        if report["all_power_of_2"]:
            print("所有 enabled integer scale 均为 Power-of-2")
        else:
            print(f"非 Po2 quantizer: {report['not_power_of_2']}")
    return report
