"""Apply the Torch JSON-2 mixed-precision file to an ONNX QuantSim.

``quick_start_full_quant.json`` 的 key 是 ``QuantizedConv2d`` 这类 Torch
模块名。ONNX 图里没有这些模块：能映射到的算子就套用，对不上的类型和
``GRU_config`` 直接忽略，不报错。
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# JSON 2 的 Torch 类名 → 当前图里可能出现的 ONNX op_type。
# 没有对应节点的项在 apply 时跳过。
_TORCH_TYPE_TO_ONNX = {
    "QuantizedConv1d": ("Conv",),
    "QuantizedConv2d": ("Conv",),
    "QuantizedConv3d": ("Conv",),
    "QuantizedConvTranspose1d": ("ConvTranspose",),
    "QuantizedConvTranspose2d": ("ConvTranspose",),
    "QuantizedConvTranspose3d": ("ConvTranspose",),
    "QuantizedLinear": ("Gemm", "MatMul"),
    "QuantizedTanh": ("Tanh",),
    "QuantizedSigmoid": ("Sigmoid",),
    "QuantizedAdd": ("Add",),
    "QuantizedMultiply": ("Mul",),
    "QuantizedSubtract": ("Sub",),
    "QuantizedDivide": ("Div",),
    "QuantizedPow": ("Pow",),
    "QuantizedSqrt": ("Sqrt",),
    "QuantizedMean": ("ReduceMean", "Mean"),
    "QuantizedConcat": ("Concat",),
    "QuantizedAbs": ("Abs",),
    "QuantizedSign": ("Sign",),
    "QuantizedClip": ("Clip",),
    "QuantizedRelu": ("Relu",),
    "QuantizedHardSwish": ("HardSwish",),
}

# JSON2 的 QuantizedPad disable 是 Torch MRNN 的整数 size Pad，
# 不是 ONNX 里的空间 ZeroPad。不映射，避免把 yolo 的 Pad I/O 整段关掉。


def _load_json(config_file: str) -> Dict[str, Any]:
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_comment_key(key: str) -> bool:
    return key.startswith("_") or key in {"comment", "examples", "description"}


def map_json2_types_to_onnx(
    layer_type_config: Dict[str, Any], present_onnx_types: Iterable[str]
) -> Tuple[Dict[str, Any], List[str]]:
    """Return ONNX-op configs plus JSON2 keys that this graph cannot use."""
    present = set(present_onnx_types)
    mapped: Dict[str, Any] = {}
    ignored: List[str] = []
    for torch_name, layer_config in layer_type_config.items():
        if _is_comment_key(torch_name):
            continue
        onnx_types = _TORCH_TYPE_TO_ONNX.get(torch_name)
        if not onnx_types:
            ignored.append(torch_name)
            continue
        hits = [op_type for op_type in onnx_types if op_type in present]
        if not hits:
            ignored.append(torch_name)
            continue
        for op_type in hits:
            mapped[op_type] = layer_config
    return mapped, ignored


def _quantizers_for_op(sim: Any, op: Any) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    inputs = []
    for product in getattr(op, "inputs", []) or []:
        quantizer = sim.qc_quantize_op_dict.get(getattr(product, "name", None))
        if quantizer is not None:
            inputs.append(quantizer)
    outputs = []
    for product in getattr(op, "outputs", []) or []:
        quantizer = sim.qc_quantize_op_dict.get(getattr(product, "name", None))
        if quantizer is not None:
            outputs.append(quantizer)
    params: Dict[str, Any] = {}
    getter = getattr(sim, "_get_weight_and_bias", None)
    if getter is not None:
        weight, bias = getter(op)
        if weight is not None:
            params["weight"] = sim.qc_quantize_op_dict.get(weight.name)
        if bias is not None:
            params["bias"] = sim.qc_quantize_op_dict.get(bias.name)
    return inputs, outputs, params


def _set_bitwidth(quantizer: Any, bitwidth: Optional[int]) -> None:
    if quantizer is None or bitwidth is None:
        return
    setter = getattr(quantizer, "set_bitwidth", None)
    if setter is not None:
        setter(int(bitwidth))
    else:
        quantizer.bitwidth = int(bitwidth)


def _set_symmetric(quantizer: Any, symmetric: Optional[bool]) -> None:
    if quantizer is None or symmetric is None:
        return
    if hasattr(quantizer, "use_symmetric_encodings"):
        quantizer.use_symmetric_encodings = bool(symmetric)
    elif hasattr(quantizer, "symmetric"):
        quantizer.symmetric = bool(symmetric)


def _disable(quantizer: Any) -> None:
    if quantizer is not None:
        quantizer.enabled = False


def _apply_layer_config(sim: Any, op: Any, layer_config: Dict[str, Any]) -> str:
    inputs, outputs, params = _quantizers_for_op(sim, op)
    if layer_config.get("disable_quantization", False):
        for quantizer in inputs + outputs + [q for q in params.values() if q is not None]:
            _disable(quantizer)
        return "disabled"

    _set_bitwidth(params.get("weight"), layer_config.get("weight_bitwidth"))
    _set_symmetric(params.get("weight"), layer_config.get("weight_symmetric"))
    _set_bitwidth(params.get("bias"), layer_config.get("bias_bitwidth"))
    _set_symmetric(params.get("bias"), layer_config.get("bias_symmetric"))

    input_bws = layer_config.get("input_bitwidths")
    default_input_bw = layer_config.get("input_bitwidth")
    input_sym = layer_config.get("input_symmetric")
    for index, quantizer in enumerate(inputs):
        bitwidth = (
            input_bws[index]
            if input_bws is not None and index < len(input_bws)
            else default_input_bw
        )
        _set_bitwidth(quantizer, bitwidth)
        _set_symmetric(quantizer, input_sym)

    output_bw = layer_config.get("output_bitwidth")
    output_sym = layer_config.get("output_symmetric")
    for quantizer in outputs:
        _set_bitwidth(quantizer, output_bw)
        _set_symmetric(quantizer, output_sym)
    return "applied"


def apply_onnx_mixed_precision_bitwidth(
    sim: Any, config_file: str, verbose: bool = True
) -> Dict[str, Any]:
    """Reuse ``quick_start_full_quant.json`` on an ONNX QuantizationSimModel."""
    config = _load_json(config_file)
    default_bitwidth = config.get("default_bitwidth", {})
    default_disable = bool(
        config.get("default_config", {}).get("disable_quantization", False)
    )
    layer_type_config = {
        key: value
        for key, value in config.get("layer_type_config", {}).items()
        if not _is_comment_key(key)
    }
    layer_name_config = {
        key: value
        for key, value in config.get("layer_name_config", {}).items()
        if not _is_comment_key(key)
    }

    ops = list(sim.connected_graph.get_all_ops().values())
    present_types = {op.type for op in ops}
    mapped, ignored_types = map_json2_types_to_onnx(layer_type_config, present_types)

    if verbose:
        print("\n应用混合精度位宽配置（ONNX，复用 Torch JSON 2）")
        print(f"  配置文件: {config_file}")
        if ignored_types:
            print(f"  忽略当前图没有的类型: {ignored_types}")
        if config.get("GRU_config"):
            print("  当前 ONNX 图没有 QuantGRU，忽略 GRU_config")

    stats: Dict[str, Any] = {
        "applied_ops": 0,
        "disabled_ops": 0,
        "default_ops": 0,
        "ignored_types": ignored_types,
        "name_matched": 0,
        "type_matched": 0,
    }

    for op in ops:
        layer_config = None
        match = None
        if op.name in layer_name_config:
            layer_config = layer_name_config[op.name]
            match = "name"
            stats["name_matched"] += 1
        if layer_config is None and op.type in mapped:
            layer_config = mapped[op.type]
            match = "type"
            stats["type_matched"] += 1
        if layer_config is None:
            for pattern, candidate in layer_name_config.items():
                if "*" in pattern and fnmatch.fnmatch(op.name, pattern):
                    layer_config = candidate
                    match = "pattern"
                    stats["name_matched"] += 1
                    break
        if layer_config is None:
            if default_disable:
                action = _apply_layer_config(
                    sim, op, {"disable_quantization": True}
                )
                if action == "disabled":
                    stats["disabled_ops"] += 1
                continue
            layer_config = {
                "weight_bitwidth": default_bitwidth.get("weight"),
                "output_bitwidth": default_bitwidth.get("output"),
                "input_bitwidth": default_bitwidth.get("output"),
            }
            match = "default"
            stats["default_ops"] += 1

        action = _apply_layer_config(sim, op, layer_config)
        if action == "disabled":
            stats["disabled_ops"] += 1
        else:
            stats["applied_ops"] += 1
            if verbose and match in {"name", "pattern"}:
                print(f"  [{match}] {op.name} ({op.type})")

    if verbose:
        print(
            f"  应用 {stats['applied_ops']} 个 op，"
            f"禁用 {stats['disabled_ops']}，"
            f"默认 {stats['default_ops']}"
        )
    return stats
