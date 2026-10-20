"""Apply the Torch JSON-2 mixed-precision file to an ONNX QuantSim.

``quick_start_full_quant.json`` 的 key 是 ``QuantizedConv2d`` 这类 Torch
模块名。算子映射由本模块适配，配置解析、匹配和冲突检查由公共模块负责。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from aimet_common.mixed_precision import (
    QuantizerConfigPlan, is_config_comment, load_mixed_precision_config, select_layer_config,
)

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
    "QuantizedMatMul": ("MatMul",),
    "QuantizedBatchNorm1d": ("BatchNormalization",),
    "QuantizedBatchNorm2d": ("BatchNormalization",),
    "QuantizedBatchNorm3d": ("BatchNormalization",),
    "QuantizedLayerNorm": ("LayerNormalization",),
    "QuantizedGroupNorm": ("GroupNormalization",),
    "QuantizedInstanceNorm1d": ("InstanceNormalization",),
    "QuantizedInstanceNorm2d": ("InstanceNormalization",),
    "QuantizedInstanceNorm3d": ("InstanceNormalization",),
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
    "QuantizedClamp": ("Clip",),
    "QuantizedHardtanh": ("Clip",),
    "QuantizedLog": ("Log",),
    "QuantizedSoftmax": ("Softmax",),
    "QuantizedLogSoftmax": ("LogSoftmax",),
    "QuantizedRelu": ("Relu",),
    "QuantizedReLU": ("Relu",),
    "QuantizedReLU6": ("Clip",),
    "QuantizedHardSwish": ("HardSwish",),
    "QuantizedHardswish": ("HardSwish",),
}

_INT_ONLY_TYPES = frozenset({"QuantizedPad", "QuantizedFloorDivide", "QuantizedUnfold"})
_UNMAPPED_TYPES = _INT_ONLY_TYPES | {"QuantizedVar", "QuantizedSnake2d"}

# JSON2 的 QuantizedPad disable 是 Torch MRNN 的整数 size Pad，
# 不是 ONNX 里的空间 ZeroPad。不映射，避免把 yolo 的 Pad I/O 整段关掉。


def _is_comment_key(key: str) -> bool:
    return is_config_comment(key)


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
        layer_config = {key: value for key, value in layer_config.items() if not is_config_comment(key)}
        onnx_types = _TORCH_TYPE_TO_ONNX.get(torch_name)
        if not onnx_types:
            if torch_name in _UNMAPPED_TYPES:
                ignored.append(torch_name)
                continue
            raise ValueError(f"Unknown or unsupported layer_type_config key: {torch_name}")
        hits = [op_type for op_type in onnx_types if op_type in present]
        if not hits:
            ignored.append(torch_name)
            continue
        for op_type in hits:
            if op_type == "MatMul" and "QuantizedMatMul" in layer_type_config:
                mapped[op_type] = {key: value for key, value in layer_type_config["QuantizedMatMul"].items()
                                   if not is_config_comment(key)}
                continue
            if op_type in mapped and mapped[op_type] != layer_config:
                raise ValueError(f"Ambiguous layer type configs for ONNX {op_type}")
            mapped[op_type] = layer_config
    return mapped, ignored


def _quantizers_for_op(sim: Any, op: Any) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    params = {
        role or name: sim.qc_quantize_op_dict.get(name)
        for name, (_product, role) in getattr(op, "parameters", {}).items()
    }
    parameter_names = set(getattr(op, "parameters", {}))
    getter = getattr(sim, "_get_weight_and_bias", None)
    if getter is not None:
        weight, bias = getter(op)
        if weight is not None:
            parameter_names.add(weight.name)
            params["weight"] = sim.qc_quantize_op_dict.get(weight.name)
        if bias is not None:
            parameter_names.add(bias.name)
            params["bias"] = sim.qc_quantize_op_dict.get(bias.name)
    # Keep empty activation slots so input_bitwidths retains its input indices.
    inputs = [
        sim.qc_quantize_op_dict.get(product.name)
        for product in op.inputs
        if not product.is_parm and product.name not in parameter_names
    ]
    outputs = [sim.qc_quantize_op_dict.get(product.name) for product in op.outputs]
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


def _apply_layer_config(sim: Any, op: Any, layer_config: Dict[str, Any],
                        plan: QuantizerConfigPlan, *, fallback: bool = False) -> str:
    inputs, outputs, params = _quantizers_for_op(sim, op)
    def request(quantizer, field, value, role):
        plan.request(quantizer, field, value, f"{op.name}.{role}", fallback=fallback)

    if layer_config.get("disable_quantization", False):
        for quantizer in inputs + outputs + [q for q in params.values() if q is not None]:
            request(quantizer, "enabled", False, "disable_quantization")
        return "disabled"

    if not fallback:
        for quantizer in inputs + outputs + list(params.values()):
            if quantizer is not None and quantizer.enabled:
                request(quantizer, "enabled", True, "quantization")
    for role, quantizer in params.items():
        prefix = role if role in {"weight", "bias"} else "param"
        request(quantizer, "bitwidth", layer_config.get(f"{prefix}_bitwidth"), role)
        request(quantizer, "symmetric", layer_config.get(f"{prefix}_symmetric"), role)
    request(params.get("weight"), "per_channel_quantization",
            layer_config.get("per_channel_quantization"), "weight")

    input_bws = layer_config.get("input_bitwidths")
    default_input_bw = layer_config.get("input_bitwidth")
    input_sym = layer_config.get("input_symmetric")
    for index, quantizer in enumerate(inputs):
        bitwidth = (
            input_bws[index]
            if input_bws is not None and index < len(input_bws)
            else None if input_bws is not None else default_input_bw
        )
        request(quantizer, "bitwidth", bitwidth, f"input[{index}]")
        request(quantizer, "symmetric", input_sym, f"input[{index}]")

    output_bw = layer_config.get("output_bitwidth")
    output_sym = layer_config.get("output_symmetric")
    for quantizer in outputs:
        request(quantizer, "bitwidth", output_bw, "output")
        request(quantizer, "symmetric", output_sym, "output")
    return "applied"


def _type_config_for_op(sim, op, type_configs):
    candidates = {name: {key: value for key, value in settings.items() if not is_config_comment(key)}
                  for name, settings in type_configs.items()
                  if op.type in _TORCH_TYPE_TO_ONNX.get(name, ())}
    if op.type in {"Conv", "ConvTranspose"} and len(op.inputs) > 1:
        rank = len(op.inputs[1].shape) - 2
        candidates = {name: value for name, value in candidates.items() if name.endswith(f"{rank}d")}
    if op.type == "BatchNormalization" and op.inputs[0].shape:
        rank = len(op.inputs[0].shape)
        dimensions = 2 if rank == 4 else 3 if rank == 5 else 1
        candidates = {name: value for name, value in candidates.items()
                      if name.endswith(f"{dimensions}d")}
    if op.type == "MatMul":
        if "QuantizedMatMul" in candidates:
            candidates = {"QuantizedMatMul": candidates["QuantizedMatMul"]}
        elif sim._get_weight_and_bias(op)[0] is None:
            candidates = {}
    configs = list(candidates.values())
    if configs and any(value != configs[0] for value in configs[1:]):
        raise ValueError(f"{op.name}: ambiguous JSON2 aliases for {op.type}: {list(candidates)}")
    return {op.type: configs[0]} if configs else {}


def _apply_plan(plan):
    entries = list(plan.entries())
    for quantizer, settings in entries:
        if settings.get("per_channel_quantization"):
            params = quantizer.tensor_quantizer_params
            if params is None or params.channel_axis is None:
                raise ValueError("Weight quantizer does not support per-channel quantization")
        if "per_channel_quantization" in settings:
            if (quantizer.quant_info.usePerChannelMode != settings["per_channel_quantization"]
                    and quantizer.is_initialized()):
                raise ValueError("Configure per_channel_quantization before calibration")
    for quantizer, settings in entries:
        _set_bitwidth(quantizer, settings.get("bitwidth"))
        _set_symmetric(quantizer, settings.get("symmetric"))
        if "per_channel_quantization" in settings:
            enabled = settings["per_channel_quantization"]
            if quantizer.quant_info.usePerChannelMode != enabled:
                quantizer.enable_per_channel_quantization(enabled)
        if "enabled" in settings:
            quantizer.enabled = settings["enabled"]


def apply_onnx_mixed_precision_bitwidth(
    sim: Any, config_file: str, verbose: bool = True
) -> Dict[str, Any]:
    """Reuse ``quick_start_full_quant.json`` on an ONNX QuantizationSimModel."""
    config = load_mixed_precision_config(config_file)
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
    unknown = set(layer_type_config) - _TORCH_TYPE_TO_ONNX.keys() - _UNMAPPED_TYPES
    if unknown:
        raise ValueError(f"Unknown or unsupported layer_type_config key(s): {sorted(unknown)}")
    ignored_types = sorted(name for name in layer_type_config
                           if not (set(_TORCH_TYPE_TO_ONNX.get(name, ())) & present_types))

    if verbose:
        print("\n应用混合精度位宽配置（ONNX，复用 Torch JSON 2）")
        print(f"  配置文件: {config_file}")
        if ignored_types:
            print(f"  当前图不适用的类型: {ignored_types}")
        unsupported_types = sorted(set(layer_type_config) & (_UNMAPPED_TYPES - _INT_ONLY_TYPES))
        if unsupported_types:
            print(f"  没有单节点 ONNX 对应的类型配置: {unsupported_types}")
        if config.get("LSTM_config"):
            print("  ONNX PTQ 不执行 QuantLSTM 内部校准，忽略 LSTM_config")
        if config.get("GRU_config"):
            print("  当前 ONNX 图没有 QuantGRU，忽略 GRU_config")

    stats: Dict[str, Any] = {
        "applied_ops": 0,
        "disabled_ops": 0,
        "default_ops": 0,
        "ignored_types": ignored_types,
        "unsupported_types": sorted(set(layer_type_config) & (_UNMAPPED_TYPES - _INT_ONLY_TYPES)),
        "name_matched": 0,
        "type_matched": 0,
    }

    plan = QuantizerConfigPlan()
    for op in ops:
        type_config = {} if op.name in layer_name_config else _type_config_for_op(
            sim, op, layer_type_config
        )
        layer_config, match, _ = select_layer_config(
            op.name, op.type, type_config, layer_name_config
        )
        if match in {"name", "pattern"}:
            stats["name_matched"] += 1
        elif match == "type":
            stats["type_matched"] += 1
        if layer_config is None:
            if default_disable:
                action = _apply_layer_config(
                    sim, op, {"disable_quantization": True}, plan, fallback=True
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

        action = _apply_layer_config(sim, op, layer_config, plan, fallback=match == "default")
        if action == "disabled":
            stats["disabled_ops"] += 1
        else:
            stats["applied_ops"] += 1
            if verbose and match in {"name", "pattern"}:
                print(f"  [{match}] {op.name} ({op.type})")

    _apply_plan(plan)
    if verbose:
        print(
            f"  应用 {stats['applied_ops']} 个 op，"
            f"禁用 {stats['disabled_ops']}，"
            f"默认 {stats['default_ops']}"
        )
    return stats
