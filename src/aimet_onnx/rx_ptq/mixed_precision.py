"""Apply the Torch JSON-2 mixed-precision file to an ONNX QuantSim.

``quick_start_full_quant.json`` 的 key 是 ``QuantizedConv2d`` 这类 Torch
模块名。算子映射由本模块适配，配置解析、匹配和冲突检查由公共模块负责。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from aimet_common.mixed_precision import (
    QuantizerConfigPlan, default_layer_config, is_config_comment,
    load_mixed_precision_config, select_layer_config,
)
from .input_quantization import input_quantizer
from aimet_common.run_context import log_details, log_print as print, logged_operation

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


def _quantizers_for_op(sim: Any, op: Any, *, matmul_operands: bool = False
                       ) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
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
    # ConnectedGraph omits optional empty ports; preserve the original ONNX indices.
    roles = getattr(sim, "_input_roles", None)
    names = roles.input_names(op.name) if roles is not None else None
    if names is None:
        names = [product.name for product in op.inputs]
    products = {product.name: product for product in op.inputs}
    inputs = [
        input_quantizer(sim, op, index) if name and (matmul_operands or (
            not products[name].is_parm and name not in parameter_names)) else None
        for index, name in enumerate(names)
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
                        plan: QuantizerConfigPlan, *, fallback: bool = False,
                        matmul_operands: bool = False) -> str:
    inputs, outputs, params = _quantizers_for_op(sim, op, matmul_operands=matmul_operands)
    weight = params.get("weight")
    # MatMul's constant matrix is still an operand, despite ONNX storing it as
    # an initializer. Explicit weight fields opt back into parameter semantics.
    weight_is_operand = (matmul_operands and weight is not None
                         and any(q is weight for q in inputs)
                         and not any(layer_config.get(field) is not None
                                     for field in ("weight_bitwidth", "weight_symmetric", "weight_signed")))
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
    request(weight, "calibration_passthrough", weight_is_operand, "weight_calibration")
    for role, quantizer in params.items():
        prefix = role if role in {"weight", "bias"} else "param"
        request(quantizer, "bitwidth", layer_config.get(f"{prefix}_bitwidth"), role)
        request(quantizer, "symmetric", layer_config.get(f"{prefix}_symmetric"), role)
        request(quantizer, "signed", layer_config.get(f"{prefix}_signed"), role)
    granularity = layer_config.get("per_channel_quantization")
    if weight_is_operand and granularity is None:
        granularity = False
    request(weight, "per_channel_quantization", granularity, "weight")

    input_bws = layer_config.get("input_bitwidths")
    default_input_bw = layer_config.get("input_bitwidth")
    input_sym = layer_config.get("input_symmetric")
    for index, quantizer in enumerate(inputs):
        if quantizer is weight and not weight_is_operand:
            continue
        bitwidth = (
            input_bws[index]
            if input_bws is not None and index < len(input_bws)
            else None if input_bws is not None else default_input_bw
        )
        request(quantizer, "bitwidth", bitwidth, f"input[{index}]")
        request(quantizer, "symmetric", input_sym, f"input[{index}]")
        signs = layer_config.get('input_signeds')
        signed = (signs[index] if index < len(signs) else None) if signs is not None else layer_config.get('input_signed')
        request(quantizer, 'signed', signed, f'input[{index}]')

    output_bw = layer_config.get("output_bitwidth")
    output_sym = layer_config.get("output_symmetric")
    for quantizer in outputs:
        request(quantizer, "bitwidth", output_bw, "output")
        request(quantizer, "symmetric", output_sym, "output")
        request(quantizer, "signed", layer_config.get('output_signed'), "output")
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
        if ("calibration_passthrough" in settings and quantizer.is_initialized()
                and bool(getattr(quantizer, "_rx_calibration_passthrough", False))
                != settings["calibration_passthrough"]):
            raise ValueError("Configure operand calibration roles before calibration")
    for quantizer, settings in entries:
        _set_bitwidth(quantizer, settings.get("bitwidth"))
        if 'signed' in settings:
            quantizer.signed = settings['signed']
        _set_symmetric(quantizer, settings.get("symmetric"))
        if "per_channel_quantization" in settings:
            enabled = settings["per_channel_quantization"]
            if quantizer.quant_info.usePerChannelMode != enabled:
                quantizer.enable_per_channel_quantization(enabled)
        if "enabled" in settings:
            quantizer.enabled = settings["enabled"]
        if "calibration_passthrough" in settings:
            quantizer._rx_calibration_passthrough = settings["calibration_passthrough"]


@logged_operation
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

    with log_details(verbose):
        print("\n应用混合精度位宽配置（ONNX，复用 Torch JSON 2）")
        print(f"  配置文件: {config_file}")
        if ignored_types:
            print(f"  当前图不适用的类型: {ignored_types}")
        unsupported_types = sorted(set(layer_type_config) & (_UNMAPPED_TYPES - _INT_ONLY_TYPES))
        if unsupported_types:
            print(f"  没有单节点 ONNX 对应的类型配置: {unsupported_types}")
        for kind in ("GRU", "LSTM"):
            if config.get(f"{kind}_config") and kind not in present_types:
                print(f"  当前 ONNX 图没有 {kind}，忽略 {kind}_config")

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
    for name, quantizer in sim.qc_quantize_op_dict.items():
        plan.request(quantizer, 'signed', config.get('default_config', {}).get('signed'),
                     f'default_config.{name}', fallback=True)
    for op in ops:
        if op.name in getattr(sim, "_rx_native_recurrent", {}):
            continue
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
            layer_config = default_layer_config(default_bitwidth)
            match = "default"
            stats["default_ops"] += 1

        matmul_operands = (op.type == "MatMul" and (
            "QuantizedMatMul" in layer_type_config or "QuantizedLinear" not in layer_type_config))
        action = _apply_layer_config(sim, op, layer_config, plan, fallback=match == "default",
                                     matmul_operands=matmul_operands)
        if action == "disabled":
            stats["disabled_ops"] += 1
        else:
            stats["applied_ops"] += 1
            if match in {'name', 'pattern'}:
                with log_details(verbose):
                    print(f"  [{match}] {op.name} ({op.type})")

    _apply_plan(plan)
    for adapter in getattr(sim, "_rx_native_recurrent", {}).values():
        adapter.configure(config_file)
    with log_details(verbose):
        print(
            f"  应用 {stats['applied_ops']} 个 op，"
            f"禁用 {stats['disabled_ops']}，"
            f"默认 {stats['default_ops']}"
        )
    return stats
