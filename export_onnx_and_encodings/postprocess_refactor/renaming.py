from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import onnx

from .io_utils import load_encodings, save_encodings

logger = logging.getLogger(__name__)


def _rename_value_references(model: onnx.ModelProto, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return

    for node in model.graph.node:
        node.input[:] = [new_name if name == old_name else name for name in node.input]
        node.output[:] = [new_name if name == old_name else name for name in node.output]

    for value in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if value.name == old_name:
            value.name = new_name


def _strip_hash_suffix(name: str) -> str:
    if "#" in name:
        return name.split("#", 1)[0]
    return name


def _match_gru_name_by_json(base_name: str, json_keys: list[str]) -> str | None:
    if base_name in json_keys:
        return base_name
    prefix = base_name + "."
    candidates = [k for k in json_keys if k.startswith(prefix)]
    if candidates:
        candidates.sort(key=len)
        return candidates[0]

    # 单向 GRU 的特殊情况：ExportOptimizedQuantizableGRU 用 self.gru = nn.GRU(...)，
    # 因此 ONNX 节点名为 "...seq_t.gru"，而 encodings 键名为 "...seq_t.cells.0"。
    # 回退：当节点名以 ".gru" 结尾时，尝试匹配对应的 ".cells.0" 键。
    if base_name.endswith(".gru"):
        parent = base_name[: -len(".gru")]
        cells_key = parent + ".cells.0"
        if cells_key in json_keys:
            return cells_key
        cells_prefix = parent + ".cells."
        cells_candidates = [k for k in json_keys if k.startswith(cells_prefix)]
        if cells_candidates:
            cells_candidates.sort(key=len)
            return cells_candidates[0]

    return None


def _rename_initializer_and_refs(model: onnx.ModelProto, old_name: str, new_name: str) -> bool:
    if old_name == new_name:
        return False
    renamed = False
    for init in model.graph.initializer:
        if init.name == old_name:
            init.name = new_name
            renamed = True
            break
    for inp in model.graph.input:
        if inp.name == old_name:
            inp.name = new_name
            renamed = True
    if renamed:
        _rename_value_references(model, old_name, new_name)
    return renamed


def _rename_param_encoding(enc: Dict[str, Any], old: str, new: str) -> bool:
    if "param_encodings" not in enc or not isinstance(enc["param_encodings"], dict):
        return False
    pe = enc["param_encodings"]
    if old not in pe or old == new:
        return False
    pe[new] = pe.pop(old)
    return True


def rename_gru_initializers_and_change_gru_json_format(
    onnx_path: str,
    encodings_path: str,
    output_onnx_path: str | None = None,
    output_encodings_path: str | None = None,
    verbose: bool = True,
) -> Tuple[str, str]:
    enc_out = output_encodings_path or encodings_path
    enc = load_encodings(encodings_path)
    act_keys = list(enc.get("activation_encodings", {}).keys()) if isinstance(enc.get("activation_encodings"), dict) else []

    onnx_out = output_onnx_path or onnx_path.replace(".onnx", "_renamed.onnx")
    try:
        model = onnx.load(onnx_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法加载 ONNX: {exc}") from exc

    gru_renamed = False
    param_renamed = False
    for node in model.graph.node:
        if node.op_type == "GRU":
            base_name = _strip_hash_suffix(node.name)
            matched = _match_gru_name_by_json(base_name, act_keys)
            new_name = matched or base_name
            if new_name != node.name:
                if verbose:
                    logger.info("重命名 GRU 节点: %s -> %s", node.name, new_name)
                node.name = new_name
                gru_renamed = True

            inputs = list(node.input)
            if len(inputs) >= 3:
                weight_ih_old = inputs[1]
                weight_hh_old = inputs[2]
                weight_ih_new = f"{node.name}.weight_ih.weight"
                weight_hh_new = f"{node.name}.weight_hh.weight"
                if _rename_initializer_and_refs(model, weight_ih_old, weight_ih_new):
                    if verbose:
                        logger.info("重命名 GRU weight_ih: %s -> %s", weight_ih_old, weight_ih_new)
                    _rename_param_encoding(enc, weight_ih_old, weight_ih_new)
                    inputs[1] = weight_ih_new
                    param_renamed = True
                if _rename_initializer_and_refs(model, weight_hh_old, weight_hh_new):
                    if verbose:
                        logger.info("重命名 GRU weight_hh: %s -> %s", weight_hh_old, weight_hh_new)
                    _rename_param_encoding(enc, weight_hh_old, weight_hh_new)
                    inputs[2] = weight_hh_new
                    param_renamed = True
            if len(inputs) >= 4:
                bias_old = inputs[3]
                bias_new = f"{node.name}.bias"
                if _rename_initializer_and_refs(model, bias_old, bias_new):
                    if verbose:
                        logger.info("重命名 GRU bias: %s -> %s", bias_old, bias_new)
                    _rename_param_encoding(enc, bias_old, bias_new)
                    inputs[3] = bias_new
                    param_renamed = True
            node.input[:] = inputs

        if node.op_type == "Conv" and node.name:
            inputs = list(node.input)
            if len(inputs) >= 2:
                w_old = inputs[1]
                w_new = f"{node.name}.weight"
                if _rename_initializer_and_refs(model, w_old, w_new):
                    if verbose:
                        logger.info("重命名 Conv weight: %s -> %s", w_old, w_new)
                    _rename_param_encoding(enc, w_old, w_new)
                    inputs[1] = w_new
                    param_renamed = True
            if len(inputs) >= 3:
                b_old = inputs[2]
                b_new = f"{node.name}.bias"
                if _rename_initializer_and_refs(model, b_old, b_new):
                    if verbose:
                        logger.info("重命名 Conv bias: %s -> %s", b_old, b_new)
                    _rename_param_encoding(enc, b_old, b_new)
                    inputs[2] = b_new
                    param_renamed = True
            node.input[:] = inputs

    if gru_renamed or param_renamed or onnx_out != onnx_path:
        onnx.save(model, onnx_out)
    else:
        onnx_out = onnx_path

    if param_renamed or enc_out != encodings_path:
        save_encodings(enc, enc_out)
    else:
        enc_out = encodings_path

    if verbose:
        logger.info("ONNX 输出: %s", onnx_out)
        logger.info("Encodings 输出: %s", enc_out)

    return onnx_out, enc_out
