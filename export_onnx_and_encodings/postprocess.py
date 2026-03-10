"""
ONNX 导出后的图清理与重命名工具。
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Any, List, Optional

import math
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
from onnxsim import simplify


logger = logging.getLogger(__name__)

# ======================================================================================
# ONNX 图后处理（mandatory）
# - onnxsim 简化
# - shape inference
# - 折叠 Reshape 的动态 shape 子图 -> static initializer
# - 折叠常量 If 分支（compute_encodings 后更容易出现）
# ======================================================================================


def _get_static_shape_tuple(name: str, graph) -> Optional[Tuple[int, ...]]:
    """读取 value_info/input/output 中的静态 shape（若有动态 dim 则返回 None）。"""

    def _scan(vis):
        for vi in vis:
            if vi.name != name:
                continue
            if not vi.type.HasField("tensor_type"):
                return None
            tt = vi.type.tensor_type
            if not tt.HasField("shape"):
                return None
            dims = []
            for d in tt.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                else:
                    return None
            return tuple(dims)
        return None

    return _scan(graph.value_info) or _scan(graph.output) or _scan(graph.input)


def _fold_static_reshape_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """把 Reshape 的第二输入（shape）子图尽可能折叠为静态 initializer。"""
    g = model.graph
    produced_by = {o: n for n in g.node for o in n.output}
    init_map = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    memo: Dict[str, Any] = {}

    def _const_from_node(node):
        for a in node.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
        return None

    def _tensor_shape(name: str) -> Optional[Tuple[int, ...]]:
        if name in init_map:
            return tuple(init_map[name].shape)
        return _get_static_shape_tuple(name, g)

    def _eval(name):
        if isinstance(name, int):
            return name
        if name in memo:
            return memo[name]
        if name in init_map:
            arr = init_map[name]
            memo[name] = int(arr) if arr.ndim == 0 else [int(x) for x in arr.flatten().tolist()]
            return memo[name]
        node = produced_by.get(name)
        if node is None:
            memo[name] = None
            return None
        op = node.op_type
        if op == "Shape":
            shp = _tensor_shape(node.input[0])
            memo[name] = list(shp) if shp is not None else None
        elif op == "Gather":
            data = _eval(node.input[0])
            idx = _eval(node.input[1])
            memo[name] = int(data[idx]) if isinstance(data, list) and isinstance(idx, int) else None
        elif op == "Unsqueeze":
            memo[name] = _eval(node.input[0])
        elif op == "Concat":
            out: List[int] = []
            for inp in node.input:
                v = _eval(inp)
                if isinstance(v, int):
                    out.append(v)
                elif isinstance(v, list):
                    out.extend(v)
                else:
                    memo[name] = None
                    return None
            memo[name] = out
        elif op == "Mul":
            a = _eval(node.input[0])
            b = _eval(node.input[1])
            memo[name] = int(a * b) if isinstance(a, int) and isinstance(b, int) else None
        elif op == "Constant":
            arr = _const_from_node(node)
            memo[name] = None if arr is None else (
                int(arr) if arr.ndim == 0 else [int(x) for x in arr.flatten().tolist()]
            )
        else:
            memo[name] = None
        return memo[name]

    existing_init = {init.name for init in g.initializer}
    to_add = []
    for node in g.node:
        if node.op_type != "Reshape" or len(node.input) < 2 or not node.output:
            continue
        shape_in = node.input[1]
        if shape_in in existing_init:
            continue
        shape_val = _eval(shape_in)
        if not isinstance(shape_val, list) or not shape_val:
            continue
        const_name = f"{node.output[0]}_shape_const"
        if const_name not in existing_init:
            to_add.append(numpy_helper.from_array(np.asarray(shape_val, dtype=np.int64), name=const_name))
            existing_init.add(const_name)
        node.input[1] = const_name
    if to_add:
        g.initializer.extend(to_add)
    return model


def _fold_constant_if_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """折叠 If(cond) 且 cond 可被静态求值为 True/False 的分支。"""
    g = model.graph
    produced_by = {o: n for n in g.node for o in n.output}
    init_map = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    memo: Dict[str, Any] = {}

    def _const_from_node(node):
        for a in node.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
        return None

    def _tensor_shape(name: str) -> Optional[Tuple[int, ...]]:
        if name in init_map:
            return tuple(init_map[name].shape)
        return _get_static_shape_tuple(name, g)

    def _eval(name):
        if isinstance(name, bool):
            return bool(name)
        if isinstance(name, int):
            return int(name)
        if name in memo:
            return memo[name]
        if name in init_map:
            arr = init_map[name]
            memo[name] = int(arr) if arr.ndim == 0 else [int(x) for x in arr.flatten().tolist()]
            return memo[name]
        node = produced_by.get(name)
        if node is None:
            memo[name] = None
            return None
        op = node.op_type
        if op == "Constant":
            arr = _const_from_node(node)
            memo[name] = None if arr is None else (
                int(arr) if arr.ndim == 0 else [int(x) for x in arr.flatten().tolist()]
            )
        elif op == "Shape":
            shp = _tensor_shape(node.input[0])
            memo[name] = list(shp) if shp is not None else None
        elif op == "Gather":
            data = _eval(node.input[0])
            idx = _eval(node.input[1])
            memo[name] = int(data[idx]) if isinstance(data, list) and isinstance(idx, int) else None
        elif op == "Equal":
            a = _eval(node.input[0])
            b = _eval(node.input[1])
            memo[name] = bool(a == b) if isinstance(a, int) and isinstance(b, int) else None
        elif op == "Cast":
            v = _eval(node.input[0])
            memo[name] = v if isinstance(v, (bool, int, list)) else None
        elif op == "Identity":
            memo[name] = _eval(node.input[0])
        else:
            memo[name] = None
        return memo[name]

    def _get_attr_graph(node, name: str):
        for a in node.attribute:
            if a.name == name:
                return a.g
        return None

    new_nodes = []
    removed = set()
    for i, node in enumerate(list(g.node)):
        if node.op_type != "If" or not node.input:
            continue
        cond = _eval(node.input[0])
        if not isinstance(cond, bool):
            continue
        chosen = _get_attr_graph(node, "then_branch" if cond else "else_branch")
        if chosen is None:
            continue
        prefix = (node.name or f"If_{i}") + ("/then" if cond else "/else")
        rename: Dict[str, str] = {}
        for bn in chosen.node:
            for o in bn.output:
                rename[o] = f"{prefix}/{o}"
        for o in chosen.output:
            rename[o.name] = f"{prefix}/{o.name}"
        import copy as _copy

        for bn in chosen.node:
            nn_ = _copy.deepcopy(bn)
            nn_.output[:] = [rename.get(o, o) for o in bn.output]
            nn_.input[:] = [rename.get(i_, i_) for i_ in bn.input]
            new_nodes.append(nn_)
        for j, out_name in enumerate(node.output):
            if j >= len(chosen.output):
                break
            br_old = chosen.output[j].name
            br_new = rename.get(br_old, br_old)
            new_nodes.append(
                onnx.helper.make_node("Identity", [br_new], [out_name], name=f"{prefix}/Identity_{j}")
            )
        removed.add(id(node))

    if not removed:
        return model

    kept = [n for n in g.node if id(n) not in removed]
    g.ClearField("node")
    g.node.extend(kept + new_nodes)
    return model


def postprocess_onnx_graph_inplace(onnx_path: str, input_shape: Tuple[int, int, int]) -> None:
    """
    对导出的 ONNX 执行强制图清理（inplace 写回原文件）。
    """
    m = onnx.load(onnx_path)
    m, ok = simplify(m, overwrite_input_shapes={"input": list(input_shape)})
    if not ok:
        raise RuntimeError("onnx-simplifier 校验失败")
    m = shape_inference.infer_shapes(m, data_prop=True)
    m = _fold_static_reshape_shapes(m)
    m = _fold_constant_if_nodes(m)

    # 二次清理
    m, ok = simplify(m, overwrite_input_shapes={"input": list(input_shape)})
    if not ok:
        raise RuntimeError("onnx-simplifier（二次）校验失败")
    m = shape_inference.infer_shapes(m, data_prop=True)

    # 三次 If 折叠收敛（compute_encodings 后更容易出现）
    m = _fold_constant_if_nodes(m)
    m, ok = simplify(m, overwrite_input_shapes={"input": list(input_shape)})
    if not ok:
        raise RuntimeError("onnx-simplifier（三次）校验失败")
    m = shape_inference.infer_shapes(m, data_prop=True)
    onnx.save(m, onnx_path)



def _get_shape_from_value_info(value_info) -> list:
    """从 ONNX ValueInfoProto 中提取 shape 信息（包含动态维度）。"""
    dims = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    return dims


def fix_slice_end_constants(onnx_path: str, verbose: bool = True) -> bool:
    """
    遍历 ONNX 中的 Slice 节点，将 `INT64_MAX` 形式的 ends 改写为真实维度。
    """
    try:
        model = onnx.load(onnx_path)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            logger.warning("无法加载ONNX进行Slice修正: %s", exc)
        return False

    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        inferred = None
        if verbose:
            logger.warning("Slice修正时形状推断失败: %s", exc)

    value_info_map: Dict[str, onnx.ValueInfoProto] = {}
    if inferred is not None:
        for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
            value_info_map[vi.name] = vi

    init_map = {init.name: init for init in model.graph.initializer}
    int64_max = np.iinfo(np.int64).max
    changed = False

    for node in model.graph.node:
        if node.op_type != "Slice" or len(node.input) < 3:
            continue

        data_name = node.input[0]
        starts_name = node.input[1]
        ends_name = node.input[2]
        axes_name = node.input[3] if len(node.input) >= 4 else None
        steps_name = node.input[4] if len(node.input) >= 5 else None

        if ends_name not in init_map:
            continue

        ends_arr = numpy_helper.to_array(init_map[ends_name]).astype(np.int64)
        if axes_name and axes_name in init_map:
            axes_arr = numpy_helper.to_array(init_map[axes_name]).astype(np.int64)
        else:
            axes_arr = np.arange(len(ends_arr), dtype=np.int64)

        starts_arr = numpy_helper.to_array(init_map[starts_name]).astype(np.int64) if starts_name in init_map else None

        if steps_name and steps_name in init_map:
            steps_arr = numpy_helper.to_array(init_map[steps_name]).astype(np.int64)
        else:
            steps_arr = None

        vi = value_info_map.get(data_name)
        if vi is None:
            continue
        data_shape = _get_shape_from_value_info(vi)

        updated = False
        for idx, end_val in enumerate(ends_arr):
            if end_val != int64_max or idx >= len(axes_arr):
                continue

            axis = int(axes_arr[idx])
            if axis < 0 or axis >= len(data_shape):
                continue

            dim = data_shape[axis]
            if dim is None:
                continue

            start_val = 0
            if starts_arr is not None and idx < len(starts_arr):
                start_val = int(starts_arr[idx])
                if start_val < 0:
                    start_val = dim + start_val

            step_val = 1
            if steps_arr is not None and idx < len(steps_arr):
                step_val = int(steps_arr[idx])
                if step_val <= 0:
                    continue

            new_end = dim if step_val > 0 else -1
            if new_end <= start_val:
                new_end = start_val

            ends_arr[idx] = new_end
            updated = True

        if updated:
            new_tensor = numpy_helper.from_array(ends_arr.astype(np.int64), ends_name)
            init_index = next(i for i, init in enumerate(model.graph.initializer) if init.name == ends_name)
            model.graph.initializer[init_index].CopyFrom(new_tensor)
            init_map[ends_name] = model.graph.initializer[init_index]
            changed = True
            if verbose:
                logger.info("✅ 调整Slice %s 的ends: %s", node.name, ends_arr.tolist())

    if changed:
        onnx.save(model, onnx_path)

    return changed


def _rename_value_references(model: onnx.ModelProto, old_name: str, new_name: str) -> None:
    """在整张图里把旧张量名切换为新名字，保持引用一致性。"""
    if old_name == new_name:
        return

    for node in model.graph.node:
        node.input[:] = [new_name if name == old_name else name for name in node.input]
        node.output[:] = [new_name if name == old_name else name for name in node.output]

    for value in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if value.name == old_name:
            value.name = new_name


def _strip_hash_suffix(name: str) -> str:
    """去掉 FX/ONNX 节点名中的 #数字 后缀。"""
    if "#" in name:
        return name.split("#", 1)[0]
    return name


def _match_gru_name_by_json(base_name: str, json_keys: list[str]) -> str | None:
    """
    根据 JSON 中的键名寻找与 GRU 节点最匹配的名称。
    规则：
      1) 精确匹配优先（json_keys 中已有 base_name）。
      2) 其次选择以 base_name + "." 开头的键（比如 json 有 enc_seqs.0.seq_t.cells.0，
         而 ONNX 节点名是 enc_seqs.0.seq_t）。若有多个候选，取最短的键（更可能是直接上一级）。
      3) 找不到则返回 None。
    """
    if base_name in json_keys:
        return base_name
    prefix = base_name + "."
    candidates = [k for k in json_keys if k.startswith(prefix)]
    if not candidates:
        return None
    # 取最短候选，兼顾通用性（可能出现 .cells.0 或其他后缀）
    candidates.sort(key=len)
    return candidates[0]


def _rename_initializer_and_refs(model: onnx.ModelProto, old_name: str, new_name: str) -> bool:
    """
    重命名 initializer 以及所有引用它的节点输入/输出/value_info。
    返回是否发生修改。
    """
    if old_name == new_name:
        return False
    # 更新 initializer
    renamed = False
    for init in model.graph.initializer:
        if init.name == old_name:
            init.name = new_name
            renamed = True
            break
    # 也可能在 graph.input 里（少见但防御）
    for inp in model.graph.input:
        if inp.name == old_name:
            inp.name = new_name
            renamed = True
    if renamed:
        _rename_value_references(model, old_name, new_name)
    return renamed


def _rename_param_encoding(enc: Dict[str, Any], old: str, new: str) -> bool:
    """
    若 param_encodings 中存在 old，则重命名为 new。
    返回是否发生修改。
    """
    if "param_encodings" not in enc or not isinstance(enc["param_encodings"], dict):
        return False
    pe = enc["param_encodings"]
    if old not in pe or old == new:
        return False
    pe[new] = pe.pop(old)
    return True


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def merge_gru_bias_param_encodings(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    """
    将 GRU 的两组 bias 编码合并为一组，以匹配 ONNX GRU 的单一 B 输入。

    encodings 中常见键名：
      - <gru>.weight_ih.bias   (长度 3H)
      - <gru>.weight_hh.bias   (长度 3H)
    合并后输出：
      - <gru>.bias             (长度 6H = [ih, hh] 拼接)

    注：这里的 <gru> 通常是形如 `enc_seqs.0.seq_t.cells.0` 这一层级（与你的 ONNX GRU 节点命名一致）。
    """
    if "param_encodings" not in enc or not isinstance(enc["param_encodings"], dict):
        return enc, False

    pe: Dict[str, Any] = enc["param_encodings"]
    changed = False

    # 先收集所有前缀（避免遍历时修改 dict）
    prefixes: set[str] = set()
    for k in pe.keys():
        if k.endswith(".weight_ih.bias"):
            prefixes.add(k[: -len(".weight_ih.bias")])
        elif k.endswith(".weight_hh.bias"):
            prefixes.add(k[: -len(".weight_hh.bias")])

    for prefix in sorted(prefixes):
        ih_key = prefix + ".weight_ih.bias"
        hh_key = prefix + ".weight_hh.bias"
        out_key = prefix + ".bias"

        ih_val = pe.get(ih_key)
        hh_val = pe.get(hh_key)
        if ih_val is None and hh_val is None:
            continue

        merged = _as_list(ih_val) + _as_list(hh_val)
        if not merged:
            continue

        if out_key in pe and pe[out_key] != merged:
            if verbose:
                logger.warning("GRU bias encodings 冲突：%s 已存在，将覆盖为合并后的结果", out_key)
        pe[out_key] = merged

        if ih_key in pe:
            del pe[ih_key]
            changed = True
        if hh_key in pe:
            del pe[hh_key]
            changed = True
        changed = True

    enc["param_encodings"] = pe
    return enc, changed


def compact_per_channel_param_encodings(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    """
    将 per-channel 的 param_encodings 从 list[dict] 压缩为单个 dict：

    现状（AIMET 常见输出）：
      "xxx.weight": [
        {"bitwidth":8,"dtype":"int","is_symmetric":"True","min":...,"max":...,"offset":...,"scale":...},
        {"bitwidth":8,"dtype":"int","is_symmetric":"True","min":...,"max":...,"offset":...,"scale":...},
        ...
      ]

    目标（你希望的格式）：
      "xxx.weight": {
        "bitwidth": 8,
        "dtype": "int",
        "is_symmetric": "True",
        "min":   [...],
        "max":   [...],
        "offset":[...],
        "scale": [...]
      }

    规则（按你的需求）：
    - 只要原始是 per-channel 的 list[dict]，则 `min/max/offset/scale` **必须**输出为 list（即使每个 channel 数值相同，也保留长度）
    - `bitwidth/dtype/is_symmetric` 等“元数据字段”保留为标量即可（通常每个 channel 都一致）
    """
    if "param_encodings" not in enc or not isinstance(enc["param_encodings"], dict):
        return enc, False

    pe: Dict[str, Any] = enc["param_encodings"]
    changed = False

    # 这些字段即使所有 channel 都相同，也要保留为 list（用于表达 per-channel 维度长度）
    ALWAYS_LIST_FIELDS = {"min", "max", "offset", "scale"}

    for k, v in list(pe.items()):
        if not (isinstance(v, list) and v and all(isinstance(e, dict) for e in v)):
            continue

        entries: List[Dict[str, Any]] = v  # type: ignore[assignment]
        keys = set(entries[0].keys())
        for e in entries[1:]:
            keys &= set(e.keys())
        if not keys:
            continue

        common: Dict[str, Any] = {}
        per_ch: Dict[str, List[Any]] = {}
        for kk in sorted(keys):
            vals = [e.get(kk) for e in entries]
            # 强制 per-channel 列表字段
            if kk in ALWAYS_LIST_FIELDS:
                per_ch[kk] = vals
                continue
            # 其余字段：若全相同则压为标量，否则保留 list
            all_same = all(val == vals[0] for val in vals)
            if all_same:
                common[kk] = vals[0]
            else:
                per_ch[kk] = vals

        # 也把“非交集”的键当做 per-channel（更防御）
        all_keys = set().union(*(set(e.keys()) for e in entries))
        extra_keys = all_keys - keys
        for kk in sorted(extra_keys):
            per_ch[kk] = [e.get(kk) for e in entries]

        compacted: Dict[str, Any] = {}
        compacted.update(common)
        compacted.update(per_ch)

        pe[k] = compacted
        changed = True

    if changed:
        if verbose:
            logger.info("✅ 已将 param_encodings 中的 per-channel list[dict] 压缩为 dict 格式")
        enc["param_encodings"] = pe
    return enc, changed


def merge_gru_activation_encodings(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    """
    将 activation_encodings 下 GRU 拆解小算子合并为大 GRU 节点，形如：
    enc_seqs.0.seq_t.cells.0.mul_old_contribution -> enc_seqs.0.seq_t.cells.0.internal_ops.mul_old_contribution
    合并后删除旧键，保留全部原字段。
    """
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act = enc["activation_encodings"]
    grouped: Dict[str, Dict[str, Any]] = {}
    for key, val in list(act.items()):
        # 识别形如 xxx.cells.N.<op> 的键
        parts = key.split(".")
        if len(parts) < 4 or parts[-3] != "cells":
            continue
        gru_prefix = ".".join(parts[:-1])  # 去掉最后的小算子名
        op_name = parts[-1]
        grouped.setdefault(gru_prefix, {})[op_name] = val

    if not grouped:
        return enc, False

    changed = False
    # 删除旧键
    for gru_prefix, ops in grouped.items():
        for op in ops:
            old_key = f"{gru_prefix}.{op}"
            if old_key in act:
                del act[old_key]
                changed = True
        # 新增合并后的键
        if gru_prefix not in act:
            act[gru_prefix] = {"internal_ops": ops}
        else:
            # 若已存在，补充 internal_ops
            entry = act[gru_prefix]
            if not isinstance(entry, dict):
                entry = {}
            internal_ops = entry.get("internal_ops", {})
            internal_ops.update(ops)
            entry["internal_ops"] = internal_ops
            act[gru_prefix] = entry
        changed = True

    return enc, changed


def rename_gru_internal_ops_keys(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    """
    将 GRU internal_ops 的小算子 key 从 OptimizedQuantizableGRUCell 的命名，重命名为后端期望的命名。

    OptimizedQuantizableGRUCell 名称 -> 后端名称
    - weight_ih          -> weight_ih_linear
    - weight_hh          -> weight_hh_linear
    - add_r_gate         -> add_reset_gate
    - add_u_gate         -> add_update_gate
    - mul_reset          -> mul_reset_hidden
    - add_n_gate         -> add_new_gate
    - sub_one_minus      -> sub_one_minus_update
    - add_final          -> add_final_hidden
    - sigmoid/tanh/mul_new_contribution/mul_old_contribution 保持不变

    仅重命名 activation_encodings[gru]['internal_ops'] 的 key，不改变其内部编码结构。
    """
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act: Dict[str, Any] = enc["activation_encodings"]
    changed = False

    # 兼容两套命名：
    # - 旧 OptimizedQuantizableGRUCell：add_u_gate/add_r_gate/add_n_gate/...（postprocess 负责转成后端字段名）
    # - 新对齐 HasteQDQ 的命名：update_gate_input/reset_gate_input/new_gate_input/...（通常无需再改）
    rename_map = {
        "weight_ih": "weight_ih_linear",
        "weight_hh": "weight_hh_linear",
        "add_r_gate": "reset_gate_input",
        "add_u_gate": "update_gate_input",
        "mul_reset": "mul_reset_hidden",
        "add_n_gate": "new_gate_input",
        "sub_one_minus": "sub_one_minus_update",
        "add_final": "add_final_hidden",
    }

    for gru_name, entry in list(act.items()):
        if not isinstance(entry, dict):
            continue
        internal_ops = entry.get("internal_ops")
        if not isinstance(internal_ops, dict):
            continue
        # 粗筛：更像 GRU 的 internal_ops（兼容新旧命名）
        if (
            "weight_ih" not in internal_ops
            and "weight_hh" not in internal_ops
            and "weight_ih_linear" not in internal_ops
            and "weight_hh_linear" not in internal_ops
        ):
            continue

        new_internal = dict(internal_ops)
        local_changed = False
        for old_k, new_k in rename_map.items():
            if old_k not in new_internal:
                continue
            if new_k not in new_internal:
                new_internal[new_k] = new_internal[old_k]
            del new_internal[old_k]
            local_changed = True

        if local_changed:
            entry["internal_ops"] = new_internal
            act[gru_name] = entry
            changed = True
            if verbose:
                logger.info("✅ 已重命名 GRU(%s) internal_ops key 以对齐后端字段名", gru_name)

    if changed:
        enc["activation_encodings"] = act
    return enc, changed


def add_gru_output_from_internal_ops(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    """
    在“合并后的 GRU 大算子条目”下新增 output 字段，用于表达 GRU 最终输出的量化参数。

    依据 OptimizedQuantizableGRUCell 的实现：
    - 每个 time-step 的 new_h 是最后一步 add_final（后端命名 add_final_hidden）的输出；
      整个 GRU 的输出序列就是各 time-step new_h 的堆叠/concat。
    因此用 internal_ops['add_final_hidden']['output']（或旧名 add_final）作为 GRU 大算子的 output 编码最合理。

    行为：
    - 若 gru_entry 已有 output，则不覆盖
    - 若 internal_ops 里缺 add_final 或其 output，则跳过
    """
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act: Dict[str, Any] = enc["activation_encodings"]
    changed = False

    for gru_name, entry in list(act.items()):
        if not isinstance(entry, dict):
            continue
        internal_ops = entry.get("internal_ops")
        if not isinstance(internal_ops, dict):
            continue
        if "output" in entry:
            continue

        add_final = internal_ops.get("add_final_hidden") or internal_ops.get("add_final")
        if not isinstance(add_final, dict):
            continue
        out_enc = add_final.get("output")
        if out_enc is None:
            continue

        entry["output"] = out_enc
        act[gru_name] = entry
        changed = True
        if verbose:
            logger.info("✅ 已为 GRU(%s) 补齐 output（来源 internal_ops.add_final_hidden.output）", gru_name)

    if changed:
        enc["activation_encodings"] = act
    return enc, changed


def add_gru_input_from_internal_ops(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    """
    在“合并后的 GRU 大算子条目”下新增 input 字段，用于表达 GRU 原始输入张量 x 的量化参数。

    依据 OptimizedQuantizableGRUCell 的实现：
    - 原始输入 x_t 首先进入 weight_ih（Linear），得到 gi，再切分为 i_r/i_z/i_n。
    因此最贴近“GRU 输入 x”的编码就是 internal_ops['weight_ih_linear']（或旧名 weight_ih）的 input 编码。

    行为：
    - 若 gru_entry 已有 input，则不覆盖
    - 若 internal_ops 中找不到候选 op 或其 input，则跳过
    """
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act: Dict[str, Any] = enc["activation_encodings"]
    changed = False

    # 候选 key：以 weight_ih 为主，兼容少数命名差异
    candidates = ("weight_ih_linear", "weight_ih", "weight_ih_l0", "ih", "linear_ih")

    for gru_name, entry in list(act.items()):
        if not isinstance(entry, dict):
            continue
        internal_ops = entry.get("internal_ops")
        if not isinstance(internal_ops, dict):
            continue
        if "input" in entry:
            continue

        picked = None
        used_key = None
        for ck in candidates:
            op_entry = internal_ops.get(ck)
            if isinstance(op_entry, dict) and "input" in op_entry:
                picked = op_entry.get("input")
                if picked is not None:
                    used_key = ck
                    break

        if picked is None:
            continue

        entry["input"] = picked
        act[gru_name] = entry
        changed = True
        if verbose:
            logger.info("✅ 已为 GRU(%s) 补齐 input（来源 internal_ops.%s.input）", gru_name, used_key or "weight_ih_linear")

    if changed:
        enc["activation_encodings"] = act
    return enc, changed


def merge_bn_activation_encodings(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    """
    将 activation_encodings 下 BatchNorm2d 的拆解小算子合并：
    - 针对键名包含 pre_bn 或 rnn2d_bn（可能带前缀，如 neck_seqs.0.rnn2d_bn.module_sub_3）。
    - 把 *.module_* 聚合到 *.internal_ops 下，并删除原始小算子键。
    """
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act = enc["activation_encodings"]
    grouped: Dict[str, Dict[str, Any]] = {}
    for key, val in list(act.items()):
        parts = key.split(".")
        if len(parts) < 2:
            continue
        # 只处理含 pre_bn 或 rnn2d_bn 的键
        if "pre_bn" not in parts and "rnn2d_bn" not in parts:
            continue
        prefix = ".".join(parts[:-1])  # 去掉最后的小算子名
        op_name = parts[-1]
        grouped.setdefault(prefix, {})[op_name] = val

    if not grouped:
        return enc, False

    changed = False
    for bn_prefix, ops in grouped.items():
        for op in ops:
            old_key = f"{bn_prefix}.{op}"
            if old_key in act:
                del act[old_key]
                changed = True
        if bn_prefix not in act:
            act[bn_prefix] = {"internal_ops": ops}
        else:
            entry = act[bn_prefix]
            if not isinstance(entry, dict):
                entry = {}
            internal_ops = entry.get("internal_ops", {})
            internal_ops.update(ops)
            entry["internal_ops"] = internal_ops
            act[bn_prefix] = entry
        changed = True

    enc["activation_encodings"] = act
    return enc, changed


def load_encodings(encodings_path: str) -> Dict[str, Any]:
    try:
        import json

        with open(encodings_path, "r") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法读取 encodings: {exc}") from exc


def save_encodings(enc: Dict[str, Any], encodings_path: str) -> None:
    import json

    with open(encodings_path, "w") as f:
        json.dump(enc, f, indent=2)


def postprocess_encodings_inplace(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    """
    encodings 后处理流水线（只处理 encodings dict，不碰 ONNX）：
    - 合并 GRU/BN activation_encodings -> internal_ops
    - 合并 GRU bias: *.weight_ih.bias + *.weight_hh.bias -> *.bias
    - 压缩 per-channel param_encodings 结构

    返回 (enc, changed)。
    """
    """
    执行顺序非常重要（避免互相“踩字段/踩结构”）：
    1) merge_gru_activation_encodings / merge_bn_activation_encodings
       - 先把 GRU/BN 的拆解算子聚合到 internal_ops，形成稳定层级
    1.2) rename_gru_internal_ops_keys
       - 将 GRU internal_ops 的小算子命名对齐到后端期望的字段名
    1.5) add_gru_output_from_internal_ops
       - 从 GRU internal_ops 中抽取最终输出对应的小算子输出（add_final.output），在 GRU 大算子下新增 output 字段
    1.6) add_gru_input_from_internal_ops
       - 从 GRU internal_ops 中抽取“最贴近原始输入张量 x”的小算子输入（优先 weight_ih.input），在 GRU 大算子下新增 input 字段
    2) merge_gru_bias_param_encodings
       - 合并 GRU 两组 bias（仍保持 list[dict] 形态，便于后续 compact）
    3) compact_per_channel_param_encodings
       - 必须在字段改名前执行（依赖 min/max/offset/scale 原字段名）
    4) flatten_activation_io_index_dict
       - 统一把 activation_encodings 的 input/output（以及 internal_ops）变成 list[dict]
    5) normalize_quant_fields_add_n
       - 最后统一改字段名 + 生成 n（递归覆盖 internal_ops）
    6) normalize_dtype_with_bitwidth
       - dtype 转大写，并把 bitwidth 拼接到 dtype 后（如 INT8/INT16/INT32）
    """
    changed_any = False

    enc, c1 = merge_gru_activation_encodings(enc, verbose=verbose)
    changed_any |= c1

    enc, c1_2 = rename_gru_internal_ops_keys(enc, verbose=verbose)
    changed_any |= c1_2

    enc, c1_5 = add_gru_output_from_internal_ops(enc, verbose=verbose)
    changed_any |= c1_5

    enc, c1_6 = add_gru_input_from_internal_ops(enc, verbose=verbose)
    changed_any |= c1_6
    
    # enc, c2 = merge_bn_activation_encodings(enc, verbose=verbose)
    # changed_any |= c2

    enc, c3 = merge_gru_bias_param_encodings(enc, verbose=verbose)
    changed_any |= c3

    enc, c4 = compact_per_channel_param_encodings(enc, verbose=verbose)
    changed_any |= c4

    # 可选：从 ONNX initializer 反推 BN 参数 encodings（尤其是 running_mean/running_var）
    # 说明：
    # - AIMET 常常不会给 running_mean/running_var 生成 param_encodings（它们是 buffer 时更常见）
    # - 但部署/硬件验证时我们仍希望 encodings 里具备它们的量化参数
    # - 这里通过读取 ONNX BatchNormalization 节点的 initializer，并按 min/max 生成对称 Po2 编码来补齐
    # - 默认只在调用方提供 onnx_path 时启用（见 postprocess_all）

    enc, c5 = flatten_activation_io_index_dict(enc, verbose=verbose)
    changed_any |= c5

    enc, c6 = normalize_quant_fields_add_n(enc, verbose=verbose)
    changed_any |= c6

    enc, c7 = normalize_dtype_with_bitwidth(enc, verbose=verbose)
    changed_any |= c7

    return enc, changed_any


def _find_closest_power_of_2_scale(scale: float) -> Tuple[float, int]:
    """
    将 scale 四舍五入到最接近的 2^{-n}。
    """
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError(f"scale must be finite positive, got {scale}")
    n = -math.log2(scale)
    n_rounded = int(round(n))
    return float(2.0 ** (-n_rounded)), int(n_rounded)


def _build_symmetric_po2_param_encoding(
    arr: np.ndarray,
    *,
    bitwidth: int,
    dtype: str = "int",
    symmetric: bool = True,
) -> Dict[str, Any]:
    """
    从参数张量数值范围构造“对称量化 + Po2 scale”的 encoding dict（AIMET encodings 常见字段）：
      {bitwidth, dtype, is_symmetric, min, max, offset, scale}

    注意：
    - 这里使用“unsigned 存储域 + offset=-2^(bw-1)”的等价表达，和你现有 encodings 文件保持一致：
        real = (q + offset) * scale, 其中 q in [0, 2^bw-1]
      对应的 representable 实数范围为：
        [offset*scale, (2^bw-1+offset)*scale] = [-2^(bw-1)*scale, (2^(bw-1)-1)*scale]
    """
    # 目前仅支持对称量化（与硬件/文档一致）
    if not symmetric:
        raise ValueError("only symmetric=True is supported for BN params in this helper")

    bw = int(bitwidth)
    if bw <= 0:
        raise ValueError(f"bitwidth must be > 0, got {bw}")

    qmin_s = -(2 ** (bw - 1))
    qmax_s = (2 ** (bw - 1)) - 1
    offset = int(qmin_s)  # 与现有 encodings 一致：offset=-128 (int8)

    arr_f = np.asarray(arr).astype(np.float64)
    max_abs = float(np.max(np.abs(arr_f))) if arr_f.size else 0.0
    # 避免全 0 导致 scale=0
    max_abs = max(max_abs, 1e-12)
    scale0 = max_abs / float(qmax_s)
    scale, _ = _find_closest_power_of_2_scale(scale0)

    real_min = float(offset) * float(scale)
    real_max = float(qmax_s) * float(scale)

    return {
        "bitwidth": bw,
        "dtype": dtype,
        "is_symmetric": "True",
        "min": real_min,
        "max": real_max,
        "offset": offset,
        "scale": float(scale),
    }


def add_bn_param_encodings_from_onnx(
    enc: Dict[str, Any],
    *,
    onnx_path: str,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], bool]:
    """
    从 ONNX `BatchNormalization` 节点的 initializer 读取参数值，并在 encodings 中补齐缺失的 BN param_encodings。

    当前策略：
    - 仅在 enc['param_encodings'] 缺少以下键时补齐：
        <bn>.running_mean
        <bn>.running_var
      （weight/bias 通常 AIMET 会导出；若缺少也可以一并补齐）
    - 生成方式：对称量化 + Po2 scale（scale=2^{-n}），bitwidth 默认对齐现有 weight/bias 的 bitwidth
    """
    if "param_encodings" not in enc or not isinstance(enc["param_encodings"], dict):
        enc["param_encodings"] = {}
    pe: Dict[str, Any] = enc["param_encodings"]

    try:
        model = onnx.load(onnx_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法加载 ONNX 用于补齐 BN param_encodings: {exc}") from exc

    init_map = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    changed = False
    for node in model.graph.node:
        if node.op_type != "BatchNormalization":
            continue
        bn_name = node.name or (node.output[0] if node.output else "")
        if not bn_name:
            continue

        # BN 输入：X, scale(weight), B(bias), mean, var
        if len(node.input) < 5:
            continue

        # 推断 bitwidth：优先从已有 weight/bias encodings 继承，否则从 quantizer_args.param_bitwidth
        bw = None
        for k in (f"{bn_name}.weight", f"{bn_name}.bias"):
            v = pe.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict) and "bitwidth" in v[0]:
                bw = int(v[0]["bitwidth"])
                break
            if isinstance(v, dict) and "bitwidth" in v:
                bw = int(v["bitwidth"])
                break
        if bw is None:
            qa = enc.get("quantizer_args", {})
            bw = int(qa.get("param_bitwidth", 8))

        # 需要补齐的参数
        targets = {
            "weight": node.input[1],
            "bias": node.input[2],
            "running_mean": node.input[3],
            "running_var": node.input[4],
        }

        for suffix, init_name in targets.items():
            key = f"{bn_name}.{suffix}"
            if key in pe:
                continue
            if init_name not in init_map:
                continue
            arr = init_map[init_name]
            # 只补 running_mean/var（默认）；若你想连 weight/bias 也补齐，可在这里去掉限制
            if suffix not in ("running_mean", "running_var"):
                continue

            enc_one = _build_symmetric_po2_param_encoding(arr, bitwidth=bw, dtype="int", symmetric=True)
            pe[key] = [enc_one]
            changed = True
            if verbose:
                logger.info("✅ 补齐 BN param_encodings: %s (from ONNX initializer=%s)", key, init_name)

    enc["param_encodings"] = pe
    return enc, changed


def flatten_activation_io_index_dict(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    """
    将 activation_encodings 下每个算子的 input/output 从带索引的 dict 规范化为 list[dict]：

    现状（AIMET 常见输出）：
      act[name]["input"]  = {"0": {...}} 或 {"0": {...}, "1": {...}, ...}
      act[name]["output"] = {"0": {...}}

    目标（你要求的格式）：
      - 单输入： act[name]["input"]  = [{...}]
      - 多输入： act[name]["input"]  = [{...}, {...}, ...]  （顺序按 "0","1",...）
      - 输出：   act[name]["output"] = [{...}]（一般只需要 1 个；若导出出现多个则保留顺序）

    同时递归处理：
      - act[name]["internal_ops"][op]["input"/"output"]（用于 GRU internal_ops，消除内部的 "0"/"1" 字段）
    """
    if "activation_encodings" not in enc or not isinstance(enc["activation_encodings"], dict):
        return enc, False

    act: Dict[str, Any] = enc["activation_encodings"]
    changed = False

    def _is_index_dict(x: Any) -> bool:
        if not isinstance(x, dict) or not x:
            return False
        # 只要 key 全是数字字符串，就认为是 index dict
        return all(isinstance(k, str) and k.isdigit() for k in x.keys())

    def _sorted_index_items(d: Dict[str, Any]) -> List[Any]:
        return [d[k] for k in sorted(d.keys(), key=lambda s: int(s))]

    def _to_list(v: Any) -> Any:
        """
        - index dict -> list (sorted by index)
        - single dict -> [dict]
        - list -> list (unchanged)
        """
        if _is_index_dict(v):
            items = _sorted_index_items(v)
            return items
        if isinstance(v, dict):
            return [v]
        if isinstance(v, list):
            return v
        return v

    def _normalize_entry_io(entry: Dict[str, Any]) -> bool:
        local_changed = False
        for io_key in ("input", "output"):
            if io_key not in entry:
                continue
            before = entry[io_key]
            after = _to_list(before)
            if after is not before:
                entry[io_key] = after
                local_changed = True
        # 递归 internal_ops（如 GRU）
        internal = entry.get("internal_ops")
        if isinstance(internal, dict):
            for _, op_entry in internal.items():
                if isinstance(op_entry, dict):
                    if _normalize_entry_io(op_entry):
                        local_changed = True
        return local_changed

    for op_name, entry in list(act.items()):
        if not isinstance(entry, dict):
            continue
        if _normalize_entry_io(entry):
            changed = True

        act[op_name] = entry

    if changed:
        enc["activation_encodings"] = act
        if verbose:
            logger.info("✅ 已规范化 activation_encodings 的 input/output 为 list[dict]（并递归处理 internal_ops，删除 '0'/'1' ...）")
    return enc, changed


def normalize_quant_fields_add_n(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    """
    给每组量化参数新增字段 n，并做字段名规范化：
    - n = -log2(scale)（scale 为 list 则 n 为 list）
    - offset -> zero_point
    - min -> real_min
    - max -> real_max

    覆盖范围：
    - activation_encodings：每个算子的 input/output（input 可能是 dict 或 list）
    - param_encodings：每个参数的量化 dict（含 per-channel 列表字段）
    """
    changed = False

    def _to_n(scale_val: Any) -> Any:
        if isinstance(scale_val, list):
            return [_to_n(x) for x in scale_val]
        try:
            s = float(scale_val)
        except Exception:  # noqa: BLE001
            return None
        if s <= 0:
            return None
        n_f = -math.log2(s)
        n_i = int(round(n_f))
        if abs(n_f - n_i) > 1e-6 and verbose:
            logger.warning("scale=%s 得到的 n 非整数：%s（将 round 为 %d）", scale_val, n_f, n_i)
        return n_i

    def _norm_dict(d: Dict[str, Any]) -> bool:
        local_changed = False

        # n
        if "scale" in d and "n" not in d:
            n_val = _to_n(d.get("scale"))
            if n_val is not None:
                d["n"] = n_val
                local_changed = True

        # rename
        rename_map = {
            "offset": "zero_point",
            "min": "real_min",
            "max": "real_max",
        }
        for old, new in rename_map.items():
            if old in d and new not in d:
                d[new] = d.pop(old)
                local_changed = True
            elif old in d and new in d:
                d.pop(old, None)
                local_changed = True

        return local_changed

    def _is_index_dict(x: Any) -> bool:
        return isinstance(x, dict) and bool(x) and all(isinstance(k, str) and k.isdigit() for k in x.keys())

    def _visit(obj: Any) -> None:
        """
        递归遍历 encodings 结构中的“量化参数 dict”，并对其做字段改名 + n 生成。
        - 支持 list / dict
        - 支持旧结构：input/output 为 {"0": {...}, "1": {...}}
        - 支持新结构：input/output 为 [{...}, {...}]
        - 支持 internal_ops 递归
        """
        nonlocal changed
        if obj is None:
            return
        if isinstance(obj, list):
            for it in obj:
                _visit(it)
            return
        if isinstance(obj, dict):
            # 如果是 {"0":{...},"1":{...}} 这种索引 dict，就按 value 递归
            if _is_index_dict(obj):
                for k in sorted(obj.keys(), key=lambda s: int(s)):
                    _visit(obj[k])
                return
            # 尝试把它当做“量化参数 dict”
            if any(k in obj for k in ("scale", "offset", "min", "max", "zero_point", "real_min", "real_max")):
                if _norm_dict(obj):
                    changed = True
            # 继续递归子字段（覆盖 internal_ops / input / output 等）
            for v in obj.values():
                _visit(v)
            return

    # activation_encodings（含 internal_ops）
    _visit(enc.get("activation_encodings"))

    # param_encodings
    _visit(enc.get("param_encodings"))

    if changed and verbose:
        logger.info(
            "✅ 已新增 n=-log2(scale)，并重命名字段 offset/min/max -> zero_point/real_min/real_max"
        )
    return enc, changed


def normalize_dtype_with_bitwidth(enc: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, Any], bool]:
    """
    修改每组量化参数的 dtype 字段：
    - dtype 字符串转大写
    - 将 bitwidth 追加到 dtype 后缀：
      bitwidth=8  -> INT8
      bitwidth=16 -> INT16
      bitwidth=32 -> INT32

    覆盖范围：
    - activation_encodings：每个算子的 input/output（含 internal_ops 递归）
    - param_encodings：每个参数的量化 dict
    """
    changed = False

    def _norm_one(d: Dict[str, Any]) -> bool:
        if "dtype" not in d or "bitwidth" not in d:
            return False
        dt = d.get("dtype")
        bw = d.get("bitwidth")
        if not isinstance(dt, str):
            return False
        try:
            bw_i = int(bw)
        except Exception:  # noqa: BLE001
            return False
        dt_up = dt.upper()
        base = dt_up.rstrip("0123456789")
        dt_new = f"{base}{bw_i}"
        if dt_new != dt:
            d["dtype"] = dt_new
            return True
        return False

    def _is_index_dict(x: Any) -> bool:
        return isinstance(x, dict) and bool(x) and all(isinstance(k, str) and k.isdigit() for k in x.keys())

    def _visit(obj: Any) -> None:
        nonlocal changed
        if obj is None:
            return
        if isinstance(obj, list):
            for it in obj:
                _visit(it)
            return
        if isinstance(obj, dict):
            if _is_index_dict(obj):
                for k in sorted(obj.keys(), key=lambda s: int(s)):
                    _visit(obj[k])
                return
            if _norm_one(obj):
                changed = True
            for v in obj.values():
                _visit(v)
            return

    _visit(enc.get("activation_encodings"))
    _visit(enc.get("param_encodings"))

    if changed and verbose:
        logger.info("✅ 已规范化 dtype：转大写并拼接 bitwidth（如 INT8/INT16/INT32）")
    return enc, changed


def postprocess_encodings_file(
    encodings_path: str,
    output_encodings_path: str,
    onnx_path: str | None = None,
    verbose: bool = True,
) -> str:
    """
    对 encodings 文件做后处理并写到 output_encodings_path。
    （与 ONNX 解耦，方便你单独 debug/启用/禁用某些步骤）
    """
    enc = load_encodings(encodings_path)
    enc, changed = postprocess_encodings_inplace(enc, verbose=verbose)
    # 可选：若提供 onnx_path，则补齐 BN running_mean/running_var 的 param_encodings
    if onnx_path:
        enc, bn_added = add_bn_param_encodings_from_onnx(enc, onnx_path=onnx_path, verbose=verbose)
        changed = bool(changed or bn_added)
    if changed or output_encodings_path != encodings_path:
        save_encodings(enc, output_encodings_path)
    return output_encodings_path


def rename_gru_initializers_and_change_gru_json_format(
    onnx_path: str,
    encodings_path: str,
    output_onnx_path: str | None = None,
    output_encodings_path: str | None = None,
    verbose: bool = True,
) -> Tuple[str, str]:
    """
    - ONNX：重命名 GRU 节点（去掉 # 后缀，并尽量匹配 encodings 中已有 key）
    - ONNX：重命名 GRU/Conv 的 initializer 名称为 <node>.<param> 形式
    - encodings：仅同步 param_encodings 的键名重命名（不做 merge/compact 等结构变换）

    返回 (onnx_out, enc_out) 路径。
    """
    enc_out = output_encodings_path or encodings_path
    enc = load_encodings(encodings_path)
    # 仅用于“匹配 GRU 节点名字”（不改 enc）
    act_keys = list(enc.get("activation_encodings", {}).keys()) if isinstance(enc.get("activation_encodings"), dict) else []

    # ------------ ONNX 处理（使用 JSON 键名做匹配） ------------
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

            # 处理 GRU 权重/偏置重命名：input[1]=weight_ih, input[2]=weight_hh, input[3]=bias
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

        # 通用 Conv 权重/偏置重命名：input[1]=weight, input[2]=bias
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

    # 写回 ONNX（保持行为确定性）
    if gru_renamed or param_renamed or onnx_out != onnx_path:
        onnx.save(model, onnx_out)
    else:
        onnx_out = onnx_path

    # ------------ encodings 写回（只同步 param_encodings 键名） ------------
    if param_renamed or enc_out != encodings_path:
        save_encodings(enc, enc_out)
    else:
        enc_out = encodings_path

    if verbose:
        logger.info("ONNX 输出: %s", onnx_out)
        logger.info("Encodings 输出: %s", enc_out)

    return onnx_out, enc_out


def postprocess_all(
    *,
    onnx_path: str,
    encodings_path: str,
    input_shape: Tuple[int, int, int],
    output_onnx_path: str | None = None,
    output_encodings_path: str | None = None,
    verbose: bool = True,
) -> Tuple[str, str]:
    """
    一键完成所有后处理（推荐调用的唯一入口）。

    PipelineStep 顺序（有依赖关系）：
    - G1: ONNX 图清理（onnxsim + shape inference + Reshape 形状子图折叠 + 常量 If 折叠）
    - G2: Slice ends 修正（INT64_MAX -> 真维度）
    - E1~E6: encodings 结构整理（GRU internal_ops 合并/改名、GRU bias 合并、per-channel 压缩）
    - X1: 从 ONNX initializer 补齐 BN running_mean/running_var 的 param_encodings（必须在 normalize 之前）
    - E7~E9: encodings 规范化（input/output 结构统一、字段改名+生成 n、dtype 标准化）
    - X2: ONNX initializer 重命名 + 同步 encodings 的 param_encodings key
    """
    # ------------------------
    # G1: ONNX 图清理（inplace）
    # ------------------------
    if verbose:
        logger.info("=== Pipeline G1: postprocess_onnx_graph_inplace ===")
    postprocess_onnx_graph_inplace(onnx_path, input_shape)

    # ------------------------
    # G2: Slice ends 修正（inplace）
    # ------------------------
    if verbose:
        logger.info("=== Pipeline G2: fix_slice_end_constants ===")
    fix_slice_end_constants(onnx_path, verbose=verbose)

    # ------------------------
    # E: encodings 结构整理（只处理 enc dict，不碰 ONNX）
    # ------------------------
    if verbose:
        logger.info("=== Pipeline E*: load_encodings ===")
    enc = load_encodings(encodings_path)
    changed_any = False

    # E1
    if verbose:
        logger.info("=== Pipeline E1: merge_gru_activation_encodings ===")
    enc, c = merge_gru_activation_encodings(enc, verbose=verbose)
    changed_any |= c

    # E2
    if verbose:
        logger.info("=== Pipeline E2: rename_gru_internal_ops_keys ===")
    enc, c = rename_gru_internal_ops_keys(enc, verbose=verbose)
    changed_any |= c

    # E3
    if verbose:
        logger.info("=== Pipeline E3: add_gru_output_from_internal_ops ===")
    enc, c = add_gru_output_from_internal_ops(enc, verbose=verbose)
    changed_any |= c

    # E4
    if verbose:
        logger.info("=== Pipeline E4: add_gru_input_from_internal_ops ===")
    enc, c = add_gru_input_from_internal_ops(enc, verbose=verbose)
    changed_any |= c

    # （可选）BN activation_encodings 合并：当前默认关闭（历史上容易与后端约定冲突）
    # enc, c = merge_bn_activation_encodings(enc, verbose=verbose)
    # changed_any |= c

    # E5
    if verbose:
        logger.info("=== Pipeline E5: merge_gru_bias_param_encodings ===")
    enc, c = merge_gru_bias_param_encodings(enc, verbose=verbose)
    changed_any |= c

    # E6
    if verbose:
        logger.info("=== Pipeline E6: compact_per_channel_param_encodings ===")
    enc, c = compact_per_channel_param_encodings(enc, verbose=verbose)
    changed_any |= c

    # ------------------------
    # X1: 从 ONNX 补齐 BN 参数 encodings（必须在 normalize 前）
    # ------------------------
    if verbose:
        logger.info("=== Pipeline X1: add_bn_param_encodings_from_onnx ===")
    enc, c = add_bn_param_encodings_from_onnx(enc, onnx_path=onnx_path, verbose=verbose)
    changed_any |= c

    # ------------------------
    # E7~E9: encodings 规范化
    # ------------------------
    if verbose:
        logger.info("=== Pipeline E7: flatten_activation_io_index_dict ===")
    enc, c = flatten_activation_io_index_dict(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E8: normalize_quant_fields_add_n ===")
    enc, c = normalize_quant_fields_add_n(enc, verbose=verbose)
    changed_any |= c

    if verbose:
        logger.info("=== Pipeline E9: normalize_dtype_with_bitwidth ===")
    enc, c = normalize_dtype_with_bitwidth(enc, verbose=verbose)
    changed_any |= c

    # 写回 encodings（在 X2 前先落盘，确保后续只做 key 同步不丢结构）
    enc_out = output_encodings_path or encodings_path
    if changed_any or enc_out != encodings_path:
        if verbose:
            logger.info("=== Pipeline E*: save_encodings -> %s ===", enc_out)
        save_encodings(enc, enc_out)
    else:
        enc_out = encodings_path

    # ------------------------
    # X2: ONNX initializer 重命名 + 同步 param_encodings 键名
    # ------------------------
    if verbose:
        logger.info("=== Pipeline X2: rename_gru_initializers_and_change_gru_json_format ===")
    onnx_out = output_onnx_path or onnx_path
    return rename_gru_initializers_and_change_gru_json_format(
        onnx_path=onnx_path,
        encodings_path=enc_out,
        output_onnx_path=onnx_out,
        output_encodings_path=enc_out,
        verbose=verbose,
    )



__all__ = [
    "postprocess_onnx_graph_inplace",
    "fix_slice_end_constants",
    "load_encodings",
    "save_encodings",
    "merge_gru_activation_encodings",
    "rename_gru_internal_ops_keys",
    "add_gru_output_from_internal_ops",
    "add_gru_input_from_internal_ops",
    "merge_bn_activation_encodings",
    "merge_gru_bias_param_encodings",
    "compact_per_channel_param_encodings",
    "flatten_activation_io_index_dict",
    "normalize_quant_fields_add_n",
    "normalize_dtype_with_bitwidth",
    "postprocess_encodings_inplace",
    "postprocess_encodings_file",
    "rename_gru_initializers_and_change_gru_json_format",
    "postprocess_all",
]

