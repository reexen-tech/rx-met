"""
从 AIMET QuantizationSimModel 导出干净ONNX + encodings（sidecar）。

核心特性：
- 使用 AIMET 的导出链路（节点名/张量名尽量与 encodings 对齐）
- 将 `OptimizedQuantizableGRU` 替换为单节点导出（导出为标准 ONNX `GRU` 节点）
- ONNX 图不包含 Q/DQ/fakequant（encodings 单独文件）
- 导出后强制执行：onnx-simplifier + 静态 shape inference + 折叠 Reshape shape 子图 + 折叠常量 If
"""

from __future__ import annotations

from typing import Tuple
import os
import shutil
from collections import Counter

import onnx
import torch
import torch.nn as nn
from .custom_gru_onnx import (
    ensure_custom_gru_op_registered,
    replace_optimized_gru_modules,
)
from .postprocess import postprocess_all

# 尝试导入 QuantGRU（如果已通过 pip install 安装，可以直接导入）
try:
    from quant_gru import QuantGRU
except ImportError:
    QuantGRU = None  # 如果未安装，设为 None

try:
    from aimet_torch.optimized_quantizable_gru import OptimizedQuantizableGRU
except ImportError:
    try:
        from aimet_rx.aimet_torch.optimized_quantizable_gru import OptimizedQuantizableGRU
    except ImportError:
        OptimizedQuantizableGRU = None  # 如果导入失败，设为 None

# 静态输入（你要求固定）
STATIC_DUMMY_INPUT_SHAPE: Tuple[int, int, int] = (1, 16000, 1)


# ---------- Internal helpers ----------
def _replace_quantgru_with_optimized(module: nn.Module, parent_name: str = "") -> list:
    """
    递归地将 QuantGRU 替换为 OptimizedQuantizableGRU
    
    QuantGRU 是 CUDA 实现，无法在 CPU 上导出 ONNX，因此需要临时替换为 CPU 兼容的版本。
    
    Args:
        module: 要处理的模块
        parent_name: 父模块名称（用于调试）
    
    Returns:
        replacements: [(full_name, original_module, replacement), ...] 替换列表
    """
    if QuantGRU is None or OptimizedQuantizableGRU is None:
        return []  # 如果导入失败，跳过替换
    
    replacements = []
    for name, child in list(module.named_children()):
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, QuantGRU):
            # 创建 OptimizedQuantizableGRU 替换
            input_size = child.input_size
            hidden_size = child.hidden_size
            num_layers = child.num_layers
            batch_first = child.batch_first
            
            replacement = OptimizedQuantizableGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=batch_first,
                num_layers=num_layers
            )
            
            # 复制权重（从 QuantGRU 的 state_dict 转换）
            # QuantGRU 的权重格式：weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
            # OptimizedQuantizableGRU 的权重格式：cells.0.weight_ih.weight, etc.
            try:
                child_state = child.state_dict()
                
                # 从 QuantGRU 的 state_dict 获取权重（避免使用 or 操作符，因为张量不能直接用于布尔判断）
                weight_ih = child_state.get('weight_ih_l0')
                if weight_ih is None:
                    weight_ih = child_state.get('_weight_ih_l0')
                
                weight_hh = child_state.get('weight_hh_l0')
                if weight_hh is None:
                    weight_hh = child_state.get('_weight_hh_l0')
                
                bias_ih = child_state.get('bias_ih_l0')
                if bias_ih is None:
                    bias_ih = child_state.get('_bias_ih_l0')
                
                bias_hh = child_state.get('bias_hh_l0')
                if bias_hh is None:
                    bias_hh = child_state.get('_bias_hh_l0')
                
                # 复制权重到 OptimizedQuantizableGRU
                # OptimizedQuantizableGRU 使用 cells 属性（不是 gru_cells）
                if weight_ih is not None:
                    replacement.cells[0].weight_ih.weight.data = weight_ih.cpu().clone()
                if weight_hh is not None:
                    replacement.cells[0].weight_hh.weight.data = weight_hh.cpu().clone()
                if bias_ih is not None:
                    replacement.cells[0].weight_ih.bias.data = bias_ih.cpu().clone()
                if bias_hh is not None:
                    replacement.cells[0].weight_hh.bias.data = bias_hh.cpu().clone()
                
                print(f"    ✅ {full_name}: 权重已复制")
            except Exception as e:
                print(f"  ⚠️ 警告：无法复制 {full_name} 的权重: {e}")
                print(f"     将使用随机初始化的权重（导出可能不准确）")
            
            # 替换模块
            setattr(module, name, replacement)
            replacements.append((full_name, child, replacement))
            print(f"  ✅ {full_name}: QuantGRU -> OptimizedQuantizableGRU")
        else:
            # 递归处理子模块
            replacements.extend(_replace_quantgru_with_optimized(child, full_name))
    return replacements


def _is_truthy_env(name: str) -> bool:
    v = os.getenv(name, "").strip()
    return v not in ("", "0", "false", "False", "FALSE", "no", "No", "NO")


def _pick_encodings_input_path(export_dir: str, filename_prefix: str) -> str:
    """
    选择 AIMET 导出的 encodings 输入文件路径：
    1) 优先 *_torch.encodings*（更接近 module-name 的层级格式，利于后处理）
    2) 其次是 {prefix}.encodings / {prefix}.json
    3) 再不行就挑 export_dir 下最新修改的 encodings/json
    """
    torch_cand = [
        os.path.join(export_dir, filename_prefix + "_torch.encodings"),
        os.path.join(export_dir, filename_prefix + "_torch.encodings.json"),
        os.path.join(export_dir, filename_prefix + "_torch.json"),
    ]
    enc_in_path = next((p for p in torch_cand if os.path.exists(p)), "")
    if enc_in_path:
        return enc_in_path

    cand = [
        os.path.join(export_dir, filename_prefix + ".encodings"),
        os.path.join(export_dir, filename_prefix + ".encodings.json"),
        os.path.join(export_dir, filename_prefix + ".json"),
    ]
    enc_in_path = next((p for p in cand if os.path.exists(p)), "")
    if enc_in_path:
        return enc_in_path

    files = [os.path.join(export_dir, f) for f in os.listdir(export_dir)]
    files = [
        f
        for f in files
        if f.endswith(".encodings") or f.endswith(".encodings.json") or f.endswith(".json")
    ]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0] if files else ""


def _maybe_keep_raw_exports(*, export_dir: str, filename_prefix: str, onnx_path: str, enc_in_path: str) -> None:
    """
    Debug：保留 raw 导出物（用于判断后处理/constant folding 是否把某些节点折叠掉）。
    用法：AIMET_EXPORT_KEEP_RAW=1 python your_script.py
    """
    if not _is_truthy_env("AIMET_EXPORT_KEEP_RAW"):
        return

    raw_onnx = os.path.join(export_dir, filename_prefix + "_raw.onnx")
    try:
        shutil.copyfile(onnx_path, raw_onnx)
        print(f"🧪 已保留 raw ONNX: {raw_onnx}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 保留 raw ONNX 失败：{e}")

    try:
        base, ext = os.path.splitext(enc_in_path)
        # 兼容 .encodings / .json / .encodings.json 之类
        ext2 = ""
        if base.endswith(".encodings") and ext == ".json":
            ext2 = ".encodings.json"
        raw_enc = os.path.join(export_dir, filename_prefix + "_raw" + (ext2 or ext or ".encodings"))
        shutil.copyfile(enc_in_path, raw_enc)
        print(f"🧪 已保留 raw encodings: {raw_enc}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 保留 raw encodings 失败：{e}")


def _postprocess_with_retry(
    *,
    onnx_path: str,
    enc_in_path: str,
    enc_out_path: str,
    input_shape: Tuple[int, int, int],
) -> None:
    def _postprocess_and_verify(tag: str) -> None:
        postprocess_all(
            onnx_path=onnx_path,
            encodings_path=enc_in_path,
            input_shape=input_shape,
            output_onnx_path=onnx_path,
            output_encodings_path=enc_out_path,
            verbose=True,
        )
        mm = onnx.load(onnx_path)
        ops_cnt = Counter([(n.domain, n.op_type) for n in mm.graph.node])
        print(
            f"✅ postprocess[{tag}] ops: "
            f"GRU={ops_cnt.get(('', 'GRU'), 0)}, "
            f"If={ops_cnt.get(('', 'If'), 0)}, "
            f"Shape={ops_cnt.get(('', 'Shape'), 0)}, "
            f"Gather={ops_cnt.get(('', 'Gather'), 0)}"
        )

    try:
        _postprocess_and_verify("first")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 第一次后处理失败：{e}，尝试再次后处理…")
        _postprocess_and_verify("second")


# ---------- Public API ----------
def export_onnx_json(
    sim,
    export_dir: str,
    filename_prefix: str,
    dummy_input_shape: Tuple[int, int, int] = STATIC_DUMMY_INPUT_SHAPE,
    opset: int = 18,
) -> Tuple[str, str]:
    """
    从 AIMET sim 导出 ONNX + encodings，并强制后处理 ONNX（去除辅助节点/If 并补齐静态 shape）。

    - ONNX 保持“干净”：不导出 Q/DQ/fakequant（固定为 sidecar encodings）
    - torch.onnx.export 强制 legacy：dynamo=False
    """
    os.makedirs(export_dir, exist_ok=True)
    # if tuple(dummy_input_shape) != STATIC_DUMMY_INPUT_SHAPE:
    #     raise ValueError(f"导出要求静态 dummy_input_shape={STATIC_DUMMY_INPUT_SHAPE}，当前为 {dummy_input_shape}")

    dummy_input = torch.zeros(*dummy_input_shape, device="cpu")
    ensure_custom_gru_op_registered(opset=opset)

    if not hasattr(sim, "get_original_model"):
        raise RuntimeError("sim 没有 get_original_model()，无法走 sim.export 链路")
    
    #---------------------------------------------------------------------------------------------------------------------
    # 临时替换 QuantGRU 为 OptimizedQuantizableGRU（QuantGRU 是 CUDA 实现，无法在 CPU 上导出）
    #---------------------------------------------------------------------------------------------------------------------
    quant_gru_replacements = []
    if QuantGRU is not None:
        # 检查是否有 QuantGRU 模块
        has_quantgru = any(isinstance(m, QuantGRU) for m in sim.model.modules())
        if has_quantgru:
            print("⚠️ 检测到 QuantGRU 模块，临时替换为 OptimizedQuantizableGRU 以便在 CPU 上导出 ONNX...")
            quant_gru_replacements = _replace_quantgru_with_optimized(sim.model)
            if quant_gru_replacements:
                print(f"✅ 共替换 {len(quant_gru_replacements)} 个 QuantGRU 模块")
    
    #---------------------------------------------------------------------------------------------------------------------
    # 将 OptimizedQuantizableGRU替换为 ExportOptimizedQuantizableGRU
    #---------------------------------------------------------------------------------------------------------------------
    # 确保模型在 CPU 上
    sim.model.cpu()
    sim.model.eval()

    original_model = sim.get_original_model(sim.model, qdq_weights=True)  
    original_model = original_model.cpu().eval()
    original_model = replace_optimized_gru_modules(original_model)
    original_model = original_model.cpu().eval()

    onnx_export_args = {
        "opset_version": opset,
        "input_names": ["input"],
        "output_names": ["output"],
        "do_constant_folding": True,
        # 关键：强制 legacy exporter，避免 dynamo/export 路径导致自定义 symbolic 不命中
        "dynamo": False,
        # 关键：告诉 exporter 这些自定义 domain 的 opset version
        "custom_opsets": {"custom_gru": 1},
    }

    onnx_path = os.path.join(export_dir, filename_prefix + ".onnx")
    sim.__class__.export_onnx_model_and_encodings(  #这个是aimet官方库的导出函数
        export_dir,
        filename_prefix,
        original_model,
        sim.model,
        dummy_input,
        onnx_export_args,
        propagate_encodings=False,
        module_marker_map=getattr(sim, "_module_marker_map", None),
        is_conditional=getattr(sim, "_is_conditional", False),
        excluded_layer_names=getattr(sim, "_excluded_layer_names", None),
        quantizer_args=getattr(sim, "quant_args", None),
        export_model=True,
        filename_prefix_encodings=filename_prefix,
    )

    #---------------------------------------------------------------------------------------------------------------------
    # 后处理 ONNX（去除辅助节点/If 并补齐静态 shape）
    #---------------------------------------------------------------------------------------------------------------------

    enc_in_path = _pick_encodings_input_path(export_dir, filename_prefix)
    if not enc_in_path:
        raise RuntimeError("未找到 AIMET 导出的 encodings 文件（.encodings/.json）")

    # 最终对外输出路径固定为 {prefix}.encodings（避免调用侧需要跟随 *_torch.encodings）
    enc_out_path = os.path.join(export_dir, filename_prefix + ".encodings")

    _maybe_keep_raw_exports(
        export_dir=export_dir,
        filename_prefix=filename_prefix,
        onnx_path=onnx_path,
        enc_in_path=enc_in_path,
    )
    _postprocess_with_retry(
        onnx_path=onnx_path,
        enc_in_path=enc_in_path,
        enc_out_path=enc_out_path,
        input_shape=dummy_input_shape,
    )
    
    #---------------------------------------------------------------------------------------------------------------------
    # 从 ONNX 模型中提取 GRU 节点名称并存储到 QuantGRU 模块
    #---------------------------------------------------------------------------------------------------------------------
    if quant_gru_replacements:
        try:
            onnx_model = onnx.load(onnx_path)
            # 提取所有 GRU 节点的名称（建立名称映射）
            gru_node_names = {}
            for node in onnx_model.graph.node:
                if node.op_type == 'GRU':
                    # ONNX 节点名称通常是模块路径名称
                    node_name = node.name
                    gru_node_names[node_name] = node_name
            
            # 将名称映射到 QuantGRU 模块（在恢复之前存储）
            print("\n📝 提取 AIMET 分配的 GRU 节点名称...")
            for full_name, original_module, replacement in quant_gru_replacements:
                # 尝试匹配 ONNX 节点名称（通常与模块路径名称一致）
                onnx_name = gru_node_names.get(full_name, full_name)
                
                # 存储到 QuantGRU 模块
                if hasattr(original_module, 'aimet_onnx_name'):
                    original_module.aimet_onnx_name = onnx_name
                    print(f"  ✅ {full_name}: ONNX 名称 = {onnx_name}")
                else:
                    # 如果 QuantGRU 没有这个属性，动态添加
                    original_module.aimet_onnx_name = onnx_name
                    print(f"  ✅ {full_name}: ONNX 名称 = {onnx_name} (动态添加)")
        except Exception as e:
            print(f"  ⚠️ 警告：无法提取 ONNX 节点名称: {e}")
    
    #---------------------------------------------------------------------------------------------------------------------
    # 导出 QuantGRU 的量化参数到 AIMET encodings 格式
    #---------------------------------------------------------------------------------------------------------------------
    if quant_gru_replacements:
        try:
            import json
            # 加载 AIMET encodings 文件
            with open(enc_out_path, 'r') as f:
                aimet_encodings = json.load(f)
            
            # 遍历所有 QuantGRU 模块，导出量化参数
            print("\n📤 导出 QuantGRU 量化参数到 AIMET 格式...")
            for full_name, original_module, replacement in quant_gru_replacements:
                if isinstance(original_module, QuantGRU) and original_module.is_calibrated():
                    module_name = original_module.aimet_onnx_name
                    if not module_name:
                        module_name = full_name
                        print(f"  ⚠️ 警告：{full_name} 的 aimet_onnx_name 未设置，使用模块路径名称作为后备")
                    
                    original_module.export_quant_params_to_aimet_format(
                        aimet_encodings,
                        module_name=module_name,
                        verbose=True
                    )
            
            # 保存更新后的 encodings
            with open(enc_out_path, 'w') as f:
                json.dump(aimet_encodings, f, indent=2, ensure_ascii=False)
            print(f"  ✅ 已保存更新后的 encodings 到 {enc_out_path}")
        except Exception as e:
            print(f"  ⚠️ 警告：导出 QuantGRU 量化参数失败: {e}")
        
    #---------------------------------------------------------------------------------------------------------------------
    # 恢复 QuantGRU 模块（如果需要的话）
    #---------------------------------------------------------------------------------------------------------------------
    if quant_gru_replacements:
        print("\n♻️ 恢复 QuantGRU 模块...")
        for full_name, original_module, replacement in quant_gru_replacements:
            # 找到父模块和子模块名称
            parts = full_name.split('.')
            parent = sim.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            # 恢复原始模块（此时 aimet_onnx_name 已经设置）
            setattr(parent, parts[-1], original_module)
            print(f"  ✅ {full_name}: OptimizedQuantizableGRU -> QuantGRU")

    return onnx_path, enc_out_path


