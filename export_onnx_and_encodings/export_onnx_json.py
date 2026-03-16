"""
从 AIMET QuantizationSimModel 导出 **干净** ONNX + encodings（sidecar）。

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
from .custom_gru_onnx import (
    ensure_custom_gru_op_registered,
    replace_optimized_gru_modules,
)
from .custom_bn_onnx import replace_quantizable_batchnorm_modules
from .postprocess_refactor import postprocess_all

# 静态输入（你要求固定）
STATIC_DUMMY_INPUT_SHAPE: Tuple[int, int, int] = (1, 16000, 1)


# ---------- Internal helpers ----------
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
            f"BatchNormalization={ops_cnt.get(('', 'BatchNormalization'), 0)}, "
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
    # 将 OptimizedQuantizableGRU替换为 ExportOptimizedQuantizableGRU
    #---------------------------------------------------------------------------------------------------------------------

    original_model = sim.get_original_model(sim.model, qdq_weights=False)  
    original_model = original_model.cpu().eval()
    original_model = replace_optimized_gru_modules(original_model)
    original_model = replace_quantizable_batchnorm_modules(original_model)

    onnx_export_args = {
        "opset_version": opset,
        "input_names": ["input"],
        "output_names": ["output"],
        "do_constant_folding": True,
        # 关键：强制 legacy exporter，避免 dynamo/export 路径导致自定义 symbolic 不命中
        "dynamo": False,
        # 关键：告诉 exporter 这些自定义 domain 的 opset version
        "custom_opsets": {"custom_gru": 1, "custom_bn": 1},
    }

    onnx_path = os.path.join(export_dir, filename_prefix + ".onnx")
    sim.__class__.export_onnx_model_and_encodings(  # type: ignore[attr-defined]
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

    return onnx_path, enc_out_path


