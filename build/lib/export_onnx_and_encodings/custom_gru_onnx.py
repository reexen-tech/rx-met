"""
自定义 GRU 的 ONNX 导出支持。

职责：
- 注册 torch.library 自定义算子 `custom_gru::custom_gru`
- 注册 legacy ONNX symbolic，把自定义算子导出为标准 ONNX `GRU` 节点
- 提供占位模块 `ExportOptimizedQuantizableGRU`，用于把 Python 展开版 GRU 替换为“单节点导出”
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.onnx import register_custom_op_symbolic, symbolic_helper

from aimet_torch.optimized_quantizable_gru import OptimizedQuantizableGRU

# torch.library 必须全局持有，否则 op 可能“消失”
_CUSTOM_GRU_LIB: Optional[torch.library.Library] = None


def ensure_custom_gru_op_registered(opset: int = 18) -> None:
    """注册 runtime op + ONNX symbolic。必须在导出 ONNX 前调用一次。"""
    if getattr(ensure_custom_gru_op_registered, "_done", False):
        return

    global _CUSTOM_GRU_LIB
    if _CUSTOM_GRU_LIB is None:
        _CUSTOM_GRU_LIB = torch.library.Library("custom_gru", "DEF")

    _CUSTOM_GRU_LIB.define(
        # 约定对齐 ONNX GRU：x 为 time-major [T,B,I]
        # 为了导出为标准 ONNX::GRU，我们在这里直接传入已打包/重排好的 W/R/B（ONNX 格式）。
        "custom_gru(Tensor x, Tensor h0, Tensor W, Tensor R, Tensor B, "
        "int hidden_size, int num_layers) -> (Tensor, Tensor)"
    )

    def _custom_gru_cpu(x, h0, W, R, B, hidden_size, num_layers):
        t, b = x.shape[0], x.shape[1]
        out = x.new_zeros(t, b, hidden_size)
        h_n = x.new_zeros(num_layers, b, hidden_size)
        return out, h_n

    try:
        _CUSTOM_GRU_LIB.impl("custom_gru", _custom_gru_cpu, dispatch_key="CPU")
    except TypeError:
        _CUSTOM_GRU_LIB.impl("custom_gru", "CPU", _custom_gru_cpu)

    def _symbolic_custom_gru(g, x, h0, W, R, B, hidden_size, num_layers):
        hidden_size_i = symbolic_helper._maybe_get_const(hidden_size, "i")
        num_layers_i = symbolic_helper._maybe_get_const(num_layers, "i")
        if hidden_size_i is None or num_layers_i is None:
            raise RuntimeError("custom_gru attributes must be constants for ONNX export")

        # 导出为标准 ONNX GRU（domain=""，op_type="GRU"）
        # 输入格式（ONNX）：X[T,B,I], W[1,3H,I], R[1,3H,H], B[1,6H], sequence_lens(optional), initial_h(optional)
        Y, Y_h = g.op(
            "GRU",
            x,
            W,
            R,
            B,
            symbolic_helper._optional_input_placeholder_tensor(g),
            h0,
            hidden_size_i=int(hidden_size_i),
            direction_s="forward",
            outputs=2,
        )

        # Y: [T, 1, B, H] -> squeeze axis=1 -> [T,B,H]
        axes = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        out = g.op("Squeeze", Y, axes)
        return out, Y_h

    register_custom_op_symbolic("custom_gru::custom_gru", _symbolic_custom_gru, opset)
    ensure_custom_gru_op_registered._done = True


class ExportOptimizedQuantizableGRU(nn.Module):
    """
    占位模块：把 Python GRU 替换为单节点 custom_gru 调用。

    关键点：
    - custom_gru 约定输入为 time-major: [T,B,I]（对齐 ONNX GRU）
    - 这里会根据源模块 `src.batch_first` 自动适配：
      - src.batch_first=True  : forward 接受 [B,T,I]，内部转为 [T,B,I]，输出再转回 [B,T,H]
      - src.batch_first=False : forward 接受 [T,B,I]，不做转置，输出保持 [T,B,H]
    因此未来你想把上游改成 [T,B,C] 输入也无需改导出逻辑。
    """

    def __init__(self, src: OptimizedQuantizableGRU):
        super().__init__()
        assert src.num_layers == 1
        # 输出布局：与源模块一致
        self.batch_first = bool(getattr(src, "batch_first", False))
        # 输入布局：若源模块支持 input_batch_first，则以它为准；否则默认与 batch_first 一致
        self.input_batch_first = bool(getattr(src, "input_batch_first", self.batch_first))
        self.hidden_size = src.hidden_size
        self.num_layers = src.num_layers

        cell = src.cells[0]
        # PyTorch GRU gate order 通常为 (r, z, n)，ONNX GRU 规范为 (z, r, h)
        # 这里在 Python 侧一次性重排并打包成 ONNX 需要的 W/R/B，避免导出图里出现 Gather/Concat 等辅助节点。
        H = int(self.hidden_size)

        def _reorder_rzn_to_zrh(t: torch.Tensor) -> torch.Tensor:
            assert t.shape[0] == 3 * H
            r, z, n = t[:H], t[H : 2 * H], t[2 * H :]
            return torch.cat([z, r, n], dim=0)

        W_ih = _reorder_rzn_to_zrh(cell.weight_ih.weight.detach())
        W_hh = _reorder_rzn_to_zrh(cell.weight_hh.weight.detach())
        b_ih = _reorder_rzn_to_zrh(cell.weight_ih.bias.detach())
        b_hh = _reorder_rzn_to_zrh(cell.weight_hh.bias.detach())

        # ONNX: W[1,3H,I], R[1,3H,H], B[1,6H] = [Wb(3H), Rb(3H)]
        self.register_buffer("W", W_ih.unsqueeze(0).contiguous())
        self.register_buffer("R", W_hh.unsqueeze(0).contiguous())
        self.register_buffer("B", torch.cat([b_ih, b_hh], dim=0).unsqueeze(0).contiguous())

    def forward(self, x, h0=None):
        # 统一喂给 custom_gru 的 time-major 输入：[T,B,I]
        # - 若 input_batch_first=True ：x 是 [B,T,I] -> [T,B,I]
        # - 若 input_batch_first=False：x 已是 [T,B,I]
        x_tb = x.transpose(0, 1) if self.input_batch_first else x
        if h0 is None:
            # initial_h: [1, B, H]
            b_dim = x_tb.shape[1]
            try:
                b = int(b_dim)
                h0 = torch.zeros((1, b, int(self.hidden_size)), device=x.device, dtype=x.dtype)
            except Exception:
                b = x_tb.size(1)
                h0 = x.new_zeros((1, b, int(self.hidden_size)))

        out_tb, h_n = torch.ops.custom_gru.custom_gru(
            x_tb,
            h0,
            self.W,
            self.R,
            self.B,
            int(self.hidden_size),
            int(self.num_layers),
        )
        # 输出 layout 与原模块保持一致
        return (out_tb.transpose(0, 1) if self.batch_first else out_tb), h_n


def replace_optimized_gru_modules(module: nn.Module) -> nn.Module:
    """递归替换模型中的 OptimizedQuantizableGRU 为 ExportOptimizedQuantizableGRU。"""
    for name, child in module.named_children():
        is_gru = isinstance(child, OptimizedQuantizableGRU)
        is_gru = is_gru or (hasattr(child, "cells") and hasattr(child, "hidden_size") and hasattr(child, "num_layers"))
        if is_gru:
            setattr(module, name, ExportOptimizedQuantizableGRU(child))  # type: ignore[arg-type]
        else:
            replace_optimized_gru_modules(child)
    return module


__all__ = [
    "ensure_custom_gru_op_registered",
    "replace_optimized_gru_modules",
    "ExportOptimizedQuantizableGRU",
]


