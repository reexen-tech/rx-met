# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Quantized definitions for custom modules of AIMET"""

import copy
from typing import Optional
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from aimet_torch._base.nn.modules.custom import *  # pylint: disable=wildcard-import, unused-wildcard-import
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase
from aimet_torch.v2.quantization.affine import AffineQuantizerBase, QuantizeDequantize
from aimet_torch.v2.quantization.affine.backends.torch_builtins import (
    get_encoding_shape_with_blocks,
    reshape_tensor_for_blocks,
)
from ..true_quant import (
    QuantizationMixin,
    _DispatchMixin,
    _quantize_if_applicable,
    _quantize_dequantize_if_applicable,
    _is_computing_encodings,
)

# NOTE: Disabling due to pylint false alarm in ModuleList
# pylint: disable=not-callable

from aimet_torch.native_recurrent import recurrent_operators


class _QuantizedSingularUnary(torch.autograd.Function):
    """保持一元算子的前向结果，约定量化训练在奇点处的反向行为。"""

    @staticmethod
    def forward(ctx, x, is_sqrt, zero_code):
        output = torch.sqrt(x) if is_sqrt else torch.log(x)
        ctx.save_for_backward(x, output)
        ctx.is_sqrt = is_sqrt
        ctx.zero_code = zero_code
        return output

    @staticmethod
    def backward(ctx, grad):
        x, output = ctx.saved_tensors
        denominator = 2 * output if ctx.is_sqrt else x
        # 下游截断可能给 log/sqrt(0) 传回零梯度，此时应避开 0/0。
        # TensorFlow 的 SqrtGrad 对零上游梯度也采用这一保护。
        inactive = (denominator == 0) & (grad == 0)
        if ctx.zero_code:
            # 待审核的 QAT 近似梯度：Sqrt 输入为零码时，即使收到有限非零梯度也返回零。
            # 这可防止无穷梯度传到上游平方，与平方的零导数相乘后产生 NaN。
            # 这不是数学上的 Sqrt 零点导数，也不是量化截断的 STE mask：
            # 被舍入为零的小非零值同样会失去梯度。
            # 目前仅验证波形梯度变为有限，尚未证实训练收益；该策略仍待研究、审核。
            # 正值沿用原导数，非有限的上游梯度不被屏蔽。
            inactive = (denominator == 0) & torch.isfinite(grad)
        # 除法前先为无效分支选择安全分母；先算无穷导数再乘零 mask 仍会产生 NaN。
        safe = torch.where(inactive, torch.ones_like(denominator), denominator)
        return torch.where(inactive, torch.zeros_like(grad), grad / safe), None, None


def _singular_unary_qat_fn(module, fn, is_sqrt):
    """校准期间和所有量化器均禁用时，保持 Torch 原有行为。"""
    quantizers = list(module.input_quantizers) + list(module.output_quantizers)
    if (not torch.is_grad_enabled() or _is_computing_encodings(module)
            or not any(isinstance(q, AffineQuantizerBase) and q.is_initialized()
                       for q in quantizers)):
        return fn
    input_quantizer = module.input_quantizers[0]
    # 仅在输入 QDQ 已初始化时启用实验性的 Sqrt 零码规则。
    # 只有输出 QDQ 时，零点收到非零梯度仍保留原有奇点行为。
    zero_code = (is_sqrt and isinstance(input_quantizer, AffineQuantizerBase)
                 and input_quantizer.is_initialized())

    def call(x, **kwargs):
        if kwargs or not x.is_floating_point():
            return fn(x, **kwargs)
        return _QuantizedSingularUnary.apply(x, is_sqrt, zero_code)

    return call


@QuantizationMixin.implements(Sin)
class QuantizedSin(_DispatchMixin, QuantizationMixin, Sin):
    """Quantized Sin"""

    _builtin_torch_fn = torch.sin


@QuantizationMixin.implements(Cos)
class QuantizedCos(_DispatchMixin, QuantizationMixin, Cos):
    """Quantized Cos"""

    _builtin_torch_fn = torch.cos


@QuantizationMixin.implements(AvgPool2d)
class QuantizedAvgPool2d(_DispatchMixin, QuantizationMixin, AvgPool2d):
    """Quantized AvgPool2d"""

    _builtin_torch_fn = F.avg_pool2d

    def _is_dynamo_traceable(self):
        # F.avg_pool2d isn't dynamo-traceable
        return False


@QuantizationMixin.implements(Reshape)
class QuantizedReshape(_DispatchMixin, QuantizationMixin, Reshape):
    """Quantized Reshape"""

    _builtin_torch_fn = torch.reshape

    def _is_dynamo_traceable(self):
        # torch.reshape isn't dynamo-traceable
        return False


@QuantizationMixin.implements(RSqrt)
class QuantizedRSqrt(_DispatchMixin, QuantizationMixin, RSqrt):
    """Quantized RSqrt"""

    _builtin_torch_fn = torch.rsqrt


@QuantizationMixin.implements(MatMul)
class QuantizedMatMul(_DispatchMixin, QuantizationMixin, MatMul):
    """Quantized MatMul"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.matmul


@QuantizationMixin.implements(Add)
class QuantizedAdd(_DispatchMixin, QuantizationMixin, Add):
    """Quantized Add"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.add


@QuantizationMixin.implements(Multiply)
class QuantizedMultiply(_DispatchMixin, QuantizationMixin, Multiply):
    """Quantized Multiply"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.mul


@QuantizationMixin.implements(Subtract)
class QuantizedSubtract(_DispatchMixin, QuantizationMixin, Subtract):
    """Quantized Subtract"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.sub


@QuantizationMixin.implements(Divide)
class QuantizedDivide(_DispatchMixin, QuantizationMixin, Divide):
    """Quantized Divide"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.div


@QuantizationMixin.implements(Outer)
class QuantizedOuter(_DispatchMixin, QuantizationMixin, Outer):
    """Quantized Outer"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.outer


@QuantizationMixin.implements(Concat)
class QuantizedConcat(_DispatchMixin, QuantizationMixin, Concat):
    """Quantized Concat"""

    _builtin_torch_fn = torch.cat

    # pylint: disable=attribute-defined-outside-init
    def __quant_init__(self):
        super().__quant_init__()
        self._num_inputs = 1

    def export_input_encodings(self, encoding_version: str):
        """
        Extends super().export to repeat input quantizer's encodings :attr:`self._num_inputs` times
        """
        input_encodings = super().export_input_encodings(encoding_version)
        # Create separate encoding objects to avoid overriding of attributes added/updated later while exporting encodings
        return [
            copy.deepcopy(encoding) for encoding in input_encodings * self._num_inputs
        ]

    def import_input_encodings(
        self,
        encodings,
        strict: bool,
        partial: bool,
        requires_grad: Optional[bool],
        allow_overwrite: bool,
    ):
        """
        Extends super().import_input_encodings to set `self._num_inputs` based on length of encodings.
        """
        self._num_inputs = len(encodings)
        super().import_input_encodings(
            encodings,
            strict=strict,
            partial=partial,
            requires_grad=requires_grad,
            allow_overwrite=allow_overwrite,
        )

    def forward(self, *x):  # pylint: disable=arguments-differ
        """
        Quantized forward impl for custom.Concat.
        """
        self._num_inputs = len(x)
        return super().forward(*x)

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        def cat(tensors, dim=0, *, out=None):
            input_qtzr = self.input_quantizers[0]
            tensors = tuple(
                _quantize_dequantize_if_applicable(x, input_qtzr) for x in tensors
            )
            output = fn(tensors, dim=dim, out=out)
            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return cat

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        def cat(tensors, dim=0, *, out=None):
            input_qtzr = self.input_quantizers[0]
            tensors = tuple(_quantize_if_applicable(x, input_qtzr) for x in tensors)
            output_encodings = (
                self.output_quantizers[0].get_encodings()
                if self.output_quantizers[0]
                else None
            )
            return fn(tensors, dim=dim, out=out, output_encodings=output_encodings)

        return cat


@QuantizationMixin.implements(FloorDivide)
class QuantizedFloorDivide(_DispatchMixin, QuantizationMixin, FloorDivide):
    """ Quantized FloorDivide """
    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.floor_divide
#
#
@QuantizationMixin.implements(Norm)
class QuantizedNorm(_DispatchMixin, QuantizationMixin, Norm):
    """Quantized Norm"""

    _builtin_torch_fn = torch.norm


@QuantizationMixin.implements(Exponential)
class QuantizedExponential(_DispatchMixin, QuantizationMixin, Exponential):
    """Quantized Exponential"""

    _builtin_torch_fn = torch.exp


@QuantizationMixin.implements(Erf)
class QuantizedErf(_DispatchMixin, QuantizationMixin, Erf):
    """Quantized Erf"""

    _builtin_torch_fn = torch.erf


@QuantizationMixin.implements(Sqrt)
class QuantizedSqrt(_DispatchMixin, QuantizationMixin, Sqrt):
    """Quantized Sqrt"""

    _builtin_torch_fn = torch.sqrt

    def _builtin_torch_fn_helper(self, fn):
        return super()._builtin_torch_fn_helper(_singular_unary_qat_fn(self, fn, True))


# @QuantizationMixin.implements(Maximum)
# class QuantizedMaximum(_DispatchMixin, QuantizationMixin, Maximum):
#     """ Quantized Maximum """
#     _builtin_torch_fn = torch.maximum
#
#
# @QuantizationMixin.implements(Max)
# class QuantizedMax(_DispatchMixin, QuantizationMixin, Max):
#     """ Quantized Max """
#     _builtin_torch_fn = torch.max
#
# @QuantizationMixin.implements(AMax)
# class QuantizedAMax(_DispatchMixin, QuantizationMixin, AMax):
#     """ Quantized AMax """
#     _builtin_torch_fn = torch.amax
#
#
# @QuantizationMixin.implements(Minimum)
# class QuantizedMinimum(_DispatchMixin, QuantizationMixin, Minimum):
#     """ Quantized Minimum """
#     _builtin_torch_fn = torch.minimum
#
#
# @QuantizationMixin.implements(Min)
# class QuantizedMin(_DispatchMixin, QuantizationMixin, Min):
#     """ Quantized Min """
#     _builtin_torch_fn = torch.min
#
# @QuantizationMixin.implements(AMin)
# class QuantizedAMin(_DispatchMixin, QuantizationMixin, AMin):
#     """ Quantized AMin """
#     _builtin_torch_fn = torch.amin
#
#
# @QuantizationMixin.implements(Where)
# class QuantizedWhere(_DispatchMixin, QuantizationMixin, Where):
#     """ Quantized Where """
#     _builtin_torch_fn = torch.where
#
#
# @QuantizationMixin.implements(Greater)
# class QuantizedGreater(_DispatchMixin, QuantizationMixin, Greater):
#     """ Quantized Greater """
#     _builtin_torch_fn = torch.gt
#
#
# @QuantizationMixin.implements(Less)
# class QuantizedLess(_DispatchMixin, QuantizationMixin, Less):
#     """ Quantized Less """
#     _builtin_torch_fn = torch.lt
#
#
# @QuantizationMixin.implements(GreaterEqual)
# class QuantizedGreaterEqual(_DispatchMixin, QuantizationMixin, GreaterEqual):
#     """ Quantized GreaterEqual """
#     _builtin_torch_fn = torch.ge
#
#
# @QuantizationMixin.implements(LessEqual)
# class QuantizedLessEqual(_DispatchMixin, QuantizationMixin, LessEqual):
#     """ Quantized LessEqual """
#     _builtin_torch_fn = torch.le
#
#
# @QuantizationMixin.implements(NotEqual)
# class QuantizedNotEqual(_DispatchMixin, QuantizationMixin, NotEqual):
#     """ Quantized NotEqual """
#     _builtin_torch_fn = torch.ne
#
#
# @QuantizationMixin.implements(Equal)
# class QuantizedEqual(_DispatchMixin, QuantizationMixin, Equal):
#     """ Quantized Equal """
#     _builtin_torch_fn = torch.eq


@QuantizationMixin.implements(Bmm)
class QuantizedBmm(_DispatchMixin, QuantizationMixin, Bmm):
    """Quantized Bmm"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.bmm


@QuantizationMixin.implements(CumSum)
class QuantizedCumSum(_DispatchMixin, QuantizationMixin, CumSum):
    """Quantized CumSum"""

    _builtin_torch_fn = torch.cumsum

    def _is_dynamo_traceable(self):
        # torch.cumsum isn't dynamo-traceable
        return False

@QuantizationMixin.implements(Reciprocal)
class QuantizedReciprocal(_DispatchMixin, QuantizationMixin, Reciprocal):
    """Quantized Reciprocal"""
    _builtin_torch_fn = torch.reciprocal


@QuantizationMixin.implements(Clamp)
class QuantizedClamp(_DispatchMixin, QuantizationMixin, Clamp):
    """Quantized Clamp"""
    _builtin_torch_fn = torch.clamp


@QuantizationMixin.implements(Clip)
class QuantizedClip(_DispatchMixin, QuantizationMixin, Clip):
    """Quantized Clip"""
    _builtin_torch_fn = torch.clip



# @QuantizationMixin.implements(MaskedFill)
# class QuantizedMaskedFill(_DispatchMixin, QuantizationMixin, MaskedFill):
#     """ Quantized MaskedFill """
#     _builtin_torch_fn = torch.Tensor.masked_fill_
#
#
@QuantizationMixin.implements(Mean)
class QuantizedMean(_DispatchMixin, QuantizationMixin, Mean):
    """Quantized Mean"""
    
    _builtin_torch_fn = torch.mean


@QuantizationMixin.implements(Var)
class QuantizedVar(_DispatchMixin, QuantizationMixin, Var):
    """Quantized Var"""

    _builtin_torch_fn = torch.var

    def __quant_init__(self):
        super().__quant_init__()
        # torch.var 没有标准 ONNX op 映射，JSON config 无法自动创建 input quantizer。
        # 此处保持 None 占位（避免 sim 初始化 trace 阶段因未 calibrate 而报错）；
        # 需由 apply_mixed_precision_bitwidth 在 sim 创建后按需创建真实 quantizer。
        if not self.input_quantizers or len(self.input_quantizers) == 0:
            self.input_quantizers = nn.ModuleList([None])


# @QuantizationMixin.implements(Sum)
# class QuantizedSum(_DispatchMixin, QuantizationMixin, Sum):
#     """ Quantized Sum """
#     _builtin_torch_fn = torch.sum
#
#
# @QuantizationMixin.implements(Prod)
# class QuantizedProd(_DispatchMixin, QuantizationMixin, Prod):
#     """ Quantized Prod """
#     _builtin_torch_fn = torch.prod
#
#
@QuantizationMixin.implements(Log)
class QuantizedLog(_DispatchMixin, QuantizationMixin, Log):
    """Quantized Log"""

    _builtin_torch_fn = torch.log

    def _builtin_torch_fn_helper(self, fn):
        fn = _singular_unary_qat_fn(self, fn, False)
        output_quantizer = self.output_quantizers[0]
        # 有限饱和处理要求已初始化的 affine 输出编码。
        # 校准仍需观察原始 Log，不能用此前的输出编码覆盖校准样本。
        if (not isinstance(output_quantizer, AffineQuantizerBase)
                or not output_quantizer.is_initialized()
                or _is_computing_encodings(self)):
            return super()._builtin_torch_fn_helper(fn)

        def saturated_log(x, **kwargs):
            if kwargs or not x.is_floating_point():
                return fn(x, **kwargs)
            # 此处的 x 已经过输入 QDQ，基类稍后仍会执行输出 QDQ。
            # 因 Qout(log(0)) == Qout(y_min)，零码直接取输出编码下限，
            # 可保持前向结果一致，并避免饱和分支的输入梯度被无穷中间值污染。
            # 不移动 QDQ、不改变 epsilon、不读取量化前输入，也不恢复已压零的小值信息。
            zero = x == 0
            # torch.where 会计算两个分支。1 仅用于未选中的 Log 分支，避免 log(0)
            # 污染反向；零码实际返回 y_min，而非 log(1)。保留 y_min 的编码参数梯度。
            safe = torch.where(zero, torch.ones_like(x), x)
            lower = output_quantizer.get_min(dtype=x.dtype)
            if output_quantizer.block_size is not None:
                # 分块编码下每块有独立下限，不能按普通通道广播，也不能共用全局最小值。
                # 沿用 QDQ 后端的分块布局，再映射回输入形状；保留 lower 的梯度，
                # 使零码对编码参数的梯度与有限的饱和输入一致，避开 -Inf 污染。
                blocked = reshape_tensor_for_blocks(x, lower.shape, output_quantizer.block_size)
                lower = lower.view(get_encoding_shape_with_blocks(
                    lower.shape, output_quantizer.block_size)).expand_as(blocked).reshape(x.shape)
            return torch.where(zero, lower, fn(safe))

        return super()._builtin_torch_fn_helper(saturated_log)


@QuantizationMixin.implements(Abs)
class QuantizedAbs(_DispatchMixin, QuantizationMixin, Abs):
    """Quantized Abs"""

    _builtin_torch_fn = torch.abs


@QuantizationMixin.implements(Neg)
class QuantizedNeg(_DispatchMixin, QuantizationMixin, Neg):
    """Quantized Neg"""

    _builtin_torch_fn = torch.neg


#
#
# @QuantizationMixin.implements(Argmin)
# class QuantizedArgmin(_DispatchMixin, QuantizationMixin, Argmin):
#     """ Quantized Argmin """
#     _builtin_torch_fn = torch.argmin
#
#
# @QuantizationMixin.implements(Argmax)
# class QuantizedArgmax(_DispatchMixin, QuantizationMixin, Argmax):
#     """ Quantized Argmax """
#     _builtin_torch_fn = torch.argmax
#
#
# @QuantizationMixin.implements(ElementwiseCeil)
# class QuantizedElementwiseCeil(_DispatchMixin, QuantizationMixin, ElementwiseCeil):
#     """ Quantized ElementwiseCeil """
#     _builtin_torch_fn = torch.ceil
#
#
# @QuantizationMixin.implements(ElementwiseFloor)
# class QuantizedElementwiseFloor(_DispatchMixin, QuantizationMixin, ElementwiseFloor):
#     """ Quantized ElementwiseFloor """
#     _builtin_torch_fn = torch.floor
#
#
# @QuantizationMixin.implements(Asin)
# class QuantizedAsin(_DispatchMixin, QuantizationMixin, Asin):
#     """ Quantized Asin """
#     _builtin_torch_fn = torch.asin
#
#
# @QuantizationMixin.implements(Atan)
# class QuantizedAtan(_DispatchMixin, QuantizationMixin, Atan):
#     """ Quantized Atan """
#     _builtin_torch_fn = torch.atan
#
#
# @QuantizationMixin.implements(Round)
# class QuantizedRound(_DispatchMixin, QuantizationMixin, Round):
#     """ Quantized Round """
#     _builtin_torch_fn = torch.round
#
#
# @QuantizationMixin.implements(Gather)
# class QuantizedGather(_DispatchMixin, QuantizationMixin, Gather):
#     """ Quantized Gather """
#     _builtin_torch_fn = torch.gather
#
#
# @QuantizationMixin.implements(LogicalOr)
# class QuantizedLogicalOr(_DispatchMixin, QuantizationMixin, LogicalOr):
#     """ Quantized LogicalOr """
#     _builtin_torch_fn = torch.logical_or
#
#
# @QuantizationMixin.implements(LogicalAnd)
# class QuantizedLogicalAnd(_DispatchMixin, QuantizationMixin, LogicalAnd):
#     """ Quantized LogicalAnd """
#     _builtin_torch_fn = torch.logical_and
#
#
# @QuantizationMixin.implements(LogicalNot)
# class QuantizedLogicalNot(_DispatchMixin, QuantizationMixin, LogicalNot):
#     """ Quantized LogicalNot """
#     _builtin_torch_fn = torch.logical_not
#
#
# @QuantizationMixin.implements(Split)
# class QuantizedSplit(_DispatchMixin, QuantizationMixin, Split):
#     """ Quantized Split """
#     _builtin_torch_fn = torch.split
#
#
# @QuantizationMixin.implements(Permute)
# class QuantizedPermute(_DispatchMixin, QuantizationMixin, Permute):
#     """ Quantized Permute """
#     _builtin_torch_fn = torch.permute
#
#
# @QuantizationMixin.implements(Remainder)
# class QuantizedRemainder(_DispatchMixin, QuantizationMixin, Remainder):
#     """ Quantized Remainder """
#     _builtin_torch_fn = torch.remainder
#
#
# @QuantizationMixin.implements(IndexSelect)
# class QuantizedIndexSelect(_DispatchMixin, QuantizationMixin, IndexSelect):
#     """ Quantized IndexSelect """
#     _builtin_torch_fn = torch.index_select
#
#
# @QuantizationMixin.implements(Fmod)
# class QuantizedFmod(_DispatchMixin, QuantizationMixin, Fmod):
#     """ Quantized Fmod """
#     _builtin_torch_fn = torch.fmod
#
#
# @QuantizationMixin.implements(NonZero)
# class QuantizedNonZero(_DispatchMixin, QuantizationMixin, NonZero):
#     """ Quantized NonZero """
#     _builtin_torch_fn = torch.nonzero
#
#
# @QuantizationMixin.implements(TopK)
# class QuantizedTopK(_DispatchMixin, QuantizationMixin, TopK):
#     """ Quantized TopK """
#     _builtin_torch_fn = torch.topk
#
#
# @QuantizationMixin.implements(Shape)
# class QuantizedShape(_DispatchMixin, QuantizationMixin, Shape):
#     """ Quantized Shape """
#     _builtin_torch_fn = torch.Tensor.size
#
#
# @QuantizationMixin.implements(Tile)
# class QuantizedTile(_DispatchMixin, QuantizationMixin, Tile):
#     """ Quantized Tile """
#     _builtin_torch_fn = torch.tile

@QuantizationMixin.implements(ElementwiseUnarySign)
class QuantizedElementwiseUnarySign(_DispatchMixin, QuantizationMixin, ElementwiseUnarySign):
    """Quantized ElementwiseUnarySign (Sign function)"""
    
    _builtin_torch_fn = torch.sign


@QuantizationMixin.implements(Baddbmm)
class QuantizedBaddbmm(_DispatchMixin, QuantizationMixin, Baddbmm):
    """Quantized Baddbmm"""

    __quant_init__ = QuantizationMixin.__ternary__
    _builtin_torch_fn = torch.baddbmm


@QuantizationMixin.implements(Addmm)
class QuantizedAddmm(_DispatchMixin, QuantizationMixin, Addmm):
    """Quantized Addmm"""

    __quant_init__ = QuantizationMixin.__ternary__
    _builtin_torch_fn = torch.addmm


@QuantizationMixin.implements(RmsNorm)
class QuantizedRmsNorm(QuantizationMixin, RmsNorm):
    """Custom module for RmsNorm"""

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RmsNorm
        """
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            out = super().forward(x)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(HadamardRotation)
class QuantizedHadamardRotation(QuantizationMixin, HadamardRotation):
    """Custom module for HadamardRotation"""

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for HadamardRotation
        """
        # Quantize input tensors
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward(x)

        # Quantize output tensors
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret


# @QuantizationMixin.implements(Square)
# class QuantizedSquare(_DispatchMixin, QuantizationMixin, Square):
#     """ Quantized Square """
#     _builtin_torch_fn = torch.square
#
#
# @QuantizationMixin.implements(Select)
# class QuantizedSelect(_DispatchMixin, QuantizationMixin, Select):
#     """ Quantized Select """
#     _builtin_torch_fn = torch.select
#
#
#
# # modules for functional operations defined under torch.nn.functional package
# @QuantizationMixin.implements(Interpolate)
# class QuantizedInterpolate(_DispatchMixin, QuantizationMixin, Interpolate):
#     """ Quantized Interpolate """
#     _builtin_torch_fn = torch.nn.functional.interpolate
#
#
# @QuantizationMixin.implements(MaxPool2d)
# class QuantizedMaxPool2d(_DispatchMixin, QuantizationMixin, MaxPool2d):
#     """ Quantized MaxPool2d """
#     _builtin_torch_fn = torch.nn.functional.max_pool2d
#
#
# @QuantizationMixin.implements(AdaptiveAvgPool2d)
# class QuantizedAdaptiveAvgPool2d(_DispatchMixin, QuantizationMixin, AdaptiveAvgPool2d):
#     """ Quantized AdaptiveAvgPool2d """
#     _builtin_torch_fn = torch.nn.functional.adaptive_avg_pool2d
#
#
@QuantizationMixin.implements(BatchNorm)
class QuantizedBatchNorm(_DispatchMixin, QuantizationMixin, BatchNorm):
    """Quantized BatchNorm"""

    _builtin_torch_fn = torch.nn.functional.batch_norm

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None, None, None])

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        # pylint: disable=redefined-builtin
        def batch_norm_wrapper(
            input: Tensor,
            running_mean: Optional[Tensor],
            running_var: Optional[Tensor],
            weight: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            training: bool = False,
            momentum: float = 0.1,
            eps: float = 1e-5,
        ) -> Tensor:
            if training:
                if (
                    self.input_quantizers[1] is not None
                    or self.input_quantizers[2] is not None
                ):
                    raise RuntimeError(
                        f"{self.__class__} doesn't support quantizing running_mean or running_var in training mode"
                    )

            input = _quantize_dequantize_if_applicable(input, self.input_quantizers[0])
            running_mean = _quantize_dequantize_if_applicable(
                running_mean, self.input_quantizers[1]
            )
            running_var = _quantize_dequantize_if_applicable(
                running_var, self.input_quantizers[2]
            )
            weight = _quantize_dequantize_if_applicable(
                weight, self.input_quantizers[3]
            )
            bias = _quantize_dequantize_if_applicable(bias, self.input_quantizers[4])

            # PyTorch doesn't support gradient calculation of running_mean/var
            output = fn(
                input,
                running_mean.detach(),
                running_var.detach(),
                weight,
                bias,
                training,
                momentum,
                eps,
            )

            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return batch_norm_wrapper

    def _custom_kernel_helper(self, fn: Callable[..., Tensor]):
        # pylint: disable=redefined-builtin
        def batch_norm_wrapper(
            input: Tensor,
            running_mean: Optional[Tensor],
            running_var: Optional[Tensor],
            weight: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            training: bool = False,
            momentum: float = 0.1,
            eps: float = 1e-5,
        ) -> Tensor:
            if training:
                if (
                    self.input_quantizers[1] is not None
                    or self.input_quantizers[2] is not None
                ):
                    raise RuntimeError(
                        f"{self.__class__} doesn't support quantizing running_mean or running_var in training mode"
                    )

            input = _quantize_if_applicable(input, self.input_quantizers[0])
            running_mean = _quantize_if_applicable(
                running_mean, self.input_quantizers[1]
            )
            running_var = _quantize_if_applicable(running_var, self.input_quantizers[2])
            weight = _quantize_if_applicable(weight, self.input_quantizers[3])
            bias = _quantize_if_applicable(bias, self.input_quantizers[4])

            # PyTorch doesn't support gradient calculation of running_mean/var
            output = fn(
                input,
                running_mean.detach(),
                running_var.detach(),
                weight,
                bias,
                training,
                momentum,
                eps,
            )
            return _quantize_if_applicable(output, self.output_quantizers[0])

        return batch_norm_wrapper


@QuantizationMixin.implements(GroupNorm)
class QuantizedGroupNorm(_DispatchMixin, QuantizationMixin, GroupNorm):
    """Quantized GroupNorm"""

    _builtin_torch_fn = F.group_norm

    def _is_dynamo_traceable(self):
        # F.group_norm isn't dynamo-traceable
        return False


@QuantizationMixin.implements(Normalize)
class QuantizedNormalize(_DispatchMixin, QuantizationMixin, Normalize):
    """Quantized Normalize"""

    _builtin_torch_fn = torch.nn.functional.normalize


@QuantizationMixin.implements(NullRequant)
class QuantizedNullRequant(QuantizationMixin, NullRequant):
    """Quantized module for NullRequant"""

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor, shape: list) -> torch.Tensor:
        """
        Forward pass for NullRequant
        """
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            out = super().forward(x, shape)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


@QuantizationMixin.implements(Pad)
class QuantizedPad(_DispatchMixin, QuantizationMixin, Pad):
    """ Quantized Pad """
    _builtin_torch_fn = torch.nn.functional.pad


@QuantizationMixin.implements(GridSample)
class QuantizedGridSample(_DispatchMixin, QuantizationMixin, GridSample):
    """Quantized GridSample"""

    __quant_init__ = QuantizationMixin.__binary__
    _builtin_torch_fn = torch.nn.functional.grid_sample


@QuantizationMixin.implements(Snake2d)
class QuantizedSnake2d(QuantizationMixin, Snake2d):
    """Quantized Snake2d activation"""

    def __quant_init__(self):
        super().__quant_init__()
        # input_quantizers[0]: None 占位，对应输入 x（可由 apply_mixed_precision_bitwidth 创建）
        # input_quantizers[1]: None，不量化 alpha（alpha 由父 Module 的 param_quantizer 处理）
        self.input_quantizers = nn.ModuleList([None, None])

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        input_qtzr = self.input_quantizers[0] if self.input_quantizers else None
        output_qtzr = self.output_quantizers[0] if self.output_quantizers else None

        if input_qtzr:
            x = input_qtzr(x)

        out = super().forward(x, alpha)

        if output_qtzr:
            out = output_qtzr(out)

        return out


# @QuantizationMixin.implements(DynamicConv2d)
# class QuantizedDynamicConv2d(QuantizationMixin, DynamicConv2d):
#     """ Quantized DynamicConv2d """
#
#
@QuantizationMixin.implements(Pow)
class QuantizedPow(QuantizationMixin, Pow):
    """Quantized Pow"""

    __quant_init__ = QuantizationMixin.__binary__
    
    def forward(self, x, exponent):  # pylint: disable=arguments-differ
        """Quantized forward for Pow operation"""
        # Get quantizers
        input_qtzr_x = self.input_quantizers[0] if self.input_quantizers else None
        input_qtzr_exp = self.input_quantizers[1] if len(self.input_quantizers) > 1 else None
        output_qtzr = self.output_quantizers[0] if self.output_quantizers else None
        
        # Quantize inputs
        if input_qtzr_x:
            x = input_qtzr_x(x)
        
        # Only quantize exponent if it's a tensor
        if input_qtzr_exp and isinstance(exponent, torch.Tensor):
            exponent = input_qtzr_exp(exponent)
        
        # Compute
        out = torch.pow(x, exponent)
        
        # Quantize output
        if output_qtzr:
            out = output_qtzr(out)
        
        return out


@QuantizationMixin.implements(CustomSiLU)
class QuantizedCustomSiLU(QuantizationMixin, CustomSiLU):
    """Quantized CustomSiLU"""

    __quant_init__ = QuantizationMixin.__unary__

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        (input_qtzr,) = self.input_quantizers
        (output_qtzr,) = self.output_quantizers

        if input_qtzr:
            x = input_qtzr(x)

        out = super().forward(x)

        if output_qtzr:
            out = output_qtzr(out)

        return out


class _NativeRecurrentQuantizationMixin(QuantizationMixin):
    """Pass through native execution and keep generic quantizers disabled."""

    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None, None])
        self.param_quantizers = nn.ModuleDict({
            name: None for name, _ in self.named_parameters(recurse=False)
        })

    def forward(self, input: torch.Tensor, hx=None):
        return super().forward(input, hx)


for _operator in recurrent_operators():
    _name = "Quantized" + _operator.module_type.__name__
    _wrapper = type(_name, (_NativeRecurrentQuantizationMixin, _operator.module_type),
                    {"__module__": __name__})
    globals()[_name] = QuantizationMixin.implements(_operator.module_type)(_wrapper)


# @QuantizationMixin.implements(StridedSlice)
# class QuantizedStridedSlice(QuantizationMixin, StridedSlice):
#     """ Quantized StridedSlice """
#
#
# @QuantizationMixin.implements(ChannelShuffle)
# class QuantizedChannelShuffle(QuantizationMixin, ChannelShuffle):
#     """ Quantized ChannelShuffle """
#
#
# @QuantizationMixin.implements(Cast)
# class QuantizedCast(QuantizationMixin, Cast):
#     """ Quantized Cast """
#
#
# @QuantizationMixin.implements(CustomGather)
# class QuantizedCustomGather(QuantizationMixin, CustomGather):
#     """ Quantized CustomGather """
#
#
# @QuantizationMixin.implements(DepthToSpaceCRDMode)
# class QuantizedDepthToSpaceCRDMode(QuantizationMixin, DepthToSpaceCRDMode):
#     """ Quantized DepthToSpaceCRDMode """
#
#
# @QuantizationMixin.implements(DepthToSpaceDCRMode)
# class QuantizedDepthToSpaceDCRMode(QuantizationMixin, DepthToSpaceDCRMode):
#     """ Quantized DepthToSpaceDCRMode """
#
#
# @QuantizationMixin.implements(CustomSparseConv3DLayer)
# class QuantizedCustomSparseConv3DLayer(QuantizationMixin, CustomSparseConv3DLayer):
#     """ Quantized CustomSparseConv3DLayer """
#
#
# @QuantizationMixin.implements(SparseTensorWrapper)
# class QuantizedSparseTensorWrapper(QuantizationMixin, SparseTensorWrapper):
#     """ Quantized SparseTensorWrapper """
#
#
# @QuantizationMixin.implements(ScatterDense)
# class QuantizedScatterDense(QuantizationMixin, ScatterDense):
#     """ Quantized ScatterDense """
#
#
# @QuantizationMixin.implements(ScatterND)
# class QuantizedScatterND(QuantizationMixin, ScatterND):
#     """ Quantized ScatterND """
#
#
# @QuantizationMixin.implements(RoiAlign)
# class QuantizedRoiAlign(QuantizationMixin, RoiAlign):
#     """ Quantized RoiAlign """
#
#
# @QuantizationMixin.implements(NonMaxSuppression)
# class QuantizedNonMaxSuppression(QuantizationMixin, NonMaxSuppression):
#     """ Quantized NonMaxSuppression """
#
#
# @QuantizationMixin.implements(GatherNd)
# class QuantizedGatherNd(QuantizationMixin, GatherNd):
#     """ Quantized GatherNd """
#
#
# @QuantizationMixin.implements(ScatterElements)
# class QuantizedScatterElements(QuantizationMixin, ScatterElements):
#     """ Quantized ScatterElements """
#
#
# @QuantizationMixin.implements(OneHot)
# class QuantizedOneHot(QuantizationMixin, OneHot):
#     """ Quantized OneHot """
#
#
# @QuantizationMixin.implements(Expand)
# class QuantizedExpand(QuantizationMixin, Expand):
#     """ Quantized Expand """
#
#
# @QuantizationMixin.implements(DynamicLinear)
# class QuantizedDynamicLinear(QuantizationMixin, DynamicLinear):
#     """ Quantized DynamicLinear """
