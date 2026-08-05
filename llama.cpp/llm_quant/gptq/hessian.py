"""Single-linear GPTQ using a physical Q64 runtime grid."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .formats import Q4_0_64, Q4_1_64, Q8_0_64, Q64Format, format_for_bits


@dataclass(frozen=True)
class GPTQResult:
    dequant: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor
    packed: torch.Tensor
    fixed_point: torch.Tensor
    loss: float | None
    dead_columns: int | None
    damp: float | None
    method: str
    fallback_reason: str | None


class GPTQQuantizer:
    """Accumulate a full Hessian and quantize one ``nn.Linear`` in place."""

    def __init__(
        self,
        layer: nn.Linear,
        *,
        block_format: Q64Format,
        name: str | None = None,
    ):
        if not isinstance(layer, nn.Linear):
            raise TypeError("GPTQ only supports torch.nn.Linear")
        if layer.in_features % block_format.group_size:
            raise ValueError(
                f"in_features={layer.in_features} is not divisible by "
                f"{block_format.group_size}"
            )
        self.layer = layer
        self.block_format = block_format
        self.name = name or "<unnamed>"
        self.device = layer.weight.device
        self.columns = layer.in_features
        self.rows = layer.out_features
        self.H_sum = torch.zeros(
            (self.columns, self.columns), dtype=torch.float32, device=self.device
        )
        self.n_tokens = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.shape[-1] != self.columns:
            raise ValueError(
                f"expected input width {self.columns}, got {inp.shape[-1]}"
            )
        x = inp.detach().reshape(-1, self.columns).to(
            device=self.device, dtype=torch.float32
        )
        if not torch.isfinite(x).all():
            raise ValueError("calibration input contains NaN or Inf")
        self.H_sum.addmm_(x.transpose(0, 1), x)
        self.n_tokens += x.shape[0]

    @torch.no_grad()
    def fasterquant(
        self,
        *,
        lazy_block_size: int = 128,
        damp_percent: float = 0.01,
    ) -> GPTQResult:
        group_size = self.block_format.group_size
        if lazy_block_size <= 0 or lazy_block_size % group_size:
            raise ValueError(
                f"lazy_block_size must be a positive multiple of {group_size}"
            )
        if damp_percent < 0:
            raise ValueError("damp_percent must be non-negative")
        if self.n_tokens == 0:
            raise RuntimeError(
                f"GPTQ calibration has zero samples for {self.name}"
            )

        W = self.layer.weight.detach().to(torch.float32).clone()
        H = self.H_sum * (2.0 / self.n_tokens)
        if not torch.isfinite(W).all() or not torch.isfinite(H).all():
            raise ValueError("weight or Hessian contains NaN or Inf")

        diagonal = torch.diagonal(H)
        dead = diagonal == 0
        dead_columns = int(dead.sum().item())
        if dead_columns:
            indices = torch.arange(self.columns, device=self.device)
            H[indices[dead], indices[dead]] = 1
            W[:, dead] = 0

        damp = damp_percent * torch.diagonal(H).mean()
        indices = torch.arange(self.columns, device=self.device)
        H[indices, indices] += damp
        try:
            chol = torch.linalg.cholesky(H)
            inverse = torch.cholesky_inverse(chol)
            Hinv = torch.linalg.cholesky(inverse, upper=True)
        except torch.linalg.LinAlgError as exc:
            diag = torch.diagonal(H)
            raise RuntimeError(
                f"GPTQ Cholesky failed for {self.name}: "
                f"K={self.columns}, damp={float(damp):.6g}, "
                f"diag_min={float(diag.min()):.6g}, "
                f"diag_max={float(diag.max()):.6g}"
            ) from exc

        Q = torch.zeros_like(W)
        codes = torch.zeros_like(W, dtype=torch.int8)
        scale_shape = (self.rows, self.columns // group_size)
        if self.block_format.parameter_count > 1:
            scale_shape += (self.block_format.parameter_count,)
        scales = torch.empty(
            scale_shape,
            dtype=torch.float16,
            device=self.device,
        )
        losses = torch.zeros_like(W)

        for block_start in range(0, self.columns, lazy_block_size):
            block_end = min(block_start + lazy_block_size, self.columns)
            count = block_end - block_start
            W1 = W[:, block_start:block_end].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[block_start:block_end, block_start:block_end]

            for local_i in range(count):
                global_i = block_start + local_i
                group_index = global_i // group_size
                if global_i % group_size == 0:
                    group_end = global_i + group_size
                    if group_end > block_end:
                        raise RuntimeError("a physical group crosses a lazy block")
                    _, stored_scales = self.block_format.find_scales(
                        W1[:, local_i : local_i + group_size]
                    )
                    if self.block_format.parameter_count == 1:
                        stored_scales = stored_scales.squeeze(-1)
                    scales[:, group_index] = stored_scales

                w = W1[:, local_i]
                group_scales = scales[:, group_index]
                q_code = self.block_format.quantize_with_scales(
                    w, group_scales
                )
                q = self.block_format.dequantize_with_scales(
                    q_code, group_scales
                )
                sensitivity = Hinv1[local_i, local_i]
                if not torch.isfinite(sensitivity) or sensitivity == 0:
                    raise RuntimeError(
                        f"invalid inverse-Hessian diagonal at column {global_i}"
                    )
                err = (w - q) / sensitivity
                W1[:, local_i:] -= torch.outer(
                    err, Hinv1[local_i, local_i:]
                )
                Q1[:, local_i] = q
                codes[:, global_i] = q_code
                losses[:, global_i] = (w - q).square() / (sensitivity * sensitivity)
                Err1[:, local_i] = err

            Q[:, block_start:block_end] = Q1
            W[:, block_end:] -= Err1 @ Hinv[block_start:block_end, block_end:]

        dequant = self.block_format.dequantize(codes, scales)
        if not torch.equal(Q, dequant):
            raise RuntimeError(
                f"internal {self.block_format.name} dequantization mismatch"
            )
        self.layer.weight.copy_(dequant.to(self.layer.weight.dtype))
        packed = self.block_format.pack(codes, scales)
        fixed_point = self.block_format.fixed_point_mask(codes, scales)
        return GPTQResult(
            dequant=dequant,
            codes=codes,
            scales=scales,
            packed=packed,
            fixed_point=fixed_point,
            loss=float((losses / 2).sum().item()),
            dead_columns=dead_columns,
            damp=float(damp.item()),
            method="gptq",
            fallback_reason=None,
        )

    @torch.no_grad()
    def rtn_fallback(self, *, reason: str) -> GPTQResult:
        """Use reference RTN when GPTQ has no calibration observations."""

        result = self.block_format.quantize(self.layer.weight.detach())
        self.layer.weight.copy_(result.dequant.to(self.layer.weight.dtype))
        return GPTQResult(
            dequant=result.dequant,
            codes=result.codes,
            scales=result.scales,
            packed=result.packed,
            fixed_point=self.block_format.fixed_point_mask(
                result.codes, result.scales
            ),
            loss=None,
            dead_columns=None,
            damp=None,
            method="rtn",
            fallback_reason=reason,
        )

    def free(self) -> None:
        self.H_sum = None  # type: ignore[assignment]
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class GPTQQ4064(GPTQQuantizer):
    def __init__(self, layer: nn.Linear, *, name: str | None = None):
        super().__init__(layer, block_format=Q4_0_64, name=name)


class GPTQQ4164(GPTQQuantizer):
    def __init__(self, layer: nn.Linear, *, name: str | None = None):
        super().__init__(layer, block_format=Q4_1_64, name=name)


class GPTQQ8064(GPTQQuantizer):
    def __init__(self, layer: nn.Linear, *, name: str | None = None):
        super().__init__(layer, block_format=Q8_0_64, name=name)


def quantizer_class_for_format(
    block_format: Q64Format,
) -> type[GPTQQuantizer]:
    return {
        "Q4_0_64": GPTQQ4064,
        "Q4_1_64": GPTQQ4164,
        "Q8_0_64": GPTQQ8064,
    }[block_format.name]


def quantizer_class_for_bits(bits: int) -> type[GPTQQuantizer]:
    return quantizer_class_for_format(format_for_bits(bits))
