"""Single-linear GPTQ using the physical Q4_0_64 runtime grid."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .q4_0_64 import (
    GROUP_SIZE,
    dequantize_q4_0_64,
    find_scales,
    fixed_point_mask,
    pack_q4_0_64,
    quantize_q4_0_64,
    quantize_with_scales,
)


@dataclass(frozen=True)
class GPTQResult:
    dequant: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor
    packed: torch.Tensor
    fixed_point: torch.Tensor
    loss: float
    dead_columns: int
    damp: float
    method: str
    fallback_reason: str | None


class GPTQQ4064:
    """Accumulate a full Hessian and quantize one ``nn.Linear`` in place."""

    def __init__(self, layer: nn.Linear):
        if not isinstance(layer, nn.Linear):
            raise TypeError("GPTQQ4064 only supports torch.nn.Linear")
        if layer.in_features % GROUP_SIZE:
            raise ValueError(
                f"in_features={layer.in_features} is not divisible by {GROUP_SIZE}"
            )
        self.layer = layer
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
    def _rtn_fallback(self, reason: str, *, damp: float = 0.0) -> GPTQResult:
        quantized = quantize_q4_0_64(self.layer.weight.detach())
        self.layer.weight.copy_(quantized.dequant.to(self.layer.weight.dtype))
        return GPTQResult(
            dequant=quantized.dequant.to(torch.float32),
            codes=quantized.codes,
            scales=quantized.scales,
            packed=quantized.packed,
            fixed_point=fixed_point_mask(quantized.codes, quantized.scales),
            loss=0.0,
            dead_columns=0,
            damp=damp,
            method="rtn",
            fallback_reason=reason,
        )

    @torch.no_grad()
    def fasterquant(
        self,
        *,
        lazy_block_size: int = 128,
        damp_percent: float = 0.01,
    ) -> GPTQResult:
        if lazy_block_size <= 0 or lazy_block_size % GROUP_SIZE:
            raise ValueError("lazy_block_size must be a positive multiple of 64")
        if damp_percent < 0:
            raise ValueError("damp_percent must be non-negative")
        if self.n_tokens == 0:
            return self._rtn_fallback("zero_samples")

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
        except torch.linalg.LinAlgError:
            diag = torch.diagonal(H)
            reason = (
                "cholesky_failure: "
                f"K={self.columns}, damp={float(damp):.6g}, "
                f"diag_min={float(diag.min()):.6g}, "
                f"diag_max={float(diag.max()):.6g}"
            )
            return self._rtn_fallback(reason, damp=float(damp))

        Q = torch.zeros_like(W)
        codes = torch.zeros_like(W, dtype=torch.int8)
        scales = torch.empty(
            (self.rows, self.columns // GROUP_SIZE),
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
                group_index = global_i // GROUP_SIZE
                if global_i % GROUP_SIZE == 0:
                    group_end = global_i + GROUP_SIZE
                    if group_end > block_end:
                        raise RuntimeError("a physical group crosses a lazy block")
                    _, stored_scale = find_scales(
                        W1[:, local_i : local_i + GROUP_SIZE]
                    )
                    scales[:, group_index] = stored_scale.squeeze(-1)

                w = W1[:, local_i]
                q_code = quantize_with_scales(w, scales[:, group_index])
                q = (
                    q_code.to(torch.float32)
                    * scales[:, group_index].to(torch.float32)
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

        dequant = dequantize_q4_0_64(codes, scales)
        if not torch.equal(Q, dequant):
            raise RuntimeError("internal Q4_0_64 dequantization mismatch")
        self.layer.weight.copy_(dequant.to(self.layer.weight.dtype))
        packed = pack_q4_0_64(codes, scales)
        fixed_point = fixed_point_mask(codes, scales)
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

    def free(self) -> None:
        self.H_sum = None  # type: ignore[assignment]
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
