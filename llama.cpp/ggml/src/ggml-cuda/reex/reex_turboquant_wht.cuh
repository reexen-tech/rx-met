// SPDX-License-Identifier: MIT
//
// reex_turboquant_wht.cuh — REEX_TURBOQUANT only.
//
// CUDA implementation of GGML_OP_REEX_WHT — the structured Walsh–Hadamard
// rotation used by TurboQuant.  Mirrors `ggml/src/ggml-cpu/reex/reex_turboquant_wht.c`
// bit-for-bit (same sign tables, same (1/sqrt(d)) normalisation, same optional
// per-channel `scale_inv` multiplier applied AFTER the global scale).
//
// One CUDA block processes one row of length `dim`; dim ∈ {64, 128}.  The
// sign tables live in `__constant__` memory so they're cached on-chip and
// shared across all rows.  scale_inv (when present) is read from global
// memory in a strided coalesced pattern.
//
// Public entry point:
//   reex_turboquant_cuda_wht_apply_rows(...)
// is called from ggml-cuda.cu inside the GGML_OP_REEX_WHT compute case.

#pragma once

#ifdef REEX_TURBOQUANT

#include "../common.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

// `direction`:
//   0 = forward   (s1 → butterfly → s2)
//   1 = inverse   (s2 → butterfly → s1)
//
// Input/output buffers are contiguous F32; in-place is allowed (`x_d == y_d`).
// `scale_inv_d` is either NULL (skip) or a `dim`-long F32 buffer.
// `sq_accum_d` is either NULL (skip) or a `dim`-long F64 buffer that the
// kernel atomically accumulates per-channel `(double)y * (double)y` into,
// matching the CPU path's `+= (double)y * (double)y` post-rotation
// semantics.  Required for InnerQ EMA calibration on the CUDA backend
// (P4 A2.4 Stage 4).
//
// The launcher routes between dim==64 / dim==128 specialisations.  Other
// dims abort exactly like the CPU path.
void reex_turboquant_cuda_wht_apply_rows(
        const float * x_d,
        float       * y_d,
        int64_t       rows,
        int64_t       dim,
        int           direction,
        const float * scale_inv_d,
        double      * sq_accum_d,
        cudaStream_t  stream);

void ggml_cuda_op_reex_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#endif  // REEX_TURBOQUANT
