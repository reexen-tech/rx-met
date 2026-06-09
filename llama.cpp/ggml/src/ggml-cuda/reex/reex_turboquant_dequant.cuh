// SPDX-License-Identifier: MIT
//
// reex_turboquant_dequant.cuh — REEX_TURBOQUANT only.
//
// CUDA-side dequantization primitives for GGML_TYPE_TQ_K3 / TQ_V2 / TQ_V4.
//
// Layered:
//   - per-element __device__ __forceinline__ helpers (used by getrows / FA / mmvf)
//   - __global__ block-level kernels (used by ggml_get_to_fp32_cuda)
//   - host-callable launchers (called from getrows.cu / convert.cu)
//
// All bit decoding logic must remain bit-identical to the CPU reference path
// in ggml/src/reex_turboquant_quants_ref.c (`dequantize_row_tq_*`).
//
// Codebook values mirror ggml/src/ggml-cpu/reex/reex_turboquant_codebook_tq_k3.inc;
// the static_assert at the bottom of the .cu file ensures they stay in sync.
//
// P4 A2.2 K3 P3 upgrade (TheTom turbo3_0 path, 2026-04-30):
//   idx (3-bit) = low_bits | (high_bit << 2),
//     where low_bits = (qs[j>>2] >> (2*(j&3))) & 0x3
//           high_bit = (signs[j>>3] >> (j&7)) & 0x1
//   value = centroids[idx] * b->norm  (b->norm holds the corrected norm
//                                      ‖x‖/‖recon_unit‖, see plan doc §0.3)
// No residual-sign term anymore; `per_elt_residual_unit` is gone.

#pragma once

#ifdef REEX_TURBOQUANT

#include "../common.cuh"
#include "../convert.cuh"

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Per-element decode helpers (__device__ __forceinline__).
//
// Mirror ggml/src/reex_turboquant_quants_ref.c::dequantize_row_tq_* one-element-
// at-a-time. `centroids` is an 8-entry table (K_BITS=3 full PolarQuant); the
// caller is responsible for supplying it from __constant__ memory so the
// broadcast load is shared across the warp.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float reex_dequant_tq_k3_elt(
        const block_tq_k3 * __restrict__ b, int j,
        const float * __restrict__ centroids) {
    const float corr     = __half2float(b->norm);                       // corrected norm
    const int   low_bits = (b->qs   [j >> 2] >> ((j & 0x3) * 2)) & 0x3;
    const int   high_bit = (b->signs[j >> 3] >> ( j & 0x7))      & 0x1;
    const int   idx      = low_bits | (high_bit << 2);
    return centroids[idx] * corr;
}

__device__ __forceinline__ float reex_dequant_tq_v2_elt(
        const block_tq_v2 * __restrict__ b, int j) {
    const float scale = __half2float(b->scale);
    const float zero  = __half2float(b->zero);
    const int   q     = (b->qs[j >> 2] >> ((j & 0x3) * 2)) & 0x3;
    return (float) q * scale + zero;
}

__device__ __forceinline__ float reex_dequant_tq_v4_elt(
        const block_tq_v4 * __restrict__ b, int j) {
    const float scale = __half2float(b->scale);
    const float zero  = __half2float(b->zero);
    const int   q     = (b->qs[j >> 1] >> ((j & 0x1) * 4)) & 0xF;
    return (float) q * scale + zero;
}

// P4 A2.5 V-β (TheTom turbo2_0 / turbo4_0 algorithmic alignment):
//   v = centroids[idx] * corrected_norm
// where corrected_norm = ‖x‖ / ‖recon_unit‖ is stored in `b->norm` (same
// idempotency trick as K3).  block_tq_v_polar4 carries a reserved `rnorm`
// field for byte-level TheTom alignment — V-β does not use it.

__device__ __forceinline__ float reex_dequant_tq_v_polar2_elt(
        const block_tq_v_polar2 * __restrict__ b, int j,
        const float * __restrict__ centroids) {
    const float corr = __half2float(b->norm);                       // corrected norm
    const int   idx  = (b->qs[j >> 2] >> ((j & 0x3) * 2)) & 0x3;
    return centroids[idx] * corr;
}

__device__ __forceinline__ float reex_dequant_tq_v_polar4_elt(
        const block_tq_v_polar4 * __restrict__ b, int j,
        const float * __restrict__ centroids) {
    const float corr = __half2float(b->norm);                       // corrected norm
    const int   idx  = (b->qs[j >> 1] >> ((j & 0x1) * 4)) & 0xF;
    return centroids[idx] * corr;
}

// ---------------------------------------------------------------------------
// Host-callable launchers.
//
// `k` is the total number of dequantized elements (must be a multiple of QK_*).
// `dst` is a contiguous float / half / bfloat16 buffer of length `k`.
// ---------------------------------------------------------------------------

template <typename dst_t>
void reex_turboquant_to_t_tq_k3_cuda(
        const void * __restrict__ src,
        dst_t      * __restrict__ dst,
        int64_t                   k,
        cudaStream_t              stream);

template <typename dst_t>
void reex_turboquant_to_t_tq_v2_cuda(
        const void * __restrict__ src,
        dst_t      * __restrict__ dst,
        int64_t                   k,
        cudaStream_t              stream);

template <typename dst_t>
void reex_turboquant_to_t_tq_v4_cuda(
        const void * __restrict__ src,
        dst_t      * __restrict__ dst,
        int64_t                   k,
        cudaStream_t              stream);

template <typename dst_t>
void reex_turboquant_to_t_tq_v_polar2_cuda(
        const void * __restrict__ src,
        dst_t      * __restrict__ dst,
        int64_t                   k,
        cudaStream_t              stream);

template <typename dst_t>
void reex_turboquant_to_t_tq_v_polar4_cuda(
        const void * __restrict__ src,
        dst_t      * __restrict__ dst,
        int64_t                   k,
        cudaStream_t              stream);

// ---------------------------------------------------------------------------
// GET_ROWS launchers.
//
// Mirror the signature of `get_rows_cuda_q<>` in getrows.cu. Each ggml row is
// `ne00` elements wide and lives at byte offset `i01*nb01 + i11*nb02 + i12*nb03`
// inside the source buffer; rows to extract come from the I32 index tensor.
// ---------------------------------------------------------------------------

template <typename dst_t>
void reex_turboquant_get_rows_tq_k3_cuda(
        const void *  src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00,  size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10,  int64_t ne11, int64_t ne12,
        size_t  nb10,  size_t nb11, size_t nb12,
        size_t  nb1,   size_t nb2,  size_t nb3,
        cudaStream_t stream);

template <typename dst_t>
void reex_turboquant_get_rows_tq_v2_cuda(
        const void *  src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00,  size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10,  int64_t ne11, int64_t ne12,
        size_t  nb10,  size_t nb11, size_t nb12,
        size_t  nb1,   size_t nb2,  size_t nb3,
        cudaStream_t stream);

template <typename dst_t>
void reex_turboquant_get_rows_tq_v4_cuda(
        const void *  src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00,  size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10,  int64_t ne11, int64_t ne12,
        size_t  nb10,  size_t nb11, size_t nb12,
        size_t  nb1,   size_t nb2,  size_t nb3,
        cudaStream_t stream);

template <typename dst_t>
void reex_turboquant_get_rows_tq_v_polar2_cuda(
        const void *  src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00,  size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10,  int64_t ne11, int64_t ne12,
        size_t  nb10,  size_t nb11, size_t nb12,
        size_t  nb1,   size_t nb2,  size_t nb3,
        cudaStream_t stream);

template <typename dst_t>
void reex_turboquant_get_rows_tq_v_polar4_cuda(
        const void *  src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00,  size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10,  int64_t ne11, int64_t ne12,
        size_t  nb10,  size_t nb11, size_t nb12,
        size_t  nb1,   size_t nb2,  size_t nb3,
        cudaStream_t stream);

#endif  // REEX_TURBOQUANT
