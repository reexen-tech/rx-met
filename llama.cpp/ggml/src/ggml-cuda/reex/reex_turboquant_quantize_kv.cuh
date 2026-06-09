// SPDX-License-Identifier: MIT
//
// reex_turboquant_quantize_kv.cuh — REEX_TURBOQUANT only.
//
// CUDA-side per-block F32 -> TQ_K3 / TQ_V2 / TQ_V4 quantizers.
//
// These mirror ggml/src/reex_turboquant_quants_ref.c::quantize_row_tq_*_ref
// one block at a time, matching the cpy-utils.cuh / set-rows.cu pattern of
// `void quantize_f32_<type>_block(const float *, block_<type> *)`.
//
// The launcher wrappers (cpy.cu / set-rows.cu) take care of slicing the
// flattened nelems / nrows-by-nblocks_per_row index space; here we focus
// purely on encoding QK_TQ_* contiguous floats into a single ggml block.
//
// All math must remain bit-identical to the CPU `*_ref` routines so the
// CPU-vs-GPU correctness tests (test-backend-ops CPY/SET_ROWS) compare
// element-by-element after a F32->TQ->F32 round trip.
//
// Note: `block_tq_*` and `QK_TQ_*` are visible because the including TUs
// pull in ggml-common.h via common.cuh under GGML_COMMON_DECL_CUDA.

#pragma once

#ifdef REEX_TURBOQUANT

#include "../common.cuh"

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// TQ_K3 codebook tables — mirror of
// reex_turboquant_codebook_tq_k3_centroids[8] / _decision_boundaries[7]
// in the .inc file (see reex_turboquant_dequant.cu for the matching
// __constant__ array on the dequant side).
//
// P4 A2.2 K3 P3 upgrade (TheTom turbo3_0): full 3-bit PolarQuant + norm
// correction, no QJL split.  Both centroids and boundaries are needed at
// quantize time because we (1) decide the bucket via lower_bound on the 7
// boundaries, and (2) accumulate Σ centroid² for the norm-correction trick.
// ---------------------------------------------------------------------------

__device__ static const float reex_quantize_tq_k3_centroids[8] = {
    -0.1861817092f, -0.1154012680f, -0.0635419339f, -0.0184134189f,
     0.0247995369f,  0.0696418509f,  0.1208978295f,  0.1906363368f,
};
__device__ static const float reex_quantize_tq_k3_boundaries[7] = {
    -0.1507914960f, -0.0894716009f, -0.0409776755f, 0.0031930597f,
     0.0472206958f,  0.0952698439f,  0.1557670832f,
};

// ---------------------------------------------------------------------------
// quantize_f32_tq_k3_block — bit-identical to
// ggml/src/reex_turboquant_quants_ref.c::quantize_row_tq_k3_ref one block at
// a time (TheTom turbo3_0 path).  Idempotent: a second pass produces the
// same indices and the same `corrected` norm because ‖dequant‖ ≡ ‖x‖.
// ---------------------------------------------------------------------------

static __device__ void quantize_f32_tq_k3_block(
        const float * __restrict__ x, block_tq_k3 * __restrict__ y) {
    float n2 = 0.0f;
#pragma unroll
    for (int j = 0; j < QK_TQ_K3; ++j) {
        n2 += x[j] * x[j];
    }
    const float grp_norm = sqrtf(n2 > 1e-12f ? n2 : 1e-12f);
    const float inv_norm = 1.0f / grp_norm;

#pragma unroll
    for (int j = 0; j < QK_TQ_K3 / 4; ++j) y->qs[j]    = 0;
#pragma unroll
    for (int j = 0; j < QK_TQ_K3 / 8; ++j) y->signs[j] = 0;

    float recon_sq = 0.0f;
#pragma unroll
    for (int j = 0; j < QK_TQ_K3; ++j) {
        const float u = x[j] * inv_norm;

        int idx = 0;
        if (u >= reex_quantize_tq_k3_boundaries[0]) idx = 1;
        if (u >= reex_quantize_tq_k3_boundaries[1]) idx = 2;
        if (u >= reex_quantize_tq_k3_boundaries[2]) idx = 3;
        if (u >= reex_quantize_tq_k3_boundaries[3]) idx = 4;
        if (u >= reex_quantize_tq_k3_boundaries[4]) idx = 5;
        if (u >= reex_quantize_tq_k3_boundaries[5]) idx = 6;
        if (u >= reex_quantize_tq_k3_boundaries[6]) idx = 7;

        y->qs[j >> 2] |= (uint8_t)((idx & 0x3) << ((j & 0x3) * 2));
        if (idx & 0x4) {
            y->signs[j >> 3] |= (uint8_t)(1u << (j & 0x7));
        }

        const float c = reex_quantize_tq_k3_centroids[idx];
        recon_sq += c * c;
    }

    // Norm-correction trick: store ‖x‖ / ‖recon_unit‖ so dequant ‖y‖ ≡ ‖x‖
    // and a second quantize pass lands on exactly the same indices (drift = 0).
    const float recon_norm = sqrtf(recon_sq > 1e-20f ? recon_sq : 1e-20f);
    const float corrected  = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;
    y->norm = __float2half(corrected);
}

// ---------------------------------------------------------------------------
// quantize_f32_tq_v2_block / quantize_f32_tq_v4_block
//
// Per-block linear quantization. Mirror of reex_turboquant_quants_ref.c.
// ---------------------------------------------------------------------------

template <int BITS, typename block_t, int QK>
static __device__ void quantize_f32_tq_v_block_impl(
        const float * __restrict__ x, block_t * __restrict__ y) {
    constexpr int LEVELS = (1 << BITS) - 1;

    float xmin = +FLT_MAX, xmax = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < QK; ++j) {
        const float v = x[j];
        if (v < xmin) xmin = v;
        if (v > xmax) xmax = v;
    }

    float scale = (xmax - xmin) / (float) LEVELS;
    if (scale < 1e-30f) scale = 1.0f;
    const float zero      = xmin;
    const float inv_scale = 1.0f / scale;

    y->scale = __float2half(scale);
    y->zero  = __float2half(zero);

    if constexpr (BITS == 2) {
#pragma unroll
        for (int j = 0; j < QK / 4; ++j) y->qs[j] = 0;
#pragma unroll
        for (int j = 0; j < QK; ++j) {
            int q = (int) roundf((x[j] - zero) * inv_scale);
            if (q < 0)      q = 0;
            if (q > LEVELS) q = LEVELS;
            y->qs[j >> 2] |= (uint8_t)((q & 0x3) << ((j & 0x3) * 2));
        }
    } else {
#pragma unroll
        for (int j = 0; j < QK / 2; ++j) y->qs[j] = 0;
#pragma unroll
        for (int j = 0; j < QK; ++j) {
            int q = (int) roundf((x[j] - zero) * inv_scale);
            if (q < 0)      q = 0;
            if (q > LEVELS) q = LEVELS;
            y->qs[j >> 1] |= (uint8_t)((q & 0xF) << ((j & 0x1) * 4));
        }
    }
}

static __device__ void quantize_f32_tq_v2_block(
        const float * __restrict__ x, block_tq_v2 * __restrict__ y) {
    quantize_f32_tq_v_block_impl<2, block_tq_v2, QK_TQ_V2>(x, y);
}

static __device__ void quantize_f32_tq_v4_block(
        const float * __restrict__ x, block_tq_v4 * __restrict__ y) {
    quantize_f32_tq_v_block_impl<4, block_tq_v4, QK_TQ_V4>(x, y);
}

// ---------------------------------------------------------------------------
// TQ_V_POLAR2 / TQ_V_POLAR4 codebook tables — per-TU copy of the centroids
// and decision boundaries.  Must stay bit-identical to
// ggml/src/ggml-cpu/reex/reex_turboquant_codebook_tq_v_polar.inc.
//
// P4 A2.5 V-β (TheTom turbo2_0 / turbo4_0): Lloyd-Max codebook + norm
// correction.  Boundaries are needed at quantize time to decide buckets;
// centroids are needed to accumulate Σ centroid² for the norm-correction
// trick (same idempotency mechanism as K3 P3).
// ---------------------------------------------------------------------------

__device__ static const float reex_quantize_tq_v_polar2_centroids[4] = {
    -0.1323675364f, -0.0391626097f, 0.0408212505f, 0.1337173581f,
};
__device__ static const float reex_quantize_tq_v_polar2_boundaries[3] = {
    -0.0857650712f, 0.0008293193f, 0.0872692987f,
};

__device__ static const float reex_quantize_tq_v_polar4_centroids[16] = {
    -0.2315807939f, -0.1735078096f, -0.1336077154f, -0.1014359593f,
    -0.0734821036f, -0.0480686277f, -0.0242136829f, -0.0012462543f,
     0.0213646460f,  0.0441083387f,  0.0675085932f,  0.0922231078f,
     0.1192075014f,  0.1500890255f,  0.1882706583f,  0.2438714951f,
};
__device__ static const float reex_quantize_tq_v_polar4_boundaries[15] = {
    -0.2025443017f, -0.1535577625f, -0.1175218374f, -0.0874590278f,
    -0.0607753657f, -0.0361411572f, -0.0127299689f, 0.0100591956f,
     0.0327364914f, 0.0558084659f, 0.0798658505f, 0.1057153046f,
     0.1346482635f, 0.1691798419f, 0.2160710692f,
};

// ---------------------------------------------------------------------------
// quantize_f32_tq_v_polar2_block / quantize_f32_tq_v_polar4_block
//
// Mirror ggml/src/reex_turboquant_quants_ref.c::quantize_row_tq_v_polar{2,4}_ref
// one block at a time.  Norm-correction (corrected = ‖x‖ / ‖recon_unit‖)
// pins ‖dequant‖ ≡ ‖x‖ and lands a second-pass on the same indices.
// ---------------------------------------------------------------------------

static __device__ void quantize_f32_tq_v_polar2_block(
        const float * __restrict__ x, block_tq_v_polar2 * __restrict__ y) {
    float n2 = 0.0f;
#pragma unroll
    for (int j = 0; j < QK_TQ_V_POLAR2; ++j) {
        n2 += x[j] * x[j];
    }
    const float grp_norm = sqrtf(n2 > 1e-12f ? n2 : 1e-12f);
    const float inv_norm = 1.0f / grp_norm;

#pragma unroll
    for (int j = 0; j < QK_TQ_V_POLAR2 / 4; ++j) y->qs[j] = 0;

    float recon_sq = 0.0f;
#pragma unroll
    for (int j = 0; j < QK_TQ_V_POLAR2; ++j) {
        const float u = x[j] * inv_norm;

        int idx = 0;
        if (u >= reex_quantize_tq_v_polar2_boundaries[0]) idx = 1;
        if (u >= reex_quantize_tq_v_polar2_boundaries[1]) idx = 2;
        if (u >= reex_quantize_tq_v_polar2_boundaries[2]) idx = 3;

        y->qs[j >> 2] |= (uint8_t)((idx & 0x3) << ((j & 0x3) * 2));

        const float c = reex_quantize_tq_v_polar2_centroids[idx];
        recon_sq += c * c;
    }

    const float recon_norm = sqrtf(recon_sq > 1e-20f ? recon_sq : 1e-20f);
    const float corrected  = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;
    y->norm = __float2half(corrected);
}

static __device__ void quantize_f32_tq_v_polar4_block(
        const float * __restrict__ x, block_tq_v_polar4 * __restrict__ y) {
    float n2 = 0.0f;
#pragma unroll
    for (int j = 0; j < QK_TQ_V_POLAR4; ++j) {
        n2 += x[j] * x[j];
    }
    const float grp_norm = sqrtf(n2 > 1e-12f ? n2 : 1e-12f);
    const float inv_norm = 1.0f / grp_norm;

#pragma unroll
    for (int j = 0; j < QK_TQ_V_POLAR4 / 2; ++j) y->qs[j] = 0;

    float recon_sq = 0.0f;
#pragma unroll
    for (int j = 0; j < QK_TQ_V_POLAR4; ++j) {
        const float u = x[j] * inv_norm;

        int idx = 0;
#pragma unroll
        for (int b = 0; b < 15; ++b) {
            if (u >= reex_quantize_tq_v_polar4_boundaries[b]) {
                idx = b + 1;
            }
        }

        y->qs[j >> 1] |= (uint8_t)((idx & 0xF) << ((j & 0x1) * 4));

        const float c = reex_quantize_tq_v_polar4_centroids[idx];
        recon_sq += c * c;
    }

    const float recon_norm = sqrtf(recon_sq > 1e-20f ? recon_sq : 1e-20f);
    const float corrected  = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;
    y->norm  = __float2half(corrected);
    y->rnorm = __float2half(0.0f);  // reserved (TheTom byte-level alignment)
}

#endif  // REEX_TURBOQUANT
