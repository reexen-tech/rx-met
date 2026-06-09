// SPDX-License-Identifier: MIT
//
// reex_turboquant_fattn_helpers.cuh — REEX_TURBOQUANT only.
//
// Flash-Attention vec-kernel helpers for GGML_TYPE_TQ_K3 / TQ_V2 / TQ_V4.
//
// This header provides the per-call helpers consumed by
// `fattn-common.cuh::get_dequantize_V<>` (and, in P3.5.b, `get_vec_dot_KQ<>`):
//
//   - `dequantize_V_tq_v2<T, ne>(vx, dst, i0)` — V-cache dequant for 2-bit V
//   - `dequantize_V_tq_v4<T, ne>(vx, dst, i0)` — V-cache dequant for 4-bit V
//   - `vec_dot_fattn_vec_KQ_tq_k3<D, nthreads>(...)` — added in P3.5.b
//
// Layout / shape conventions follow `dequantize_V_q8_0` and the TheTom
// reference (`fattn-common.cuh::dequantize_V_turbo*_0`):
//   - `<T>` selects `half` or `float` output element type
//   - `<ne>` ∈ {2, 4} = elements written per call
//   - `i0`  = absolute element offset within the K/V tensor's contiguous run
//   - `dst` is half2[ne/2] or float2[ne/2] for the fast (ne == 4) path
//
// Algorithm parameters (per-block scale + zero, NOT a centroid LUT) come from
// `02_data_layout.md` §3 and match `reex_dequant_tq_v2_elt / _tq_v4_elt` in
// `reex_turboquant_dequant.cuh`. No bit-level rewriting is allowed without
// updating the .cuh helper and the CPU reference path together.

#pragma once

#ifdef REEX_TURBOQUANT

#include "reex_turboquant_dequant.cuh"

#include <cuda_runtime.h>
#include <type_traits>

// ---------------------------------------------------------------------------
// Per-translation-unit copies of the TQ_K3 dequant constants.
//
// `__constant__` symbols defined in reex_turboquant_dequant.cu are NOT visible
// from this header's includers (the .cu / .cuh that pulls us in lives in a
// different TU). Per 02 §6, each TU that decodes TQ_K3 elements must carry
// its own `static __constant__` copies. Values must stay bit-identical to
// `reex_tq_k3_centroids_d` in reex_turboquant_dequant.cu — keep the literals
// here in lockstep when regenerating reex_turboquant_codebook_tq_k3.inc
// (see 02 §6.1).
//
// P4 A2.2 K3 P3 upgrade (TheTom turbo3_0): 8 centroids, no residual_unit.
// ---------------------------------------------------------------------------

static __constant__ float reex_fattn_tq_k3_centroids[8] = {
    -0.1861817092f, -0.1154012680f, -0.0635419339f, -0.0184134189f,
     0.0247995369f,  0.0696418509f,  0.1208978295f,  0.1906363368f,
};

// V_POLAR2 / V_POLAR4 codebook constants — per-TU __constant__ copies, same
// reason as `reex_fattn_tq_k3_centroids` above.  Values must stay bit-identical
// to reex_turboquant_codebook_tq_v_polar.inc and the __constant__ arrays in
// reex_turboquant_dequant.cu (`reex_tq_v_polar{2,4}_centroids_d`).
//
// P4 A2.5 V-β (TheTom turbo2_0 / turbo4_0).  Decode:
//     v = centroids[idx] * corrected_norm
// where corrected_norm = ‖x‖ / ‖recon_unit‖ is stored in `b->norm`.

static __constant__ float reex_fattn_tq_v_polar2_centroids[4] = {
    -0.1323675364f, -0.0391626097f, 0.0408212505f, 0.1337173581f,
};

static __constant__ float reex_fattn_tq_v_polar4_centroids[16] = {
    -0.2315807939f, -0.1735078096f, -0.1336077154f, -0.1014359593f,
    -0.0734821036f, -0.0480686277f, -0.0242136829f, -0.0012462543f,
     0.0213646460f,  0.0441083387f,  0.0675085932f,  0.0922231078f,
     0.1192075014f,  0.1500890255f,  0.1882706583f,  0.2438714951f,
};

// ---------------------------------------------------------------------------
// dequantize_V_tq_v2 — TQ_V2 V-cache dequant.
//
// `block_tq_v2` (12 B / 32 elt):
//     half scale; half zero; uint8_t qs[8];   // 2-bit indices, 4 elt/byte
//
// Algorithm: v = (float) q * scale + zero      (see 02 §3)
// ---------------------------------------------------------------------------

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tq_v2(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tq_v2 * x = (const block_tq_v2 *) vx;

    const int64_t ib    = i0 / QK_TQ_V2;
    const int     j0    = (int) (i0 % QK_TQ_V2);
    const float   scale = __half2float(x[ib].scale);
    const float   zero  = __half2float(x[ib].zero);

    static_assert(ne == 2 || ne == 4, "bad ne");

    if constexpr (ne == 4) {
        // VEC kernel access pattern guarantees j0 % 4 == 0, so a single qs
        // byte holds all 4 indices we need.
        const uint8_t qs_byte = x[ib].qs[j0 >> 2];
        const int q0 = (qs_byte >> 0) & 0x3;
        const int q1 = (qs_byte >> 2) & 0x3;
        const int q2 = (qs_byte >> 4) & 0x3;
        const int q3 = (qs_byte >> 6) & 0x3;

#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *) dst)[0] = make_half2(
                __float2half((float) q0 * scale + zero),
                __float2half((float) q1 * scale + zero));
            ((half2 *) dst)[1] = make_half2(
                __float2half((float) q2 * scale + zero),
                __float2half((float) q3 * scale + zero));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float2 *) dst)[0] = make_float2(
                (float) q0 * scale + zero,
                (float) q1 * scale + zero);
            ((float2 *) dst)[1] = make_float2(
                (float) q2 * scale + zero,
                (float) q3 * scale + zero);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    } else { // ne == 2
#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            const float v0 = reex_dequant_tq_v2_elt(&x[ib], j0);
            const float v1 = reex_dequant_tq_v2_elt(&x[ib], j0 + 1);
            ((half2 *) dst)[0] = make_half2(__float2half(v0), __float2half(v1));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[0] = reex_dequant_tq_v2_elt(&x[ib], j0);
            ((float *) dst)[1] = reex_dequant_tq_v2_elt(&x[ib], j0 + 1);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    }
}

// ---------------------------------------------------------------------------
// dequantize_V_tq_v4 — TQ_V4 V-cache dequant.
//
// `block_tq_v4` (20 B / 32 elt):
//     half scale; half zero; uint8_t qs[16];  // 4-bit indices, 2 elt/byte
//
// Algorithm: v = (float) q * scale + zero      (see 02 §3)
// ---------------------------------------------------------------------------

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tq_v4(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tq_v4 * x = (const block_tq_v4 *) vx;

    const int64_t ib    = i0 / QK_TQ_V4;
    const int     j0    = (int) (i0 % QK_TQ_V4);
    const float   scale = __half2float(x[ib].scale);
    const float   zero  = __half2float(x[ib].zero);

    static_assert(ne == 2 || ne == 4, "bad ne");

    if constexpr (ne == 4) {
        // j0 % 4 == 0 from VEC kernel access pattern; 4 nibbles span 2 bytes.
        const uint8_t qs_byte0 = x[ib].qs[j0 >> 1];        // elements j0, j0+1
        const uint8_t qs_byte1 = x[ib].qs[(j0 >> 1) + 1];  // elements j0+2, j0+3
        const int q0 = (qs_byte0 >> 0) & 0xF;
        const int q1 = (qs_byte0 >> 4) & 0xF;
        const int q2 = (qs_byte1 >> 0) & 0xF;
        const int q3 = (qs_byte1 >> 4) & 0xF;

#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *) dst)[0] = make_half2(
                __float2half((float) q0 * scale + zero),
                __float2half((float) q1 * scale + zero));
            ((half2 *) dst)[1] = make_half2(
                __float2half((float) q2 * scale + zero),
                __float2half((float) q3 * scale + zero));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float2 *) dst)[0] = make_float2(
                (float) q0 * scale + zero,
                (float) q1 * scale + zero);
            ((float2 *) dst)[1] = make_float2(
                (float) q2 * scale + zero,
                (float) q3 * scale + zero);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    } else { // ne == 2
#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            const float v0 = reex_dequant_tq_v4_elt(&x[ib], j0);
            const float v1 = reex_dequant_tq_v4_elt(&x[ib], j0 + 1);
            ((half2 *) dst)[0] = make_half2(__float2half(v0), __float2half(v1));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[0] = reex_dequant_tq_v4_elt(&x[ib], j0);
            ((float *) dst)[1] = reex_dequant_tq_v4_elt(&x[ib], j0 + 1);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    }
}

// ---------------------------------------------------------------------------
// dequantize_V_tq_v_polar2 — TQ_V_POLAR2 V-cache dequant.
//
// `block_tq_v_polar2` (34 B / 128 elt):
//     half norm;  uint8_t qs[32];      // 2-bit Lloyd-Max idx, 4 elt/byte
//
// Algorithm: v = centroids[idx] * corrected_norm   (P4 A2.5 V-β, TheTom
// turbo2_0; same idempotency mechanism as K3 P3).
// ---------------------------------------------------------------------------

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tq_v_polar2(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tq_v_polar2 * x = (const block_tq_v_polar2 *) vx;

    const int64_t ib   = i0 / QK_TQ_V_POLAR2;
    const int     j0   = (int) (i0 % QK_TQ_V_POLAR2);
    const float   corr = __half2float(x[ib].norm);

    static_assert(ne == 2 || ne == 4, "bad ne");

    if constexpr (ne == 4) {
        // VEC kernel guarantees j0 % 4 == 0 → all 4 indices live in one byte.
        const uint8_t qs_byte = x[ib].qs[j0 >> 2];
        const int q0 = (qs_byte >> 0) & 0x3;
        const int q1 = (qs_byte >> 2) & 0x3;
        const int q2 = (qs_byte >> 4) & 0x3;
        const int q3 = (qs_byte >> 6) & 0x3;

#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *) dst)[0] = make_half2(
                __float2half(reex_fattn_tq_v_polar2_centroids[q0] * corr),
                __float2half(reex_fattn_tq_v_polar2_centroids[q1] * corr));
            ((half2 *) dst)[1] = make_half2(
                __float2half(reex_fattn_tq_v_polar2_centroids[q2] * corr),
                __float2half(reex_fattn_tq_v_polar2_centroids[q3] * corr));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float2 *) dst)[0] = make_float2(
                reex_fattn_tq_v_polar2_centroids[q0] * corr,
                reex_fattn_tq_v_polar2_centroids[q1] * corr);
            ((float2 *) dst)[1] = make_float2(
                reex_fattn_tq_v_polar2_centroids[q2] * corr,
                reex_fattn_tq_v_polar2_centroids[q3] * corr);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    } else { // ne == 2
#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            const float v0 = reex_dequant_tq_v_polar2_elt(&x[ib], j0,     reex_fattn_tq_v_polar2_centroids);
            const float v1 = reex_dequant_tq_v_polar2_elt(&x[ib], j0 + 1, reex_fattn_tq_v_polar2_centroids);
            ((half2 *) dst)[0] = make_half2(__float2half(v0), __float2half(v1));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[0] = reex_dequant_tq_v_polar2_elt(&x[ib], j0,     reex_fattn_tq_v_polar2_centroids);
            ((float *) dst)[1] = reex_dequant_tq_v_polar2_elt(&x[ib], j0 + 1, reex_fattn_tq_v_polar2_centroids);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    }
}

// ---------------------------------------------------------------------------
// dequantize_V_tq_v_polar4 — TQ_V_POLAR4 V-cache dequant.
//
// `block_tq_v_polar4` (68 B / 128 elt):
//     half norm; half rnorm; uint8_t qs[64];    // 4-bit Lloyd-Max idx
//
// Algorithm: v = centroids[idx] * corrected_norm   (P4 A2.5 V-β, TheTom
// turbo4_0; rnorm is reserved / always 0).
// ---------------------------------------------------------------------------

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tq_v_polar4(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tq_v_polar4 * x = (const block_tq_v_polar4 *) vx;

    const int64_t ib   = i0 / QK_TQ_V_POLAR4;
    const int     j0   = (int) (i0 % QK_TQ_V_POLAR4);
    const float   corr = __half2float(x[ib].norm);

    static_assert(ne == 2 || ne == 4, "bad ne");

    if constexpr (ne == 4) {
        // j0 % 4 == 0 from VEC kernel; 4 nibbles span 2 bytes.
        const uint8_t qs_byte0 = x[ib].qs[j0 >> 1];        // elements j0, j0+1
        const uint8_t qs_byte1 = x[ib].qs[(j0 >> 1) + 1];  // elements j0+2, j0+3
        const int q0 = (qs_byte0 >> 0) & 0xF;
        const int q1 = (qs_byte0 >> 4) & 0xF;
        const int q2 = (qs_byte1 >> 0) & 0xF;
        const int q3 = (qs_byte1 >> 4) & 0xF;

#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *) dst)[0] = make_half2(
                __float2half(reex_fattn_tq_v_polar4_centroids[q0] * corr),
                __float2half(reex_fattn_tq_v_polar4_centroids[q1] * corr));
            ((half2 *) dst)[1] = make_half2(
                __float2half(reex_fattn_tq_v_polar4_centroids[q2] * corr),
                __float2half(reex_fattn_tq_v_polar4_centroids[q3] * corr));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float2 *) dst)[0] = make_float2(
                reex_fattn_tq_v_polar4_centroids[q0] * corr,
                reex_fattn_tq_v_polar4_centroids[q1] * corr);
            ((float2 *) dst)[1] = make_float2(
                reex_fattn_tq_v_polar4_centroids[q2] * corr,
                reex_fattn_tq_v_polar4_centroids[q3] * corr);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    } else { // ne == 2
#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            const float v0 = reex_dequant_tq_v_polar4_elt(&x[ib], j0,     reex_fattn_tq_v_polar4_centroids);
            const float v1 = reex_dequant_tq_v_polar4_elt(&x[ib], j0 + 1, reex_fattn_tq_v_polar4_centroids);
            ((half2 *) dst)[0] = make_half2(__float2half(v0), __float2half(v1));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[0] = reex_dequant_tq_v_polar4_elt(&x[ib], j0,     reex_fattn_tq_v_polar4_centroids);
            ((float *) dst)[1] = reex_dequant_tq_v_polar4_elt(&x[ib], j0 + 1, reex_fattn_tq_v_polar4_centroids);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    }
}

// ---------------------------------------------------------------------------
// vec_dot_fattn_vec_KQ_tq_k3 — partial KQ dot for TQ_K3 K cache.
//
// P4 A2.2 K3 P3 upgrade (TheTom turbo3_0):
//   idx_j  = (qs_low_2_bits) | (signs_high_1_bit << 2)              // 3 bits
//   K[j]   = corrected_norm * centroids[idx_j]                      // 8-entry LUT
//   <K,Q>  = Σ_j K[j] * Q[j]
//
// `corrected_norm = ‖x‖ / ‖recon_unit‖` is stored in `block_tq_k3::norm`
// (see plan doc §0.3 / reex_turboquant_quantize_kv.cuh).  This is now a
// single-LUT decode — there is no residual term — so each pair is
//   kv = corrected_norm * (centroids[idx0], centroids[idx1]).
//
// Each thread accumulates partial pairs; the FA vec kernel's reduction across
// the `nthreads` thread group yields the full row dot. Q is consumed via the
// fp16/float path (`Q_v` half2 / float2). Q8_1 inputs (`Q_q8`, `Q_ds_v`) are
// unused: the TQ_K3 codebook is fp32, so an integer-SIMD path (`__dp4a`) is
// not applicable to a single-step decoder. (See 01 §0.3.2 #11 for the
// rationale; revisit as a P4 optimization if profiling justifies a quantize-
// to-int8 step in front of `__dp4a`.)
//
// Loop structure cribbed from TheTom's `vec_dot_fattn_vec_KQ_turbo3_0`
// (fattn-common.cuh:301-350, feature/turboquant-kv-cache @ 11a241d): outer
// `k_KQ_0` strides by `nthreads * cpy_ne`, inner `k_KQ_1` walks `cpy_ne`
// pairs, `elem0 = 2*k_KQ` is always even so the qs / signs bytes for j0 and
// j0+1 are shared and read once per iteration.
//
// `D % QK_TQ_K3 == 0` is enforced because the WHT block (and therefore the
// shared `norm`) is sized to 128 in this project. The kernel itself supports
// any positive multiple of QK_TQ_K3 because `ib = elem0 / QK_TQ_K3` walks the
// 128-element blocks one-after-another (mirroring how `vec_dot_q8_0_q8_1`
// strides 32-element blocks for D=256). P3.5.c instantiates D ∈ {128, 256}.
// Adding more (e.g. D=512 if a future model needs it) is purely a CMake +
// `DECL_FATTN_VEC_CASE` change — no further kernel work.
// ---------------------------------------------------------------------------

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tq_k3(
        const char * __restrict__ K_c, const void * __restrict__ Q_v,
        const int  * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    static_assert(D >= QK_TQ_K3 && D % QK_TQ_K3 == 0,
                  "vec_dot_fattn_vec_KQ_tq_k3 requires D % QK_TQ_K3 == 0 (head_dim = N x 128)");

    const block_tq_k3 * K_tq = (const block_tq_k3 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads * cpy_ne) {
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int k_KQ = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne + k_KQ_1;

            // elem0 is always even; elem0 and elem0+1 are in the same block,
            // the same qs byte (j0 % 4 ∈ {0, 2}), and the same signs byte
            // (j0 % 8 ∈ {0, 2, 4, 6}) so we can load each byte once.
            const int elem0 = k_KQ * 2;
            const int ib    = elem0 / QK_TQ_K3;
            const int j0    = elem0 % QK_TQ_K3;

            const float   corr     = __half2float(K_tq[ib].norm);  // corrected norm
            const uint8_t qs_byte  = K_tq[ib].qs   [j0 >> 2];
            const uint8_t sgn_byte = K_tq[ib].signs[j0 >> 3];

            const int shift_qs  = (j0 & 0x3) * 2;   // 0 or 4
            const int shift_sgn =  j0 & 0x7;        // 0, 2, 4 or 6

            // 3-bit idx = low 2 bits (qs) | (high 1 bit (signs) << 2).
            const int low0  = (qs_byte  >> (shift_qs    )) & 0x3;
            const int low1  = (qs_byte  >> (shift_qs + 2)) & 0x3;
            const int high0 = (sgn_byte >> (shift_sgn    )) & 0x1;
            const int high1 = (sgn_byte >> (shift_sgn + 1)) & 0x1;
            const int idx0  = low0 | (high0 << 2);
            const int idx1  = low1 | (high1 << 2);

            float2 kv;
            kv.x = reex_fattn_tq_k3_centroids[idx0] * corr;
            kv.y = reex_fattn_tq_k3_centroids[idx1] * corr;

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 qv = ((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            ggml_cuda_mad(sum, make_float2(kv.x, kv.y), __half22float2(qv));
#else
            const float2 qv = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            sum += kv.x * qv.x + kv.y * qv.y;
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

#endif  // REEX_TURBOQUANT
