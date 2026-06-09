// SPDX-License-Identifier: MIT
//
// reex_turboquant_quants_ref.c — REEX_TURBOQUANT only.
//
// Reference per-row quantize implementations (`*_ref` variants used as
// `from_float_ref` in `type_traits[]`) and the dequantize implementations
// (used as `to_float`). Built into libggml-base so that both libggml-cpu and
// any other backend's tests can reference them via `ggml_get_type_traits()`.
//
// Production (potentially SIMD-accelerated) `quantize_row_*` variants live
// in ggml-cpu/reex/reex_turboquant_quants.c (libggml-cpu).
//
// All functions guarded by REEX_TURBOQUANT — when the macro is OFF the
// translation unit is empty and contributes nothing to libggml-base.

#include "reex_turboquant_quants.h"

#ifdef REEX_TURBOQUANT

#include "ggml-impl.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ggml-cpu/reex/reex_turboquant_codebook_tq_k3.inc"
#include "ggml-cpu/reex/reex_turboquant_codebook_tq_v_polar.inc"

// Make .inc statics reachable from this TU even if the compiler decides the
// .inc is "unused" in some configurations.  QJL_SCALE is a legacy emit
// kept for backward link compatibility (see .inc header comment).
#define REEX_TQ_K3_USE_CB ((void)reex_turboquant_codebook_tq_k3_centroids,            \
                          (void)reex_turboquant_codebook_tq_k3_decision_boundaries,    \
                          (void)REEX_TURBOQUANT_QJL_SCALE_TQ_K3)
#define REEX_TQ_V_POLAR_USE_CB ((void)reex_turboquant_codebook_tq_v_polar2_centroids,            \
                               (void)reex_turboquant_codebook_tq_v_polar2_decision_boundaries,    \
                               (void)reex_turboquant_codebook_tq_v_polar4_centroids,              \
                               (void)reex_turboquant_codebook_tq_v_polar4_decision_boundaries)

// ---------------------------------------------------------------------------
// TQ_K3 — P4 A2.2 K3 P3 upgrade (TheTom turbo3_0 path: full 3-bit PolarQuant
//         + norm correction; no QJL split)
//
// On-disk `block_tq_k3` layout (50 B / 128 elt, sizeof unchanged from P1):
//   - norm  (ggml_half)               : corrected_norm = grp_norm / recon_norm
//                                       so dequantize output ‖y‖ ≡ grp_norm
//                                       (idempotency guarantee).
//   - qs[QK_TQ_K3 / 4]                : 8 centroids → 3-bit idx; the LOW 2
//                                       bits of each idx land here (4 elts /
//                                       byte, LSB first).
//   - signs[QK_TQ_K3 / 8]             : the HIGH 1 bit of each 3-bit idx
//                                       (8 elts / byte, LSB first).  P1 used
//                                       this field as a sign(u−c) QJL stub;
//                                       P3 reinterprets it as the high bit
//                                       of the centroid index.  The byte
//                                       stream is therefore not bitwise
//                                       compatible with P1 K cache GGUFs —
//                                       see plan doc §0.7 Q1.3.
//
// References:
//   - docs/turboquant/p4.a22_k3_p3_upgrade_plan.md §0 / §5 / §11
//   - llama-cpp-turboquant ggml/src/ggml-turbo-quant.c::quantize_row_turbo3_0_ref
// ---------------------------------------------------------------------------

void quantize_row_tq_k3_ref(const float * GGML_RESTRICT x,
                            block_tq_k3 * GGML_RESTRICT y,
                            int64_t k) {
    assert(k % QK_TQ_K3 == 0);
    REEX_TQ_K3_USE_CB;

    const int64_t nb = k / QK_TQ_K3;

    for (int64_t i = 0; i < nb; ++i) {
        const float       * GGML_RESTRICT xi = x + i * QK_TQ_K3;
        block_tq_k3       * GGML_RESTRICT yi = y + i;

        // Stage 1: group-wide L2 norm.  The caller is responsible for forward-
        // WHT rotation (graph op `GGML_OP_REEX_WHT`); since WHT is orthogonal
        // the per-block ‖x‖ here equals the un-rotated K vector's ‖x‖.
        float n2 = 0.0f;
        for (int j = 0; j < QK_TQ_K3; ++j) {
            n2 += xi[j] * xi[j];
        }
        const float grp_norm = sqrtf(n2 > 1e-12f ? n2 : 1e-12f);
        const float inv_norm = 1.0f / grp_norm;

        memset(yi->qs,    0, sizeof yi->qs);
        memset(yi->signs, 0, sizeof yi->signs);

        // Stage 2: per-element 3-bit centroid quantize (8 buckets).  We do
        // `lower_bound` against the 7 decision boundaries — same convention
        // as oracle `TurboQuantMSE` — and accumulate Σ centroid² as we go,
        // which Stage 3 needs for the norm correction trick.
        float recon_sq = 0.0f;
        for (int j = 0; j < QK_TQ_K3; ++j) {
            const float u = xi[j] * inv_norm;

            int idx = 0;
            // Open-coded 3-level branch (faster than std::lower_bound on a
            // 7-elt array; mirrors TheTom `nearest_centroid_3bit`).
            if (u >= reex_turboquant_codebook_tq_k3_decision_boundaries[0]) idx = 1;
            if (u >= reex_turboquant_codebook_tq_k3_decision_boundaries[1]) idx = 2;
            if (u >= reex_turboquant_codebook_tq_k3_decision_boundaries[2]) idx = 3;
            if (u >= reex_turboquant_codebook_tq_k3_decision_boundaries[3]) idx = 4;
            if (u >= reex_turboquant_codebook_tq_k3_decision_boundaries[4]) idx = 5;
            if (u >= reex_turboquant_codebook_tq_k3_decision_boundaries[5]) idx = 6;
            if (u >= reex_turboquant_codebook_tq_k3_decision_boundaries[6]) idx = 7;

            // Pack low 2 bits into qs (4 elt/byte), high 1 bit into signs (8 elt/byte).
            yi->qs   [j >> 2] |= (uint8_t)((idx & 0x3) << ((j & 0x3) * 2));
            if (idx & 0x4) {
                yi->signs[j >> 3] |= (uint8_t)(1u << (j & 0x7));
            }

            const float c = reex_turboquant_codebook_tq_k3_centroids[idx];
            recon_sq += c * c;
        }

        // Stage 3: norm correction — see plan doc §0.3.  This is the entire
        // idempotency mechanism: storing `corrected = grp_norm / recon_norm`
        // makes ‖dequant‖ ≡ grp_norm so a second quantize lands on exactly
        // the same bucket vector and `multi-pass cos_sim drift = 0`.
        const float recon_norm = sqrtf(recon_sq > 1e-20f ? recon_sq : 1e-20f);
        const float corrected  = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;
        yi->norm = GGML_FP32_TO_FP16(corrected);
    }
}

void dequantize_row_tq_k3(const block_tq_k3 * GGML_RESTRICT x,
                          float             * GGML_RESTRICT y,
                          int64_t k) {
    assert(k % QK_TQ_K3 == 0);

    const int64_t nb = k / QK_TQ_K3;

    for (int64_t i = 0; i < nb; ++i) {
        const block_tq_k3 * GGML_RESTRICT xi = x + i;
        float             * GGML_RESTRICT yi = y + i * QK_TQ_K3;

        // norm here is the corrected_norm stored at quantize time;
        // ‖y‖ ≡ original ‖x‖ falls out of (centroid · corrected_norm)²
        // summing back to grp_norm² — see plan doc §0.3.
        const float corrected_norm = GGML_FP16_TO_FP32(xi->norm);

        for (int j = 0; j < QK_TQ_K3; ++j) {
            const int low2 = (xi->qs   [j >> 2] >> ((j & 0x3) * 2)) & 0x3;
            const int hi1  = (xi->signs[j >> 3] >> (j & 0x7))       & 0x1;
            const int idx  = low2 | (hi1 << 2);

            yi[j] = reex_turboquant_codebook_tq_k3_centroids[idx] * corrected_norm;
        }
    }
}

// ---------------------------------------------------------------------------
// TQ_V2
// ---------------------------------------------------------------------------

void quantize_row_tq_v2_ref(const float * GGML_RESTRICT x,
                            block_tq_v2 * GGML_RESTRICT y,
                            int64_t k) {
    assert(k % QK_TQ_V2 == 0);
    static const int LEVELS = (1 << 2) - 1;  // 3 → 4 buckets (0..3)
    const int64_t nb = k / QK_TQ_V2;

    for (int64_t i = 0; i < nb; ++i) {
        const float * GGML_RESTRICT xi = x + i * QK_TQ_V2;
        block_tq_v2 * GGML_RESTRICT yi = y + i;

        float xmin = +FLT_MAX, xmax = -FLT_MAX;
        for (int j = 0; j < QK_TQ_V2; ++j) {
            if (xi[j] < xmin) xmin = xi[j];
            if (xi[j] > xmax) xmax = xi[j];
        }

        float scale = (xmax - xmin) / (float) LEVELS;
        if (scale < 1e-30f) scale = 1.0f;
        const float zero     = xmin;
        const float inv_scale = 1.0f / scale;

        yi->scale = GGML_FP32_TO_FP16(scale);
        yi->zero  = GGML_FP32_TO_FP16(zero);
        memset(yi->qs, 0, sizeof yi->qs);

        for (int j = 0; j < QK_TQ_V2; ++j) {
            int q = (int) roundf((xi[j] - zero) * inv_scale);
            if (q < 0)       q = 0;
            if (q > LEVELS)  q = LEVELS;
            yi->qs[j >> 2] |= (uint8_t)((q & 0x3) << ((j & 0x3) * 2));
        }
    }
}

void dequantize_row_tq_v2(const block_tq_v2 * GGML_RESTRICT x,
                          float             * GGML_RESTRICT y,
                          int64_t k) {
    assert(k % QK_TQ_V2 == 0);
    const int64_t nb = k / QK_TQ_V2;

    for (int64_t i = 0; i < nb; ++i) {
        const block_tq_v2 * GGML_RESTRICT xi = x + i;
        float             * GGML_RESTRICT yi = y + i * QK_TQ_V2;

        const float scale = GGML_FP16_TO_FP32(xi->scale);
        const float zero  = GGML_FP16_TO_FP32(xi->zero);

        for (int j = 0; j < QK_TQ_V2; ++j) {
            const int q = (xi->qs[j >> 2] >> ((j & 0x3) * 2)) & 0x3;
            yi[j] = (float) q * scale + zero;
        }
    }
}

// ---------------------------------------------------------------------------
// TQ_V4
// ---------------------------------------------------------------------------

void quantize_row_tq_v4_ref(const float * GGML_RESTRICT x,
                            block_tq_v4 * GGML_RESTRICT y,
                            int64_t k) {
    assert(k % QK_TQ_V4 == 0);
    static const int LEVELS = (1 << 4) - 1;  // 15 → 16 buckets (0..15)
    const int64_t nb = k / QK_TQ_V4;

    for (int64_t i = 0; i < nb; ++i) {
        const float * GGML_RESTRICT xi = x + i * QK_TQ_V4;
        block_tq_v4 * GGML_RESTRICT yi = y + i;

        float xmin = +FLT_MAX, xmax = -FLT_MAX;
        for (int j = 0; j < QK_TQ_V4; ++j) {
            if (xi[j] < xmin) xmin = xi[j];
            if (xi[j] > xmax) xmax = xi[j];
        }

        float scale = (xmax - xmin) / (float) LEVELS;
        if (scale < 1e-30f) scale = 1.0f;
        const float zero      = xmin;
        const float inv_scale = 1.0f / scale;

        yi->scale = GGML_FP32_TO_FP16(scale);
        yi->zero  = GGML_FP32_TO_FP16(zero);
        memset(yi->qs, 0, sizeof yi->qs);

        for (int j = 0; j < QK_TQ_V4; ++j) {
            int q = (int) roundf((xi[j] - zero) * inv_scale);
            if (q < 0)       q = 0;
            if (q > LEVELS)  q = LEVELS;
            yi->qs[j >> 1] |= (uint8_t)((q & 0xF) << ((j & 0x1) * 4));
        }
    }
}

void dequantize_row_tq_v4(const block_tq_v4 * GGML_RESTRICT x,
                          float             * GGML_RESTRICT y,
                          int64_t k) {
    assert(k % QK_TQ_V4 == 0);
    const int64_t nb = k / QK_TQ_V4;

    for (int64_t i = 0; i < nb; ++i) {
        const block_tq_v4 * GGML_RESTRICT xi = x + i;
        float             * GGML_RESTRICT yi = y + i * QK_TQ_V4;

        const float scale = GGML_FP16_TO_FP32(xi->scale);
        const float zero  = GGML_FP16_TO_FP32(xi->zero);

        for (int j = 0; j < QK_TQ_V4; ++j) {
            const int q = (xi->qs[j >> 1] >> ((j & 0x1) * 4)) & 0xF;
            yi[j] = (float) q * scale + zero;
        }
    }
}

// ---------------------------------------------------------------------------
// TQ_V_POLAR2 — P4 A2.5 V-β (TheTom turbo2_0 algorithmic alignment):
//   2-bit PolarQuant + Lloyd-Max codebook + norm correction.
//
// On-disk `block_tq_v_polar2` layout (34 B / 128 elt, TheTom-aligned):
//   - norm  (ggml_half)                  : corrected_norm = grp_norm / recon_norm
//                                          so dequantize output ‖y‖ ≡ grp_norm
//                                          (idempotency guarantee, same trick as K3).
//   - qs[QK_TQ_V_POLAR2 / 4]             : 4 centroids → 2-bit idx packed 4 elt /
//                                          byte, LSB first.  No sign-flip /
//                                          signs[] field (V is consumed by
//                                          softmax · V, not by Q·K inner-product;
//                                          centroid sign is preserved by
//                                          unit-vector normalization).
//
// Input is expected to be in the WHT-rotated frame (caller responsibility,
// via GGML_OP_REEX_WHT in cpy_v under tq_v_active, same as legacy tq_v2/tq_v4).
// Dequantized output is still in the WHT-rotated frame — the graph inserts
// inverse WHT at attention output (build_attn_mha).
//
// References:
//   - docs/turboquant/p4.a25_v_polar_design.md §3 / §5
//   - llama-cpp-turboquant ggml/src/ggml-turbo-quant.c::quantize_row_turbo2_0_ref
// ---------------------------------------------------------------------------

void quantize_row_tq_v_polar2_ref(const float       * GGML_RESTRICT x,
                                  block_tq_v_polar2 * GGML_RESTRICT y,
                                  int64_t k) {
    assert(k % QK_TQ_V_POLAR2 == 0);
    REEX_TQ_V_POLAR_USE_CB;

    const int64_t nb = k / QK_TQ_V_POLAR2;

    for (int64_t i = 0; i < nb; ++i) {
        const float       * GGML_RESTRICT xi = x + i * QK_TQ_V_POLAR2;
        block_tq_v_polar2 * GGML_RESTRICT yi = y + i;

        float n2 = 0.0f;
        for (int j = 0; j < QK_TQ_V_POLAR2; ++j) {
            n2 += xi[j] * xi[j];
        }
        const float grp_norm = sqrtf(n2 > 1e-12f ? n2 : 1e-12f);
        const float inv_norm = 1.0f / grp_norm;

        memset(yi->qs, 0, sizeof yi->qs);

        float recon_sq = 0.0f;
        for (int j = 0; j < QK_TQ_V_POLAR2; ++j) {
            const float u = xi[j] * inv_norm;

            int idx = 0;
            if (u >= reex_turboquant_codebook_tq_v_polar2_decision_boundaries[0]) idx = 1;
            if (u >= reex_turboquant_codebook_tq_v_polar2_decision_boundaries[1]) idx = 2;
            if (u >= reex_turboquant_codebook_tq_v_polar2_decision_boundaries[2]) idx = 3;

            yi->qs[j >> 2] |= (uint8_t)((idx & 0x3) << ((j & 0x3) * 2));

            const float c = reex_turboquant_codebook_tq_v_polar2_centroids[idx];
            recon_sq += c * c;
        }

        const float recon_norm = sqrtf(recon_sq > 1e-20f ? recon_sq : 1e-20f);
        const float corrected  = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;
        yi->norm = GGML_FP32_TO_FP16(corrected);
    }
}

void dequantize_row_tq_v_polar2(const block_tq_v_polar2 * GGML_RESTRICT x,
                                float                   * GGML_RESTRICT y,
                                int64_t k) {
    assert(k % QK_TQ_V_POLAR2 == 0);

    const int64_t nb = k / QK_TQ_V_POLAR2;

    for (int64_t i = 0; i < nb; ++i) {
        const block_tq_v_polar2 * GGML_RESTRICT xi = x + i;
        float                   * GGML_RESTRICT yi = y + i * QK_TQ_V_POLAR2;

        const float corrected_norm = GGML_FP16_TO_FP32(xi->norm);

        for (int j = 0; j < QK_TQ_V_POLAR2; ++j) {
            const int idx = (xi->qs[j >> 2] >> ((j & 0x3) * 2)) & 0x3;
            yi[j] = reex_turboquant_codebook_tq_v_polar2_centroids[idx] * corrected_norm;
        }
    }
}

// ---------------------------------------------------------------------------
// TQ_V_POLAR4 — P4 A2.5 V-β (TheTom turbo4_0 algorithmic alignment):
//   4-bit PolarQuant + Lloyd-Max codebook + norm correction.
//
// On-disk `block_tq_v_polar4` layout (68 B / 128 elt, TheTom-aligned):
//   - norm  (ggml_half)                  : corrected_norm = grp_norm / recon_norm
//                                          (same idempotency guarantee as V_POLAR2).
//   - rnorm (ggml_half)                  : reserved (always 0).  Kept for
//                                          byte-level alignment with TheTom
//                                          turbo4_0 layout; V-β does not use it.
//   - qs[QK_TQ_V_POLAR4 / 2]             : 16 centroids → 4-bit idx packed 2
//                                          elt / byte, LSB first.
//
// `nearest centroid` here is linear scan over the 15 boundaries since the
// 4-bit branch count is small enough that branchless lower_bound is faster
// only with SIMD; the SIMD variant lives in ggml-cpu/reex/.
// ---------------------------------------------------------------------------

void quantize_row_tq_v_polar4_ref(const float       * GGML_RESTRICT x,
                                  block_tq_v_polar4 * GGML_RESTRICT y,
                                  int64_t k) {
    assert(k % QK_TQ_V_POLAR4 == 0);
    REEX_TQ_V_POLAR_USE_CB;

    const int64_t nb = k / QK_TQ_V_POLAR4;

    for (int64_t i = 0; i < nb; ++i) {
        const float       * GGML_RESTRICT xi = x + i * QK_TQ_V_POLAR4;
        block_tq_v_polar4 * GGML_RESTRICT yi = y + i;

        float n2 = 0.0f;
        for (int j = 0; j < QK_TQ_V_POLAR4; ++j) {
            n2 += xi[j] * xi[j];
        }
        const float grp_norm = sqrtf(n2 > 1e-12f ? n2 : 1e-12f);
        const float inv_norm = 1.0f / grp_norm;

        memset(yi->qs, 0, sizeof yi->qs);

        float recon_sq = 0.0f;
        for (int j = 0; j < QK_TQ_V_POLAR4; ++j) {
            const float u = xi[j] * inv_norm;

            // Linear scan of 15 boundaries (open-coded; matches K3's open-
            // coded 7-boundary lower_bound style).
            int idx = 0;
            for (int b = 0; b < 15; ++b) {
                if (u >= reex_turboquant_codebook_tq_v_polar4_decision_boundaries[b]) {
                    idx = b + 1;
                } else {
                    break;
                }
            }

            yi->qs[j >> 1] |= (uint8_t)((idx & 0xF) << ((j & 0x1) * 4));

            const float c = reex_turboquant_codebook_tq_v_polar4_centroids[idx];
            recon_sq += c * c;
        }

        const float recon_norm = sqrtf(recon_sq > 1e-20f ? recon_sq : 1e-20f);
        const float corrected  = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;
        yi->norm  = GGML_FP32_TO_FP16(corrected);
        yi->rnorm = GGML_FP32_TO_FP16(0.0f);  // reserved (TheTom byte-level alignment)
    }
}

void dequantize_row_tq_v_polar4(const block_tq_v_polar4 * GGML_RESTRICT x,
                                float                   * GGML_RESTRICT y,
                                int64_t k) {
    assert(k % QK_TQ_V_POLAR4 == 0);

    const int64_t nb = k / QK_TQ_V_POLAR4;

    for (int64_t i = 0; i < nb; ++i) {
        const block_tq_v_polar4 * GGML_RESTRICT xi = x + i;
        float                   * GGML_RESTRICT yi = y + i * QK_TQ_V_POLAR4;

        const float corrected_norm = GGML_FP16_TO_FP32(xi->norm);

        for (int j = 0; j < QK_TQ_V_POLAR4; ++j) {
            const int idx = (xi->qs[j >> 1] >> ((j & 0x1) * 4)) & 0xF;
            yi[j] = reex_turboquant_codebook_tq_v_polar4_centroids[idx] * corrected_norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Whole-row chunk wrappers used by ggml_quantize_chunk()
// ---------------------------------------------------------------------------
// These mirror the upstream `quantize_q4_0(...)` style: take a (nrow,
// n_per_row) layout in float, write nrow * row_size bytes of blocks, and
// return that byte count.  TurboQuant does not consume the imatrix
// argument (no per-channel re-weighting in Phase 1) so it is silently
// ignored, matching how upstream Q*_0 wrappers handle a NULL imatrix.

size_t quantize_tq_k3(const float * GGML_RESTRICT src,
                            void  * GGML_RESTRICT dst,
                            int64_t                nrow,
                            int64_t                n_per_row,
                            const float * GGML_RESTRICT quant_weights) {
    (void) quant_weights;
    quantize_row_tq_k3_ref(src, (block_tq_k3 *) dst, (int64_t) nrow * n_per_row);
    return (size_t) nrow * ggml_row_size(GGML_TYPE_TQ_K3, n_per_row);
}

size_t quantize_tq_v2(const float * GGML_RESTRICT src,
                            void  * GGML_RESTRICT dst,
                            int64_t                nrow,
                            int64_t                n_per_row,
                            const float * GGML_RESTRICT quant_weights) {
    (void) quant_weights;
    quantize_row_tq_v2_ref(src, (block_tq_v2 *) dst, (int64_t) nrow * n_per_row);
    return (size_t) nrow * ggml_row_size(GGML_TYPE_TQ_V2, n_per_row);
}

size_t quantize_tq_v4(const float * GGML_RESTRICT src,
                            void  * GGML_RESTRICT dst,
                            int64_t                nrow,
                            int64_t                n_per_row,
                            const float * GGML_RESTRICT quant_weights) {
    (void) quant_weights;
    quantize_row_tq_v4_ref(src, (block_tq_v4 *) dst, (int64_t) nrow * n_per_row);
    return (size_t) nrow * ggml_row_size(GGML_TYPE_TQ_V4, n_per_row);
}

size_t quantize_tq_v_polar2(const float * GGML_RESTRICT src,
                                  void  * GGML_RESTRICT dst,
                                  int64_t                nrow,
                                  int64_t                n_per_row,
                                  const float * GGML_RESTRICT quant_weights) {
    (void) quant_weights;
    quantize_row_tq_v_polar2_ref(src, (block_tq_v_polar2 *) dst, (int64_t) nrow * n_per_row);
    return (size_t) nrow * ggml_row_size(GGML_TYPE_TQ_V_POLAR2, n_per_row);
}

size_t quantize_tq_v_polar4(const float * GGML_RESTRICT src,
                                  void  * GGML_RESTRICT dst,
                                  int64_t                nrow,
                                  int64_t                n_per_row,
                                  const float * GGML_RESTRICT quant_weights) {
    (void) quant_weights;
    quantize_row_tq_v_polar4_ref(src, (block_tq_v_polar4 *) dst, (int64_t) nrow * n_per_row);
    return (size_t) nrow * ggml_row_size(GGML_TYPE_TQ_V_POLAR4, n_per_row);
}

#endif  // REEX_TURBOQUANT
