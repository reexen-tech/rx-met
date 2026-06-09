// SPDX-License-Identifier: MIT
//
// ggml-cpu/reex/reex_turboquant_quants.c — REEX_TURBOQUANT only.
//
// Production per-row quantize implementations for GGML_TYPE_TQ_K3 / TQ_V2 /
// TQ_V4 (the `from_float` slot in `type_traits_cpu[]`). The Phase 1 versions
// simply forward to the `_ref` implementations in libggml-base; SIMD variants
// will be added incrementally in Phase 2.
//
// All functions guarded by REEX_TURBOQUANT — when the macro is OFF the TU is
// empty and contributes nothing to libggml-cpu.

#include "reex_turboquant_quants.h"

#ifdef REEX_TURBOQUANT

#include "ggml-impl.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

void quantize_row_tq_k3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq_k3_ref(x, (block_tq_k3 *) y, k);
}

void quantize_row_tq_v2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq_v2_ref(x, (block_tq_v2 *) y, k);
}

void quantize_row_tq_v4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq_v4_ref(x, (block_tq_v4 *) y, k);
}

void quantize_row_tq_v_polar2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq_v_polar2_ref(x, (block_tq_v_polar2 *) y, k);
}

void quantize_row_tq_v_polar4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq_v_polar4_ref(x, (block_tq_v_polar4 *) y, k);
}

// ---------------------------------------------------------------------------
// vec_dot — Phase 2 scalar dequantize-then-dot.
// ---------------------------------------------------------------------------
// All three follow the same pattern:
//   1. assert n == nb * QK
//   2. for each block: dequantize 1 block into a stack-local F32 buffer,
//      accumulate sum_j(x_dq[j] * y[j]).
// nrc==1 is asserted; bx/by/bs unused (single-row dot product).
//
// We re-use the existing `dequantize_row_*` per-block code by calling it once
// per block (k = QK).  That keeps the kernel correct-by-construction at the
// cost of an extra stack buffer; a fused dequant+dot SIMD variant is a
// future optimization once we have a perf baseline from the PPL gate.

#include "ggml-common.h"  // already pulled in indirectly but be explicit

void ggml_vec_dot_tq_k3_f32(int n, float * GGML_RESTRICT s, size_t bs,
                            const void * GGML_RESTRICT vx, size_t bx,
                            const void * GGML_RESTRICT vy, size_t by,
                            int nrc) {
    (void) bs; (void) bx; (void) by;
    assert(nrc == 1);
    (void) nrc;
    assert(n % QK_TQ_K3 == 0);

    const block_tq_k3 * GGML_RESTRICT x = (const block_tq_k3 *) vx;
    const float       * GGML_RESTRICT y = (const float       *) vy;

    const int nb = n / QK_TQ_K3;
    float acc = 0.0f;
    float buf[QK_TQ_K3];

    for (int ib = 0; ib < nb; ++ib) {
        dequantize_row_tq_k3(x + ib, buf, QK_TQ_K3);
        const float * yi = y + ib * QK_TQ_K3;
        float local = 0.0f;
        for (int j = 0; j < QK_TQ_K3; ++j) {
            local += buf[j] * yi[j];
        }
        acc += local;
    }
    *s = acc;
}

void ggml_vec_dot_tq_v2_f32(int n, float * GGML_RESTRICT s, size_t bs,
                            const void * GGML_RESTRICT vx, size_t bx,
                            const void * GGML_RESTRICT vy, size_t by,
                            int nrc) {
    (void) bs; (void) bx; (void) by;
    assert(nrc == 1);
    (void) nrc;
    assert(n % QK_TQ_V2 == 0);

    const block_tq_v2 * GGML_RESTRICT x = (const block_tq_v2 *) vx;
    const float       * GGML_RESTRICT y = (const float       *) vy;

    const int nb = n / QK_TQ_V2;
    float acc = 0.0f;
    float buf[QK_TQ_V2];

    for (int ib = 0; ib < nb; ++ib) {
        dequantize_row_tq_v2(x + ib, buf, QK_TQ_V2);
        const float * yi = y + ib * QK_TQ_V2;
        float local = 0.0f;
        for (int j = 0; j < QK_TQ_V2; ++j) {
            local += buf[j] * yi[j];
        }
        acc += local;
    }
    *s = acc;
}

void ggml_vec_dot_tq_v4_f32(int n, float * GGML_RESTRICT s, size_t bs,
                            const void * GGML_RESTRICT vx, size_t bx,
                            const void * GGML_RESTRICT vy, size_t by,
                            int nrc) {
    (void) bs; (void) bx; (void) by;
    assert(nrc == 1);
    (void) nrc;
    assert(n % QK_TQ_V4 == 0);

    const block_tq_v4 * GGML_RESTRICT x = (const block_tq_v4 *) vx;
    const float       * GGML_RESTRICT y = (const float       *) vy;

    const int nb = n / QK_TQ_V4;
    float acc = 0.0f;
    float buf[QK_TQ_V4];

    for (int ib = 0; ib < nb; ++ib) {
        dequantize_row_tq_v4(x + ib, buf, QK_TQ_V4);
        const float * yi = y + ib * QK_TQ_V4;
        float local = 0.0f;
        for (int j = 0; j < QK_TQ_V4; ++j) {
            local += buf[j] * yi[j];
        }
        acc += local;
    }
    *s = acc;
}

void ggml_vec_dot_tq_v_polar2_f32(int n, float * GGML_RESTRICT s, size_t bs,
                                  const void * GGML_RESTRICT vx, size_t bx,
                                  const void * GGML_RESTRICT vy, size_t by,
                                  int nrc) {
    (void) bs; (void) bx; (void) by;
    assert(nrc == 1);
    (void) nrc;
    assert(n % QK_TQ_V_POLAR2 == 0);

    const block_tq_v_polar2 * GGML_RESTRICT x = (const block_tq_v_polar2 *) vx;
    const float             * GGML_RESTRICT y = (const float             *) vy;

    const int nb = n / QK_TQ_V_POLAR2;
    float acc = 0.0f;
    float buf[QK_TQ_V_POLAR2];

    for (int ib = 0; ib < nb; ++ib) {
        dequantize_row_tq_v_polar2(x + ib, buf, QK_TQ_V_POLAR2);
        const float * yi = y + ib * QK_TQ_V_POLAR2;
        float local = 0.0f;
        for (int j = 0; j < QK_TQ_V_POLAR2; ++j) {
            local += buf[j] * yi[j];
        }
        acc += local;
    }
    *s = acc;
}

void ggml_vec_dot_tq_v_polar4_f32(int n, float * GGML_RESTRICT s, size_t bs,
                                  const void * GGML_RESTRICT vx, size_t bx,
                                  const void * GGML_RESTRICT vy, size_t by,
                                  int nrc) {
    (void) bs; (void) bx; (void) by;
    assert(nrc == 1);
    (void) nrc;
    assert(n % QK_TQ_V_POLAR4 == 0);

    const block_tq_v_polar4 * GGML_RESTRICT x = (const block_tq_v_polar4 *) vx;
    const float             * GGML_RESTRICT y = (const float             *) vy;

    const int nb = n / QK_TQ_V_POLAR4;
    float acc = 0.0f;
    float buf[QK_TQ_V_POLAR4];

    for (int ib = 0; ib < nb; ++ib) {
        dequantize_row_tq_v_polar4(x + ib, buf, QK_TQ_V_POLAR4);
        const float * yi = y + ib * QK_TQ_V_POLAR4;
        float local = 0.0f;
        for (int j = 0; j < QK_TQ_V_POLAR4; ++j) {
            local += buf[j] * yi[j];
        }
        acc += local;
    }
    *s = acc;
}

#endif  // REEX_TURBOQUANT
