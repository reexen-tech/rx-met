#ifdef GGML_USE_REEX_Q64

#include "reex/ggml-reex-q64.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>

/* Fixed-point Psum truncation bit-width, read once from REEX_Q64_PSUM_BITS.
 * <= 0 (unset/invalid) disables truncation. Benign init race across threads. */
int reex_q64_psum_bits(void) {
    static int cached = -2; /* -2 = uninitialized */
    if (cached == -2) {
        const char * e = getenv("REEX_Q64_PSUM_BITS");
        int v = e ? atoi(e) : 0;
        cached = v > 0 ? v : -1;
    }
    return cached;
}

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/* Self-contained copies of ggml-quants.c helpers (kept private here so the REEX
 * K-quant path stays decoupled from the upstream static symbols). */
static inline int reex_nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// Mirror of make_qkx2_quants (ggml-quants.c): weighted scale+min search.
static float reex_make_qkx2_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = reex_nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = reex_nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

#ifndef GGML_Q64_GROUP_MAX_EPS
#define GGML_Q64_GROUP_MAX_EPS 1e-15f
#endif

// Mirror of make_qx_quants (ggml-quants.c): symmetric scale search (Q6_K path).
static float reex_make_qx_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L,
        int rmse_type, const float * GGML_RESTRICT qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GGML_Q64_GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = reex_nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) { rmse_type = -rmse_type; return_early = true; }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = reex_nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = reex_nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = reex_nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

// Mirror of make_q3_quants (ggml-quants.c): symmetric scale search (Q3_K path).
static float reex_make_q3_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GGML_Q64_GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = reex_nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            L[i] = l;
            float w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i]*x[i];
                float slx = sumlx - w*x[i]*L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w*L[i]*L[i];
                    int new_l = reex_nearest_int(x[i] * sl2 / slx);
                    new_l = MAX(-nmax, MIN(nmax-1, new_l));
                    if (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) break;
        }
        for (int i = 0; i < n; ++i) L[i] += nmax;
        return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }
    for (int i = 0; i < n; ++i) {
        int l = reex_nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
    }
    return 1/iscale;
}

/* ============================================================
 *  Q4_0_64: symmetric 4-bit, block=64, fp16 scale.
 *  Mirrors native quantize_row_q4_0_ref / dequantize_row_q4_0 with QK=64.
 * ============================================================ */

void quantize_row_q4_0_64_ref(const float * GGML_RESTRICT x, block_q4_0_64 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0_64;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void dequantize_row_q4_0_64(const block_q4_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0_64;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

size_t quantize_q4_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights; // legacy round-to-nearest: imatrix not used
    assert(n_per_row % QK4_0_64 == 0);
    const int64_t nblk = n_per_row / QK4_0_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q4_0_64));
    quantize_row_q4_0_64_ref(src, (block_q4_0_64 *) dst, nrow * n_per_row);
    return nrow * row_size;
}

/* ============================================================
 *  Q8_0_64: symmetric 8-bit, block=64, fp16 scale.
 *  Used both as a standalone weight type and as the activation
 *  companion (vec_dot_type) for Q4_0_64.
 * ============================================================ */

void quantize_row_q8_0_64_ref(const float * GGML_RESTRICT x, block_q8_0_64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0_64 == 0);
    const int nb = k / QK8_0_64;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0_64; j++) {
            const float v = x[i*QK8_0_64 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0_64; ++j) {
            const float x0 = x[i*QK8_0_64 + j]*id;
            y[i].qs[j] = roundf(x0);
        }
    }
}

void dequantize_row_q8_0_64(const block_q8_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_0_64;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

size_t quantize_q8_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK8_0_64 == 0);
    const int64_t nblk = n_per_row / QK8_0_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q8_0_64));
    quantize_row_q8_0_64_ref(src, (block_q8_0_64 *) dst, nrow * n_per_row);
    return nrow * row_size;
}

/* ============================================================
 *  Q4_1_64: asymmetric 4-bit (delta + min), block=64.
 * ============================================================ */

void quantize_row_q4_1_64_ref(const float * GGML_RESTRICT x, block_q4_1_64 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_1_64;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + 0    + j] - min)*id;
            const float x1 = (x[i*qk + qk/2 + j] - min)*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void dequantize_row_q4_1_64(const block_q4_1_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_1_64;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

size_t quantize_q4_1_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK4_1_64 == 0);
    const int64_t nblk = n_per_row / QK4_1_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q4_1_64));
    quantize_row_q4_1_64_ref(src, (block_q4_1_64 *) dst, nrow * n_per_row);
    return nrow * row_size;
}

/* ============================================================
 *  Q5_0_64: symmetric 5-bit, block=64. 5th bit packed in qh (64 bits).
 * ============================================================ */

void quantize_row_q5_0_64_ref(const float * GGML_RESTRICT x, block_q5_0_64 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK5_0_64;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) { amax = fabsf(v); max = v; }
        }

        const float d  = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        uint64_t qh = 0;
        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= ((uint64_t)((xi0 & 0x10u) >> 4)) << (j + 0);
            qh |= ((uint64_t)((xi1 & 0x10u) >> 4)) << (j + qk/2);
        }

        memcpy(y[i].qh, &qh, sizeof(qh));
    }
}

void dequantize_row_q5_0_64(const block_q5_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK5_0_64;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        uint64_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0))            << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + qk/2 - 4))         ) & 0x10;

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

size_t quantize_q5_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK5_0_64 == 0);
    const int64_t nblk = n_per_row / QK5_0_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q5_0_64));
    quantize_row_q5_0_64_ref(src, (block_q5_0_64 *) dst, nrow * n_per_row);
    return nrow * row_size;
}

/* ============================================================
 *  Q5_1_64: asymmetric 5-bit (delta + min), block=64.
 * ============================================================ */

void quantize_row_q5_1_64_ref(const float * GGML_RESTRICT x, block_q5_1_64 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK5_1_64;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 5) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        uint64_t qh = 0;
        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + 0    + j] - min)*id;
            const float x1 = (x[i*qk + qk/2 + j] - min)*id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= ((uint64_t)((xi0 & 0x10u) >> 4)) << (j + 0);
            qh |= ((uint64_t)((xi1 & 0x10u) >> 4)) << (j + qk/2);
        }

        memcpy(y[i].qh, &qh, sizeof(qh));
    }
}

void dequantize_row_q5_1_64(const block_q5_1_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK5_1_64;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);

        uint64_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0))            << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + qk/2 - 4))         ) & 0x10;

            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >>   4) | xh_1;

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

size_t quantize_q5_1_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK5_1_64 == 0);
    const int64_t nblk = n_per_row / QK5_1_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q5_1_64));
    quantize_row_q5_1_64_ref(src, (block_q5_1_64 *) dst, nrow * n_per_row);
    return nrow * row_size;
}

/* ============================================================
 *  Q8_1_64: symmetric 8-bit with running sum, block=64.
 *  s = d * sum(qs) feeds the min-correction in q4_1/q5_1 dots.
 * ============================================================ */

void quantize_row_q8_1_64_ref(const float * GGML_RESTRICT x, block_q8_1_64 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_1_64;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) {
            amax = MAX(amax, fabsf(x[i*qk + j]));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        int sum = 0;
        for (int j = 0; j < qk/2; ++j) {
            const float v0 = x[i*qk +      j]*id;
            const float v1 = x[i*qk + qk/2 + j]*id;

            y[i].qs[       j] = roundf(v0);
            y[i].qs[qk/2 + j] = roundf(v1);

            sum += y[i].qs[       j];
            sum += y[i].qs[qk/2 + j];
        }

        y[i].s = GGML_FP32_TO_FP16(sum*d);
    }
}

void dequantize_row_q8_1_64(const block_q8_1_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_1_64;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

size_t quantize_q8_1_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK8_1_64 == 0);
    const int64_t nblk = n_per_row / QK8_1_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q8_1_64));
    quantize_row_q8_1_64_ref(src, (block_q8_1_64 *) dst, nrow * n_per_row);
    return nrow * row_size;
}

/* ============================================================
 *  Q4_K_64: K-quant, super-block 256, sub-block 64 (4 per super-block).
 *  Mirrors quantize_row_q4_K_ref with the sub-block loop at 64 elements
 *  and the compact 6-byte scale/min packing.
 * ============================================================ */

void quantize_row_q4_K_64_ref(const float * GGML_RESTRICT x, block_q4_K_64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb   = k / QK_K_64;
    const int nsub = QK_K_64 / 64; // 4 sub-blocks of 64

    uint8_t L[QK_K_64];
    uint8_t Laux[64];
    float   weights[64];
    float   mins[QK_K_64/64];
    float   scales[QK_K_64/64];
    uint8_t sc6[QK_K_64/64], mn6[QK_K_64/64];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // we deduct the min, so scales stay positive
        float max_min   = 0;
        for (int j = 0; j < nsub; ++j) {
            float sum_x2 = 0;
            for (int l = 0; l < 64; ++l) sum_x2 += x[64*j + l] * x[64*j + l];
            float av_x = sqrtf(sum_x2/64);
            for (int l = 0; l < 64; ++l) weights[l] = av_x + fabsf(x[64*j + l]);
            scales[j] = reex_make_qkx2_quants(64, 15, x + 64*j, weights, L + 64*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j]   > max_min)   max_min   = mins[j];
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < nsub; ++j) {
            uint8_t ls = reex_nearest_int(inv_scale*scales[j]);
            uint8_t lm = reex_nearest_int(inv_min*mins[j]);
            sc6[j] = MIN(63, ls);
            mn6[j] = MIN(63, lm);
        }
        q4_K_64_pack_scales(y[i].scales, sc6, mn6);
        y[i].d    = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < nsub; ++j) {
            get_scale_min_k4_64(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 64; ++ii) {
                int l = reex_nearest_int((x[64*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[64*j + ii] = l;
            }
        }

        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K_64; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K_64;
    }
}

void dequantize_row_q4_K_64(const block_q4_K_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb = k / QK_K_64;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;
        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        uint8_t sc, m;
        for (int j = 0; j < QK_K_64/64; ++j) {
            get_scale_min_k4_64(j, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l]  >> 4) - m1;
            q += 32;
        }
    }
}

size_t quantize_q4_K_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights; // minimal closed loop: round-to-nearest, imatrix TODO
    assert(n_per_row % QK_K_64 == 0);
    const int64_t nblk = n_per_row / QK_K_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q4_K_64));
    quantize_row_q4_K_64_ref(src, (block_q4_K_64 *) dst, nrow * n_per_row);
    return nrow * row_size;
}

/* ============================================================
 *  Q2_K_64: 2-bit K-quant, super-block 256, sub-block 64.
 *  x = d*scale*q - dmin*min. scales[j] = 4-bit scale | 4-bit min.
 * ============================================================ */

void quantize_row_q2_K_64_ref(const float * GGML_RESTRICT x, block_q2_K_64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb   = k / QK_K_64;
    const int nsub = QK_K_64 / 64; // 4

    uint8_t L[QK_K_64];
    uint8_t Laux[64];
    float   weights[64];
    float   mins[QK_K_64/64];
    float   scales[QK_K_64/64];
    const float q4scale = 15.f;

    for (int i = 0; i < nb; i++) {
        float max_scale = 0;
        float max_min   = 0;
        for (int j = 0; j < nsub; ++j) {
            for (int l = 0; l < 64; ++l) weights[l] = fabsf(x[64*j + l]);
            scales[j] = reex_make_qkx2_quants(64, 3, x + 64*j, weights, L + 64*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j]   > max_min)   max_min   = mins[j];
        }

        uint8_t sc[QK_K_64/64], mn[QK_K_64/64];
        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < nsub; ++j) sc[j] = (uint8_t) reex_nearest_int(iscale*scales[j]);
            y[i].d = GGML_FP32_TO_FP16(max_scale/q4scale);
        } else {
            for (int j = 0; j < nsub; ++j) sc[j] = 0;
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < nsub; ++j) mn[j] = (uint8_t) reex_nearest_int(iscale*mins[j]);
            y[i].dmin = GGML_FP32_TO_FP16(max_min/q4scale);
        } else {
            for (int j = 0; j < nsub; ++j) mn[j] = 0;
            y[i].dmin = GGML_FP32_TO_FP16(0.f);
        }
        for (int j = 0; j < nsub; ++j) y[i].scales[j] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4);

        for (int j = 0; j < nsub; ++j) {
            const float d = GGML_FP16_TO_FP32(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 64; ++ii) {
                int l = reex_nearest_int((x[64*j + ii] + dm)/d);
                l = MAX(0, MIN(3, l));
                L[64*j + ii] = l;
            }
        }

        memset(y[i].qs, 0, QK_K_64/4);
        for (int e = 0; e < QK_K_64; ++e) {
            y[i].qs[e >> 2] |= (uint8_t)((L[e] & 3) << (2*(e & 3)));
        }

        x += QK_K_64;
    }
}

void dequantize_row_q2_K_64(const block_q2_K_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb = k / QK_K_64;

    for (int i = 0; i < nb; i++) {
        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);
        for (int j = 0; j < QK_K_64/64; ++j) {
            const uint8_t sc = x[i].scales[j];
            const float dl = d   * (sc & 0xF);
            const float ml = min * (sc >> 4);
            for (int ii = 0; ii < 64; ++ii) {
                const int e = 64*j + ii;
                const int q = (x[i].qs[e >> 2] >> (2*(e & 3))) & 3;
                *y++ = dl * q - ml;
            }
        }
    }
}

size_t quantize_q2_K_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK_K_64 == 0);
    const int64_t nblk = n_per_row / QK_K_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q2_K_64));
    quantize_row_q2_K_64_ref(src, (block_q2_K_64 *) dst, nrow * n_per_row);
    return nrow * row_size;
}

/* ============================================================
 *  Q3_K_64: 3-bit K-quant, super-block 256, sub-block 64.
 *  x = d*scale*(q-4), scale signed [-32,31]. low 2 bits in qs, 3rd bit in hmask.
 * ============================================================ */

void quantize_row_q3_K_64_ref(const float * GGML_RESTRICT x, block_q3_K_64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb   = k / QK_K_64;
    const int nsub = QK_K_64 / 64; // 4

    int8_t L[QK_K_64];
    float  scales[QK_K_64/64];

    for (int i = 0; i < nb; i++) {
        memset(y[i].scales, 0, 4);

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < nsub; ++j) {
            scales[j] = reex_make_q3_quants(64, 4, x + 64*j, L + 64*j, true);
            float a = fabsf(scales[j]);
            if (a > amax) { amax = a; max_scale = scales[j]; }
        }

        uint8_t sc6[QK_K_64/64];
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < nsub; ++j) {
                int l = reex_nearest_int(iscale*scales[j]);
                l = MAX(-32, MIN(31, l)) + 32; // [0,63]
                sc6[j] = (uint8_t) l;
            }
            y[i].d = GGML_FP32_TO_FP16(1/iscale);
        } else {
            for (int j = 0; j < nsub; ++j) sc6[j] = 32; // biased zero
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        q64_pack4x6(y[i].scales, sc6);

        for (int j = 0; j < nsub; ++j) {
            const int sc = (int) q64_unpack4x6(j, y[i].scales) - 32;
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            for (int ii = 0; ii < 64; ++ii) {
                int l = reex_nearest_int(x[64*j + ii]/d);
                l = MAX(-4, MIN(3, l));
                L[64*j + ii] = (int8_t)(l + 4); // [0,7]
            }
        }

        memset(y[i].hmask, 0, QK_K_64/8);
        memset(y[i].qs,    0, QK_K_64/4);
        for (int e = 0; e < QK_K_64; ++e) {
            const uint8_t v = (uint8_t) L[e]; // [0,7]
            y[i].qs[e >> 2] |= (uint8_t)((v & 3) << (2*(e & 3)));
            if (v & 4) y[i].hmask[e >> 3] |= (uint8_t)(1u << (e & 7));
        }

        x += QK_K_64;
    }
}

void dequantize_row_q3_K_64(const block_q3_K_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb = k / QK_K_64;

    for (int i = 0; i < nb; i++) {
        const float d_all = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK_K_64/64; ++j) {
            const int sc = (int) q64_unpack4x6(j, x[i].scales) - 32;
            const float dl = d_all * sc;
            for (int ii = 0; ii < 64; ++ii) {
                const int e = 64*j + ii;
                const int low2 = (x[i].qs[e >> 2] >> (2*(e & 3))) & 3;
                const int hbit = (x[i].hmask[e >> 3] >> (e & 7)) & 1;
                const int q3 = low2 | (hbit << 2);
                *y++ = dl * (q3 - 4);
            }
        }
    }
}

size_t quantize_q3_K_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK_K_64 == 0);
    const int64_t nblk = n_per_row / QK_K_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q3_K_64));
    quantize_row_q3_K_64_ref(src, (block_q3_K_64 *) dst, nrow * n_per_row);
    return nrow * row_size;
}

/* ============================================================
 *  Q5_K_64: 5-bit K-quant, super-block 256, sub-block 64.
 *  x = d*scale*q - dmin*min. low 4 bits in qs, 5th bit in qh.
 * ============================================================ */

void quantize_row_q5_K_64_ref(const float * GGML_RESTRICT x, block_q5_K_64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb   = k / QK_K_64;
    const int nsub = QK_K_64 / 64; // 4

    uint8_t L[QK_K_64];
    uint8_t Laux[64];
    float   weights[64];
    float   mins[QK_K_64/64];
    float   scales[QK_K_64/64];
    uint8_t sc6[QK_K_64/64], mn6[QK_K_64/64];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0;
        float max_min   = 0;
        for (int j = 0; j < nsub; ++j) {
            float sum_x2 = 0;
            for (int l = 0; l < 64; ++l) sum_x2 += x[64*j + l] * x[64*j + l];
            float av_x = sqrtf(sum_x2/64);
            for (int l = 0; l < 64; ++l) weights[l] = av_x + fabsf(x[64*j + l]);
            scales[j] = reex_make_qkx2_quants(64, 31, x + 64*j, weights, L + 64*j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j]   > max_min)   max_min   = mins[j];
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < nsub; ++j) {
            uint8_t ls = (uint8_t) reex_nearest_int(inv_scale*scales[j]);
            uint8_t lm = (uint8_t) reex_nearest_int(inv_min*mins[j]);
            sc6[j] = MIN(63, ls);
            mn6[j] = MIN(63, lm);
        }
        q4_K_64_pack_scales(y[i].scales, sc6, mn6);
        y[i].d    = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < nsub; ++j) {
            get_scale_min_k4_64(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 64; ++ii) {
                int l = reex_nearest_int((x[64*j + ii] + dm)/d);
                l = MAX(0, MIN(31, l));
                L[64*j + ii] = l;
            }
        }

        memset(y[i].qh, 0, QK_K_64/8);
        uint8_t * GGML_RESTRICT ql = y[i].qs;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        for (int n = 0; n < QK_K_64; n += 64) {
            for (int l = 0; l < 32; ++l) {
                ql[l] = (uint8_t)((L[n + l] & 0xF) | ((L[n + l + 32] & 0xF) << 4));
            }
            for (int l = 0; l < 64; ++l) {
                if (L[n + l] & 0x10) qh[l >> 3] |= (uint8_t)(1u << (l & 7));
            }
            ql += 32;
            qh += 8;
        }

        x += QK_K_64;
    }
}

void dequantize_row_q5_K_64(const block_q5_K_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb = k / QK_K_64;

    for (int i = 0; i < nb; i++) {
        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);
        const uint8_t * GGML_RESTRICT ql = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        uint8_t sc, m;
        for (int j = 0; j < QK_K_64/64; ++j) {
            get_scale_min_k4_64(j, x[i].scales, &sc, &m);
            const float d1 = d   * sc;
            const float m1 = min * m;
            for (int l = 0; l < 32; ++l) {
                const int hbit = (qh[l >> 3] >> (l & 7)) & 1;
                const int q5 = (ql[l] & 0xF) | (hbit << 4);
                y[l] = d1 * q5 - m1;
            }
            for (int l = 0; l < 32; ++l) {
                const int e = l + 32;
                const int hbit = (qh[e >> 3] >> (e & 7)) & 1;
                const int q5 = (ql[l] >> 4) | (hbit << 4);
                y[e] = d1 * q5 - m1;
            }
            y  += 64;
            ql += 32;
            qh += 8;
        }
    }
}

size_t quantize_q5_K_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK_K_64 == 0);
    const int64_t nblk = n_per_row / QK_K_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q5_K_64));
    quantize_row_q5_K_64_ref(src, (block_q5_K_64 *) dst, nrow * n_per_row);
    return nrow * row_size;
}

/* ============================================================
 *  Q6_K_64: 6-bit K-quant, super-block 256, sub-block 64.
 *  x = d*scale*(q-32), scale int8. low 4 bits in ql, high 2 bits in qh.
 * ============================================================ */

void quantize_row_q6_K_64_ref(const float * GGML_RESTRICT x, block_q6_K_64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb   = k / QK_K_64;
    const int nsub = QK_K_64 / 64; // 4

    int8_t L[QK_K_64];
    float  scales[QK_K_64/64];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0;
        float max_abs_scale = 0;
        for (int j = 0; j < nsub; ++j) {
            const float scale = reex_make_qx_quants(64, 32, x + 64*j, L + 64*j, 1, NULL);
            scales[j] = scale;
            const float a = fabsf(scale);
            if (a > max_abs_scale) { max_abs_scale = a; max_scale = scale; }
        }

        if (max_abs_scale < GGML_Q64_GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K_64));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K_64;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int j = 0; j < nsub; ++j) {
            y[i].scales[j] = (int8_t) MIN(127, reex_nearest_int(iscale*scales[j]));
        }

        for (int j = 0; j < nsub; ++j) {
            const float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) continue;
            for (int ii = 0; ii < 64; ++ii) {
                int l = reex_nearest_int(x[64*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[64*j + ii] = (int8_t)(l + 32); // [0,63]
            }
        }

        memset(y[i].ql, 0, QK_K_64/2);
        memset(y[i].qh, 0, QK_K_64/4);
        for (int e = 0; e < QK_K_64; ++e) {
            const uint8_t v = (uint8_t) L[e]; // [0,63]
            y[i].ql[e >> 1] |= (uint8_t)((v & 0xF) << (4*(e & 1)));
            y[i].qh[e >> 2] |= (uint8_t)(((v >> 4) & 3) << (2*(e & 3)));
        }

        x += QK_K_64;
    }
}

void dequantize_row_q6_K_64(const block_q6_K_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb = k / QK_K_64;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK_K_64/64; ++j) {
            const int sc = x[i].scales[j];
            const float dl = d * sc;
            for (int ii = 0; ii < 64; ++ii) {
                const int e = 64*j + ii;
                const int low4 = (x[i].ql[e >> 1] >> (4*(e & 1))) & 0xF;
                const int hi2  = (x[i].qh[e >> 2] >> (2*(e & 3))) & 3;
                const int q6 = low4 | (hi2 << 4);
                *y++ = dl * (q6 - 32);
            }
        }
    }
}

size_t quantize_q6_K_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK_K_64 == 0);
    const int64_t nblk = n_per_row / QK_K_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q6_K_64));
    quantize_row_q6_K_64_ref(src, (block_q6_K_64 *) dst, nrow * n_per_row);
    return nrow * row_size;
}

/* ============================================================
 *  SYMMETRIC K-quant block-64 (signed sub-block scale, no min):
 *    x = d * scale * (q - mid), scale signed-biased (int6 -> +32 / int4 -> +8).
 *  Scale quantization mirrors Q3_K_64 (iscale = -range/max_scale, biased pack).
 *  Q2_K_64S: 2-bit/int4 scale, Q4_K_64S/Q5_K_64S: 4/5-bit/int6 scale.
 * ============================================================ */

// Q5_K_64S: 5-bit, int6 signed scale, mid=16. low 4 bits in qs, 5th bit in qh.
void quantize_row_q5_K_64S_ref(const float * GGML_RESTRICT x, block_q5_K_64S * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb   = k / QK_K_64;
    const int nsub = QK_K_64 / 64; // 4

    int8_t L[QK_K_64];
    float  scales[QK_K_64/64];

    for (int i = 0; i < nb; i++) {
        memset(y[i].scales, 0, 4);

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < nsub; ++j) {
            scales[j] = reex_make_qx_quants(64, 16, x + 64*j, L + 64*j, 1, NULL);
            const float a = fabsf(scales[j]);
            if (a > amax) { amax = a; max_scale = scales[j]; }
        }

        uint8_t sc6[QK_K_64/64];
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < nsub; ++j) {
                int l = reex_nearest_int(iscale*scales[j]);
                l = MAX(-32, MIN(31, l)) + 32; // [0,63]
                sc6[j] = (uint8_t) l;
            }
            y[i].d = GGML_FP32_TO_FP16(1/iscale);
        } else {
            for (int j = 0; j < nsub; ++j) sc6[j] = 32; // biased zero
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        q64_pack4x6(y[i].scales, sc6);

        for (int j = 0; j < nsub; ++j) {
            const int sc = (int) q64_unpack4x6(j, y[i].scales) - 32;
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            for (int ii = 0; ii < 64; ++ii) {
                int l = reex_nearest_int(x[64*j + ii]/d);
                l = MAX(-16, MIN(15, l));
                L[64*j + ii] = (int8_t)(l + 16); // [0,31]
            }
        }

        memset(y[i].qs, 0, QK_K_64/2);
        memset(y[i].qh, 0, QK_K_64/8);
        uint8_t * GGML_RESTRICT ql = y[i].qs;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        for (int n = 0; n < QK_K_64; n += 64) {
            for (int l = 0; l < 32; ++l) {
                ql[l] = (uint8_t)((L[n + l] & 0xF) | ((L[n + l + 32] & 0xF) << 4));
            }
            for (int l = 0; l < 64; ++l) {
                if (L[n + l] & 0x10) qh[l >> 3] |= (uint8_t)(1u << (l & 7));
            }
            ql += 32;
            qh += 8;
        }

        x += QK_K_64;
    }
}

void dequantize_row_q5_K_64S(const block_q5_K_64S * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb = k / QK_K_64;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * GGML_RESTRICT ql = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        for (int j = 0; j < QK_K_64/64; ++j) {
            const int sc = (int) q64_unpack4x6(j, x[i].scales) - 32;
            const float dl = d * sc;
            for (int l = 0; l < 32; ++l) {
                const int hbit = (qh[l >> 3] >> (l & 7)) & 1;
                const int q5 = (ql[l] & 0xF) | (hbit << 4);
                y[l] = dl * (q5 - 16);
            }
            for (int l = 0; l < 32; ++l) {
                const int e = l + 32;
                const int hbit = (qh[e >> 3] >> (e & 7)) & 1;
                const int q5 = (ql[l] >> 4) | (hbit << 4);
                y[e] = dl * (q5 - 16);
            }
            y  += 64;
            ql += 32;
            qh += 8;
        }
    }
}

size_t quantize_q5_K_64S(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK_K_64 == 0);
    const int64_t nblk = n_per_row / QK_K_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q5_K_64S));
    quantize_row_q5_K_64S_ref(src, (block_q5_K_64S *) dst, nrow * n_per_row);
    return nrow * row_size;
}

// Q4_K_64S: 4-bit, int6 signed scale, mid=8. low/high nibble in qs.
void quantize_row_q4_K_64S_ref(const float * GGML_RESTRICT x, block_q4_K_64S * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb   = k / QK_K_64;
    const int nsub = QK_K_64 / 64; // 4

    int8_t L[QK_K_64];
    float  scales[QK_K_64/64];

    for (int i = 0; i < nb; i++) {
        memset(y[i].scales, 0, 4);

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < nsub; ++j) {
            scales[j] = reex_make_qx_quants(64, 8, x + 64*j, L + 64*j, 1, NULL);
            const float a = fabsf(scales[j]);
            if (a > amax) { amax = a; max_scale = scales[j]; }
        }

        uint8_t sc6[QK_K_64/64];
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < nsub; ++j) {
                int l = reex_nearest_int(iscale*scales[j]);
                l = MAX(-32, MIN(31, l)) + 32; // [0,63]
                sc6[j] = (uint8_t) l;
            }
            y[i].d = GGML_FP32_TO_FP16(1/iscale);
        } else {
            for (int j = 0; j < nsub; ++j) sc6[j] = 32;
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        q64_pack4x6(y[i].scales, sc6);

        for (int j = 0; j < nsub; ++j) {
            const int sc = (int) q64_unpack4x6(j, y[i].scales) - 32;
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            for (int ii = 0; ii < 64; ++ii) {
                int l = reex_nearest_int(x[64*j + ii]/d);
                l = MAX(-8, MIN(7, l));
                L[64*j + ii] = (int8_t)(l + 8); // [0,15]
            }
        }

        memset(y[i].qs, 0, QK_K_64/2);
        uint8_t * GGML_RESTRICT ql = y[i].qs;
        for (int n = 0; n < QK_K_64; n += 64) {
            for (int l = 0; l < 32; ++l) {
                ql[l] = (uint8_t)((L[n + l] & 0xF) | ((L[n + l + 32] & 0xF) << 4));
            }
            ql += 32;
        }

        x += QK_K_64;
    }
}

void dequantize_row_q4_K_64S(const block_q4_K_64S * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb = k / QK_K_64;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * GGML_RESTRICT ql = x[i].qs;
        for (int j = 0; j < QK_K_64/64; ++j) {
            const int sc = (int) q64_unpack4x6(j, x[i].scales) - 32;
            const float dl = d * sc;
            for (int l = 0; l < 32; ++l) {
                y[l]      = dl * ((ql[l] & 0xF) - 8);
                y[l + 32] = dl * ((ql[l] >>  4) - 8);
            }
            y  += 64;
            ql += 32;
        }
    }
}

size_t quantize_q4_K_64S(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK_K_64 == 0);
    const int64_t nblk = n_per_row / QK_K_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q4_K_64S));
    quantize_row_q4_K_64S_ref(src, (block_q4_K_64S *) dst, nrow * n_per_row);
    return nrow * row_size;
}

// Q2_K_64S: 2-bit, int4 signed scale, mid=2. 2-bit quants sequential.
void quantize_row_q2_K_64S_ref(const float * GGML_RESTRICT x, block_q2_K_64S * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb   = k / QK_K_64;
    const int nsub = QK_K_64 / 64; // 4

    int8_t L[QK_K_64];
    float  scales[QK_K_64/64];

    for (int i = 0; i < nb; i++) {
        memset(y[i].scales, 0, 2);

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < nsub; ++j) {
            scales[j] = reex_make_qx_quants(64, 2, x + 64*j, L + 64*j, 1, NULL);
            const float a = fabsf(scales[j]);
            if (a > amax) { amax = a; max_scale = scales[j]; }
        }

        uint8_t sc4[QK_K_64/64];
        if (max_scale) {
            float iscale = -8.f/max_scale;
            for (int j = 0; j < nsub; ++j) {
                int l = reex_nearest_int(iscale*scales[j]);
                l = MAX(-8, MIN(7, l)) + 8; // [0,15]
                sc4[j] = (uint8_t) l;
            }
            y[i].d = GGML_FP32_TO_FP16(1/iscale);
        } else {
            for (int j = 0; j < nsub; ++j) sc4[j] = 8; // biased zero
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        q64_pack4x4(y[i].scales, sc4);

        for (int j = 0; j < nsub; ++j) {
            const int sc = (int) q64_unpack4x4(j, y[i].scales) - 8;
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            for (int ii = 0; ii < 64; ++ii) {
                int l = reex_nearest_int(x[64*j + ii]/d);
                l = MAX(-2, MIN(1, l));
                L[64*j + ii] = (int8_t)(l + 2); // [0,3]
            }
        }

        memset(y[i].qs, 0, QK_K_64/4);
        for (int e = 0; e < QK_K_64; ++e) {
            y[i].qs[e >> 2] |= (uint8_t)((L[e] & 3) << (2*(e & 3)));
        }

        x += QK_K_64;
    }
}

void dequantize_row_q2_K_64S(const block_q2_K_64S * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K_64 == 0);
    const int nb = k / QK_K_64;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK_K_64/64; ++j) {
            const int sc = (int) q64_unpack4x4(j, x[i].scales) - 8;
            const float dl = d * sc;
            for (int ii = 0; ii < 64; ++ii) {
                const int e = 64*j + ii;
                const int q = (x[i].qs[e >> 2] >> (2*(e & 3))) & 3;
                *y++ = dl * (q - 2);
            }
        }
    }
}

size_t quantize_q2_K_64S(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    assert(n_per_row % QK_K_64 == 0);
    const int64_t nblk = n_per_row / QK_K_64;
    const size_t row_size = (size_t)(nblk * sizeof(block_q2_K_64S));
    quantize_row_q2_K_64S_ref(src, (block_q2_K_64S *) dst, nrow * n_per_row);
    return nrow * row_size;
}

#endif /* GGML_USE_REEX_Q64 */
