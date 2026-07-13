#ifdef GGML_USE_REEX_Q64

#include "ggml-quants.h"      // pulls ggml-common.h (DECL_C) -> upstream block_q8_K

#include "reex/reex_q64_quants.h"
#include "reex/ggml-reex-q64.h"
#include "reex/ggml-reex-q64-common.h"

#include "simd-mappings.h"

#include <assert.h>
#include <string.h>

#define UNUSED GGML_UNUSED

/* from_float wrappers (scalar; reuse base reference quantizers) */
void quantize_row_q4_0_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_0_64_ref(x, (block_q4_0_64 *) y, k);
}

void quantize_row_q8_0_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_0_64_ref(x, (block_q8_0_64 *) y, k);
}

void quantize_row_q4_1_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_1_64_ref(x, (block_q4_1_64 *) y, k);
}

void quantize_row_q5_0_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_0_64_ref(x, (block_q5_0_64 *) y, k);
}

void quantize_row_q5_1_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_1_64_ref(x, (block_q5_1_64 *) y, k);
}

void quantize_row_q8_1_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_1_64_ref(x, (block_q8_1_64 *) y, k);
}

void quantize_row_q4_K_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_K_64_ref(x, (block_q4_K_64 *) y, k);
}

void quantize_row_q2_K_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q2_K_64_ref(x, (block_q2_K_64 *) y, k);
}

void quantize_row_q3_K_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q3_K_64_ref(x, (block_q3_K_64 *) y, k);
}

void quantize_row_q5_K_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_K_64_ref(x, (block_q5_K_64 *) y, k);
}

void quantize_row_q6_K_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q6_K_64_ref(x, (block_q6_K_64 *) y, k);
}

void quantize_row_q5_K_64S(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_K_64S_ref(x, (block_q5_K_64S *) y, k);
}

void quantize_row_q4_K_64S(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_K_64S_ref(x, (block_q4_K_64S *) y, k);
}

void quantize_row_q2_K_64S(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q2_K_64S_ref(x, (block_q2_K_64S *) y, k);
}

/* W4(block=64) x A8(block=64) integer-domain dot, scalar reference.
 * Mirrors ggml_vec_dot_q4_0_q8_0_generic with QK=64. */
void ggml_vec_dot_q4_0_64_q8_0_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    const int qk = QK8_0_64;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0_64 * GGML_RESTRICT x = (const block_q4_0_64 *) vx;
    const block_q8_0_64 * GGML_RESTRICT y = (const block_q8_0_64 *) vy;

    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = ((x[ib].qs[j] & 0x0F) ^ 0x08) - 0x08; // sign-extend signed 4-bit
            const int v1 = ((x[ib].qs[j] >>   4) ^ 0x08) - 0x08;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, pb);
        sumf += sumi * GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

/* W8(block=64) x A8(block=64) integer-domain dot, scalar reference.
 * Mirrors ggml_vec_dot_q8_0_q8_0_generic with QK=64. */
void ggml_vec_dot_q8_0_64_q8_0_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    const int qk = QK8_0_64;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0_64 * GGML_RESTRICT x = (const block_q8_0_64 *) vx;
    const block_q8_0_64 * GGML_RESTRICT y = (const block_q8_0_64 *) vy;

    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }
        sumi = reex_q64_psum_trunc_b(sumi, pb);
        sumf += sumi * (GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

/* W4_1(block=64) x A8_1(block=64), scalar. Mirrors ggml_vec_dot_q4_1_q8_1_generic. */
void ggml_vec_dot_q4_1_64_q8_1_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    const int qk = QK8_1_64;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q4_1_64 * GGML_RESTRICT x = (const block_q4_1_64 *) vx;
    const block_q8_1_64 * GGML_RESTRICT y = (const block_q8_1_64 *) vy;

    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F);
            const int v1 = (x[ib].qs[j] >>   4);
            sumi0 += v0 * y[ib].qs[j];
            sumi1 += v1 * y[ib].qs[j + qk/2];
        }
        const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, pb);
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi
              +  GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }
    *s = sumf;
}

/* W5_0(block=64) x A8_0(block=64), scalar. Mirrors ggml_vec_dot_q5_0_q8_0_generic. */
void ggml_vec_dot_q5_0_64_q8_0_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    const int qk = QK8_0_64;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q5_0_64 * GGML_RESTRICT x = (const block_q5_0_64 *) vx;
    const block_q8_0_64 * GGML_RESTRICT y = (const block_q8_0_64 *) vy;

    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        uint64_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0))         << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + qk/2 - 4))      ) & 0x10;

            const int u0 = (x[ib].qs[j] & 0x0F) | xh_0;
            const int u1 = (x[ib].qs[j] >>   4) | xh_1;
            const int32_t x0 = (u0 ^ 0x10) - 0x10; // sign-extend signed 5-bit
            const int32_t x1 = (u1 ^ 0x10) - 0x10;

            sumi0 += x0 * y[ib].qs[j];
            sumi1 += x1 * y[ib].qs[j + qk/2];
        }
        const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, pb);
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }
    *s = sumf;
}

/* W5_1(block=64) x A8_1(block=64), scalar. Mirrors ggml_vec_dot_q5_1_q8_1_generic. */
void ggml_vec_dot_q5_1_64_q8_1_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    const int qk = QK8_1_64;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q5_1_64 * GGML_RESTRICT x = (const block_q5_1_64 *) vx;
    const block_q8_1_64 * GGML_RESTRICT y = (const block_q8_1_64 *) vy;

    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        uint64_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0))         << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + qk/2 - 4))      ) & 0x10;

            const int32_t x0 = (x[ib].qs[j] & 0x0F) | xh_0;
            const int32_t x1 = (x[ib].qs[j] >>   4) | xh_1;

            sumi0 += x0 * y[ib].qs[j];
            sumi1 += x1 * y[ib].qs[j + qk/2];
        }
        const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, pb);
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi
              +  GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }
    *s = sumf;
}

/* W8_1(block=64) x A8_1(block=64), scalar symmetric (s field unused). */
void ggml_vec_dot_q8_1_64_q8_1_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    const int qk = QK8_1_64;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q8_1_64 * GGML_RESTRICT x = (const block_q8_1_64 *) vx;
    const block_q8_1_64 * GGML_RESTRICT y = (const block_q8_1_64 *) vy;

    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }
        sumi = reex_q64_psum_trunc_b(sumi, pb);
        sumf += sumi * (GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));
    }
    *s = sumf;
}

/* W(Q4_K_64) x A(Q8_K), scalar. Mirrors ggml_vec_dot_q4_K_q8_K_generic with the
 * sub-block loop at 64 elements (4 per super-block). The q8_K activation keeps
 * its native bsums[16] (16-element sums); each 64-element weight sub-block folds
 * 4 adjacent bsums for the min-correction term. */
void ggml_vec_dot_q4_K_64_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    assert(n % QK_K_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q4_K_64 * GGML_RESTRICT x = (const block_q4_K_64 *) vx;
    const block_q8_K    * GGML_RESTRICT y = (const block_q8_K    *) vy;

    const int nb = n / QK_K_64;

    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;

        // min-correction: sum over sub-blocks of min * (sum of that sub-block's q8)
        int sumi_min = 0;
        for (int j = 0; j < QK_K_64/64; ++j) {
            uint8_t sc, m;
            get_scale_min_k4_64(j, x[i].scales, &sc, &m);
            int bsum = 0;
            for (int b = 0; b < 4; ++b) bsum += y[i].bsums[4*j + b]; // 4x 16-element sums = 64
            sumi_min += bsum * m;
        }

        int32_t sumi = 0;
        for (int j = 0; j < QK_K_64/64; ++j) {
            uint8_t sc, m;
            get_scale_min_k4_64(j, x[i].scales, &sc, &m);
            int32_t acc = 0;
            for (int l = 0; l < 32; ++l) acc += (int32_t)(q4[l] & 0xF) * q8[l];
            for (int l = 0; l < 32; ++l) acc += (int32_t)(q4[l] >>  4) * q8[l + 32];
            acc = reex_q64_psum_trunc_b(acc, pb);
            sumi += sc * acc;
            q4 += 32;
            q8 += 64;
        }

        const float d    = GGML_CPU_FP16_TO_FP32(x[i].d)    * y[i].d;
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf += d * sumi - dmin * sumi_min;
    }
    *s = sumf;
}

/* W(Q2_K_64) x A(Q8_K), scalar. x = d*scale*q - dmin*min, 2-bit sequential quants. */
void ggml_vec_dot_q2_K_64_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    assert(n % QK_K_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q2_K_64 * GGML_RESTRICT x = (const block_q2_K_64 *) vx;
    const block_q8_K    * GGML_RESTRICT y = (const block_q8_K    *) vy;

    const int nb = n / QK_K_64;

    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const int8_t * GGML_RESTRICT q8 = y[i].qs;

        int     sumi_min = 0;
        int32_t sumi     = 0;
        for (int j = 0; j < QK_K_64/64; ++j) {
            const uint8_t scb = x[i].scales[j];
            const int sc = scb & 0xF;
            const int m  = scb >> 4;

            int bsum = 0;
            for (int b = 0; b < 4; ++b) bsum += y[i].bsums[4*j + b];
            sumi_min += bsum * m;

            int32_t acc = 0;
            for (int ii = 0; ii < 64; ++ii) {
                const int e = 64*j + ii;
                const int q = (x[i].qs[e >> 2] >> (2*(e & 3))) & 3;
                acc += q * q8[e];
            }
            acc = reex_q64_psum_trunc_b(acc, pb);
            sumi += sc * acc;
        }

        const float d    = GGML_CPU_FP16_TO_FP32(x[i].d)    * y[i].d;
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf += d * sumi - dmin * sumi_min;
    }
    *s = sumf;
}

/* Shared integer-domain dot for the SYMMETRIC K-quant block-64 types (no min):
 * y = d * sum_s sub_scale_s * sum_e code_{s,e} * q8_{s,e}. Weights are read
 * straight from the HW bit-stream (d, sub_scale, codes) via the q64 bit codec;
 * q8 is the block_q8_K activation (sequential, 64 elems / sub-block). */
static inline float reex_q64_sym_dot_q8_K(const uint8_t * GGML_RESTRICT qs,
    const int8_t * GGML_RESTRICT q8, float y_d, int scale_bits, int w_bits, int pb) {
    const float d = GGML_CPU_FP16_TO_FP32((ggml_fp16_t) q64_get_bits(qs, 0, 16));
    int32_t sumi = 0;
    for (int sblk = 0; sblk < QK_K_64/64; ++sblk) {
        const int sc = q64_get_sbits(qs, q64_sub_scale_bit(scale_bits, w_bits, sblk), scale_bits);
        int64_t bit = q64_code_bit(scale_bits, w_bits, sblk, 0);
        int32_t acc = 0;
        for (int e = 0; e < 64; ++e, bit += w_bits) {
            acc += q64_get_sbits(qs, bit, w_bits) * q8[64*sblk + e];
        }
        acc = reex_q64_psum_trunc_b(acc, pb);
        sumi += sc * acc;
    }
    return d * y_d * sumi;
}

/* W(Q3_K_64) x A(Q8_K), scalar. x = d*scale*q, signed q [-4,3] + signed scale, no min. */
void ggml_vec_dot_q3_K_64_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    assert(n % QK_K_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q3_K_64 * GGML_RESTRICT x = (const block_q3_K_64 *) vx;
    const block_q8_K    * GGML_RESTRICT y = (const block_q8_K    *) vy;

    const int nb = n / QK_K_64;
    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        sumf += reex_q64_sym_dot_q8_K(x[i].qs, y[i].qs, y[i].d, /*scale_bits*/6, /*w_bits*/3, pb);
    }
    *s = sumf;
}

/* W(Q5_K_64) x A(Q8_K), scalar. x = d*scale*q - dmin*min, 5-bit quants. */
void ggml_vec_dot_q5_K_64_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    assert(n % QK_K_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q5_K_64 * GGML_RESTRICT x = (const block_q5_K_64 *) vx;
    const block_q8_K    * GGML_RESTRICT y = (const block_q8_K    *) vy;

    const int nb = n / QK_K_64;

    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT ql = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;

        int     sumi_min = 0;
        int32_t sumi     = 0;
        for (int j = 0; j < QK_K_64/64; ++j) {
            uint8_t sc, m;
            get_scale_min_k4_64(j, x[i].scales, &sc, &m);

            int bsum = 0;
            for (int b = 0; b < 4; ++b) bsum += y[i].bsums[4*j + b];
            sumi_min += bsum * m;

            int32_t acc = 0;
            for (int l = 0; l < 32; ++l) {
                const int hbit = (qh[l >> 3] >> (l & 7)) & 1;
                const int q5 = (ql[l] & 0xF) | (hbit << 4);
                acc += q5 * q8[l];
            }
            for (int l = 0; l < 32; ++l) {
                const int e = l + 32;
                const int hbit = (qh[e >> 3] >> (e & 7)) & 1;
                const int q5 = (ql[l] >> 4) | (hbit << 4);
                acc += q5 * q8[e];
            }
            acc = reex_q64_psum_trunc_b(acc, pb);
            sumi += sc * acc;
            ql += 32;
            qh += 8;
            q8 += 64;
        }

        const float d    = GGML_CPU_FP16_TO_FP32(x[i].d)    * y[i].d;
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf += d * sumi - dmin * sumi_min;
    }
    *s = sumf;
}

/* W(Q6_K_64) x A(Q8_K), scalar. x = d*scale*q, signed q [-32,31], int8 scale, no min. */
void ggml_vec_dot_q6_K_64_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    assert(n % QK_K_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q6_K_64 * GGML_RESTRICT x = (const block_q6_K_64 *) vx;
    const block_q8_K    * GGML_RESTRICT y = (const block_q8_K    *) vy;

    const int nb = n / QK_K_64;
    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        sumf += reex_q64_sym_dot_q8_K(x[i].qs, y[i].qs, y[i].d, /*scale_bits*/8, /*w_bits*/6, pb);
    }
    *s = sumf;
}

/* W(Q5_K_64S) x A(Q8_K), scalar. x = d*scale*q, signed q [-16,15], int6 signed scale, no min. */
void ggml_vec_dot_q5_K_64S_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    assert(n % QK_K_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q5_K_64S * GGML_RESTRICT x = (const block_q5_K_64S *) vx;
    const block_q8_K     * GGML_RESTRICT y = (const block_q8_K     *) vy;

    const int nb = n / QK_K_64;
    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        sumf += reex_q64_sym_dot_q8_K(x[i].qs, y[i].qs, y[i].d, /*scale_bits*/6, /*w_bits*/5, pb);
    }
    *s = sumf;
}

/* W(Q4_K_64S) x A(Q8_K), scalar. x = d*scale*q, signed q [-8,7], int6 signed scale, no min. */
void ggml_vec_dot_q4_K_64S_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    assert(n % QK_K_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q4_K_64S * GGML_RESTRICT x = (const block_q4_K_64S *) vx;
    const block_q8_K     * GGML_RESTRICT y = (const block_q8_K     *) vy;

    const int nb = n / QK_K_64;
    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        sumf += reex_q64_sym_dot_q8_K(x[i].qs, y[i].qs, y[i].d, /*scale_bits*/6, /*w_bits*/4, pb);
    }
    *s = sumf;
}

/* W(Q2_K_64S) x A(Q8_K), scalar. x = d*scale*q, signed q [-2,1], int4 signed scale, no min. */
void ggml_vec_dot_q2_K_64S_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    assert(n % QK_K_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q2_K_64S * GGML_RESTRICT x = (const block_q2_K_64S *) vx;
    const block_q8_K     * GGML_RESTRICT y = (const block_q8_K     *) vy;

    const int nb = n / QK_K_64;
    const int pb = reex_q64_psum_bits();

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        sumf += reex_q64_sym_dot_q8_K(x[i].qs, y[i].qs, y[i].d, /*scale_bits*/4, /*w_bits*/2, pb);
    }
    *s = sumf;
}

#endif /* GGML_USE_REEX_Q64 */
