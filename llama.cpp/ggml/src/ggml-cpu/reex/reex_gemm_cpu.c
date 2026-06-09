#ifdef GGML_USE_REEX_GEMM

#include "reex/reex_gemm_cpu.h"
#include "ggml-quants.h"
#include "quants.h"
#include "simd-mappings.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define REEX_Q16_SCALE 32767.f

/* ================================================================
 *  Q16 (INT16) 激活
 * ================================================================ */

void ggml_quantize_row_f32_to_q16_scale_row_reex(
    const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k)
{
    assert(k > 0 && (k % QK_K) == 0);
    const int64_t nb = k / QK_K;
    block_q16_K * GGML_RESTRICT yb = (block_q16_K *)y;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * QK_K;
        float amax = 0.f;
        for (int j = 0; j < QK_K; j++) {
            float a = fabsf(xb[j]);
            if (a > amax) amax = a;
        }
        if (amax < 1e-9f) amax = 1e-9f;
        float scale = REEX_Q16_SCALE / amax;
        yb[i].d = amax / REEX_Q16_SCALE;
        for (int j = 0; j < QK_K; j++) {
            yb[i].qs[j] = (int16_t)(roundf(xb[j] * scale));
        }
    }
}

/*
 * 整数域实现，参考原生 ggml_vec_dot_q4_K_q8_K_generic 的模式：
 *
 *   Q4_K:  value[i] = d * scale_s * q4[i] - dmin * min_s
 *   Q16:   value[i] = y_scale * q16[i]
 *
 *   dot = y_scale * [ d * Σ(scale_s * q4[i] * q16[i])       // 主项：整数域累加
 *                    - dmin * Σ(min_s * q16_bsum[j]) ]       // min 偏移项
 *
 * 与 Q8 版本的差异：
 *   - q4×q16 乘积为 int32（Q8 版是 int16），主累加器用 int64 确保安全
 *   - Q16 无预存 bsums，需在线计算每 16 元素的和
 */
void ggml_vec_dot_q4_K_q16_reex(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    (void)bx; (void)by; (void)bs; (void)nrc;
    assert(nrc == 1);
    assert(n > 0 && (n % QK_K) == 0);

    const block_q4_K * GGML_RESTRICT x = (const block_q4_K *)vx;
    const block_q16_K * GGML_RESTRICT yblk = (const block_q16_K *)vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    const uint8_t * scales = (const uint8_t *)&utmp[0];
    const uint8_t * mins   = (const uint8_t *)&utmp[2];

    int8_t  aux8[QK_K];
    float   sums[8];
    int64_t aux64[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const float y_scale = yblk[i].d;
        const int16_t * GGML_RESTRICT q16 = yblk[i].qs;
        memset(aux64, 0, 8 * sizeof(int64_t));

        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            a += 32; q4 += 32;
        }

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int64_t sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            int32_t bsum = 0;
            for (int l = 0; l < 16; ++l) bsum += q16[j * 16 + l];
            sumi += (int64_t)bsum * mins[j / 2];
        }

        a = aux8;
        const int16_t * q16p = q16;
        int is = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux64[l] += (int64_t)scale * ((int32_t)q16p[l] * a[l]);
            q16p += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux64[l] += (int64_t)scale * ((int32_t)q16p[l] * a[l]);
            q16p += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux64[l] += (int64_t)scale * ((int32_t)q16p[l] * a[l]);
            q16p += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux64[l] += (int64_t)scale * ((int32_t)q16p[l] * a[l]);
            q16p += 8; a += 8;
        }

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y_scale;
        for (int l = 0; l < 8; ++l) sums[l] += d * (float)aux64[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y_scale;
        sumf -= dmin * (float)sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

/* ================================================================
 *  Q8 (INT8) 激活：直接复用原生 ggml_vec_dot_q4_K_q8_K
 *  原生实现在量化整数域直接做 int8×int4 乘积累加 + scale，
 *  有 AVX2/NEON 等 SIMD 优化，性能远高于 dequant-dot。
 *  注意：调用前须确保 ggml_cpu_init() 已执行（初始化 fp16→fp32 查表）。
 * ================================================================ */

void ggml_quantize_row_f32_to_q8_scale_row_reex(
    const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k)
{
    quantize_row_q8_K_ref(x, (block_q8_K *)y, k);
}

void ggml_vec_dot_q4_K_q8_reex(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    ggml_vec_dot_q4_K_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

/* ================================================================
 *  FLOAT 激活：不量化激活，仅反量化 Q4_K 权重做 float 点积
 *  作为"仅权重量化"的精度参考基线
 * ================================================================ */

void ggml_vec_dot_q4_K_f32_reex(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc)
{
    (void)bx; (void)by; (void)bs; (void)nrc;
    assert(nrc == 1);
    assert(n > 0 && (n % QK_K) == 0);

    const block_q4_K * GGML_RESTRICT x = (const block_q4_K *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int nb = n / QK_K;

    float tmp[QK_K];
    double sum = 0.0;
    for (int i = 0; i < nb; i++) {
        dequantize_row_q4_K(x + i, tmp, QK_K);
        const float * yb = y + (size_t)i * QK_K;
        for (int j = 0; j < QK_K; j++)
            sum += (double)tmp[j] * (double)yb[j];
    }
    *s = (float)sum;
}

/* ================================================================
 *  辅助：层选择
 * ================================================================ */

int ggml_reex_get_target_layer(void) {
    static int cached = -2;
    if (cached == -2) {
        const char * v = getenv("REEX_TARGET_LAYER");
        cached = (v && v[0]) ? atoi(v) : -1;
    }
    return cached;
}

int ggml_reex_tensor_layer(const char * name) {
    if (!name || !name[0]) return -1;
    const char * p = name + strlen(name) - 1;
    while (p > name && *p >= '0' && *p <= '9') p--;
    if (*p == '-' && p[1] >= '0' && p[1] <= '9') {
        return atoi(p + 1);
    }
    return -1;
}

#endif /* GGML_USE_REEX_GEMM */
