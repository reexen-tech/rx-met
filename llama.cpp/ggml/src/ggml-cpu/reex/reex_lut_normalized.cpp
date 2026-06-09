#ifdef GGML_USE_REEX

#include "reex_lut_normalized.h"
#include "reex_lut_direct.h"
#include "reex/ggml-reex-lut-data.h"

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define GGML_REEX_LN2_F    0.69314718055994530941723212145818f
#define GGML_REEX_SQRT2_F  1.4142135623730950488016887242097f
#define GGML_REEX_INV_SQRT2_F 0.70710678118654752440084436210485f

static inline void normalize_to_1_2(float x, float *mantissa, int *exponent) {
    int e;
    *mantissa = (float)frexp((double)x, &e);
    *mantissa *= 2.0f;
    *exponent = e - 1;
}

static inline int floor_divide(int a, int b) {
    return (int)floor((double)a / (double)b);
}

static inline float reconstruct_reciprocal(float core, int exponent) {
    return core * exp2f(-(float)exponent);
}
static inline float reconstruct_rsqrt(float core, int exponent) {
    int k = floor_divide(exponent, 2);
    int parity = exponent - 2 * k;
    float base = exp2f(-(float)k);
    float parity_factor = (parity == 1) ? GGML_REEX_INV_SQRT2_F : 1.0f;
    return core * base * parity_factor;
}
static inline float reconstruct_sqrt(float core, int exponent) {
    int k = floor_divide(exponent, 2);
    int parity = exponent - 2 * k;
    float base = exp2f((float)k);
    float parity_factor = (parity == 1) ? GGML_REEX_SQRT2_F : 1.0f;
    return core * base * parity_factor;
}
static inline float reconstruct_log(float core, int exponent) {
    return core + (float)exponent * GGML_REEX_LN2_F;
}
static inline float reconstruct_power_2(float core, int exponent) {
    return core * exp2f(2.0f * (float)exponent);
}

static inline float lut_eval_fp16_normalized(float mantissa,
    const struct ggml_lut_segment_fp16_reex *tbl, int n)
{
    /* Segment selection: mantissa -> fp16 -> f32, then compare with threshold (fp16 -> f32), right-inclusive (<=). */
    float m_for_compare = ggml_fp16_to_fp32(ggml_fp32_to_fp16(mantissa));
    for (int i = 0; i < n; ++i) {
        float seg_threshold = ggml_fp16_to_fp32(tbl[i].threshold);
        if (m_for_compare <= seg_threshold || i == n - 1) {
            float b = ggml_fp16_to_fp32(tbl[i].b);
            float c = ggml_fp16_to_fp32(tbl[i].c);
            /* Full-chain FP16: use quantized mantissa for product (match Python infer_with_float_lut). */
            float bx = b * m_for_compare;
            ggml_fp16_t bx_fp16 = ggml_fp32_to_fp16(bx);
            float bx_f32 = ggml_fp16_to_fp32(bx_fp16);
            float y = bx_f32 + c;
            ggml_fp16_t y_fp16 = ggml_fp32_to_fp16(y);
            return ggml_fp16_to_fp32(y_fp16);
        }
    }
    return 0.0f;
}

static inline float lut_eval_bf16_normalized(float mantissa,
    const struct ggml_lut_segment_bf16_reex *tbl, int n)
{
    /* Segment selection: mantissa -> bf16 -> f32, then compare with threshold (bf16 -> f32), right-inclusive (<=). */
    float m_for_compare = ggml_bf16_to_fp32(ggml_fp32_to_bf16(mantissa));
    for (int i = 0; i < n; ++i) {
        float seg_threshold = ggml_bf16_to_fp32(tbl[i].threshold);
        if (m_for_compare <= seg_threshold || i == n - 1) {
            float b = ggml_bf16_to_fp32(tbl[i].b);
            float c = ggml_bf16_to_fp32(tbl[i].c);
            /* Full-chain BF16: use quantized mantissa for product (match Python infer_with_float_lut). */
            float bx = b * m_for_compare;
            ggml_bf16_t bx_bf16 = ggml_fp32_to_bf16(bx);
            float bx_f32 = ggml_bf16_to_fp32(bx_bf16);
            float y = bx_f32 + c;
            ggml_bf16_t y_bf16 = ggml_fp32_to_bf16(y);
            return ggml_bf16_to_fp32(y_bf16);
        }
    }
    return 0.0f;
}

/* Mixed-FP16: FP32 table (coefficients_float), mantissa in FP32, core = b*m+c in FP32 (match Python). */
static inline float lut_eval_mixed_fp16_normalized_fp32(float mantissa,
    const float *threshold, const float *b, const float *c)
{
    for (int i = 0; i < GGML_LUT_NUM_SEGMENTS_REEX; ++i) {
        if (mantissa <= threshold[i] || i == GGML_LUT_NUM_SEGMENTS_REEX - 1) {
            return b[i] * mantissa + c[i];
        }
    }
    return 0.0f;
}

/* x <= 0: return 0 (sqrt) or -100 (log), else normalize -> LUT -> reconstruct */
float ggml_sqrt_lut_fp16_f32_REEX(float x) {
    if (x <= 0.0f) return 0.0f;
    float m; int e;
    normalize_to_1_2(x, &m, &e);
    float core = lut_eval_fp16_normalized(m, ggml_lut_sqrt_fp16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
    return reconstruct_sqrt(core, e);
}
float ggml_sqrt_lut_bf16_f32_REEX(float x) {
    if (x <= 0.0f) return 0.0f;
    float m; int e;
    normalize_to_1_2(x, &m, &e);
    float core = lut_eval_bf16_normalized(m, ggml_lut_sqrt_bf16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
    return reconstruct_sqrt(core, e);
}

float ggml_log_lut_fp16_f32_REEX(float x) {
    if (x <= 0.0f) return -100.0f;
    float m; int e;
    normalize_to_1_2(x, &m, &e);
    float core = lut_eval_fp16_normalized(m, ggml_lut_log_fp16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
    return reconstruct_log(core, e);
}
float ggml_log_lut_bf16_f32_REEX(float x) {
    if (x <= 0.0f) return -100.0f;
    float m; int e;
    normalize_to_1_2(x, &m, &e);
    float core = lut_eval_bf16_normalized(m, ggml_lut_log_bf16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
    return reconstruct_log(core, e);
}

/* sqr (power_2 = x²): direct fp16*fp16 / bf16*bf16 multiply, no LUT. */
float ggml_sqr_fp16_f32_REEX(float x) {
    float x_q = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
    float prod = x_q * x_q;
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(prod));
}
float ggml_sqr_bf16_f32_REEX(float x) {
    float x_q = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
    float prod = x_q * x_q;
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(prod));
}

float ggml_reciprocal_lut_fp16_f32_REEX(float x) {
    if (x == 0.0f) return 0.0f;
    float ax = fabsf(x);
    float m; int e;
    normalize_to_1_2(ax, &m, &e);
    float core = lut_eval_fp16_normalized(m, ggml_lut_reciprocal_fp16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
    float r = reconstruct_reciprocal(core, e);
    return (x < 0.0f) ? -r : r;
}
float ggml_reciprocal_lut_bf16_f32_REEX(float x) {
    if (x == 0.0f) return 0.0f;
    float ax = fabsf(x);
    float m; int e;
    normalize_to_1_2(ax, &m, &e);
    float core = lut_eval_bf16_normalized(m, ggml_lut_reciprocal_bf16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
    float r = reconstruct_reciprocal(core, e);
    return (x < 0.0f) ? -r : r;
}

float ggml_rsqrt_lut_fp16_f32_REEX(float x) {
    if (x <= 0.0f) return 0.0f;
    float m; int e;
    normalize_to_1_2(x, &m, &e);
    float core = lut_eval_fp16_normalized(m, ggml_lut_rsqrt_fp16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
    return reconstruct_rsqrt(core, e);
}
float ggml_rsqrt_lut_bf16_f32_REEX(float x) {
    if (x <= 0.0f) return 0.0f;
    float m; int e;
    normalize_to_1_2(x, &m, &e);
    float core = lut_eval_bf16_normalized(m, ggml_lut_rsqrt_bf16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
    return reconstruct_rsqrt(core, e);
}

/* Mixed-FP16: FP32 compute, only final output rounded to FP16. */
static inline float mixed_fp16_out(float y_f32) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(y_f32));
}

float ggml_sqrt_lut_mixed_fp16_f32_REEX(float x) {
    if (x <= 0.0f) return 0.0f;
    float m; int e;
    normalize_to_1_2(x, &m, &e);
    float core = lut_eval_mixed_fp16_normalized_fp32(m, ggml_reex_sqrt_fp32_threshold, ggml_reex_sqrt_fp32_b, ggml_reex_sqrt_fp32_c);
    return mixed_fp16_out(reconstruct_sqrt(core, e));
}

float ggml_log_lut_mixed_fp16_f32_REEX(float x) {
    if (x <= 0.0f) return -100.0f;
    float m; int e;
    normalize_to_1_2(x, &m, &e);
    float core = lut_eval_mixed_fp16_normalized_fp32(m, ggml_reex_log_fp32_threshold, ggml_reex_log_fp32_b, ggml_reex_log_fp32_c);
    return mixed_fp16_out(reconstruct_log(core, e));
}

float ggml_sqr_mixed_fp16_f32_REEX(float x) {
    float prod_f32 = x * x;
    return mixed_fp16_out(prod_f32);
}

float ggml_reciprocal_lut_mixed_fp16_f32_REEX(float x) {
    if (x == 0.0f) return 0.0f;
    float ax = fabsf(x);
    float m; int e;
    normalize_to_1_2(ax, &m, &e);
    float core = lut_eval_mixed_fp16_normalized_fp32(m, ggml_reex_reciprocal_fp32_threshold, ggml_reex_reciprocal_fp32_b, ggml_reex_reciprocal_fp32_c);
    float r = reconstruct_reciprocal(core, e);
    return mixed_fp16_out((x < 0.0f) ? -r : r);
}

float ggml_rsqrt_lut_mixed_fp16_f32_REEX(float x) {
    if (x <= 0.0f) return 0.0f;
    float m; int e;
    normalize_to_1_2(x, &m, &e);
    float core = lut_eval_mixed_fp16_normalized_fp32(m, ggml_reex_rsqrt_fp32_threshold, ggml_reex_rsqrt_fp32_b, ggml_reex_rsqrt_fp32_c);
    return mixed_fp16_out(reconstruct_rsqrt(core, e));
}

#endif /* GGML_USE_REEX */
