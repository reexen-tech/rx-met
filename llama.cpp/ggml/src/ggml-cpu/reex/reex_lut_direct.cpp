#ifdef GGML_USE_REEX

#include "reex_lut_direct.h"
#include "reex/ggml-reex-lut-data.h"

#include <cmath>

#define SIGMOID_MIN (-6.0f)
#define SIGMOID_MAX (6.0f)
#define EXP_MIN     (-20.0f)
#define EXP_MAX     (0.0f)

static inline float lut_eval_fp16_direct(float x_clamped,
    const struct ggml_lut_segment_fp16_reex *tbl, int n)
{
    /* Segment selection: x -> fp16 -> f32, then compare with threshold (fp16 -> f32), right-inclusive (<=). */
    float x_for_compare = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x_clamped));
    for (int i = 0; i < n; ++i) {
        float seg_threshold = ggml_fp16_to_fp32(tbl[i].threshold);
        if (x_for_compare <= seg_threshold || i == n - 1) {
            float b = ggml_fp16_to_fp32(tbl[i].b);
            float c = ggml_fp16_to_fp32(tbl[i].c);
            /* Full-chain FP16: use quantized input for product (match Python infer_with_float_lut). */
            float bx = b * x_for_compare;
            ggml_fp16_t bx_fp16 = ggml_fp32_to_fp16(bx);
            float bx_f32 = ggml_fp16_to_fp32(bx_fp16);
            float y = bx_f32 + c;
            ggml_fp16_t y_fp16 = ggml_fp32_to_fp16(y);
            return ggml_fp16_to_fp32(y_fp16);
        }
    }
    return 0.0f;
}

static inline float lut_eval_bf16_direct(float x_clamped,
    const struct ggml_lut_segment_bf16_reex *tbl, int n)
{
    /* Segment selection: x -> bf16 -> f32, then compare with threshold (bf16 -> f32), right-inclusive (<=). */
    float x_for_compare = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x_clamped));
    for (int i = 0; i < n; ++i) {
        float seg_threshold = ggml_bf16_to_fp32(tbl[i].threshold);
        if (x_for_compare <= seg_threshold || i == n - 1) {
            float b = ggml_bf16_to_fp32(tbl[i].b);
            float c = ggml_bf16_to_fp32(tbl[i].c);
            /* Full-chain BF16: use quantized input for product (match Python infer_with_float_lut). */
            float bx = b * x_for_compare;
            ggml_bf16_t bx_bf16 = ggml_fp32_to_bf16(bx);
            float bx_f32 = ggml_bf16_to_fp32(bx_bf16);
            float y = bx_f32 + c;
            ggml_bf16_t y_bf16 = ggml_fp32_to_bf16(y);
            return ggml_bf16_to_fp32(y_bf16);
        }
    }
    return 0.0f;
}

/* Mixed-FP16: use FP32 table (coefficients_float), x in FP32, y = b*x+c in FP32, only output rounded to FP16 (match Python). */
static inline float lut_eval_mixed_fp16_direct_sigmoid(float x_clamped) {
    for (int i = 0; i < GGML_LUT_NUM_SEGMENTS_REEX; ++i) {
        if (x_clamped <= ggml_reex_sigmoid_fp32_threshold[i] || i == GGML_LUT_NUM_SEGMENTS_REEX - 1) {
            float b = ggml_reex_sigmoid_fp32_b[i], c = ggml_reex_sigmoid_fp32_c[i];
            float y_f32 = b * x_clamped + c;
            return ggml_fp16_to_fp32(ggml_fp32_to_fp16(y_f32));
        }
    }
    return 0.0f;
}
static inline float lut_eval_mixed_fp16_direct_exp(float x_clamped) {
    for (int i = 0; i < GGML_LUT_NUM_SEGMENTS_REEX; ++i) {
        if (x_clamped <= ggml_reex_exponential_fp32_threshold[i] || i == GGML_LUT_NUM_SEGMENTS_REEX - 1) {
            float b = ggml_reex_exponential_fp32_b[i], c = ggml_reex_exponential_fp32_c[i];
            float y_f32 = b * x_clamped + c;
            return ggml_fp16_to_fp32(ggml_fp32_to_fp16(y_f32));
        }
    }
    return 0.0f;
}

float ggml_sigmoid_lut_fp16_f32_REEX(float x) {
    if (x < SIGMOID_MIN) return 0.0f;
    if (x > SIGMOID_MAX) return 1.0f;
    return lut_eval_fp16_direct(x, ggml_lut_sigmoid_fp16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
}

float ggml_sigmoid_lut_bf16_f32_REEX(float x) {
    if (x < SIGMOID_MIN) return 0.0f;
    if (x > SIGMOID_MAX) return 1.0f;
    return lut_eval_bf16_direct(x, ggml_lut_sigmoid_bf16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
}

float ggml_exp_lut_fp16_f32_REEX(float x) {
    if (x < EXP_MIN) return 0.0f;
    float xc = fminf(EXP_MAX, x);
    return lut_eval_fp16_direct(xc, ggml_lut_exponential_fp16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
}

float ggml_exp_lut_bf16_f32_REEX(float x) {
    if (x < EXP_MIN) return 0.0f;
    float xc = fminf(EXP_MAX, x);
    return lut_eval_bf16_direct(xc, ggml_lut_exponential_bf16_reex, GGML_LUT_NUM_SEGMENTS_REEX);
}

float ggml_silu_lut_fp16_f32_REEX(float x) {
    /* Full-chain FP16: quantize x -> sigmoid(quantized x) -> quantize(x_q * sigmoid) (match Python). */
    float x_q = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
    float prod = x_q * ggml_sigmoid_lut_fp16_f32_REEX(x_q);
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(prod));
}

float ggml_silu_lut_bf16_f32_REEX(float x) {
    /* Full-chain BF16: quantize x -> sigmoid(quantized x) -> quantize(x_q * sigmoid) (match Python). */
    float x_q = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
    float prod = x_q * ggml_sigmoid_lut_bf16_f32_REEX(x_q);
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(prod));
}

float ggml_sigmoid_lut_mixed_fp16_f32_REEX(float x) {
    if (x < SIGMOID_MIN) return 0.0f;
    if (x > SIGMOID_MAX) return 1.0f;
    return lut_eval_mixed_fp16_direct_sigmoid(x);
}

float ggml_exp_lut_mixed_fp16_f32_REEX(float x) {
    if (x < EXP_MIN) return 0.0f;
    float xc = fminf(EXP_MAX, x);
    return lut_eval_mixed_fp16_direct_exp(xc);
}

float ggml_silu_lut_mixed_fp16_f32_REEX(float x) {
    /* Mixed-FP16: x in FP32, sigmoid_mixed(x), prod = x * sigmoid in FP32, only output rounded to FP16. */
    float s = ggml_sigmoid_lut_mixed_fp16_f32_REEX(x);
    float prod_f32 = x * s;
    ggml_fp16_t prod_fp16 = ggml_fp32_to_fp16(prod_f32);
    return ggml_fp16_to_fp32(prod_fp16);
}

#endif /* GGML_USE_REEX */
