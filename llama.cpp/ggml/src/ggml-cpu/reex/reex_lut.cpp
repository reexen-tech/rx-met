#ifdef GGML_USE_REEX

#include "reex_lut.h"
#include "reex/ggml-reex-lut-data.h"

#include <cmath>

float ggml_sin_lut_fp16_f32_REEX(float x) {
#ifndef NDEBUG
    GGML_ASSERT(ggml_trig_lut_initialized && "Trigonometric LUT not initialized");
#endif
    float x_mod = fmodf(x, GGML_TRIG_PERIOD);
    if (x_mod < 0.0f) x_mod += GGML_TRIG_PERIOD;
    float half_period = GGML_TRIG_PERIOD / 2.0f;
    float x_mapped = (x_mod > half_period - 1e-6f) ? (x_mod - GGML_TRIG_PERIOD) : x_mod;
    x_mapped = fmaxf(GGML_TRIG_LUT_MIN, fminf(GGML_TRIG_LUT_MAX, x_mapped));
    /* Segment selection: compare in fp16 space (x -> fp16 -> f32, threshold is fp16 -> f32), right-inclusive (<=). */
    ggml_fp16_t x_fp16 = GGML_CPU_FP32_TO_FP16(x_mapped);
    float x_for_compare = GGML_CPU_FP16_TO_FP32(x_fp16);
    for (int i = 0; i < GGML_LUT_NUM_SEGMENTS_REEX; ++i) {
        float seg_threshold = GGML_CPU_FP16_TO_FP32(ggml_lut_sin_fp16_reex[i].threshold);
        if (x_for_compare <= seg_threshold || i == GGML_LUT_NUM_SEGMENTS_REEX - 1) {
            float b = GGML_CPU_FP16_TO_FP32(ggml_lut_sin_fp16_reex[i].b);
            float c = GGML_CPU_FP16_TO_FP32(ggml_lut_sin_fp16_reex[i].c);
            /* Full-chain FP16: use quantized input for product (match Python infer_with_float_lut). */
            float bx = b * x_for_compare;
            ggml_fp16_t bx_fp16 = GGML_CPU_FP32_TO_FP16(bx);
            float bx_f32 = GGML_CPU_FP16_TO_FP32(bx_fp16);
            float y = bx_f32 + c;
            ggml_fp16_t y_fp16 = GGML_CPU_FP32_TO_FP16(y);
            return GGML_CPU_FP16_TO_FP32(y_fp16);
        }
    }
    return 0.0f;
}

float ggml_sin_lut_bf16_f32_REEX(float x) {
#ifndef NDEBUG
    GGML_ASSERT(ggml_trig_lut_initialized && "Trigonometric LUT not initialized");
#endif
    float x_mod = fmodf(x, GGML_TRIG_PERIOD);
    if (x_mod < 0.0f) x_mod += GGML_TRIG_PERIOD;
    float half_period = GGML_TRIG_PERIOD / 2.0f;
    float x_mapped = (x_mod > half_period - 1e-6f) ? (x_mod - GGML_TRIG_PERIOD) : x_mod;
    x_mapped = fmaxf(GGML_TRIG_LUT_MIN, fminf(GGML_TRIG_LUT_MAX, x_mapped));
    /* Segment selection: compare in bf16 space (x -> bf16 -> f32, threshold is bf16 -> f32), right-inclusive (<=). */
    ggml_bf16_t x_bf16 = GGML_FP32_TO_BF16(x_mapped);
    float x_for_compare = GGML_BF16_TO_FP32(x_bf16);
    for (int i = 0; i < GGML_LUT_NUM_SEGMENTS_REEX; ++i) {
        float seg_threshold = GGML_BF16_TO_FP32(ggml_lut_sin_bf16_reex[i].threshold);
        if (x_for_compare <= seg_threshold || i == GGML_LUT_NUM_SEGMENTS_REEX - 1) {
            float b = GGML_BF16_TO_FP32(ggml_lut_sin_bf16_reex[i].b);
            float c = GGML_BF16_TO_FP32(ggml_lut_sin_bf16_reex[i].c);
            /* Full-chain BF16: use quantized input for product (match Python infer_with_float_lut). */
            float bx = b * x_for_compare;
            ggml_bf16_t bx_bf16 = GGML_FP32_TO_BF16(bx);
            float bx_f32 = GGML_BF16_TO_FP32(bx_bf16);
            float y = bx_f32 + c;
            ggml_bf16_t y_bf16 = GGML_FP32_TO_BF16(y);
            return GGML_BF16_TO_FP32(y_bf16);
        }
    }
    return 0.0f;
}

/* Mixed-FP16: FP32 table (coefficients_float), x in FP32, y = b*x+c in FP32, only output rounded to FP16 (match Python). */
float ggml_sin_lut_mixed_fp16_f32_REEX(float x) {
#ifndef NDEBUG
    GGML_ASSERT(ggml_trig_lut_initialized && "Trigonometric LUT not initialized");
#endif
    float x_mod = fmodf(x, GGML_TRIG_PERIOD);
    if (x_mod < 0.0f) x_mod += GGML_TRIG_PERIOD;
    float half_period = GGML_TRIG_PERIOD / 2.0f;
    float x_mapped = (x_mod > half_period - 1e-6f) ? (x_mod - GGML_TRIG_PERIOD) : x_mod;
    x_mapped = fmaxf(GGML_TRIG_LUT_MIN, fminf(GGML_TRIG_LUT_MAX, x_mapped));
    for (int i = 0; i < GGML_LUT_NUM_SEGMENTS_REEX; ++i) {
        if (x_mapped <= ggml_reex_sin_fp32_threshold[i] || i == GGML_LUT_NUM_SEGMENTS_REEX - 1) {
            float b = ggml_reex_sin_fp32_b[i];
            float c = ggml_reex_sin_fp32_c[i];
            float y_f32 = b * x_mapped + c;
            ggml_fp16_t y_fp16 = GGML_CPU_FP32_TO_FP16(y_f32);
            return GGML_CPU_FP16_TO_FP32(y_fp16);
        }
    }
    return 0.0f;
}

/* cos(x) = sin(x + π/2): use same sin LUT with phase shift */
float ggml_cos_lut_fp16_f32_REEX(float x) {
    return ggml_sin_lut_fp16_f32_REEX(x + GGML_TRIG_HALF_PI);
}

float ggml_cos_lut_bf16_f32_REEX(float x) {
    return ggml_sin_lut_bf16_f32_REEX(x + GGML_TRIG_HALF_PI);
}

float ggml_cos_lut_mixed_fp16_f32_REEX(float x) {
    return ggml_sin_lut_mixed_fp16_f32_REEX(x + GGML_TRIG_HALF_PI);
}

#endif /* GGML_USE_REEX */
