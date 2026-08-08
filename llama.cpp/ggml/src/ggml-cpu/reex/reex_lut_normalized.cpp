#ifdef GGML_USE_REEX

#include "reex_lut_normalized.h"
#include "reex/ggml-reex-lut-data.h"
#include "ggml-impl.h"

#include <cmath>
#include <limits>

#define GGML_REEX_LN2_F          0.69314718055994530941723212145818f
#define GGML_REEX_SQRT2_F        1.4142135623730950488016887242097f
#define GGML_REEX_INV_SQRT2_F    0.70710678118654752440084436210485f

static inline float round_fp16(float value) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

static inline void normalize_to_1_2(float x, float * mantissa, int * exponent) {
    int e;
    *mantissa = static_cast<float>(std::frexp(static_cast<double>(x), &e) * 2.0);
    *exponent = e - 1;
}

static inline float normalized_core_mixed_fp16(
        float mantissa, const float * threshold, const float * slope, const float * offset) {
    for (int i = 0; i < GGML_LUT_NUM_SEGMENTS_REEX; ++i) {
        if (mantissa <= threshold[i] || i == GGML_LUT_NUM_SEGMENTS_REEX - 1) {
            volatile float product = slope[i] * mantissa;
            const float sum = product + offset[i];
            return round_fp16(sum);
        }
    }
    return std::numeric_limits<float>::quiet_NaN();
}

static inline int floor_divide_by_two(int value) {
    const int quotient = value / 2;
    return value < 0 && (value & 1) ? quotient - 1 : quotient;
}

float ggml_sqrt_lut_mixed_fp16_f32_REEX(float x) {
    if (std::isnan(x) || x < 0.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (x == 0.0f || std::isinf(x)) {
        return x;
    }

    float mantissa;
    int exponent;
    normalize_to_1_2(x, &mantissa, &exponent);
    const float core = normalized_core_mixed_fp16(
        mantissa, ggml_reex_sqrt_fp32_compare_max, ggml_reex_sqrt_fp32_b, ggml_reex_sqrt_fp32_c);
    const int k = floor_divide_by_two(exponent);
    const int parity = exponent - 2 * k;
    const float scaled = core * std::ldexp(1.0f, k);
    return scaled * (parity == 1 ? GGML_REEX_SQRT2_F : 1.0f);
}

float ggml_log_lut_mixed_fp16_f32_REEX(float x) {
    if (std::isnan(x) || x < 0.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (x == 0.0f) {
        return -std::numeric_limits<float>::infinity();
    }
    if (std::isinf(x)) {
        return std::numeric_limits<float>::infinity();
    }

    float mantissa;
    int exponent;
    normalize_to_1_2(x, &mantissa, &exponent);
    const float core = normalized_core_mixed_fp16(
        mantissa, ggml_reex_log_fp32_compare_max, ggml_reex_log_fp32_b, ggml_reex_log_fp32_c);
    const float exponent_term = static_cast<float>(exponent) * GGML_REEX_LN2_F;
    return core + exponent_term;
}

float ggml_reciprocal_lut_mixed_fp16_f32_REEX(float x) {
    if (std::isnan(x)) {
        return x;
    }
    if (x == 0.0f) {
        return std::copysign(std::numeric_limits<float>::infinity(), x);
    }
    if (std::isinf(x)) {
        return std::copysign(0.0f, x);
    }

    const float magnitude = std::fabs(x);
    float mantissa;
    int exponent;
    normalize_to_1_2(magnitude, &mantissa, &exponent);
    const float core = normalized_core_mixed_fp16(
        mantissa, ggml_reex_reciprocal_fp32_compare_max,
        ggml_reex_reciprocal_fp32_b, ggml_reex_reciprocal_fp32_c);
    const float reconstructed = core * std::ldexp(1.0f, -exponent);
    return std::copysign(reconstructed, x);
}

float ggml_rsqrt_lut_mixed_fp16_f32_REEX(float x) {
    if (std::isnan(x) || x < 0.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (x == 0.0f) {
        return std::numeric_limits<float>::infinity();
    }
    if (std::isinf(x)) {
        return 0.0f;
    }

    float mantissa;
    int exponent;
    normalize_to_1_2(x, &mantissa, &exponent);
    const float core = normalized_core_mixed_fp16(
        mantissa, ggml_reex_rsqrt_fp32_compare_max, ggml_reex_rsqrt_fp32_b, ggml_reex_rsqrt_fp32_c);
    const int k = floor_divide_by_two(exponent);
    const int parity = exponent - 2 * k;
    const float scaled = core * std::ldexp(1.0f, -k);
    return scaled * (parity == 1 ? GGML_REEX_INV_SQRT2_F : 1.0f);
}

/* power_2 is intentionally unchanged: it is a direct x*x operation, not a LUT. */
float ggml_sqr_fp16_f32_REEX(float x) {
    const float x_q = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
    return round_fp16(x_q * x_q);
}

float ggml_sqr_bf16_f32_REEX(float x) {
    const float x_q = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x_q * x_q));
}

float ggml_sqr_mixed_fp16_f32_REEX(float x) {
    return round_fp16(x * x);
}

#endif /* GGML_USE_REEX */
