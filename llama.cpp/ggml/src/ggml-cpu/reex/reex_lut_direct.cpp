#ifdef GGML_USE_REEX

#include "reex_lut_direct.h"
#include "reex_lut_normalized.h"
#include "reex/ggml-reex-lut-data.h"
#include "ggml-impl.h"

#include <cmath>
#include <limits>

#ifndef GGML_REEX_EXP_NO_LUT
static inline float round_fp16(float value) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

static inline float lut_core_output(float value) {
#ifdef GGML_REEX_LUT_FP32_OUTPUT
    return value;
#else
    return round_fp16(value);
#endif
}

static inline float exp2_core_mixed_fp16(float fraction) {
    for (int i = 0; i < GGML_LUT_NUM_SEGMENTS_REEX; ++i) {
        if (fraction <= ggml_reex_exponential_fp32_compare_max[i] ||
            i == GGML_LUT_NUM_SEGMENTS_REEX - 1) {
            volatile float product = ggml_reex_exponential_fp32_b[i] * fraction;
            const float sum = product + ggml_reex_exponential_fp32_c[i];
            return lut_core_output(sum);
        }
    }
    return std::numeric_limits<float>::quiet_NaN();
}
#endif

float ggml_exp_lut_mixed_fp16_f32_REEX(float x) {
#ifdef GGML_REEX_EXP_NO_LUT
    return std::exp(x);
#else
    if (std::isnan(x)) {
        return x;
    }
    if (std::isinf(x)) {
        return x > 0.0f ? std::numeric_limits<float>::infinity() : 0.0f;
    }

    const double reduced = static_cast<double>(x) * GGML_REEX_LOG2_E_D;
    if (reduced > 1024.0) {
        return std::numeric_limits<float>::infinity();
    }
    if (reduced < -1075.0) {
        return 0.0f;
    }

    double exponent_d = std::floor(reduced);
    float fraction = static_cast<float>(reduced - exponent_d);
    if (fraction >= 1.0f) {
        fraction = 0.0f;
        exponent_d += 1.0;
    }
    fraction = std::max(0.0f, std::min(std::nextafter(1.0f, 0.0f), fraction));

    const float core = exp2_core_mixed_fp16(fraction);
    const double reconstructed = std::ldexp(static_cast<double>(core), static_cast<int>(exponent_d));
    return static_cast<float>(reconstructed);
#endif
}

float ggml_sigmoid_lut_mixed_fp16_f32_REEX(float x) {
    const float exp_neg = ggml_exp_lut_mixed_fp16_f32_REEX(-x);
    const float denominator = 1.0f + exp_neg;
    return ggml_reciprocal_lut_mixed_fp16_f32_REEX(denominator);
}

float ggml_silu_lut_mixed_fp16_f32_REEX(float x) {
    return x * ggml_sigmoid_lut_mixed_fp16_f32_REEX(x);
}

#endif /* GGML_USE_REEX */
