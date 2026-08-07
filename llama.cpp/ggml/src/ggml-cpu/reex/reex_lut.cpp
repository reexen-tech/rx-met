#ifdef GGML_USE_REEX

#include "reex_lut.h"
#include "reex/ggml-reex-lut-data.h"
#include "ggml-impl.h"

#include <cmath>
#include <limits>

static inline float round_fp16(float value) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

static inline float sin_core_mixed_fp16(float u) {
    for (int i = 0; i < GGML_LUT_NUM_SEGMENTS_REEX; ++i) {
        if (u <= ggml_reex_sin_fp32_compare_max[i] || i == GGML_LUT_NUM_SEGMENTS_REEX - 1) {
            volatile float product = ggml_reex_sin_fp32_b[i] * u;
            const float sum = product + ggml_reex_sin_fp32_c[i];
            return round_fp16(sum);
        }
    }
    return std::numeric_limits<float>::quiet_NaN();
}

static inline float sin_like_mixed_fp16(float x, double phase) {
#ifndef NDEBUG
    GGML_ASSERT(ggml_trig_lut_initialized && "REEX LUT not initialized");
#endif
    if (!std::isfinite(x)) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    double theta = std::fmod(static_cast<double>(x) + phase, GGML_REEX_TWO_PI_D);
    if (theta < 0.0) {
        theta += GGML_REEX_TWO_PI_D;
    }
    if (theta >= GGML_REEX_TWO_PI_D) {
        theta = 0.0;
    }

    int quadrant = static_cast<int>(std::floor(theta / GGML_REEX_HALF_PI_D));
    quadrant = std::max(0, std::min(3, quadrant));
    const double remainder = theta - static_cast<double>(quadrant) * GGML_REEX_HALF_PI_D;
    const double folded = (quadrant & 1) ? GGML_REEX_HALF_PI_D - remainder : remainder;
    const float u = static_cast<float>(std::max(0.0, std::min(GGML_REEX_HALF_PI_D, folded)));
    const float core = sin_core_mixed_fp16(u);
    return quadrant <= 1 ? core : -core;
}

float ggml_sin_lut_mixed_fp16_f32_REEX(float x) {
    return sin_like_mixed_fp16(x, 0.0);
}

float ggml_cos_lut_mixed_fp16_f32_REEX(float x) {
    return sin_like_mixed_fp16(x, GGML_REEX_HALF_PI_D);
}

#endif /* GGML_USE_REEX */
