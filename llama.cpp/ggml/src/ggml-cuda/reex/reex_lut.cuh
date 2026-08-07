#pragma once

#ifdef GGML_USE_REEX

#include "../common.cuh"
#include "ggml-reex-lut.h"

#include <cstdint>

#define GGML_CUDA_LUT_NUM_SEGMENTS GGML_LUT_NUM_SEGMENTS_REEX
#define GGML_CUDA_LN2             0.69314718055994530941723212145818f
#define GGML_CUDA_SQRT2           1.4142135623730950488016887242097f
#define GGML_CUDA_INV_SQRT2       0.70710678118654752440084436210485f

#ifdef GGML_CUDA_REEX_LUT_DEFINE
#define GGML_CUDA_REEX_DECLARE_TABLE(op) \
    __constant__ float ggml_cuda_##op##_fp32_compare_max[GGML_CUDA_LUT_NUM_SEGMENTS]; \
    __constant__ float ggml_cuda_##op##_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS]; \
    __constant__ float ggml_cuda_##op##_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS]
#else
#define GGML_CUDA_REEX_DECLARE_TABLE(op) \
    extern __constant__ float ggml_cuda_##op##_fp32_compare_max[GGML_CUDA_LUT_NUM_SEGMENTS]; \
    extern __constant__ float ggml_cuda_##op##_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS]; \
    extern __constant__ float ggml_cuda_##op##_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS]
#endif

GGML_CUDA_REEX_DECLARE_TABLE(exponential);
GGML_CUDA_REEX_DECLARE_TABLE(sin);
GGML_CUDA_REEX_DECLARE_TABLE(reciprocal);
GGML_CUDA_REEX_DECLARE_TABLE(rsqrt);
GGML_CUDA_REEX_DECLARE_TABLE(sqrt);
GGML_CUDA_REEX_DECLARE_TABLE(log);

#undef GGML_CUDA_REEX_DECLARE_TABLE

static __device__ __forceinline__ float ggml_cuda_fadd_rn_ieee_reex(float a, float b) {
    float result;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

static __device__ __forceinline__ float ggml_cuda_fmul_rn_ieee_reex(float a, float b) {
    float result;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

static __device__ __forceinline__ float ggml_cuda_f64_to_f32_rn_ieee_reex(double value) {
    float result;
    asm("cvt.rn.f32.f64 %0, %1;" : "=f"(result) : "d"(value));
    return result;
}

static __device__ __forceinline__ uint32_t ggml_cuda_f32_magnitude_bits_reex(float value) {
    return __float_as_uint(value) & 0x7fffffffU;
}

static __device__ __forceinline__ bool ggml_cuda_f32_is_nan_reex(uint32_t magnitude_bits) {
    return magnitude_bits > 0x7f800000U;
}

static __device__ __forceinline__ bool ggml_cuda_f32_is_inf_reex(uint32_t magnitude_bits) {
    return magnitude_bits == 0x7f800000U;
}

static __device__ __forceinline__ float ggml_cuda_lut_core_mixed_fp16_reex(
        float x, const float * compare_max, const float * slope, const float * offset) {
    for (int i = 0; i < GGML_CUDA_LUT_NUM_SEGMENTS; ++i) {
        if (x <= compare_max[i] || i == GGML_CUDA_LUT_NUM_SEGMENTS - 1) {
            const float product = ggml_cuda_fmul_rn_ieee_reex(slope[i], x);
            const float sum = ggml_cuda_fadd_rn_ieee_reex(product, offset[i]);
            return __half2float(__float2half_rn(sum));
        }
    }
    return NAN;
}

static __device__ __forceinline__ float ggml_cuda_exp_lut_mixed_fp16_reex(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t magnitude_bits = bits & 0x7fffffffU;
    if (ggml_cuda_f32_is_nan_reex(magnitude_bits)) {
        return x;
    }
    if (ggml_cuda_f32_is_inf_reex(magnitude_bits)) {
        return (bits & 0x80000000U) == 0 ? INFINITY : 0.0f;
    }

    const double reduced = (double) x * GGML_REEX_LOG2_E_D;
    if (reduced > 1024.0) {
        return INFINITY;
    }
    if (reduced < -1075.0) {
        return 0.0f;
    }

    double exponent_d = floor(reduced);
    float fraction = (float) (reduced - exponent_d);
    if (fraction >= 1.0f) {
        fraction = 0.0f;
        exponent_d += 1.0;
    }
    fraction = fmaxf(0.0f, fminf(nextafterf(1.0f, 0.0f), fraction));
    const float core = ggml_cuda_lut_core_mixed_fp16_reex(
        fraction, ggml_cuda_exponential_fp32_compare_max,
        ggml_cuda_exponential_fp32_b, ggml_cuda_exponential_fp32_c);
    return ggml_cuda_f64_to_f32_rn_ieee_reex(ldexp((double) core, (int) exponent_d));
}

static __device__ __forceinline__ float ggml_cuda_sin_like_lut_mixed_fp16_reex(float x, double phase) {
    const uint32_t magnitude_bits = ggml_cuda_f32_magnitude_bits_reex(x);
    if (magnitude_bits >= 0x7f800000U) {
        return NAN;
    }

    double theta = fmod((double) x + phase, GGML_REEX_TWO_PI_D);
    if (theta < 0.0) {
        theta += GGML_REEX_TWO_PI_D;
    }
    if (theta >= GGML_REEX_TWO_PI_D) {
        theta = 0.0;
    }

    int quadrant = (int) floor(theta / GGML_REEX_HALF_PI_D);
    quadrant = max(0, min(3, quadrant));
    const double remainder = theta - (double) quadrant * GGML_REEX_HALF_PI_D;
    const double folded = (quadrant & 1) ? GGML_REEX_HALF_PI_D - remainder : remainder;
    const float u = (float) fmax(0.0, fmin(GGML_REEX_HALF_PI_D, folded));
    const float core = ggml_cuda_lut_core_mixed_fp16_reex(
        u, ggml_cuda_sin_fp32_compare_max, ggml_cuda_sin_fp32_b, ggml_cuda_sin_fp32_c);
    return quadrant <= 1 ? core : -core;
}

static __device__ __forceinline__ float ggml_cuda_sin_lut_mixed_fp16_reex(float x) {
    return ggml_cuda_sin_like_lut_mixed_fp16_reex(x, 0.0);
}

static __device__ __forceinline__ float ggml_cuda_cos_lut_mixed_fp16_reex(float x) {
    return ggml_cuda_sin_like_lut_mixed_fp16_reex(x, GGML_REEX_HALF_PI_D);
}

static __device__ __forceinline__ void ggml_cuda_normalize_to_1_2_reex(
        float x, float * mantissa, int * exponent) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t exponent_bits = (bits >> 23) & 0xffU;
    uint32_t fraction_bits = bits & 0x007fffffU;
    if (exponent_bits != 0) {
        *mantissa = __uint_as_float(0x3f800000U | fraction_bits);
        *exponent = (int) exponent_bits - 127;
        return;
    }

    const int shift = __clz(fraction_bits) - 8;
    fraction_bits = (fraction_bits << shift) & 0x007fffffU;
    *mantissa = __uint_as_float(0x3f800000U | fraction_bits);
    *exponent = -126 - shift;
}

static __device__ __forceinline__ int ggml_cuda_floor_divide_by_two_reex(int value) {
    const int quotient = value / 2;
    return value < 0 && (value & 1) ? quotient - 1 : quotient;
}

static __device__ __forceinline__ float ggml_cuda_reciprocal_lut_mixed_fp16_reex(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t magnitude_bits = bits & 0x7fffffffU;
    if (ggml_cuda_f32_is_nan_reex(magnitude_bits)) {
        return x;
    }
    if (magnitude_bits == 0) {
        return __uint_as_float((bits & 0x80000000U) | 0x7f800000U);
    }
    if (ggml_cuda_f32_is_inf_reex(magnitude_bits)) {
        return __uint_as_float(bits & 0x80000000U);
    }

    float mantissa;
    int exponent;
    ggml_cuda_normalize_to_1_2_reex(__uint_as_float(magnitude_bits), &mantissa, &exponent);
    const float core = ggml_cuda_lut_core_mixed_fp16_reex(
        mantissa, ggml_cuda_reciprocal_fp32_compare_max,
        ggml_cuda_reciprocal_fp32_b, ggml_cuda_reciprocal_fp32_c);
    const float scale = ggml_cuda_f64_to_f32_rn_ieee_reex(ldexp(1.0, -exponent));
    const float reconstructed = ggml_cuda_fmul_rn_ieee_reex(core, scale);
    return __uint_as_float(__float_as_uint(reconstructed) | (bits & 0x80000000U));
}

static __device__ __forceinline__ float ggml_cuda_rsqrt_lut_mixed_fp16_reex(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t magnitude_bits = bits & 0x7fffffffU;
    if (ggml_cuda_f32_is_nan_reex(magnitude_bits) ||
        ((bits & 0x80000000U) != 0 && magnitude_bits != 0)) {
        return NAN;
    }
    if (magnitude_bits == 0) {
        return INFINITY;
    }
    if (ggml_cuda_f32_is_inf_reex(magnitude_bits)) {
        return 0.0f;
    }

    float mantissa;
    int exponent;
    ggml_cuda_normalize_to_1_2_reex(x, &mantissa, &exponent);
    const float core = ggml_cuda_lut_core_mixed_fp16_reex(
        mantissa, ggml_cuda_rsqrt_fp32_compare_max,
        ggml_cuda_rsqrt_fp32_b, ggml_cuda_rsqrt_fp32_c);
    const int k = ggml_cuda_floor_divide_by_two_reex(exponent);
    const int parity = exponent - 2 * k;
    const float scale = ggml_cuda_f64_to_f32_rn_ieee_reex(ldexp(1.0, -k));
    const float scaled = ggml_cuda_fmul_rn_ieee_reex(core, scale);
    return ggml_cuda_fmul_rn_ieee_reex(
        scaled, parity == 1 ? GGML_CUDA_INV_SQRT2 : 1.0f);
}

static __device__ __forceinline__ float ggml_cuda_sqrt_lut_mixed_fp16_reex(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t magnitude_bits = bits & 0x7fffffffU;
    if (ggml_cuda_f32_is_nan_reex(magnitude_bits) ||
        ((bits & 0x80000000U) != 0 && magnitude_bits != 0)) {
        return NAN;
    }
    if (magnitude_bits == 0 || ggml_cuda_f32_is_inf_reex(magnitude_bits)) {
        return x;
    }

    float mantissa;
    int exponent;
    ggml_cuda_normalize_to_1_2_reex(x, &mantissa, &exponent);
    const float core = ggml_cuda_lut_core_mixed_fp16_reex(
        mantissa, ggml_cuda_sqrt_fp32_compare_max,
        ggml_cuda_sqrt_fp32_b, ggml_cuda_sqrt_fp32_c);
    const int k = ggml_cuda_floor_divide_by_two_reex(exponent);
    const int parity = exponent - 2 * k;
    const float scale = ggml_cuda_f64_to_f32_rn_ieee_reex(ldexp(1.0, k));
    const float scaled = ggml_cuda_fmul_rn_ieee_reex(core, scale);
    return ggml_cuda_fmul_rn_ieee_reex(
        scaled, parity == 1 ? GGML_CUDA_SQRT2 : 1.0f);
}

static __device__ __forceinline__ float ggml_cuda_log_lut_mixed_fp16_reex(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t magnitude_bits = bits & 0x7fffffffU;
    if (ggml_cuda_f32_is_nan_reex(magnitude_bits) ||
        ((bits & 0x80000000U) != 0 && magnitude_bits != 0)) {
        return NAN;
    }
    if (magnitude_bits == 0) {
        return -INFINITY;
    }
    if (ggml_cuda_f32_is_inf_reex(magnitude_bits)) {
        return INFINITY;
    }

    float mantissa;
    int exponent;
    ggml_cuda_normalize_to_1_2_reex(x, &mantissa, &exponent);
    const float core = ggml_cuda_lut_core_mixed_fp16_reex(
        mantissa, ggml_cuda_log_fp32_compare_max,
        ggml_cuda_log_fp32_b, ggml_cuda_log_fp32_c);
    const float exponent_term = ggml_cuda_fmul_rn_ieee_reex((float) exponent, GGML_CUDA_LN2);
    return ggml_cuda_fadd_rn_ieee_reex(core, exponent_term);
}

static __device__ __forceinline__ float ggml_cuda_sigmoid_lut_mixed_fp16_reex(float x) {
    const float exp_neg = ggml_cuda_exp_lut_mixed_fp16_reex(-x);
    const float denominator = ggml_cuda_fadd_rn_ieee_reex(1.0f, exp_neg);
    return ggml_cuda_reciprocal_lut_mixed_fp16_reex(denominator);
}

static __device__ __forceinline__ float ggml_cuda_silu_lut_mixed_fp16_reex(float x) {
    return ggml_cuda_fmul_rn_ieee_reex(x, ggml_cuda_sigmoid_lut_mixed_fp16_reex(x));
}

static __device__ __forceinline__ float ggml_cuda_exp_for_softmax(float x) {
    return ggml_cuda_exp_lut_mixed_fp16_reex(x);
}

void ggml_cuda_init_trigonometric_lut_REEX(int device_id);

#endif /* GGML_USE_REEX */
