#pragma once

#ifdef GGML_USE_REEX

/*
 * REEX LUT on CUDA: sin/cos (trig), sigmoid, exp (direct), rsqrt (normalized).
 * Mixed-FP16: FP32 table, float MAC, only output rounded to FP16.
 */

#include "../common.cuh"
#include "ggml-reex-lut.h"

#define GGML_CUDA_LUT_NUM_SEGMENTS  GGML_LUT_NUM_SEGMENTS_REEX
#define GGML_CUDA_TRIG_LUT_MIN      GGML_TRIG_LUT_MIN
#define GGML_CUDA_TRIG_LUT_MAX      GGML_TRIG_LUT_MAX
#define GGML_CUDA_TRIG_PERIOD       GGML_TRIG_PERIOD
#define GGML_CUDA_TRIG_HALF_PI      GGML_TRIG_HALF_PI
#define GGML_CUDA_SIGMOID_MIN       (-6.0f)
#define GGML_CUDA_SIGMOID_MAX       (6.0f)
#define GGML_CUDA_EXP_MIN           (-20.0f)
#define GGML_CUDA_EXP_MAX           (0.0f)
#define GGML_CUDA_RSQRT_INV_SQRT2   (0.70710678118654752440084436210485f)
#define GGML_CUDA_SQRT_SQRT2        (1.4142135623730950488016887242097f)
#define GGML_CUDA_LN2                (0.69314718055994530941723212145818f)

struct ggml_lut_segment_fp16_reex_device {
    half threshold;
    half b;
    half c;
};

#ifdef GGML_CUDA_REEX_LUT_DEFINE
__constant__ ggml_lut_segment_fp16_reex_device ggml_cuda_lut_sin_fp16_reex[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_sin_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_sin_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_sin_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
/* Direct LUT: sigmoid, exponential */
__constant__ float ggml_cuda_sigmoid_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_sigmoid_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_sigmoid_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_exponential_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_exponential_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_exponential_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
/* Normalized LUT: rsqrt */
__constant__ float ggml_cuda_rsqrt_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_rsqrt_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_rsqrt_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
/* Normalized LUT: sqrt, log, reciprocal */
__constant__ float ggml_cuda_sqrt_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_sqrt_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_sqrt_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_log_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_log_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_log_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_reciprocal_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_reciprocal_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
__constant__ float ggml_cuda_reciprocal_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
#else
extern __constant__ ggml_lut_segment_fp16_reex_device ggml_cuda_lut_sin_fp16_reex[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_sin_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_sin_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_sin_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_sigmoid_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_sigmoid_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_sigmoid_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_exponential_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_exponential_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_exponential_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_rsqrt_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_rsqrt_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_rsqrt_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_sqrt_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_sqrt_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_sqrt_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_log_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_log_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_log_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_reciprocal_fp32_threshold[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_reciprocal_fp32_b[GGML_CUDA_LUT_NUM_SEGMENTS];
extern __constant__ float ggml_cuda_reciprocal_fp32_c[GGML_CUDA_LUT_NUM_SEGMENTS];
#endif

/* Mixed-FP16: FP32 table, y = b*x+c in float, only output rounded to FP16 (match CPU). */
static __device__ __forceinline__ float ggml_cuda_sin_lut_mixed_fp16_reex(float x) {
    float x_mod = fmodf(x, GGML_CUDA_TRIG_PERIOD);
    if (x_mod < 0.0f) x_mod += GGML_CUDA_TRIG_PERIOD;
    float half_period = GGML_CUDA_TRIG_PERIOD * 0.5f;
    float x_mapped = (x_mod > half_period - 1e-6f) ? (x_mod - GGML_CUDA_TRIG_PERIOD) : x_mod;
    x_mapped = fmaxf(GGML_CUDA_TRIG_LUT_MIN, fminf(GGML_CUDA_TRIG_LUT_MAX, x_mapped));
    for (int i = 0; i < GGML_CUDA_LUT_NUM_SEGMENTS; ++i) {
        if (x_mapped <= ggml_cuda_sin_fp32_threshold[i] || i == GGML_CUDA_LUT_NUM_SEGMENTS - 1) {
            float b = ggml_cuda_sin_fp32_b[i];
            float c = ggml_cuda_sin_fp32_c[i];
            float y = b * x_mapped + c;
            return __half2float(__float2half(y));
        }
    }
    return 0.0f;
}

static __device__ __forceinline__ float ggml_cuda_cos_lut_mixed_fp16_reex(float x) {
    return ggml_cuda_sin_lut_mixed_fp16_reex(x + GGML_CUDA_TRIG_HALF_PI);
}

// Legacy FP16 LUT (full-chain half); kept for reference, default path uses mixed_fp16 above.
static __device__ __forceinline__ float ggml_cuda_sin_lut_fp16_reex(float x) {
    float x_mod = fmodf(x, GGML_CUDA_TRIG_PERIOD);
    if (x_mod < 0.0f) x_mod += GGML_CUDA_TRIG_PERIOD;
    float half_period = GGML_CUDA_TRIG_PERIOD * 0.5f;
    float x_mapped = (x_mod > half_period - 1e-6f) ? (x_mod - GGML_CUDA_TRIG_PERIOD) : x_mod;
    x_mapped = fmaxf(GGML_CUDA_TRIG_LUT_MIN, fminf(GGML_CUDA_TRIG_LUT_MAX, x_mapped));
    half x_h = __float2half(x_mapped);
    float x_for_compare = __half2float(x_h);
    for (int i = 0; i < GGML_CUDA_LUT_NUM_SEGMENTS; ++i) {
        float seg_end = __half2float(ggml_cuda_lut_sin_fp16_reex[i].threshold);
        if (x_for_compare <= seg_end || i == GGML_CUDA_LUT_NUM_SEGMENTS - 1) {
            half b = ggml_cuda_lut_sin_fp16_reex[i].b;
            half c = ggml_cuda_lut_sin_fp16_reex[i].c;
            return __half2float(b * x_h + c);
        }
    }
    return 0.0f;
}
static __device__ __forceinline__ float ggml_cuda_cos_lut_fp16_reex(float x) {
    return ggml_cuda_sin_lut_fp16_reex(x + GGML_CUDA_TRIG_HALF_PI);
}

/* Direct LUT: sigmoid, exp (Mixed-FP16) */
static __device__ __forceinline__ float ggml_cuda_sigmoid_lut_mixed_fp16_reex(float x) {
    if (x < GGML_CUDA_SIGMOID_MIN) return 0.0f;
    if (x > GGML_CUDA_SIGMOID_MAX) return 1.0f;
    for (int i = 0; i < GGML_CUDA_LUT_NUM_SEGMENTS; ++i) {
        if (x <= ggml_cuda_sigmoid_fp32_threshold[i] || i == GGML_CUDA_LUT_NUM_SEGMENTS - 1) {
            float y = ggml_cuda_sigmoid_fp32_b[i] * x + ggml_cuda_sigmoid_fp32_c[i];
            return __half2float(__float2half(y));
        }
    }
    return 0.0f;
}
static __device__ __forceinline__ float ggml_cuda_exp_lut_mixed_fp16_reex(float x) {
    if (x < GGML_CUDA_EXP_MIN) return 0.0f;
    float xc = fminf(GGML_CUDA_EXP_MAX, x);
    for (int i = 0; i < GGML_CUDA_LUT_NUM_SEGMENTS; ++i) {
        if (xc <= ggml_cuda_exponential_fp32_threshold[i] || i == GGML_CUDA_LUT_NUM_SEGMENTS - 1) {
            float y = ggml_cuda_exponential_fp32_b[i] * xc + ggml_cuda_exponential_fp32_c[i];
            return __half2float(__float2half(y));
        }
    }
    return 0.0f;
}
static __device__ __forceinline__ float ggml_cuda_silu_lut_mixed_fp16_reex(float x) {
    float s = ggml_cuda_sigmoid_lut_mixed_fp16_reex(x);
    float y = x * s;
    return __half2float(__float2half(y));
}

/* Normalized LUT: rsqrt (Mixed-FP16). mantissa in [1,2), core = LUT(mantissa), reconstruct. */
static __device__ __forceinline__ float ggml_cuda_rsqrt_lut_mixed_fp16_reex(float x) {
    if (x <= 0.0f) return 0.0f;
    int e;
    float m = (float)frexp((double)x, &e);
    m *= 2.0f;
    int exponent = e - 1;
    float core = 0.0f;
    for (int i = 0; i < GGML_CUDA_LUT_NUM_SEGMENTS; ++i) {
        if (m <= ggml_cuda_rsqrt_fp32_threshold[i] || i == GGML_CUDA_LUT_NUM_SEGMENTS - 1) {
            core = ggml_cuda_rsqrt_fp32_b[i] * m + ggml_cuda_rsqrt_fp32_c[i];
            break;
        }
    }
    int k = (int)floor((double)exponent / 2.0);
    int parity = exponent - 2 * k;
    float base = exp2f(-(float)k);
    float factor = (parity == 1) ? GGML_CUDA_RSQRT_INV_SQRT2 : 1.0f;
    float y = core * base * factor;
    return __half2float(__float2half(y));
}

/* Normalized LUT: sqrt (Mixed-FP16). x > 0: frexp -> mantissa in [1,2), core = LUT(mantissa), reconstruct_sqrt. */
static __device__ __forceinline__ float ggml_cuda_sqrt_lut_mixed_fp16_reex(float x) {
    if (x <= 0.0f) return 0.0f;
    int e;
    float m = (float)frexp((double)x, &e);
    m *= 2.0f;
    int exponent = e - 1;
    float core = 0.0f;
    for (int i = 0; i < GGML_CUDA_LUT_NUM_SEGMENTS; ++i) {
        if (m <= ggml_cuda_sqrt_fp32_threshold[i] || i == GGML_CUDA_LUT_NUM_SEGMENTS - 1) {
            core = ggml_cuda_sqrt_fp32_b[i] * m + ggml_cuda_sqrt_fp32_c[i];
            break;
        }
    }
    int k = (int)floor((double)exponent / 2.0);
    int parity = exponent - 2 * k;
    float base = exp2f((float)k);
    float factor = (parity == 1) ? GGML_CUDA_SQRT_SQRT2 : 1.0f;
    float y = core * base * factor;
    return __half2float(__float2half(y));
}

/* Normalized LUT: log (Mixed-FP16). x > 0: frexp -> mantissa, core = LUT(mantissa), reconstruct_log = core + exp*ln2. */
static __device__ __forceinline__ float ggml_cuda_log_lut_mixed_fp16_reex(float x) {
    if (x <= 0.0f) return -100.0f;
    int e;
    float m = (float)frexp((double)x, &e);
    m *= 2.0f;
    int exponent = e - 1;
    float core = 0.0f;
    for (int i = 0; i < GGML_CUDA_LUT_NUM_SEGMENTS; ++i) {
        if (m <= ggml_cuda_log_fp32_threshold[i] || i == GGML_CUDA_LUT_NUM_SEGMENTS - 1) {
            core = ggml_cuda_log_fp32_b[i] * m + ggml_cuda_log_fp32_c[i];
            break;
        }
    }
    float y = core + (float)exponent * GGML_CUDA_LN2;
    return __half2float(__float2half(y));
}

/* Normalized LUT: reciprocal (Mixed-FP16). x != 0: |x| -> frexp -> core, reconstruct_reciprocal = core * exp2(-exp), sign. */
static __device__ __forceinline__ float ggml_cuda_reciprocal_lut_mixed_fp16_reex(float x) {
    if (x == 0.0f) return 0.0f;
    float ax = fabsf(x);
    int e;
    float m = (float)frexp((double)ax, &e);
    m *= 2.0f;
    int exponent = e - 1;
    float core = 0.0f;
    for (int i = 0; i < GGML_CUDA_LUT_NUM_SEGMENTS; ++i) {
        if (m <= ggml_cuda_reciprocal_fp32_threshold[i] || i == GGML_CUDA_LUT_NUM_SEGMENTS - 1) {
            core = ggml_cuda_reciprocal_fp32_b[i] * m + ggml_cuda_reciprocal_fp32_c[i];
            break;
        }
    }
    float r = core * exp2f(-(float)exponent);
    if (x < 0.0f) r = -r;
    return __half2float(__float2half(r));
}

/* Softmax/other exp: REEX uses LUT, else libm */
static __device__ __forceinline__ float ggml_cuda_exp_for_softmax(float x) {
#ifdef GGML_USE_REEX
    return ggml_cuda_exp_lut_mixed_fp16_reex(x);
#else
    return expf(x);
#endif
}

void ggml_cuda_init_trigonometric_lut_REEX(int device_id);

#endif /* GGML_USE_REEX */
