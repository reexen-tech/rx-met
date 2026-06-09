/**
 * Shared Reex LUT — Sin/Cos and other piecewise-linear LUTs (FP16/BF16).
 * Cos is implemented as sin(x + π/2) using the same sin tables.
 * Table data is loaded from ggml-reex-lut-data.h (generated from JSON); no runtime fitting.
 *
 * Macros:
 *   GGML_USE_REEX     — Enable REEX LUT. Call sites use Mixed-FP16: LUT coefficients in FP32 (coefficients_float),
 *                       compute y = b*x+c in FP32, only final output rounded to FP16 (matches Python mixed_fp16).
 *   GGML_USE_BF16_LUT — Optional; when set, code that still branches on it may use BF16 LUTs. Default path is Mixed-FP16.
 *   GGML_LUT_NUM_SEGMENTS_REEX — Defined in reex/ggml-reex-lut-config.h (default 16). ggml-reex-lut-data.h contains both layouts; macro selects branch.
 *                       Override: -DGGML_LUT_NUM_SEGMENTS_REEX=31
 */
#pragma once

#ifdef GGML_USE_REEX

#include "ggml.h"
#include "reex/ggml-reex-lut-config.h"

#define GGML_TRIG_LUT_MIN (-3.14159265358979323846f)   // -π
#define GGML_TRIG_LUT_MAX (3.14159265358979323846f)    // π
#define GGML_TRIG_PERIOD  (6.28318530717958647692f)    // 2π
#define GGML_TRIG_HALF_PI (1.57079632679489661923f)    // π/2

// Trigonometric LUT segment structure (FP16)
struct ggml_lut_segment_fp16_reex {
    ggml_fp16_t threshold;
    ggml_fp16_t b;
    ggml_fp16_t c;
};

// Trigonometric LUT segment structure (BF16)
struct ggml_lut_segment_bf16_reex {
    ggml_bf16_t threshold;
    ggml_bf16_t b;
    ggml_bf16_t c;
};

// Host-side LUT tables (data from reex/ggml-reex-lut-data.h)
extern struct ggml_lut_segment_fp16_reex ggml_lut_sin_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_bf16_reex ggml_lut_sin_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_fp16_reex ggml_lut_sigmoid_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_bf16_reex ggml_lut_sigmoid_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_fp16_reex ggml_lut_exponential_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_bf16_reex ggml_lut_exponential_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_fp16_reex ggml_lut_sqrt_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_bf16_reex ggml_lut_sqrt_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_fp16_reex ggml_lut_log_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_bf16_reex ggml_lut_log_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_fp16_reex ggml_lut_power_2_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_bf16_reex ggml_lut_power_2_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_fp16_reex ggml_lut_reciprocal_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_bf16_reex ggml_lut_reciprocal_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_fp16_reex ggml_lut_rsqrt_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
extern struct ggml_lut_segment_bf16_reex ggml_lut_rsqrt_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];

extern bool ggml_trig_lut_initialized;

#ifdef __cplusplus
extern "C" {
#endif

// Fills host-side FP16/BF16 tables; call once from CPU or before CUDA LUT copy.
void ggml_init_trigonometric_lut_REEX(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_USE_REEX */
