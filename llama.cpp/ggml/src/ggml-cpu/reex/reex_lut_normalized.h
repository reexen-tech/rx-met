/**
 * REEX normalized mixed-FP16 LUT interface.
 *
 * power_2 remains a direct x*x operation and is intentionally not a LUT.
 */
#pragma once

#ifndef GGML_USE_REEX
#error "reex_lut_normalized.h must be used only when GGML_USE_REEX is defined"
#endif

#include "ggml-reex-lut.h"

#ifdef __cplusplus
extern "C" {
#endif

float ggml_sqrt_lut_mixed_fp16_f32_REEX(float x);
float ggml_log_lut_mixed_fp16_f32_REEX(float x);
float ggml_reciprocal_lut_mixed_fp16_f32_REEX(float x);
float ggml_rsqrt_lut_mixed_fp16_f32_REEX(float x);

float ggml_sqr_fp16_f32_REEX(float x);
float ggml_sqr_bf16_f32_REEX(float x);
float ggml_sqr_mixed_fp16_f32_REEX(float x);

#ifdef __cplusplus
}
#endif
