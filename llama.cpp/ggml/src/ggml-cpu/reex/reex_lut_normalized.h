/**
 * Reex normalized LUT — Sqrt, Log, Reciprocal, Rsqrt (CPU backend).
 * normalize_to_1_2(mantissa) -> LUT lookup -> reconstruct(core, exponent).
 * Sqr (power_2 = x²) is implemented as direct fp16*fp16 / bf16*bf16 multiply, no LUT.
 */
#pragma once

#ifndef GGML_USE_REEX
#error "reex_lut_normalized.h must be used only when GGML_USE_REEX is defined"
#endif

#include "ggml-reex-lut.h"

#ifdef __cplusplus
extern "C" {
#endif

float ggml_sqrt_lut_fp16_f32_REEX(float x);
float ggml_sqrt_lut_bf16_f32_REEX(float x);
float ggml_sqrt_lut_mixed_fp16_f32_REEX(float x);
float ggml_log_lut_fp16_f32_REEX(float x);
float ggml_log_lut_bf16_f32_REEX(float x);
float ggml_log_lut_mixed_fp16_f32_REEX(float x);
float ggml_sqr_fp16_f32_REEX(float x);
float ggml_sqr_bf16_f32_REEX(float x);
float ggml_sqr_mixed_fp16_f32_REEX(float x);
float ggml_reciprocal_lut_fp16_f32_REEX(float x);
float ggml_reciprocal_lut_bf16_f32_REEX(float x);
float ggml_reciprocal_lut_mixed_fp16_f32_REEX(float x);
float ggml_rsqrt_lut_fp16_f32_REEX(float x);
float ggml_rsqrt_lut_bf16_f32_REEX(float x);
float ggml_rsqrt_lut_mixed_fp16_f32_REEX(float x);

#ifdef __cplusplus
}
#endif
