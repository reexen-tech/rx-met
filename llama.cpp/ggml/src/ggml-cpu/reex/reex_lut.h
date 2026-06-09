/**
 * Reex Trigonometric LUT — Sin/Cos lookup (CPU backend).
 * When GGML_USE_REEX is defined, uses shared tables and init from ggml-reex-lut.h.
 */
#pragma once

#ifndef GGML_USE_REEX
#error "reex_lut.h must be used only when GGML_USE_REEX is defined"
#endif

#include "ggml-reex-lut.h"
#include "vec.h"

#ifdef __cplusplus
extern "C" {
#endif

float ggml_sin_lut_fp16_f32_REEX(float x);
float ggml_sin_lut_bf16_f32_REEX(float x);
float ggml_sin_lut_mixed_fp16_f32_REEX(float x);
float ggml_cos_lut_fp16_f32_REEX(float x);
float ggml_cos_lut_bf16_f32_REEX(float x);
float ggml_cos_lut_mixed_fp16_f32_REEX(float x);

#ifdef __cplusplus
}
#endif
