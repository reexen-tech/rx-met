/**
 * REEX trigonometric mixed-FP16 LUT interface.
 */
#pragma once

#ifndef GGML_USE_REEX
#error "reex_lut.h must be used only when GGML_USE_REEX is defined"
#endif

#include "ggml-reex-lut.h"

#ifdef __cplusplus
extern "C" {
#endif

float ggml_sin_lut_mixed_fp16_f32_REEX(float x);
float ggml_cos_lut_mixed_fp16_f32_REEX(float x);

#ifdef __cplusplus
}
#endif
