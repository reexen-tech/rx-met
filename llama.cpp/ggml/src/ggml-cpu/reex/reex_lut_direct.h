/**
 * REEX exp2-reduced exp, composed sigmoid, and SiLU mixed-FP16 interface.
 */
#pragma once

#ifndef GGML_USE_REEX
#error "reex_lut_direct.h must be used only when GGML_USE_REEX is defined"
#endif

#include "ggml-reex-lut.h"

#ifdef __cplusplus
extern "C" {
#endif

float ggml_sigmoid_lut_mixed_fp16_f32_REEX(float x);
float ggml_exp_lut_mixed_fp16_f32_REEX(float x);
float ggml_silu_lut_mixed_fp16_f32_REEX(float x);

static inline float ggml_reex_expf_(float x) {
    return ggml_exp_lut_mixed_fp16_f32_REEX(x);
}

#ifdef __cplusplus
}
#endif
