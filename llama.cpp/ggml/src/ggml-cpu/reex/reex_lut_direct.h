/**
 * Reex direct LUT — Sigmoid, Exponential, Silu (CPU backend).
 * Uses shared tables from ggml-reex-lut.h; clamp input to LUT range.
 */
#pragma once

#ifndef GGML_USE_REEX
#error "reex_lut_direct.h must be used only when GGML_USE_REEX is defined"
#endif

#include "ggml-reex-lut.h"

#ifdef __cplusplus
extern "C" {
#endif

float ggml_sigmoid_lut_fp16_f32_REEX(float x);
float ggml_sigmoid_lut_bf16_f32_REEX(float x);
float ggml_sigmoid_lut_mixed_fp16_f32_REEX(float x);
float ggml_exp_lut_fp16_f32_REEX(float x);
float ggml_exp_lut_bf16_f32_REEX(float x);
float ggml_exp_lut_mixed_fp16_f32_REEX(float x);
float ggml_silu_lut_fp16_f32_REEX(float x);
float ggml_silu_lut_bf16_f32_REEX(float x);
float ggml_silu_lut_mixed_fp16_f32_REEX(float x);

/*
 * Unified expf replacement using Mixed-FP16 LUT.
 * Called by GGML_EXPF() in ops.cpp to redirect all exp calls
 * through the REEX mixed-FP16 piecewise-linear LUT.
 */
static inline float ggml_reex_expf_(float x) {
    return ggml_exp_lut_mixed_fp16_f32_REEX(x);
}

#ifdef __cplusplus
}
#endif
