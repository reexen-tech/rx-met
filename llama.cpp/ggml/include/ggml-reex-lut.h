/**
 * Shared ADA300 mixed-FP16 LUT definitions.
 *
 * The generated header contains FP32 thresholds and coefficients. Each LUT
 * core performs separate FP32 multiply/add operations and rounds that core
 * output to FP16 before operator-specific reconstruction.
 */
#pragma once

#ifdef GGML_USE_REEX

#include "ggml.h"
#include "reex/ggml-reex-lut-config.h"

#define GGML_REEX_TWO_PI_D  6.283185307179586476925286766559
#define GGML_REEX_HALF_PI_D 1.5707963267948966192313216916398
#define GGML_REEX_LOG2_E_D  1.4426950408889634073599246810019

extern bool ggml_trig_lut_initialized;

#ifdef __cplusplus
extern "C" {
#endif

void ggml_init_trigonometric_lut_REEX(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_USE_REEX */
