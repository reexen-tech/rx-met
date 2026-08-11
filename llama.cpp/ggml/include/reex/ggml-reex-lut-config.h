/**
 * ADA300 REEX LUT configuration.
 *
 * Override before including generated tables:
 *   -DGGML_LUT_NUM_SEGMENTS_REEX=63
 *
 * ggml-reex-lut-data.h embeds 16-, 31-, and 63-segment FP32-coefficient
 * tables optimized for FP32 output; this macro selects the branch at compile time.
 */
#pragma once

#ifdef GGML_USE_REEX

#ifndef GGML_LUT_NUM_SEGMENTS_REEX
#define GGML_LUT_NUM_SEGMENTS_REEX 16
#endif

#if GGML_LUT_NUM_SEGMENTS_REEX != 16 && GGML_LUT_NUM_SEGMENTS_REEX != 31 && GGML_LUT_NUM_SEGMENTS_REEX != 63
#error "GGML_LUT_NUM_SEGMENTS_REEX must be 16, 31, or 63"
#endif

#endif /* GGML_USE_REEX */
