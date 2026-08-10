/**
 * ADA300 REEX LUT configuration.
 *
 * Override before including generated tables:
 *   -DGGML_LUT_NUM_SEGMENTS_REEX=31
 *
 * ggml-reex-lut-data.h embeds both 16- and 31-segment mixed-FP16 FP32 tables;
 * this macro selects the branch at compile time.
 */
#pragma once

#ifdef GGML_USE_REEX

#ifndef GGML_LUT_NUM_SEGMENTS_REEX
#define GGML_LUT_NUM_SEGMENTS_REEX 16
#endif

#if GGML_LUT_NUM_SEGMENTS_REEX != 16 && GGML_LUT_NUM_SEGMENTS_REEX != 31
#error "GGML_LUT_NUM_SEGMENTS_REEX must be 16 or 31"
#endif

#endif /* GGML_USE_REEX */
