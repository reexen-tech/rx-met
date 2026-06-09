/**
 * REEX LUT segment count — single definition shared by:
 *   - ggml-reex-lut.h (array bounds, CUDA GGML_CUDA_LUT_NUM_SEGMENTS)
 *   - ggml-reex-lut-data.h (generated tables; CPU & CUDA host paths)
 *
 * Override before any of the above: -DGGML_LUT_NUM_SEGMENTS_REEX=31
 * Valid values: 16 (default) or 31 (must match generated dual branches in ggml-reex-lut-data.h).
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
