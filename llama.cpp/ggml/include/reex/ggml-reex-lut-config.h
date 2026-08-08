/**
 * ADA300 REEX LUT configuration.
 *
 * The current production export and validation contract supports exactly the
 * 16-segment mixed-FP16 path.
 */
#pragma once

#ifdef GGML_USE_REEX

#ifndef GGML_LUT_NUM_SEGMENTS_REEX
#define GGML_LUT_NUM_SEGMENTS_REEX 16
#endif

#if GGML_LUT_NUM_SEGMENTS_REEX != 16
#error "ADA300 REEX LUT supports exactly 16 segments"
#endif

#endif /* GGML_USE_REEX */
