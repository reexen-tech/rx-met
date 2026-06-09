#ifdef GGML_USE_REEX

#include "ggml-reex-lut.h"
#include "reex/ggml-reex-lut-data.h"
#include "ggml-impl.h"

// Host-side LUT tables (filled from reex/ggml-reex-lut-data.h in init)
struct ggml_lut_segment_fp16_reex ggml_lut_sin_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_bf16_reex ggml_lut_sin_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_fp16_reex ggml_lut_sigmoid_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_bf16_reex ggml_lut_sigmoid_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_fp16_reex ggml_lut_exponential_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_bf16_reex ggml_lut_exponential_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_fp16_reex ggml_lut_sqrt_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_bf16_reex ggml_lut_sqrt_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_fp16_reex ggml_lut_log_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_bf16_reex ggml_lut_log_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_fp16_reex ggml_lut_power_2_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_bf16_reex ggml_lut_power_2_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_fp16_reex ggml_lut_reciprocal_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_bf16_reex ggml_lut_reciprocal_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_fp16_reex ggml_lut_rsqrt_fp16_reex[GGML_LUT_NUM_SEGMENTS_REEX];
struct ggml_lut_segment_bf16_reex ggml_lut_rsqrt_bf16_reex[GGML_LUT_NUM_SEGMENTS_REEX];

bool ggml_trig_lut_initialized = false;

#define GGML_REEX_COPY_OP(op) do { \
    for (int i = 0; i < GGML_LUT_NUM_SEGMENTS_REEX; ++i) { \
        ggml_lut_##op##_fp16_reex[i].threshold = (ggml_fp16_t)ggml_reex_##op##_fp16_threshold[i]; \
        ggml_lut_##op##_fp16_reex[i].b        = (ggml_fp16_t)ggml_reex_##op##_fp16_b[i]; \
        ggml_lut_##op##_fp16_reex[i].c        = (ggml_fp16_t)ggml_reex_##op##_fp16_c[i]; \
        ggml_lut_##op##_bf16_reex[i].threshold.bits = ggml_reex_##op##_bf16_threshold[i]; \
        ggml_lut_##op##_bf16_reex[i].b.bits        = ggml_reex_##op##_bf16_b[i]; \
        ggml_lut_##op##_bf16_reex[i].c.bits        = ggml_reex_##op##_bf16_c[i]; \
    } \
} while (0)

/**
 * Initialize all LUT tables from header-defined data (no runtime fitting).
 */
void ggml_init_trigonometric_lut_REEX(void) {
    if (ggml_trig_lut_initialized) {
        return;
    }
    GGML_REEX_COPY_OP(sin);
    GGML_REEX_COPY_OP(sigmoid);
    GGML_REEX_COPY_OP(exponential);
    GGML_REEX_COPY_OP(sqrt);
    GGML_REEX_COPY_OP(log);
    GGML_REEX_COPY_OP(power_2);
    GGML_REEX_COPY_OP(reciprocal);
    GGML_REEX_COPY_OP(rsqrt);
    ggml_trig_lut_initialized = true;
}

#endif /* GGML_USE_REEX */
