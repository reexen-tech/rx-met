/**
 * REEX block-64 hardware alignment dump — host-side API.
 */
#pragma once

#ifdef GGML_USE_REEX_Q64

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct reex_q64_hw_dump_cfg {
    bool  active;
    char  dir[512];
    char  tensor[128];   /* empty = any */
    int   layer;         /* -1 = any */
    int   token;         /* default 0 */
    int   max_m;         /* 0 = unlimited row cap */
    int   psum_bits;     /* from REEX_Q64_PSUM_BITS */
    bool  dump_pre;
    bool  dump_post;
} reex_q64_hw_dump_cfg;

typedef struct reex_q64_hw_psum_record {
    int32_t row;
    int32_t col;
    int32_t kbx;
    int32_t psum_pre;
    int32_t psum_post;
    float   output_f32;
} reex_q64_hw_psum_record;

const reex_q64_hw_dump_cfg * reex_q64_hw_dump_get_cfg(void);
bool reex_q64_hw_dump_should_record(const char * tensor, int layer, int token);

/** Arm CUDA device buffer before MUL_MAT (cb_eval ask=true). */
void reex_q64_hw_dump_matmul_begin(const char * name, int layer);

/** Download psum + write case bins after MUL_MAT (cb_eval ask=false). */
void reex_q64_hw_dump_matmul_end(
    const char * name, int layer,
    const void * weight_data, int64_t weight_bytes,
    const void * act_data, int64_t act_bytes,
    const float * output_data, int64_t output_count);

/** Write run-level meta.json at process exit. */
void reex_q64_hw_dump_flush(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_USE_REEX_Q64 */
