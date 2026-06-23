#pragma once

#ifdef GGML_USE_REEX_Q64

#include <stdint.h>

#define REEX_Q64_HW_MAX_RECORDS 65536

typedef struct {
    int32_t row;
    int32_t col;
    int32_t kbx;
    int32_t psum_pre;
    int32_t psum_post;
    float   output_f32;
} reex_q64_hw_dev_record;

extern __device__ int reex_q64_hw_active_dev;
extern __device__ int reex_q64_hw_max_m_dev;
extern __device__ int reex_q64_hw_count_dev;
extern __device__ int reex_q64_hw_row_dev;
extern __device__ int reex_q64_hw_col_dev;
extern __device__ int reex_q64_hw_kbx_dev;
extern __device__ reex_q64_hw_dev_record reex_q64_hw_records_dev[REEX_Q64_HW_MAX_RECORDS];

static __device__ __forceinline__ void reex_q64_hw_dump_set_ctx(
    const int row, const int col, const int kbx) {
    reex_q64_hw_row_dev = row;
    reex_q64_hw_col_dev = col;
    reex_q64_hw_kbx_dev = kbx;
}

static __device__ __forceinline__ void reex_q64_hw_dump_try_record(
    const int psum_pre, const int psum_post, const float output_f32) {
    if (!reex_q64_hw_active_dev) {
        return;
    }
    if (reex_q64_hw_max_m_dev > 0 && reex_q64_hw_row_dev >= reex_q64_hw_max_m_dev) {
        return;
    }
    const int idx = atomicAdd(&reex_q64_hw_count_dev, 1);
    if (idx >= REEX_Q64_HW_MAX_RECORDS) {
        return;
    }
    reex_q64_hw_dev_record & r = reex_q64_hw_records_dev[idx];
    r.row         = reex_q64_hw_row_dev;
    r.col         = reex_q64_hw_col_dev;
    r.kbx         = reex_q64_hw_kbx_dev;
    r.psum_pre    = psum_pre;
    r.psum_post   = psum_post;
    r.output_f32  = output_f32;
}

#endif /* GGML_USE_REEX_Q64 */
