#include "reex_q64_hw_dump.cuh"
#include "reex/ggml-reex-q64-hw-dump.h"

#include "../common.cuh"

#include <cstring>

#ifdef GGML_USE_REEX_Q64

__device__ int reex_q64_hw_active_dev;
__device__ int reex_q64_hw_max_m_dev;
__device__ int reex_q64_hw_count_dev;
__device__ int reex_q64_hw_row_dev;
__device__ int reex_q64_hw_col_dev;
__device__ int reex_q64_hw_kbx_dev;
__device__ reex_q64_hw_dev_record reex_q64_hw_records_dev[REEX_Q64_HW_MAX_RECORDS];

static reex_q64_hw_dev_record * g_hw_host_buf = nullptr;
static int                      g_hw_host_cap = 0;

extern "C" void ggml_reex_q64_hw_dump_arm(int max_m) {
    int active = 1;
    int zero   = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(reex_q64_hw_active_dev, &active, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(reex_q64_hw_max_m_dev, &max_m, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(reex_q64_hw_count_dev, &zero, sizeof(int)));
}

extern "C" void ggml_reex_q64_hw_dump_disarm(void) {
    int active = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(reex_q64_hw_active_dev, &active, sizeof(int)));
}

extern "C" int ggml_reex_q64_hw_dump_download(reex_q64_hw_psum_record * out, int max_out) {
    if (!out || max_out <= 0) {
        return 0;
    }
    int count = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&count, reex_q64_hw_count_dev, sizeof(int)));
    if (count <= 0) {
        return 0;
    }
    if (count > max_out) {
        count = max_out;
    }
    if (count > REEX_Q64_HW_MAX_RECORDS) {
        count = REEX_Q64_HW_MAX_RECORDS;
    }
    if (g_hw_host_cap < count) {
        delete[] g_hw_host_buf;
        g_hw_host_buf = new reex_q64_hw_dev_record[count];
        g_hw_host_cap = count;
    }
    CUDA_CHECK(cudaMemcpyFromSymbol(
        g_hw_host_buf, reex_q64_hw_records_dev,
        (size_t) count * sizeof(reex_q64_hw_dev_record)));
    for (int i = 0; i < count; ++i) {
        out[i].row        = g_hw_host_buf[i].row;
        out[i].col        = g_hw_host_buf[i].col;
        out[i].kbx        = g_hw_host_buf[i].kbx;
        out[i].psum_pre   = g_hw_host_buf[i].psum_pre;
        out[i].psum_post  = g_hw_host_buf[i].psum_post;
        out[i].output_f32 = g_hw_host_buf[i].output_f32;
    }
    return count;
}

#else

extern "C" void ggml_reex_q64_hw_dump_arm(int) {}
extern "C" void ggml_reex_q64_hw_dump_disarm(void) {}
extern "C" int  ggml_reex_q64_hw_dump_download(reex_q64_hw_psum_record *, int) { return 0; }

#endif
