#pragma once

/*
 * REEX GEMM CUDA — W4×INT8 / W4×INT16 量化矩阵乘法 CUDA 实现
 *
 * 与 CPU REEX 对应：
 *   - W4×INT16：本模块实现。F32 激活 → block_q16_K (per-block QK_K=256)，
 *               再与 Q4_K 权重做整数域点积 (ggml_cuda_mul_mat_reex_q16)。
 *   - W4×INT8：复用主路径：F32 → Q8_1 (quantize_row_q8_1_cuda) + Q4_K×Q8_1
 *              (ggml_cuda_mul_mat_vec_q)，见 ggml_cuda_mul_mat 中 use_mul_mat_vec_q 分支。
 *
 * 当前范围（仅 INT16）：
 *   - 支持 MUL_MAT 小 batch / MMVQ（当前 M <= 8）
 *   - 支持 MUL_MAT_ID（当前要求 src1->ne[3] == 1，即单 sample）
 *   - 更大 MUL_MAT batch 仍回退到现有 CUDA 主路径
 *
 * 架构（仅 INT16）：
 *   1. quantize_row_q16_K_cuda: F32 -> block_q16_K (per-block, QK_K=256)
 *   2. mul_mat_vec_q4_K_q16_reex: MMVQ 路径（单 token / 小 batch）
 *   3. ggml_cuda_mul_mat_reex_q16: 顶层调度入口
 */

#include "../common.cuh"
#include <cstdint>

/* ================================================================
 *  block_q16_K: CUDA 侧定义
 *  与 CPU 侧 reex_gemm_cpu.h 中的定义保持一致
 * ================================================================ */

#define REEX_Q16_SCALE 32767.0f
#define REEX_QK_K 256

struct block_q16_K_cuda {
    float   d;
    int16_t qs[REEX_QK_K];
};

static_assert(sizeof(block_q16_K_cuda) == sizeof(float) + REEX_QK_K * sizeof(int16_t),
              "block_q16_K_cuda size mismatch");

/* ================================================================
 *  CUDA kernel declarations
 * ================================================================ */

void quantize_row_q16_K_cuda(
    const float * x, void * vy,
    int64_t ne00, int64_t stride_row,
    int64_t nrows, cudaStream_t stream);

void ggml_cuda_mul_mat_reex_q16(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1,
    const ggml_tensor * ids, ggml_tensor * dst);

bool ggml_reex_cuda_should_use(void);
// Q16 CUDA trace helpers for tests/debugging.
extern "C" void ggml_reex_cuda_reset_stats(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_hits(void);
extern "C" int  ggml_reex_cuda_get_q16_batch_fallbacks(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_id_hits(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_id_unsupported(void);

// Internal trace hooks used by the CUDA dispatcher.
void ggml_reex_cuda_note_q16_batch_fallback(void);
void ggml_reex_cuda_note_q16_mul_mat_id_unsupported(void);

