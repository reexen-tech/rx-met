/**
 * Reex GEMM CPU — W4 权重 × 多版本激活（Q8 / Q16 / FLOAT）
 *
 * 仅当 GGML_USE_REEX_GEMM 定义时参与编译。构建时通过 GGML_REEX_GEMM_ACTIVATION 选激活版本：
 *   - Q8    → W4×Q8  ：CPU 主路径继续复用原生 ggml_vec_dot_q4_K_q8_K / Q8_K
 *             这里保留 ggml_vec_dot_q4_K_q8_reex / ggml_quantize_row_f32_to_q8_scale_row_reex
 *             主要用于测试对照与接口对齐，不作为 ggml-cpu.c 的独立主分发路径
 *   - Q16   → W4×Q16 ：vec_dot = ggml_vec_dot_q4_K_q16_reex
 *   - FLOAT → W4×F32 ：vec_dot = ggml_vec_dot_q4_K_f32_reex（仅权重量化，激活保持 float）
 *
 * Q16 = per-block (QK_K=256) scale + int16[]，与 Q4_K 块对齐；
 * Q8 = 复用现有 Q8_K（block_q8_K[]，QK_K=256），CPU 主 GEMM 分发仍走原生 Q4_K×Q8_K；
 * FLOAT = 不量化激活，直接反量化 Q4_K 做 float 点积，作为"仅权重量化"的精度参考。
 */
#ifndef GGML_USE_REEX_GEMM
#error "reex_gemm_cpu.h must be used only when GGML_USE_REEX_GEMM is defined"
#endif

#ifndef REEX_GEMM_CPU_H
#define REEX_GEMM_CPU_H

#include "ggml.h"

#ifndef GGML_COMMON_DECL
#define GGML_COMMON_DECL_C
#endif
#include "ggml-common.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Q16 (INT16) 激活：per-block scale, 与 Q4_K 块对齐 ========== */

typedef struct {
    float   d;           // per-block scale: amax / 32767
    int16_t qs[QK_K];   // quantized values (256 x int16)
} block_q16_K;

void ggml_quantize_row_f32_to_q16_scale_row_reex(
    const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void ggml_vec_dot_q4_K_q16_reex(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc);

static inline size_t ggml_q16_activation_row_size_reex(int64_t ne10) {
    int64_t nb = ne10 / QK_K;
    return (size_t)(nb * sizeof(block_q16_K));
}

/* ========== Q8 (INT8) 激活：复用 Q8_K；包装符号主要用于测试/对照 ========== */

void ggml_quantize_row_f32_to_q8_scale_row_reex(
    const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void ggml_vec_dot_q4_K_q8_reex(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc);

static inline size_t ggml_q8_activation_row_size_reex(int64_t ne10) {
    return (size_t)ggml_row_size(GGML_TYPE_Q8_K, ne10);
}

/* ========== FLOAT 激活（无激活量化，仅权重反量化） ========== */

void ggml_vec_dot_q4_K_f32_reex(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc);

/* ========== 辅助：层选择 ========== */

int ggml_reex_get_target_layer(void);
int ggml_reex_tensor_layer(const char * name);

#ifdef __cplusplus
}
#endif

#endif /* REEX_GEMM_CPU_H */
