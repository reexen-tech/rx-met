/**
 * REEX block-64 legacy quantization — CPU backend kernels.
 *
 * Registered from ggml-cpu.c's type_traits_cpu[] (from_float / vec_dot).
 * Scalar reference path only for now (correctness gold); SIMD can follow.
 *
 * Only compiled/declared when GGML_USE_REEX_Q64 is defined.
 */
#pragma once

#ifndef GGML_USE_REEX_Q64
#error "reex_q64_quants.h must be used only when GGML_USE_REEX_Q64 is defined"
#endif

#include "ggml.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// from_float: f32 -> block-64 (CPU). Thin wrappers over the base reference quantizers.
void quantize_row_q4_0_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_1_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_0_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_1_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_K_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q2_K_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q3_K_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_K_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q6_K_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_K_64S(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_K_64S(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q2_K_64S(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

// vec_dot: W(Q4_0_64) x A(Q8_0_64), both block=64.
void ggml_vec_dot_q4_0_64_q8_0_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc);

// vec_dot: W(Q8_0_64) x A(Q8_0_64), both block=64 (pure Q8_0_64 weight path).
void ggml_vec_dot_q8_0_64_q8_0_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc);

// vec_dot: W(Q4_1_64) x A(Q8_1_64)
void ggml_vec_dot_q4_1_64_q8_1_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
// vec_dot: W(Q5_0_64) x A(Q8_0_64)
void ggml_vec_dot_q5_0_64_q8_0_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
// vec_dot: W(Q5_1_64) x A(Q8_1_64)
void ggml_vec_dot_q5_1_64_q8_1_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
// vec_dot: W(Q8_1_64) x A(Q8_1_64) (symmetric; s field unused)
void ggml_vec_dot_q8_1_64_q8_1_64(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// vec_dot: W(Q4_K_64) x A(Q8_K) — K-quant block-64 weight against upstream q8_K activation
void ggml_vec_dot_q4_K_64_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
// vec_dot: W(Q2_K_64 / Q3_K_64 / Q5_K_64 / Q6_K_64) x A(Q8_K)
void ggml_vec_dot_q2_K_64_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_64_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_64_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_64_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
// vec_dot: W(Q5_K_64S / Q4_K_64S / Q2_K_64S) x A(Q8_K) — symmetric K-quant
void ggml_vec_dot_q5_K_64S_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_64S_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q2_K_64S_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
