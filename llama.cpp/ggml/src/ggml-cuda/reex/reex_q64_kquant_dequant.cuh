// REEX K-quant block-64 — CUDA dequantization (dequant -> FP16/FP32 path).
//
// Phase 1 of the K-quant block-64 CUDA enablement: weights are dequantized
// on-GPU and mul_mat runs through cuBLAS GEMM (MMVQ/MMQ are forced off for
// these types in ggml-cuda.cu until their kernels land). Mirrors the legacy
// reex_q64_dequant.cuh isolation pattern.
//
// The bit layouts here MUST match the CPU reference (de)quantizers in
// ggml-reex-q64.c exactly:
//   - super-block = 256, sub-block = 64 (4 sub-blocks per super-block)
//   - Q2/Q5/Q4: x = d*scale*q - dmin*min ; Q3/Q6 and *_64S: x = d*scale*q with
//     q (and the *_64S sub-block scale) stored as two's-complement signed ints
//
// Header-only (static/template) so convert.cu / getrows.cu can include it under
// #ifdef GGML_USE_REEX_Q64. Must be included AFTER reex_q64_dequant.cuh (reuses
// q64_h2f) or stand alone — q64_h2f is re-declared guarded below.
#pragma once

#include "../common.cuh"
#include "reex/ggml-reex-q64-common.h"

#ifndef REEX_Q64_H2F_DEFINED
#define REEX_Q64_H2F_DEFINED
static __device__ __forceinline__ float q64_h2f(const ggml_fp16_t h) {
    return __half2float(__ushort_as_half((unsigned short) h));
}
#endif

// device-side scale unpackers (host inline versions live in the common header).
#ifndef REEX_Q64K_SCALE_HELPERS_DEFINED
#define REEX_Q64K_SCALE_HELPERS_DEFINED
static __device__ __forceinline__ void q64k_get_scale_min(int j, const uint8_t * s, int & d, int & m) {
    const uint32_t us = (uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16);
    const uint32_t um = (uint32_t)s[3] | ((uint32_t)s[4] << 8) | ((uint32_t)s[5] << 16);
    d = (us >> (6*j)) & 0x3F;
    m = (um >> (6*j)) & 0x3F;
}

static __device__ __forceinline__ int q64k_unpack4x6(int j, const uint8_t * s) {
    const uint32_t u = (uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16);
    return (u >> (6*j)) & 0x3F;
}
static __device__ __forceinline__ int q64k_unpack4x4(int j, const uint8_t * s) {
    return (s[j >> 1] >> (4*(j & 1))) & 0xF;
}
#endif

// ============================================================================
//  Contiguous block dequant kernels. One CUDA block per super-block (256
//  elements), 64 threads; thread j handles element (sub*64 + j) for each of the
//  4 sub-blocks.
// ============================================================================

// Q4_K_64: 4-bit, x = d*scale*q - dmin*min.
// qs layout per sub-block: q[l] low nibble -> element l, high nibble -> element l+32 (l in 0..31).
template<typename dst_t>
static __global__ void dequantize_block_q4_K_64(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q4_K_64 * x = (const block_q4_K_64 *) vx + i;
    const float d   = q64_h2f(x->d);
    const float min = q64_h2f(x->dmin);
    dst_t * y = yy + i*QK_K_64;
    const int j = threadIdx.x; // 0..63
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        int sc, m;
        q64k_get_scale_min(sub, x->scales, sc, m);
        const float d1 = d * sc;
        const float m1 = min * m;
        const uint8_t * q = x->qs + sub*32;
        const int qv = (j < 32) ? (q[j] & 0xF) : (q[j-32] >> 4);
        y[sub*64 + j] = ggml_cuda_cast<dst_t>(d1 * qv - m1);
    }
}

// Q2_K_64: 2-bit, x = d*scale*q - dmin*min. scales[sub] = 4-bit scale | 4-bit min.
// qs sequential 2-bit: element e -> qs[e>>2] >> (2*(e&3)).
template<typename dst_t>
static __global__ void dequantize_block_q2_K_64(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q2_K_64 * x = (const block_q2_K_64 *) vx + i;
    const float d   = q64_h2f(x->d);
    const float min = q64_h2f(x->dmin);
    dst_t * y = yy + i*QK_K_64;
    const int j = threadIdx.x; // 0..63
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const uint8_t scb = x->scales[sub];
        const float d1 = d   * (scb & 0xF);
        const float m1 = min * (scb >> 4);
        const int e = sub*64 + j;
        const int q = (x->qs[e >> 2] >> (2*(e & 3))) & 3;
        y[e] = ggml_cuda_cast<dst_t>(d1 * q - m1);
    }
}

// Q3_K_64: 3-bit, x = d*scale*q, signed q [-4,3] + signed 6-bit scale (two's complement).
// qs low 2 bits sequential; hmask 3rd (sign) bit sequential (element e -> hmask[e>>3] bit e&7).
template<typename dst_t>
static __global__ void dequantize_block_q3_K_64(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q3_K_64 * x = (const block_q3_K_64 *) vx + i;
    const float d_all = q64_h2f(x->d);
    dst_t * y = yy + i*QK_K_64;
    const int j = threadIdx.x; // 0..63
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const int sc = q64_unpack4x6_s(sub, x->scales);
        const float dl = d_all * sc;
        const int e = sub*64 + j;
        const int low2 = (x->qs[e >> 2] >> (2*(e & 3))) & 3;
        const int hbit = (x->hmask[e >> 3] >> (e & 7)) & 1;
        const int u3 = low2 | (hbit << 2);
        y[e] = ggml_cuda_cast<dst_t>(dl * ((u3 ^ 0x4) - 0x4)); // sign-extend signed 3-bit
    }
}

// Q5_K_64: 5-bit, x = d*scale*q - dmin*min.
// qs low 4 bits per sub-block (q[l] low->elem l, high->elem l+32); qh 5th bit sequential.
template<typename dst_t>
static __global__ void dequantize_block_q5_K_64(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q5_K_64 * x = (const block_q5_K_64 *) vx + i;
    const float d   = q64_h2f(x->d);
    const float min = q64_h2f(x->dmin);
    dst_t * y = yy + i*QK_K_64;
    const int j = threadIdx.x; // 0..63
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        int sc, m;
        q64k_get_scale_min(sub, x->scales, sc, m);
        const float d1 = d * sc;
        const float m1 = min * m;
        const uint8_t * ql = x->qs + sub*32;
        const uint8_t * qh = x->qh + sub*8;
        const int low4 = (j < 32) ? (ql[j] & 0xF) : (ql[j-32] >> 4);
        const int hbit = (qh[j >> 3] >> (j & 7)) & 1;
        const int q5 = low4 | (hbit << 4);
        y[sub*64 + j] = ggml_cuda_cast<dst_t>(d1 * q5 - m1);
    }
}

// Q6_K_64: 6-bit, x = d*scale*q, signed q [-32,31] (two's complement), int8 scale.
// ql low 4 bits sequential; qh high 2 bits sequential (element e -> qh[e>>2] >> (2*(e&3))).
template<typename dst_t>
static __global__ void dequantize_block_q6_K_64(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q6_K_64 * x = (const block_q6_K_64 *) vx + i;
    const float d = q64_h2f(x->d);
    dst_t * y = yy + i*QK_K_64;
    const int j = threadIdx.x; // 0..63
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const int sc = x->scales[sub];
        const float dl = d * sc;
        const int e = sub*64 + j;
        const int low4 = (x->ql[e >> 1] >> (4*(e & 1))) & 0xF;
        const int hi2  = (x->qh[e >> 2] >> (2*(e & 3))) & 3;
        const int u6 = low4 | (hi2 << 4);
        y[e] = ggml_cuda_cast<dst_t>(dl * ((u6 ^ 0x20) - 0x20)); // sign-extend signed 6-bit
    }
}

// Q5_K_64S: 5-bit SYMMETRIC, x = d*scale*q, signed q [-16,15], int6 signed scale, no min.
// qs low 4 bits per sub-block (q[l] low->elem l, high->elem l+32); qh 5th (sign) bit sequential.
template<typename dst_t>
static __global__ void dequantize_block_q5_K_64S(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q5_K_64S * x = (const block_q5_K_64S *) vx + i;
    const float d = q64_h2f(x->d);
    dst_t * y = yy + i*QK_K_64;
    const int j = threadIdx.x; // 0..63
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const int sc = q64_unpack4x6_s(sub, x->scales);
        const float dl = d * sc;
        const uint8_t * ql = x->qs + sub*32;
        const uint8_t * qh = x->qh + sub*8;
        const int low4 = (j < 32) ? (ql[j] & 0xF) : (ql[j-32] >> 4);
        const int hbit = (qh[j >> 3] >> (j & 7)) & 1;
        const int u5 = low4 | (hbit << 4);
        y[sub*64 + j] = ggml_cuda_cast<dst_t>(dl * ((u5 ^ 0x10) - 0x10)); // sign-extend signed 5-bit
    }
}

// Q4_K_64S: 4-bit SYMMETRIC, x = d*scale*q, signed q [-8,7], int6 signed scale, no min.
template<typename dst_t>
static __global__ void dequantize_block_q4_K_64S(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q4_K_64S * x = (const block_q4_K_64S *) vx + i;
    const float d = q64_h2f(x->d);
    dst_t * y = yy + i*QK_K_64;
    const int j = threadIdx.x; // 0..63
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const int sc = q64_unpack4x6_s(sub, x->scales);
        const float dl = d * sc;
        const uint8_t * q = x->qs + sub*32;
        const int qv = (j < 32) ? (q[j] & 0xF) : (q[j-32] >> 4);
        y[sub*64 + j] = ggml_cuda_cast<dst_t>(dl * ((qv ^ 0x8) - 0x8)); // sign-extend signed 4-bit
    }
}

// Q2_K_64S: 2-bit SYMMETRIC, x = d*scale*q, signed q [-2,1], int4 signed scale, no min.
template<typename dst_t>
static __global__ void dequantize_block_q2_K_64S(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q2_K_64S * x = (const block_q2_K_64S *) vx + i;
    const float d = q64_h2f(x->d);
    dst_t * y = yy + i*QK_K_64;
    const int j = threadIdx.x; // 0..63
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const int sc = q64_unpack4x4_s(sub, x->scales);
        const float dl = d * sc;
        const int e = sub*64 + j;
        const int u2 = (x->qs[e >> 2] >> (2*(e & 3))) & 3;
        y[e] = ggml_cuda_cast<dst_t>(dl * ((u2 ^ 0x2) - 0x2)); // sign-extend signed 2-bit
    }
}

#define GGML_Q64K_DEFINE_TO_T(NAME, BLK)                                                             \
    template<typename dst_t>                                                                         \
    static void reex_q64_to_t_##NAME##_cuda(const void * vx, dst_t * y, int64_t k, cudaStream_t s) { \
        const int64_t nb = k / QK_K_64;                                                              \
        dequantize_block_##NAME<<<nb, 64, 0, s>>>(vx, y, nb);                                         \
    }

GGML_Q64K_DEFINE_TO_T(q4_K_64, block_q4_K_64)
GGML_Q64K_DEFINE_TO_T(q2_K_64, block_q2_K_64)
GGML_Q64K_DEFINE_TO_T(q3_K_64, block_q3_K_64)
GGML_Q64K_DEFINE_TO_T(q5_K_64, block_q5_K_64)
GGML_Q64K_DEFINE_TO_T(q6_K_64, block_q6_K_64)
GGML_Q64K_DEFINE_TO_T(q5_K_64S, block_q5_K_64S)
GGML_Q64K_DEFINE_TO_T(q4_K_64S, block_q4_K_64S)
GGML_Q64K_DEFINE_TO_T(q2_K_64S, block_q2_K_64S)
