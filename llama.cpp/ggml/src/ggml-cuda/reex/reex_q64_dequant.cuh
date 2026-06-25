// REEX block-64 legacy quant — CUDA dequantization (dequant -> FP16/FP32 path).
//
// Only the dequantize path is provided: weights are dequantized on-GPU and
// mul_mat runs through cuBLAS GEMM (MMVQ/MMQ are explicitly disabled for these
// types in ggml-cuda.cu). This mirrors the established REEX_TURBOQUANT isolation
// pattern. Header-only (static/template) so convert.cu and getrows.cu can both
// include it under #ifdef GGML_USE_REEX_Q64.
#pragma once

#include "../common.cuh"
#include "reex/ggml-reex-q64-common.h"

// block fp16 scale/min fields are stored as ggml_fp16_t (uint16_t bit pattern);
// reinterpret the bits as half before converting (NOT an integer->float cast).
#ifndef REEX_Q64_H2F_DEFINED
#define REEX_Q64_H2F_DEFINED
static __device__ __forceinline__ float q64_h2f(const ggml_fp16_t h) {
    return __half2float(__ushort_as_half((unsigned short) h));
}
#endif

// ============================================================================
//  getrows-style per-pair dequantizers: void(vx, ib, iqs, float2 & v)
//  iqs in [0, qk/qr); writes element iqs (v.x) and iqs + qk/2 (v.y).
// ============================================================================

static __device__ __forceinline__ void dequantize_q4_0_64(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q4_0_64 * x = (const block_q4_0_64 *) vx;
    const float d = q64_h2f(x[ib].d);
    const int vui = x[ib].qs[iqs];
    v.x = (float)(((vui & 0xF) ^ 0x8) - 0x8) * d; // sign-extend signed 4-bit
    v.y = (float)(((vui >>  4) ^ 0x8) - 0x8) * d;
}

static __device__ __forceinline__ void dequantize_q8_0_64(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q8_0_64 * x = (const block_q8_0_64 *) vx;
    const float d = q64_h2f(x[ib].d);
    v.x = x[ib].qs[iqs + 0] * d;
    v.y = x[ib].qs[iqs + 1] * d;
}

static __device__ __forceinline__ void dequantize_q4_1_64(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q4_1_64 * x = (const block_q4_1_64 *) vx;
    const float d = q64_h2f(x[ib].d);
    const float m = q64_h2f(x[ib].m);
    const int vui = x[ib].qs[iqs];
    v.x = (vui & 0xF) * d + m;
    v.y = (vui >>  4) * d + m;
}

static __device__ __forceinline__ void dequantize_q5_0_64(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q5_0_64 * x = (const block_q5_0_64 *) vx;
    const float d = q64_h2f(x[ib].d);
    uint64_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 28))     ) & 0x10; // iqs + qk/2 - 4, qk/2 = 32
    const int u0 = (x[ib].qs[iqs] & 0xf) | xh_0;
    const int u1 = (x[ib].qs[iqs] >>  4) | xh_1;
    v.x = (float)((u0 ^ 0x10) - 0x10) * d; // sign-extend signed 5-bit
    v.y = (float)((u1 ^ 0x10) - 0x10) * d;
}

static __device__ __forceinline__ void dequantize_q5_1_64(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q5_1_64 * x = (const block_q5_1_64 *) vx;
    const float d = q64_h2f(x[ib].d);
    const float m = q64_h2f(x[ib].m);
    uint64_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 28))     ) & 0x10;
    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0) * d + m;
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1) * d + m;
}

static __device__ __forceinline__ void dequantize_q8_1_64(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q8_1_64 * x = (const block_q8_1_64 *) vx;
    const float d = q64_h2f(x[ib].d);
    v.x = x[ib].qs[iqs + 0] * d;
    v.y = x[ib].qs[iqs + 1] * d;
}

// ============================================================================
//  contiguous block dequant kernels + host launchers for to_fp16 / to_fp32.
//  One CUDA block per quant block; 32 threads, each emits 2 (4/5-bit) or
//  2 (8-bit) outputs.
// ============================================================================

template<typename dst_t>
static __global__ void dequantize_block_q4_0_64(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q4_0_64 * x = (const block_q4_0_64 *) vx + i;
    const float d = q64_h2f(x->d);
    dst_t * y = yy + i*QK4_0_64;
    const int j = threadIdx.x; // 0..31
    const int vui = x->qs[j];
    y[j]              = ggml_cuda_cast<dst_t>((float)(((vui & 0xF) ^ 0x8) - 0x8) * d); // sign-extend signed 4-bit
    y[j + QK4_0_64/2] = ggml_cuda_cast<dst_t>((float)(((vui >>  4) ^ 0x8) - 0x8) * d);
}

template<typename dst_t>
static __global__ void dequantize_block_q8_0_64(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q8_0_64 * x = (const block_q8_0_64 *) vx + i;
    const float d = q64_h2f(x->d);
    dst_t * y = yy + i*QK8_0_64;
    const int j = threadIdx.x;      // 0..31
    y[j]    = ggml_cuda_cast<dst_t>(x->qs[j]    * d);
    y[j+32] = ggml_cuda_cast<dst_t>(x->qs[j+32] * d);
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1_64(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q4_1_64 * x = (const block_q4_1_64 *) vx + i;
    const float d = q64_h2f(x->d);
    const float m = q64_h2f(x->m);
    dst_t * y = yy + i*QK4_1_64;
    const int j = threadIdx.x;
    const int vui = x->qs[j];
    y[j]              = ggml_cuda_cast<dst_t>((vui & 0xF) * d + m);
    y[j + QK4_1_64/2] = ggml_cuda_cast<dst_t>((vui >>  4) * d + m);
}

template<typename dst_t>
static __global__ void dequantize_block_q5_0_64(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q5_0_64 * x = (const block_q5_0_64 *) vx + i;
    const float d = q64_h2f(x->d);
    uint64_t qh;
    memcpy(&qh, x->qh, sizeof(qh));
    dst_t * y = yy + i*QK5_0_64;
    const int j = threadIdx.x;
    const int xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (j + 28))     ) & 0x10;
    const int u0 = (x->qs[j] & 0xf) | xh_0;
    const int u1 = (x->qs[j] >>  4) | xh_1;
    y[j]              = ggml_cuda_cast<dst_t>((float)((u0 ^ 0x10) - 0x10) * d); // sign-extend signed 5-bit
    y[j + QK5_0_64/2] = ggml_cuda_cast<dst_t>((float)((u1 ^ 0x10) - 0x10) * d);
}

template<typename dst_t>
static __global__ void dequantize_block_q5_1_64(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q5_1_64 * x = (const block_q5_1_64 *) vx + i;
    const float d = q64_h2f(x->d);
    const float m = q64_h2f(x->m);
    uint64_t qh;
    memcpy(&qh, x->qh, sizeof(qh));
    dst_t * y = yy + i*QK5_1_64;
    const int j = threadIdx.x;
    const int xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (j + 28))     ) & 0x10;
    y[j]              = ggml_cuda_cast<dst_t>((((x->qs[j] & 0xf) | xh_0)) * d + m);
    y[j + QK5_1_64/2] = ggml_cuda_cast<dst_t>((((x->qs[j] >>  4) | xh_1)) * d + m);
}

template<typename dst_t>
static __global__ void dequantize_block_q8_1_64(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;
    const block_q8_1_64 * x = (const block_q8_1_64 *) vx + i;
    const float d = q64_h2f(x->d);
    dst_t * y = yy + i*QK8_1_64;
    const int j = threadIdx.x;
    y[j]    = ggml_cuda_cast<dst_t>(x->qs[j]    * d);
    y[j+32] = ggml_cuda_cast<dst_t>(x->qs[j+32] * d);
}

#define GGML_Q64_DEFINE_TO_T(NAME, BLK, QK)                                                          \
    template<typename dst_t>                                                                         \
    static void reex_q64_to_t_##NAME##_cuda(const void * vx, dst_t * y, int64_t k, cudaStream_t s) { \
        const int64_t nb = k / (QK);                                                                 \
        dequantize_block_##NAME<<<nb, 32, 0, s>>>(vx, y, nb);                                         \
    }

GGML_Q64_DEFINE_TO_T(q4_0_64, block_q4_0_64, QK4_0_64)
GGML_Q64_DEFINE_TO_T(q8_0_64, block_q8_0_64, QK8_0_64)
GGML_Q64_DEFINE_TO_T(q4_1_64, block_q4_1_64, QK4_1_64)
GGML_Q64_DEFINE_TO_T(q5_0_64, block_q5_0_64, QK5_0_64)
GGML_Q64_DEFINE_TO_T(q5_1_64, block_q5_1_64, QK5_1_64)
GGML_Q64_DEFINE_TO_T(q8_1_64, block_q8_1_64, QK8_1_64)
