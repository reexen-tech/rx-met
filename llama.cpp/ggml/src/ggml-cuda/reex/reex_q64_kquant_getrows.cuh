// REEX K-quant block-64 — CUDA get_rows (gather + dequantize).
//
// The generic k_get_rows uses a 2-value dequant functor that doesn't fit the
// K-quant per-sub-block scale layout, so each K-quant block-64 type gets a
// dedicated one-element-per-thread kernel + launcher (same shape as the
// TURBOQUANT get_rows path). Used for quantized token-embedding lookups on GPU.
//
// Per-element dequant mirrors ggml-reex-q64.c exactly: element j in 0..255 of a
// super-block lives in sub-block sb=j/64, local index l=j%64.
//
// Must be included by getrows.cu (after convert.cuh for ggml_cuda_cast) and only
// under #ifdef GGML_USE_REEX_Q64.
#pragma once

#include "reex/ggml-reex-q64-common.h"

#ifndef REEX_Q64_H2F_DEFINED
#define REEX_Q64_H2F_DEFINED
static __device__ __forceinline__ float q64_h2f(const ggml_fp16_t h) {
    return __half2float(__ushort_as_half((unsigned short) h));
}
#endif

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

// === per-element dequant (element j in 0..QK_K_64-1) =======================
static __device__ __forceinline__ float q64k_deq_elem_q4_K_64(const block_q4_K_64 * b, int j) {
    const int sb = j / 64, l = j % 64;
    int sc, mn; q64k_get_scale_min(sb, b->scales, sc, mn);
    const uint8_t * q = b->qs + sb*32;
    const int qv = (l < 32) ? (q[l] & 0xF) : (q[l - 32] >> 4);
    return q64_h2f(b->d)*sc*qv - q64_h2f(b->dmin)*mn;
}

static __device__ __forceinline__ float q64k_deq_elem_q2_K_64(const block_q2_K_64 * b, int j) {
    const int sb = j / 64, l = j % 64;
    const uint8_t s = b->scales[sb];
    const int sc = s & 0xF, mn = s >> 4;
    const uint8_t * qs = b->qs + sb*16;
    const int qv = (qs[l >> 2] >> (2*(l & 3))) & 3;
    return q64_h2f(b->d)*sc*qv - q64_h2f(b->dmin)*mn;
}

static __device__ __forceinline__ float q64k_deq_elem_q3_K_64(const block_q3_K_64 * b, int j) {
    const int sb = j / 64, l = j % 64;
    const int sc = q64k_unpack4x6(sb, b->scales) - 32;
    const uint8_t * qs = b->qs    + sb*16;
    const uint8_t * hm = b->hmask + sb*8;
    const int low2 = (qs[l >> 2] >> (2*(l & 3))) & 3;
    const int hbit = (hm[l >> 3] >> (l & 7)) & 1;
    return q64_h2f(b->d)*sc*((low2 | (hbit << 2)) - 4);
}

static __device__ __forceinline__ float q64k_deq_elem_q5_K_64(const block_q5_K_64 * b, int j) {
    const int sb = j / 64, l = j % 64;
    int sc, mn; q64k_get_scale_min(sb, b->scales, sc, mn);
    const uint8_t * q  = b->qs + sb*32;
    const uint8_t * qh = b->qh + sb*8;
    const int hb  = (qh[l >> 3] >> (l & 7)) & 1;
    const int nib = (l < 32) ? (q[l] & 0xF) : (q[l - 32] >> 4);
    const int qv  = nib | (hb << 4);
    return q64_h2f(b->d)*sc*qv - q64_h2f(b->dmin)*mn;
}

static __device__ __forceinline__ float q64k_deq_elem_q6_K_64(const block_q6_K_64 * b, int j) {
    const int sb = j / 64, l = j % 64;
    const int sc = b->scales[sb];
    const uint8_t * ql = b->ql + sb*32;
    const uint8_t * qh = b->qh + sb*16;
    const int low4 = (ql[l >> 1] >> (4*(l & 1))) & 0xF;
    const int hi2  = (qh[l >> 2] >> (2*(l & 3))) & 3;
    return q64_h2f(b->d)*sc*((low4 | (hi2 << 4)) - 32);
}

static __device__ __forceinline__ float q64k_deq_elem_q5_K_64S(const block_q5_K_64S * b, int j) {
    const int sb = j / 64, l = j % 64;
    const int sc = q64k_unpack4x6(sb, b->scales) - 32;
    const uint8_t * q  = b->qs + sb*32;
    const uint8_t * qh = b->qh + sb*8;
    const int hb  = (qh[l >> 3] >> (l & 7)) & 1;
    const int nib = (l < 32) ? (q[l] & 0xF) : (q[l - 32] >> 4);
    const int qv  = nib | (hb << 4);
    return q64_h2f(b->d)*sc*(qv - 16);
}

static __device__ __forceinline__ float q64k_deq_elem_q4_K_64S(const block_q4_K_64S * b, int j) {
    const int sb = j / 64, l = j % 64;
    const int sc = q64k_unpack4x6(sb, b->scales) - 32;
    const uint8_t * q = b->qs + sb*32;
    const int qv = (l < 32) ? (q[l] & 0xF) : (q[l - 32] >> 4);
    return q64_h2f(b->d)*sc*(qv - 8);
}

static __device__ __forceinline__ float q64k_deq_elem_q2_K_64S(const block_q2_K_64S * b, int j) {
    const int sb = j / 64, l = j % 64;
    const int sc = q64k_unpack4x4(sb, b->scales) - 8;
    const uint8_t * qs = b->qs + sb*16;
    const int qv = (qs[l >> 2] >> (2*(l & 3))) & 3;
    return q64_h2f(b->d)*sc*(qv - 2);
}

// === per-type kernel + launcher (one element per thread) ===================
#define REEX_Q64K_GETROWS(NAME, BLK)                                                                       \
template<typename dst_t>                                                                                   \
static __global__ void k_get_rows_##NAME(                                                                  \
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,       \
        const int64_t ne00, const int64_t ne11, const int64_t ne12,                                        \
        const size_t s1, const size_t s2, const size_t s3,                                                 \
        const size_t nb01, const size_t nb02, const size_t nb03,                                           \
        const size_t s10, const size_t s11, const size_t s12) {                                            \
    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {                                          \
        for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) { \
            const int i10 = blockIdx.x;                                                                    \
            const int i11 = z / ne12;                                                                      \
            const int i12 = z % ne12;                                                                      \
            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];                                             \
            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;                                               \
            const char * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;                   \
            const BLK * b = (const BLK *) src0_row + (i00 / QK_K_64);                                       \
            dst_row[i00] = ggml_cuda_cast<dst_t>(q64k_deq_elem_##NAME(b, i00 % QK_K_64));                   \
        }                                                                                                  \
    }                                                                                                      \
}                                                                                                          \
template<typename dst_t>                                                                                   \
static void reex_q64k_get_rows_##NAME##_cuda(                                                              \
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,                                        \
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,                       \
        const int64_t ne10, const int64_t ne11, const int64_t ne12,                                        \
        const size_t nb10, const size_t nb11, const size_t nb12,                                           \
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {                       \
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);                                                 \
    const int  block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;             \
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));                 \
    const size_t s1  = nb1  / sizeof(dst_t);                                                               \
    const size_t s2  = nb2  / sizeof(dst_t);                                                               \
    const size_t s3  = nb3  / sizeof(dst_t);                                                               \
    const size_t s10 = nb10 / sizeof(int32_t);                                                             \
    const size_t s11 = nb11 / sizeof(int32_t);                                                             \
    const size_t s12 = nb12 / sizeof(int32_t);                                                             \
    k_get_rows_##NAME<<<block_nums, block_dims, 0, stream>>>(                                              \
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);             \
}

REEX_Q64K_GETROWS(q4_K_64, block_q4_K_64)
REEX_Q64K_GETROWS(q2_K_64, block_q2_K_64)
REEX_Q64K_GETROWS(q3_K_64, block_q3_K_64)
REEX_Q64K_GETROWS(q5_K_64, block_q5_K_64)
REEX_Q64K_GETROWS(q6_K_64, block_q6_K_64)
REEX_Q64K_GETROWS(q5_K_64S, block_q5_K_64S)
REEX_Q64K_GETROWS(q4_K_64S, block_q4_K_64S)
REEX_Q64K_GETROWS(q2_K_64S, block_q2_K_64S)
