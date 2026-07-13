// REEX K-quant block-64 — CUDA MMVQ vec_dot (high-performance matvec path).
//
// Phase 2 of the K-quant block-64 CUDA enablement. The standard mul_mat_vec_q
// framework quantizes activations to block_q8_1 (QK8_1=32). We register the
// block-64 K-quant types with qk=256 / qi=4 / vdr=1, so the framework spawns
// EXACTLY 4 cooperating threads per super-block — one per 64-element sub-block —
// and passes iqs = sub-block index (0..3). Each super-block therefore aligns to
// 8 q8_1 activation blocks (256/32), 2 per sub-block (elements 0..31 / 32..63).
//
// vec_dot is element-wise (scalar accumulation per lane) rather than dp4a-packed:
// the sequential bit layouts (Q2/Q3/Q6) don't map cleanly onto dp4a's 4-lane
// grouping, so we favour correctness here; a dp4a fast-path can follow.
//
// Bit layouts MUST match ggml-reex-q64.c / reex_q64_kquant_dequant.cuh exactly.
// Must be included AFTER vecdotq.cuh and only under #ifdef GGML_USE_REEX_Q64.
#pragma once

#include "../common.cuh"
#include "reex/ggml-reex-q64-common.h"

// reex_q64_psum_bits_dev (device __constant__ B for fixed-point Psum truncation)
// is defined by mmvq.cu before this header is included — the only TU that uses
// it — so no extern/RDC is needed. B<=0 = off.
//
// Activations are quantized by reex_q64_quantize_q8_1_k256 with ONE scale per
// 256 super-block (CPU q8_K-aligned), so both q8_1 halves of a 64-element
// sub-block share the same scale (ds0.x == ds1.x). Each MMVQ thread owns a full
// sub-block, so sumi0+sumi1 form a single 64-element integer Psum — matching the
// CPU per-64 accumulator — which is what gets truncated before scaling.

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

// All block-64 K-quant MMVQ vec_dots use vdr = 1 (one thread per sub-block).
#define VDR_Q4_K_64_Q8_1_MMVQ 1
#define VDR_Q2_K_64_Q8_1_MMVQ 1
#define VDR_Q3_K_64_Q8_1_MMVQ 1
#define VDR_Q5_K_64_Q8_1_MMVQ 1
#define VDR_Q6_K_64_Q8_1_MMVQ 1
#define VDR_Q5_K_64S_Q8_1_MMVQ 1
#define VDR_Q4_K_64S_Q8_1_MMVQ 1
#define VDR_Q2_K_64S_Q8_1_MMVQ 1

// === Q4_K_64 : 4-bit, w = d*scale*q - dmin*min ============================
static __device__ __forceinline__ float vec_dot_q4_K_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K_64 * bq = (const block_q4_K_64 *) vbq + kbx;
    const int s = iqs; // sub-block 0..3
    int sc, m;
    q64k_get_scale_min(s, bq->scales, sc, m);
    const uint8_t * q = bq->qs + s*32;
    const block_q8_1 * b0 = bq8_1 + 2*s + 0; // elements 0..31
    const block_q8_1 * b1 = bq8_1 + 2*s + 1; // elements 32..63

    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        sumi0 += (q[l] & 0xF) * b0->qs[l];
        sumi1 += (q[l] >>  4) * b1->qs[l];
    }
    const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d    = q64_h2f(bq->d);
    const float  dmin = q64_h2f(bq->dmin);
    const float2 ds0  = __half22float2(b0->ds);
    const float2 ds1  = __half22float2(b1->ds);
    return d*sc*ds0.x*sumi - dmin*m*(ds0.y + ds1.y);
}

// === Q2_K_64 : 2-bit, w = d*scale*q - dmin*min ============================
static __device__ __forceinline__ float vec_dot_q2_K_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_K_64 * bq = (const block_q2_K_64 *) vbq + kbx;
    const int s = iqs;
    const uint8_t scb = bq->scales[s];
    const int sc = scb & 0xF;
    const int m  = scb >> 4;
    const uint8_t * qbase = bq->qs + s*16; // 64 elems * 2 bit / 8 = 16 bytes
    const block_q8_1 * b0 = bq8_1 + 2*s + 0;
    const block_q8_1 * b1 = bq8_1 + 2*s + 1;

    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int e = 0; e < 32; ++e) {
        sumi0 += ((qbase[e >> 2] >> (2*(e & 3))) & 3) * b0->qs[e];
    }
#pragma unroll
    for (int e = 32; e < 64; ++e) {
        sumi1 += ((qbase[e >> 2] >> (2*(e & 3))) & 3) * b1->qs[e - 32];
    }
    const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d    = q64_h2f(bq->d);
    const float  dmin = q64_h2f(bq->dmin);
    const float2 ds0  = __half22float2(b0->ds);
    const float2 ds1  = __half22float2(b1->ds);
    return d*sc*ds0.x*sumi - dmin*m*(ds0.y + ds1.y);
}

// === Q3_K_64 : 3-bit, w = d*scale*q, signed q [-4,3] + signed scale, no min =
static __device__ __forceinline__ float vec_dot_q3_K_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q3_K_64 * bq = (const block_q3_K_64 *) vbq + kbx;
    const int s = iqs;
    const uint8_t * qs = bq->qs;
    const int sc = q64_get_sbits(qs, q64_sub_scale_bit(6, 3, s), 6);
    const block_q8_1 * b0 = bq8_1 + 2*s + 0;
    const block_q8_1 * b1 = bq8_1 + 2*s + 1;

    int64_t bit = q64_code_bit(6, 3, s, 0);
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int e = 0; e < 32; ++e, bit += 3) {
        sumi0 += q64_get_sbits(qs, bit, 3) * b0->qs[e];
    }
#pragma unroll
    for (int e = 32; e < 64; ++e, bit += 3) {
        sumi1 += q64_get_sbits(qs, bit, 3) * b1->qs[e - 32];
    }
    const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d   = q64_h2f((ggml_fp16_t) q64_get_bits(qs, 0, 16));
    const float2 ds0 = __half22float2(b0->ds);
    const float2 ds1 = __half22float2(b1->ds);
    GGML_UNUSED(ds1);
    return d*sc*ds0.x*sumi;
}

// === Q5_K_64 : 5-bit, w = d*scale*q - dmin*min ============================
static __device__ __forceinline__ float vec_dot_q5_K_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K_64 * bq = (const block_q5_K_64 *) vbq + kbx;
    const int s = iqs;
    int sc, m;
    q64k_get_scale_min(s, bq->scales, sc, m);
    const uint8_t * q  = bq->qs + s*32;
    const uint8_t * qh = bq->qh + s*8;
    const block_q8_1 * b0 = bq8_1 + 2*s + 0;
    const block_q8_1 * b1 = bq8_1 + 2*s + 1;

    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        const int hb0 = (qh[l >> 3] >> (l & 7)) & 1;
        sumi0 += ((q[l] & 0xF) | (hb0 << 4)) * b0->qs[l];
        const int e   = l + 32;
        const int hb1 = (qh[e >> 3] >> (e & 7)) & 1;
        sumi1 += ((q[l] >> 4) | (hb1 << 4)) * b1->qs[l];
    }
    const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d    = q64_h2f(bq->d);
    const float  dmin = q64_h2f(bq->dmin);
    const float2 ds0  = __half22float2(b0->ds);
    const float2 ds1  = __half22float2(b1->ds);
    return d*sc*ds0.x*sumi - dmin*m*(ds0.y + ds1.y);
}

// === Q6_K_64 : 6-bit, w = d*scale*q, signed q [-32,31], int8 scale, no min =
static __device__ __forceinline__ float vec_dot_q6_K_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K_64 * bq = (const block_q6_K_64 *) vbq + kbx;
    const int s = iqs;
    const uint8_t * qs = bq->qs;
    const int sc = q64_get_sbits(qs, q64_sub_scale_bit(8, 6, s), 8);
    const block_q8_1 * b0 = bq8_1 + 2*s + 0;
    const block_q8_1 * b1 = bq8_1 + 2*s + 1;

    int64_t bit = q64_code_bit(8, 6, s, 0);
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int e = 0; e < 32; ++e, bit += 6) {
        sumi0 += q64_get_sbits(qs, bit, 6) * b0->qs[e];
    }
#pragma unroll
    for (int e = 32; e < 64; ++e, bit += 6) {
        sumi1 += q64_get_sbits(qs, bit, 6) * b1->qs[e - 32];
    }
    const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d   = q64_h2f((ggml_fp16_t) q64_get_bits(qs, 0, 16));
    const float2 ds0 = __half22float2(b0->ds);
    const float2 ds1 = __half22float2(b1->ds);
    GGML_UNUSED(ds1);
    return d*sc*ds0.x*sumi;
}

// === Q5_K_64S : 5-bit SYMMETRIC, w = d*scale*q, signed q [-16,15], int6 signed scale, no min ===
static __device__ __forceinline__ float vec_dot_q5_K_64S_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K_64S * bq = (const block_q5_K_64S *) vbq + kbx;
    const int s = iqs;
    const uint8_t * qs = bq->qs;
    const int sc = q64_get_sbits(qs, q64_sub_scale_bit(6, 5, s), 6);
    const block_q8_1 * b0 = bq8_1 + 2*s + 0;
    const block_q8_1 * b1 = bq8_1 + 2*s + 1;

    int64_t bit = q64_code_bit(6, 5, s, 0);
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int e = 0; e < 32; ++e, bit += 5) {
        sumi0 += q64_get_sbits(qs, bit, 5) * b0->qs[e];
    }
#pragma unroll
    for (int e = 32; e < 64; ++e, bit += 5) {
        sumi1 += q64_get_sbits(qs, bit, 5) * b1->qs[e - 32];
    }
    const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d   = q64_h2f((ggml_fp16_t) q64_get_bits(qs, 0, 16));
    const float2 ds0 = __half22float2(b0->ds);
    const float2 ds1 = __half22float2(b1->ds);
    GGML_UNUSED(ds1);
    return d*sc*ds0.x*sumi;
}

// === Q4_K_64S : 4-bit SYMMETRIC, w = d*scale*q, signed q [-8,7], int6 signed scale, no min ===
static __device__ __forceinline__ float vec_dot_q4_K_64S_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K_64S * bq = (const block_q4_K_64S *) vbq + kbx;
    const int s = iqs;
    const uint8_t * qs = bq->qs;
    const int sc = q64_get_sbits(qs, q64_sub_scale_bit(6, 4, s), 6);
    const block_q8_1 * b0 = bq8_1 + 2*s + 0;
    const block_q8_1 * b1 = bq8_1 + 2*s + 1;

    int64_t bit = q64_code_bit(6, 4, s, 0);
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int e = 0; e < 32; ++e, bit += 4) {
        sumi0 += q64_get_sbits(qs, bit, 4) * b0->qs[e];
    }
#pragma unroll
    for (int e = 32; e < 64; ++e, bit += 4) {
        sumi1 += q64_get_sbits(qs, bit, 4) * b1->qs[e - 32];
    }
    const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d   = q64_h2f((ggml_fp16_t) q64_get_bits(qs, 0, 16));
    const float2 ds0 = __half22float2(b0->ds);
    const float2 ds1 = __half22float2(b1->ds);
    GGML_UNUSED(ds1);
    return d*sc*ds0.x*sumi;
}

// === Q2_K_64S : 2-bit SYMMETRIC, w = d*scale*q, signed q [-2,1], int4 signed scale, no min ===
static __device__ __forceinline__ float vec_dot_q2_K_64S_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_K_64S * bq = (const block_q2_K_64S *) vbq + kbx;
    const int s = iqs;
    const uint8_t * qs = bq->qs;
    const int sc = q64_get_sbits(qs, q64_sub_scale_bit(4, 2, s), 4);
    const block_q8_1 * b0 = bq8_1 + 2*s + 0;
    const block_q8_1 * b1 = bq8_1 + 2*s + 1;

    int64_t bit = q64_code_bit(4, 2, s, 0);
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int e = 0; e < 32; ++e, bit += 2) {
        sumi0 += q64_get_sbits(qs, bit, 2) * b0->qs[e];
    }
#pragma unroll
    for (int e = 32; e < 64; ++e, bit += 2) {
        sumi1 += q64_get_sbits(qs, bit, 2) * b1->qs[e - 32];
    }
    const int sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d   = q64_h2f((ggml_fp16_t) q64_get_bits(qs, 0, 16));
    const float2 ds0 = __half22float2(b0->ds);
    const float2 ds1 = __half22float2(b1->ds);
    GGML_UNUSED(ds1);
    return d*sc*ds0.x*sumi;
}
