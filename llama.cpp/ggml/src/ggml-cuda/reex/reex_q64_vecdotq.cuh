// REEX block-64 legacy quant — CUDA MMVQ vec_dot (CPU-aligned matvec path).
//
// Activations are quantized by reex_q64_quantize_q8_1_g64 with ONE scale per
// 64-element block (CPU q8_0_64/q8_1_64-aligned), kept in the block_q8_1 memory
// layout (2 blocks of 32). Because both halves of a 64-block then share the
// same scale (ds0.x == ds1.x), the dot accumulates a single 64-element integer
// Psum — exactly mirroring the CPU scalar reference, including the weight
// zero-point offset folded INTO the integer products (so the fixed-point Psum
// truncation matches CPU bit-for-bit).
//
// To own a full 64-element block in one thread (required for a per-64 Psum) the
// MMVQ vdr is set to qi (= qi/vdr = 1 thread per block). This trades decode
// throughput for CPU/GPU numerical fidelity (the point of the REEX path).
//
// reex_q64_psum_bits_dev / reex_q64_psum_trunc_b: device B for Psum truncation
// (B<=0 = off), defined by mmvq.cu before this header is included.
//
// Must be included only under #ifdef GGML_USE_REEX_Q64.
#pragma once

#include "../common.cuh"
#include "reex/ggml-reex-q64-common.h"
#include "reex_q64_hw_dump.cuh"

// One thread owns the whole 64-element block: vdr = qi.
#define QI4_0_64 8
#define QR4_0_64 2
#define VDR_Q4_0_64_Q8_1_MMVQ 8

#define QI4_1_64 8
#define QR4_1_64 2
#define VDR_Q4_1_64_Q8_1_MMVQ 8

#define QI5_0_64 8
#define QR5_0_64 2
#define VDR_Q5_0_64_Q8_1_MMVQ 8

#define QI5_1_64 8
#define QR5_1_64 2
#define VDR_Q5_1_64_Q8_1_MMVQ 8

#define QI8_0_64 16
#define QR8_0_64 1
#define VDR_Q8_0_64_Q8_1_MMVQ 16

#define QI8_1_64 16
#define QR8_1_64 1
#define VDR_Q8_1_64_Q8_1_MMVQ 16

// d stored as ggml_fp16_t (uint16_t bit pattern) -> reinterpret as half.
static __device__ __forceinline__ float reex_q64_h2f(const ggml_fp16_t h) {
    return __half2float(__ushort_as_half((unsigned short) h));
}

// 8-byte little-endian load of a (possibly unaligned) qh field into uint64_t.
static __device__ __forceinline__ uint64_t reex_q64_load_qh64(const uint8_t * qh) {
    uint64_t v = 0;
#pragma unroll
    for (int t = 0; t < 8; ++t) v |= (uint64_t) qh[t] << (8*t);
    return v;
}

// === Q4_0_64 : symmetric 4-bit, w = d*q, q signed [-8,7] ===================
static __device__ __forceinline__ float vec_dot_q4_0_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    GGML_UNUSED(iqs);
    const block_q4_0_64 * bq = (const block_q4_0_64 *) vbq + kbx;
    const block_q8_1 * b0 = bq8_1 + 0; // elements 0..31  <-> low nibbles
    const block_q8_1 * b1 = bq8_1 + 1; // elements 32..63 <-> high nibbles

    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        sumi0 += (((bq->qs[j] & 0x0F) ^ 0x08) - 0x08) * b0->qs[j]; // sign-extend signed 4-bit
        sumi1 += (((bq->qs[j] >>   4) ^ 0x08) - 0x08) * b1->qs[j];
    }
    const int    sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d    = reex_q64_h2f(bq->d);
    const float2 ds0  = __half22float2(b0->ds); // ds0.x == ds1.x (per-64 scale)
    const float  out  = d * ds0.x * sumi;
    reex_q64_hw_dump_try_record(sumi0 + sumi1, sumi, out);
    return out;
}

// === Q4_1_64 : asymmetric 4-bit, w = d*q + m ===============================
static __device__ __forceinline__ float vec_dot_q4_1_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    GGML_UNUSED(iqs);
    const block_q4_1_64 * bq = (const block_q4_1_64 *) vbq + kbx;
    const block_q8_1 * b0 = bq8_1 + 0;
    const block_q8_1 * b1 = bq8_1 + 1;

    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        sumi0 += (bq->qs[j] & 0x0F) * b0->qs[j];
        sumi1 += (bq->qs[j] >>   4) * b1->qs[j];
    }
    const int    sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d    = reex_q64_h2f(bq->d);
    const float  m    = reex_q64_h2f(bq->m);
    const float2 ds0  = __half22float2(b0->ds);
    const float2 ds1  = __half22float2(b1->ds);
    return d * ds0.x * sumi + m * (ds0.y + ds1.y);
}

// === Q5_0_64 : symmetric 5-bit, w = d*q, q signed [-16,15] =================
static __device__ __forceinline__ float vec_dot_q5_0_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    GGML_UNUSED(iqs);
    const block_q5_0_64 * bq = (const block_q5_0_64 *) vbq + kbx;
    const block_q8_1 * b0 = bq8_1 + 0;
    const block_q8_1 * b1 = bq8_1 + 1;
    const uint64_t qh = reex_q64_load_qh64(bq->qh);

    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 28))     ) & 0x10; // j + qk/2 - 4 = j + 28
        const int u0 = (bq->qs[j] & 0x0F) | xh_0;
        const int u1 = (bq->qs[j] >>   4) | xh_1;
        const int x0 = (u0 ^ 0x10) - 0x10; // sign-extend signed 5-bit
        const int x1 = (u1 ^ 0x10) - 0x10;
        sumi0 += x0 * b0->qs[j];
        sumi1 += x1 * b1->qs[j];
    }
    const int    sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d    = reex_q64_h2f(bq->d);
    const float2 ds0  = __half22float2(b0->ds);
    return d * ds0.x * sumi;
}

// === Q5_1_64 : asymmetric 5-bit, w = d*q + m ===============================
static __device__ __forceinline__ float vec_dot_q5_1_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    GGML_UNUSED(iqs);
    const block_q5_1_64 * bq = (const block_q5_1_64 *) vbq + kbx;
    const block_q8_1 * b0 = bq8_1 + 0;
    const block_q8_1 * b1 = bq8_1 + 1;
    const uint64_t qh = reex_q64_load_qh64(bq->qh);

    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 28))     ) & 0x10;
        const int x0 = (bq->qs[j] & 0x0F) | xh_0;
        const int x1 = (bq->qs[j] >>   4) | xh_1;
        sumi0 += x0 * b0->qs[j];
        sumi1 += x1 * b1->qs[j];
    }
    const int    sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d    = reex_q64_h2f(bq->d);
    const float  m    = reex_q64_h2f(bq->m);
    const float2 ds0  = __half22float2(b0->ds);
    const float2 ds1  = __half22float2(b1->ds);
    return d * ds0.x * sumi + m * (ds0.y + ds1.y);
}

// === Q8_0_64 : symmetric 8-bit, w = d*q ====================================
static __device__ __forceinline__ float vec_dot_q8_0_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    GGML_UNUSED(iqs);
    const block_q8_0_64 * bq = (const block_q8_0_64 *) vbq + kbx;
    const block_q8_1 * b0 = bq8_1 + 0;
    const block_q8_1 * b1 = bq8_1 + 1;

    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        sumi0 += bq->qs[j]      * b0->qs[j];
        sumi1 += bq->qs[j + 32] * b1->qs[j];
    }
    const int    sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d    = reex_q64_h2f(bq->d);
    const float2 ds0  = __half22float2(b0->ds);
    return d * ds0.x * sumi;
}

// === Q8_1_64 (as weight) : symmetric 8-bit; s field unused on weight side ==
static __device__ __forceinline__ float vec_dot_q8_1_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    GGML_UNUSED(iqs);
    const block_q8_1_64 * bq = (const block_q8_1_64 *) vbq + kbx;
    const block_q8_1 * b0 = bq8_1 + 0;
    const block_q8_1 * b1 = bq8_1 + 1;

    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        sumi0 += bq->qs[j]      * b0->qs[j];
        sumi1 += bq->qs[j + 32] * b1->qs[j];
    }
    const int    sumi = reex_q64_psum_trunc_b(sumi0 + sumi1, reex_q64_psum_bits_dev);
    const float  d    = reex_q64_h2f(bq->d);
    const float2 ds0  = __half22float2(b0->ds);
    return d * ds0.x * sumi;
}
