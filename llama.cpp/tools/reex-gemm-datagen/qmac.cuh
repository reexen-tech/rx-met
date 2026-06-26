// Shared integer MAC kernels — host+device, byte/numerically faithful mirrors of
// reex vec_dot_* (integer Psum -> truncate -> * scale). Reused by both the GPU
// kernels (gemm.cu) and the CPU integer golden (reference.cpp).
#pragma once

#include "case.h"

#include "ggml.h"
#include "reex/ggml-reex-q64-common.h"   // block_q6_K_64 / q8_0_64 / q5_K_64S, reex_q64_psum_trunc_b

#if defined(__CUDACC__)
    #include <cuda_fp16.h>
#endif

namespace rgd {

// fp16 -> fp32 that works on host and device (intrinsic on device).
RGD_HD inline float rgd_h2f(ggml_fp16_t h) {
#if defined(__CUDA_ARCH__)
    return __half2float(__ushort_as_half((unsigned short) h));
#else
    return ggml_fp16_to_fp32(h);
#endif
}

// Read activation quant element idx from a raw container base: int16 when A_bits>8
// (A16), else int8. Returns the signed integer value.
RGD_HD inline int rgd_act_at(const uint8_t * qbase, int idx, int A_bits) {
    if (A_bits > 8) {
        int16_t v;
#if defined(__CUDA_ARCH__)
        v = ((const int16_t *) qbase)[idx];
#else
        __builtin_memcpy(&v, qbase + (size_t) idx * 2, 2);
#endif
        return (int) v;
    }
    return (int) ((const int8_t *) qbase)[idx];
}

// Activation block accessor: contiguous group { fp16 d ; intX qs[agroup] } at
// act_group_slot(m, kg). Returns the qs container base; writes scale to *d_out.
RGD_HD inline const uint8_t * rgd_act_block(
        const uint8_t * a_base, int64_t m, int64_t kg, int64_t M, int64_t K,
        const TilingSpec & ts, int A_bits, float * d_out) {
    const uint8_t * blk = a_base + (size_t) act_group_slot(m, kg, M, K, ts) * act_block_bytes(ts, A_bits);
    ggml_fp16_t dh;
#if defined(__CUDA_ARCH__)
    dh = *(const ggml_fp16_t *) blk;
#else
    __builtin_memcpy(&dh, blk, sizeof(dh));
#endif
    *d_out = rgd_h2f(dh);
    return blk + 2;
}

// Contribution of one Q6_K_64 super-block (256 K-elems, W6, w=d*scale*q, q signed
// 6-bit two's complement) of output row n to output (m,n). Mirrors vec_dot_q6_K_64,
// summed over 4 subs.
// Activation = single contiguous 256-block (one scale ad, qs[256], sub s -> [s*64,..]).
RGD_HD inline float rgd_q6k64_dot_superblock(
        const block_q6_K_64 & w, const uint8_t * a_base,
        int64_t m, int64_t sb, int64_t M, int64_t K,
        const TilingSpec & ts, int A_bits, int psum_bits) {

    const float d = rgd_h2f(w.d);
    float ad;
    const uint8_t * aq = rgd_act_block(a_base, m, sb, M, K, ts, A_bits, &ad);  // agroup == 256
    float acc = 0.0f;

#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int       sc  = w.scales[s];
        const uint8_t * qlb = w.ql + s * 32;
        const uint8_t * qhb = w.qh + s * 16;
        const int       aoff = s * 64;

        int sumi = 0;
#pragma unroll
        for (int e = 0; e < 64; ++e) {
            const int low4 = (qlb[e >> 1] >> (4 * (e & 1))) & 0xF;
            const int hi2  = (qhb[e >> 2] >> (2 * (e & 3))) & 3;
            const int u6   = low4 | (hi2 << 4);
            sumi += ((u6 ^ 0x20) - 0x20) * rgd_act_at(aq, aoff + e, A_bits); // sign-extend signed 6-bit
        }
        sumi = reex_q64_psum_trunc_b(sumi, psum_bits);
        acc += d * sc * ad * (float) sumi;
    }
    return acc;
}

// Contribution of one Q5_K_64S super-block (256 K-elems, symmetric W5,
// w = d*scale*q, q signed 5-bit + signed 6-bit scale (both two's complement), no
// min). Mirrors vec_dot_q5_K_64S_q8_K.
RGD_HD inline float rgd_q5k64s_dot_superblock(
        const block_q5_K_64S & w, const uint8_t * a_base,
        int64_t m, int64_t sb, int64_t M, int64_t K,
        const TilingSpec & ts, int A_bits, int psum_bits) {

    const float d = rgd_h2f(w.d);
    float ad;
    const uint8_t * aq = rgd_act_block(a_base, m, sb, M, K, ts, A_bits, &ad);  // agroup == 256
    float acc = 0.0f;

#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int       sc  = q64_unpack4x6_s(s, w.scales);  // signed 6-bit two's complement
        const uint8_t * q   = w.qs + s * 32;
        const uint8_t * qh  = w.qh + s * 8;
        const int       aoff = s * 64;

        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int hb0 = (qh[l >> 3] >> (l & 7)) & 1;
            const int u5l = (q[l] & 0xF) | (hb0 << 4);
            sumi += ((u5l ^ 0x10) - 0x10) * rgd_act_at(aq, aoff + l, A_bits); // sign-extend signed 5-bit
            const int e   = l + 32;
            const int hb1 = (qh[e >> 3] >> (e & 7)) & 1;
            const int u5h = (q[l] >> 4) | (hb1 << 4);
            sumi += ((u5h ^ 0x10) - 0x10) * rgd_act_at(aq, aoff + e, A_bits);
        }
        sumi = reex_q64_psum_trunc_b(sumi, psum_bits);
        acc += d * sc * ad * (float) sumi;
    }
    return acc;
}

// Contribution of one Q2_K_64S super-block (256 K-elems, symmetric W2,
// w = d*scale*q, q signed 2-bit + signed 4-bit scale (both two's complement), no
// min). qs 2-bit sequential. Mirrors vec_dot_q2_K_64S_q8_K.
RGD_HD inline float rgd_q2k64s_dot_superblock(
        const block_q2_K_64S & w, const uint8_t * a_base,
        int64_t m, int64_t sb, int64_t M, int64_t K,
        const TilingSpec & ts, int A_bits, int psum_bits) {

    const float d = rgd_h2f(w.d);
    float ad;
    const uint8_t * aq = rgd_act_block(a_base, m, sb, M, K, ts, A_bits, &ad);  // agroup == 256
    float acc = 0.0f;

#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int sc   = q64_unpack4x4_s(s, w.scales);  // signed 4-bit two's complement
        const int aoff = s * 64;

        int sumi = 0;
#pragma unroll
        for (int ii = 0; ii < 64; ++ii) {
            const int e  = s * 64 + ii;
            const int u2 = (w.qs[e >> 2] >> (2 * (e & 3))) & 3;
            sumi += ((u2 ^ 0x2) - 0x2) * rgd_act_at(aq, aoff + ii, A_bits); // sign-extend signed 2-bit
        }
        sumi = reex_q64_psum_trunc_b(sumi, psum_bits);
        acc += d * sc * ad * (float) sumi;
    }
    return acc;
}

// Contribution of one Q3_K_64 super-block (256 K-elems, signed W3, w = d*scale*q,
// q signed 3-bit (low 2 bits in qs, 3rd/sign bit in hmask) + signed 6-bit scale
// (both two's complement), no min). Mirrors vec_dot_q3_K_64_q8_K.
RGD_HD inline float rgd_q3k64_dot_superblock(
        const block_q3_K_64 & w, const uint8_t * a_base,
        int64_t m, int64_t sb, int64_t M, int64_t K,
        const TilingSpec & ts, int A_bits, int psum_bits) {

    const float d = rgd_h2f(w.d);
    float ad;
    const uint8_t * aq = rgd_act_block(a_base, m, sb, M, K, ts, A_bits, &ad);  // agroup == 256
    float acc = 0.0f;

#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int sc   = q64_unpack4x6_s(s, w.scales);  // signed 6-bit two's complement
        const int aoff = s * 64;

        int sumi = 0;
#pragma unroll
        for (int ii = 0; ii < 64; ++ii) {
            const int e    = s * 64 + ii;
            const int low2 = (w.qs[e >> 2] >> (2 * (e & 3))) & 3;
            const int hbit = (w.hmask[e >> 3] >> (e & 7)) & 1;
            const int u3   = low2 | (hbit << 2);
            sumi += ((u3 ^ 0x4) - 0x4) * rgd_act_at(aq, aoff + ii, A_bits); // sign-extend signed 3-bit
        }
        sumi = reex_q64_psum_trunc_b(sumi, psum_bits);
        acc += d * sc * ad * (float) sumi;
    }
    return acc;
}

// Contribution of one Q4_K_64S super-block (256 K-elems, symmetric W4, w = d*scale*q,
// q signed 4-bit + signed 6-bit scale (both two's complement), no min). qs low
// nibble -> elem l, high nibble -> elem l+32 (per 64-sub-block). Mirrors
// vec_dot_q4_K_64S_q8_K.
RGD_HD inline float rgd_q4k64s_dot_superblock(
        const block_q4_K_64S & w, const uint8_t * a_base,
        int64_t m, int64_t sb, int64_t M, int64_t K,
        const TilingSpec & ts, int A_bits, int psum_bits) {

    const float d = rgd_h2f(w.d);
    float ad;
    const uint8_t * aq = rgd_act_block(a_base, m, sb, M, K, ts, A_bits, &ad);  // agroup == 256
    float acc = 0.0f;

#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int       sc   = q64_unpack4x6_s(s, w.scales);  // signed 6-bit two's complement
        const uint8_t * q    = w.qs + s * 32;
        const int       aoff = s * 64;

        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int lo = ((q[l] & 0xF) ^ 0x8) - 0x8;  // elem l   (sign-extend signed 4-bit)
            const int hi = ((q[l] >>   4) ^ 0x8) - 0x8;  // elem l+32
            sumi += lo * rgd_act_at(aq, aoff + l,      A_bits);
            sumi += hi * rgd_act_at(aq, aoff + l + 32, A_bits);
        }
        sumi = reex_q64_psum_trunc_b(sumi, psum_bits);
        acc += d * sc * ad * (float) sumi;
    }
    return acc;
}

// Contribution of one Q8_0_64 block (64 K-elems, symmetric W8) of output row n to
// output (m,n). Activation = contiguous 64-block (one scale ad, qs[64]).
RGD_HD inline float rgd_q8_0_64_dot_block(
        const block_q8_0_64 & w, const uint8_t * a_base,
        int64_t m, int64_t kbw, int64_t M, int64_t K,
        const TilingSpec & ts, int A_bits, int psum_bits) {

    float ad;
    const uint8_t * aq = rgd_act_block(a_base, m, kbw, M, K, ts, A_bits, &ad);  // agroup == 64

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        sumi += w.qs[j] * rgd_act_at(aq, j, A_bits);
    }
    sumi = reex_q64_psum_trunc_b(sumi, psum_bits);
    return rgd_h2f(w.d) * ad * (float) sumi;
}

// Q8_1_64s: same symmetric W8 dot; block carries an extra running-sum field s that
// the symmetric dot ignores (kept only for the HW data format).
RGD_HD inline float rgd_q8_1_64s_dot_block(
        const block_q8_1_64 & w, const uint8_t * a_base,
        int64_t m, int64_t kbw, int64_t M, int64_t K,
        const TilingSpec & ts, int A_bits, int psum_bits) {

    float ad;
    const uint8_t * aq = rgd_act_block(a_base, m, kbw, M, K, ts, A_bits, &ad);  // agroup == 64

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        sumi += w.qs[j] * rgd_act_at(aq, j, A_bits);
    }
    sumi = reex_q64_psum_trunc_b(sumi, psum_bits);
    return rgd_h2f(w.d) * ad * (float) sumi;
}

// Q4_0_64: symmetric W4, w = d*q, q signed 4-bit two's complement, nibble-interleaved
//   (elem e<32 -> qs[e]&0xF, e>=32 -> qs[e-32]>>4).
RGD_HD inline float rgd_q4_0_64_dot_block(
        const block_q4_0_64 & w, const uint8_t * a_base,
        int64_t m, int64_t kbw, int64_t M, int64_t K,
        const TilingSpec & ts, int A_bits, int psum_bits) {

    float ad;
    const uint8_t * aq = rgd_act_block(a_base, m, kbw, M, K, ts, A_bits, &ad);  // agroup == 64

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int w0 = ((w.qs[j] & 0x0F) ^ 0x8) - 0x8;  // elem j   (sign-extend signed 4-bit)
        const int w1 = ((w.qs[j] >>   4) ^ 0x8) - 0x8;  // elem j+32
        sumi += w0 * rgd_act_at(aq, j,      A_bits);
        sumi += w1 * rgd_act_at(aq, j + 32, A_bits);
    }
    sumi = reex_q64_psum_trunc_b(sumi, psum_bits);
    return rgd_h2f(w.d) * ad * (float) sumi;
}

} // namespace rgd
