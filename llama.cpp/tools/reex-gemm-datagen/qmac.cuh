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

// 4x6-bit signed-biased scale unpack (mirrors q64k_unpack4x6).
RGD_HD inline int rgd_q64k_unpack4x6(int j, const uint8_t * s) {
    const uint32_t u = (uint32_t) s[0] | ((uint32_t) s[1] << 8) | ((uint32_t) s[2] << 16);
    return (int) ((u >> (6 * j)) & 0x3F);
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

// Contribution of one Q6_K_64 super-block (256 K-elems, W6, w=d*scale*(q-32)) of
// output row n to output (m,n). Mirrors vec_dot_q6_K_64_q8_1, summed over 4 subs.
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
            sumi += ((low4 | (hi2 << 4)) - 32) * rgd_act_at(aq, aoff + e, A_bits);
        }
        sumi = reex_q64_psum_trunc_b(sumi, psum_bits);
        acc += d * sc * ad * (float) sumi;
    }
    return acc;
}

// Contribution of one Q5_K_64S super-block (256 K-elems, symmetric W5,
// w = d*scale*(q5-16), signed 6-bit scale, no min). Mirrors vec_dot_q5_K_64S_q8_1.
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
        const int       sc  = rgd_q64k_unpack4x6(s, w.scales) - 32;
        const uint8_t * q   = w.qs + s * 32;
        const uint8_t * qh  = w.qh + s * 8;
        const int       aoff = s * 64;

        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int hb0 = (qh[l >> 3] >> (l & 7)) & 1;
            sumi += (((q[l] & 0xF) | (hb0 << 4)) - 16) * rgd_act_at(aq, aoff + l, A_bits);
            const int e   = l + 32;
            const int hb1 = (qh[e >> 3] >> (e & 7)) & 1;
            sumi += (((q[l] >> 4) | (hb1 << 4)) - 16) * rgd_act_at(aq, aoff + e, A_bits);
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

// Q4_0_64: symmetric W4, w = d*(q-8), nibble-interleaved
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
        const int w0 = (w.qs[j] & 0x0F) - 8;     // elem j
        const int w1 = (w.qs[j] >> 4)   - 8;     // elem j+32
        sumi += w0 * rgd_act_at(aq, j,      A_bits);
        sumi += w1 * rgd_act_at(aq, j + 32, A_bits);
    }
    sumi = reex_q64_psum_trunc_b(sumi, psum_bits);
    return rgd_h2f(w.d) * ad * (float) sumi;
}

} // namespace rgd
