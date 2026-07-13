/**
 * REEX block-64 legacy quantization — shared block layouts and constants.
 *
 * Hardware alignment requires a 64-element quantization block for the legacy
 * quant family, while upstream legacy types keep block=32. To stay isolated
 * from upstream (and survive future llama.cpp merges) these are NEW ggml types
 * implemented entirely under reex/ — the upstream q4_0/q8_0 path is untouched.
 *
 * Layout mirrors the native legacy blocks, only QK*=64:
 *   block_q4_0_64: fp16 scale + 64 nibbles (32 bytes)
 *   block_q8_0_64: fp16 scale + 64 int8  (64 bytes)
 *
 * This header carries only POD layout + macros, so it is always safe to include
 * (even when GGML_USE_REEX_Q64 is off) for sizeof()/blck_size in the type table.
 */
#pragma once

#include "ggml.h"

#include <stdint.h>

// portable compile-time assert (C11 / C++ / CUDA .cu)
#if defined(__cplusplus)
    #define GGML_Q64_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define GGML_Q64_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
    #define GGML_Q64_STATIC_ASSERT(cond, msg)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define QK4_0_64 64
typedef struct {
    ggml_fp16_t d;             // delta; w = d*q, q signed 4-bit [-8,7] (two's complement)
    uint8_t qs[QK4_0_64 / 2];  // signed nibbles / quants (32 bytes)
} block_q4_0_64;
GGML_Q64_STATIC_ASSERT(sizeof(block_q4_0_64) == sizeof(ggml_fp16_t) + QK4_0_64 / 2,
               "wrong q4_0_64 block size/padding");

#define QK8_0_64 64
typedef struct {
    ggml_fp16_t d;          // delta
    int8_t qs[QK8_0_64];    // quants (64 bytes)
} block_q8_0_64;
GGML_Q64_STATIC_ASSERT(sizeof(block_q8_0_64) == sizeof(ggml_fp16_t) + QK8_0_64,
               "wrong q8_0_64 block size/padding");

// 4-bit asymmetric (delta + min), block=64
#define QK4_1_64 64
typedef struct {
    ggml_fp16_t d;             // delta
    ggml_fp16_t m;             // min
    uint8_t qs[QK4_1_64 / 2];  // nibbles (32 bytes)
} block_q4_1_64;
GGML_Q64_STATIC_ASSERT(sizeof(block_q4_1_64) == 2 * sizeof(ggml_fp16_t) + QK4_1_64 / 2,
               "wrong q4_1_64 block size/padding");

// 5-bit symmetric, block=64. w = d*q, q signed 5-bit [-16,15] (two's complement);
// qh holds the 5th (sign) bit of all 64 quants -> 64 bits.
#define QK5_0_64 64
typedef struct {
    ggml_fp16_t d;             // delta
    uint8_t qh[8];             // 5-th (sign) bit of quants (64 bits)
    uint8_t qs[QK5_0_64 / 2];  // low 4 bits of signed quants (32 bytes)
} block_q5_0_64;
GGML_Q64_STATIC_ASSERT(sizeof(block_q5_0_64) == sizeof(ggml_fp16_t) + 8 + QK5_0_64 / 2,
               "wrong q5_0_64 block size/padding");

// 5-bit asymmetric (delta + min), block=64
#define QK5_1_64 64
typedef struct {
    ggml_fp16_t d;             // delta
    ggml_fp16_t m;             // min
    uint8_t qh[8];             // 5-th bit of quants (64 bits)
    uint8_t qs[QK5_1_64 / 2];  // nibbles (32 bytes)
} block_q5_1_64;
GGML_Q64_STATIC_ASSERT(sizeof(block_q5_1_64) == 2 * sizeof(ggml_fp16_t) + 8 + QK5_1_64 / 2,
               "wrong q5_1_64 block size/padding");

// 8-bit symmetric with running sum (activation companion for Q4_1_64/Q5_1_64),
// block=64. s = d * sum(qs) carries the min-correction term.
#define QK8_1_64 64
typedef struct {
    ggml_fp16_t d;          // delta
    ggml_fp16_t s;          // d * sum(qs[i])
    int8_t qs[QK8_1_64];    // quants (64 bytes)
} block_q8_1_64;
GGML_Q64_STATIC_ASSERT(sizeof(block_q8_1_64) == 2 * sizeof(ggml_fp16_t) + QK8_1_64,
               "wrong q8_1_64 block size/padding");

// ---------------------------------------------------------------------------
// K-quant block-64: super-block = 256, but sub-block (scale granularity) = 64.
// Mirrors block_q4_K's 6-bit scale + 6-bit min semantics, with only 4 sub-blocks
// per super-block instead of 8, so the packed scales shrink from 12 to 6 bytes.
// Activation companion stays the upstream block_q8_K (vec_dot folds 4 bsums).
// ---------------------------------------------------------------------------
#define QK_K_64 256
typedef struct {
    ggml_fp16_t d;            // super-block scale for the quantized scales
    ggml_fp16_t dmin;         // super-block scale for the quantized mins
    uint8_t scales[6];        // 4x 6-bit scale (low 24 bits) + 4x 6-bit min (high 24 bits)
    uint8_t qs[QK_K_64 / 2];  // 4-bit quants (128 bytes)
} block_q4_K_64;
GGML_Q64_STATIC_ASSERT(sizeof(block_q4_K_64) == 2 * sizeof(ggml_fp16_t) + 6 + QK_K_64 / 2,
               "wrong q4_K_64 block size/padding");

// Pack 4 sub-block 6-bit scales/mins into the 6-byte scales[] field.
static inline void q4_K_64_pack_scales(uint8_t * GGML_RESTRICT s,
                                       const uint8_t * GGML_RESTRICT sc, const uint8_t * GGML_RESTRICT mn) {
    const uint32_t us = (uint32_t)(sc[0] & 0x3F)
                      | ((uint32_t)(sc[1] & 0x3F) <<  6)
                      | ((uint32_t)(sc[2] & 0x3F) << 12)
                      | ((uint32_t)(sc[3] & 0x3F) << 18);
    const uint32_t um = (uint32_t)(mn[0] & 0x3F)
                      | ((uint32_t)(mn[1] & 0x3F) <<  6)
                      | ((uint32_t)(mn[2] & 0x3F) << 12)
                      | ((uint32_t)(mn[3] & 0x3F) << 18);
    s[0] = (uint8_t)(us & 0xFF); s[1] = (uint8_t)((us >> 8) & 0xFF); s[2] = (uint8_t)((us >> 16) & 0xFF);
    s[3] = (uint8_t)(um & 0xFF); s[4] = (uint8_t)((um >> 8) & 0xFF); s[5] = (uint8_t)((um >> 16) & 0xFF);
}

// Unpack the j-th (0..3) sub-block 6-bit scale and min.
static inline void get_scale_min_k4_64(int j, const uint8_t * GGML_RESTRICT s,
                                       uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    const uint32_t us = (uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16);
    const uint32_t um = (uint32_t)s[3] | ((uint32_t)s[4] << 8) | ((uint32_t)s[5] << 16);
    *d = (uint8_t)((us >> (6*j)) & 0x3F);
    *m = (uint8_t)((um >> (6*j)) & 0x3F);
}

// Pack 4 sub-block 6-bit values into 3 bytes (used for Q3_K_64 signed-biased scales).
static inline void q64_pack4x6(uint8_t * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT v) {
    const uint32_t u = (uint32_t)(v[0] & 0x3F)
                     | ((uint32_t)(v[1] & 0x3F) <<  6)
                     | ((uint32_t)(v[2] & 0x3F) << 12)
                     | ((uint32_t)(v[3] & 0x3F) << 18);
    s[0] = (uint8_t)(u & 0xFF); s[1] = (uint8_t)((u >> 8) & 0xFF); s[2] = (uint8_t)((u >> 16) & 0xFF);
}

// Unpack the j-th (0..3) 6-bit value packed by q64_pack4x6.
static inline uint8_t q64_unpack4x6(int j, const uint8_t * GGML_RESTRICT s) {
    const uint32_t u = (uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16);
    return (uint8_t)((u >> (6*j)) & 0x3F);
}

// Pack 4 sub-block 4-bit values into 2 bytes (used for Q2_K_64S signed-biased scales).
static inline void q64_pack4x4(uint8_t * GGML_RESTRICT s, const uint8_t * GGML_RESTRICT v) {
    s[0] = (uint8_t)((v[0] & 0xF) | ((v[1] & 0xF) << 4));
    s[1] = (uint8_t)((v[2] & 0xF) | ((v[3] & 0xF) << 4));
}

// Unpack the j-th (0..3) 4-bit value packed by q64_pack4x4.
static inline uint8_t q64_unpack4x4(int j, const uint8_t * GGML_RESTRICT s) {
    return (uint8_t)((s[j >> 1] >> (4*(j & 1))) & 0xF);
}

// Signed-int unpack: interpret the packed 6-bit / 4-bit field as a two's-complement
// signed value (sign-extend), used by the SYMMETRIC K-quant sub-block scales which
// now store a real signed integer (no +bias offset). host + device.
#if defined(__CUDACC__)
__host__ __device__
#endif
static inline int q64_unpack4x6_s(int j, const uint8_t * s) {
    const uint32_t u = (uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16);
    const int v = (int)((u >> (6*j)) & 0x3F);
    return (v ^ 0x20) - 0x20;  // sign-extend 6-bit -> [-32,31]
}
#if defined(__CUDACC__)
__host__ __device__
#endif
static inline int q64_unpack4x4_s(int j, const uint8_t * s) {
    const int v = (int)((s[j >> 1] >> (4*(j & 1))) & 0xF);
    return (v ^ 0x08) - 0x08;  // sign-extend 4-bit -> [-8,7]
}

// ---------------------------------------------------------------------------
// Bit-stream codec for the symmetric (no-min) K-quant block-64 layout.
//
// Q6_K_64, Q3_K_64, Q5_K_64S, Q4_K_64S, Q2_K_64S store their weights as the
// hardware-native continuous LSB-first bit-stream (byte-for-byte identical to
// reex-gemm-datagen's weight_blocks.bin, minus tiling), no byte padding:
//
//   d(fp16, 16b) then per 64-element sub-block s (0..3):
//       sub_scale(scale_bits, signed two's-complement) + 64 x code(W_bits)
//
// scale_bits / W_bits are per-type (see each block struct). sub_scale and code
// cross byte boundaries, so ALL weight (de)serialization goes through these
// three primitives — the single source of truth shared by the reference
// (de)quantizers, CPU/CUDA vec_dot, MMQ, and the offline tools.
//
// Bit offsets within a block (b = the block's byte pointer):
//   glb scale d          : bit 0                                  (16 bits)
//   sub-block s sub_scale: bit 16 + s*(scale_bits + 64*W_bits)    (scale_bits)
//   sub-block s, elem e  : that + scale_bits + e*W_bits           (W_bits)
// ---------------------------------------------------------------------------

// bit offset of sub-block s's sub_scale field.
#if defined(__CUDACC__)
__host__ __device__
#endif
static inline int64_t q64_sub_scale_bit(int scale_bits, int w_bits, int s) {
    return 16 + (int64_t) s * (scale_bits + 64 * w_bits);
}
// bit offset of element e's code within sub-block s.
#if defined(__CUDACC__)
__host__ __device__
#endif
static inline int64_t q64_code_bit(int scale_bits, int w_bits, int s, int e) {
    return q64_sub_scale_bit(scale_bits, w_bits, s) + scale_bits + (int64_t) e * w_bits;
}

// Read `nbits` (<=16) LSB-first starting at bit offset `bit`; touches only the
// bytes the field spans (<=3), never over-reads past the block.
#if defined(__CUDACC__)
__host__ __device__
#endif
static inline uint32_t q64_get_bits(const uint8_t * p, int64_t bit, int nbits) {
    int64_t   byte = bit >> 3;
    const int sh   = (int)(bit & 7);
    const int need = sh + nbits;
    uint32_t  acc  = 0;
    int       got  = 0;
    while (got < need) { acc |= (uint32_t) p[byte++] << got; got += 8; }
    return (acc >> sh) & (nbits >= 32 ? 0xFFFFFFFFu : ((1u << nbits) - 1u));
}

// Same, sign-extending the nbits-wide two's-complement field.
#if defined(__CUDACC__)
__host__ __device__
#endif
static inline int q64_get_sbits(const uint8_t * p, int64_t bit, int nbits) {
    const uint32_t u = q64_get_bits(p, bit, nbits);
    const uint32_t s = 1u << (nbits - 1);
    return (int)((u ^ s) - s);
}

// Write `nbits` of `v` LSB-first at *bit into a pre-zeroed buffer; advances *bit.
static inline void q64_put_bits(uint8_t * p, int64_t * bit, uint32_t v, int nbits) {
    for (int b = 0; b < nbits; ++b, ++(*bit))
        if ((v >> b) & 1u) p[*bit >> 3] |= (uint8_t)(1u << (*bit & 7));
}

// ---------------------------------------------------------------------------
// Q2_K_64: 2-bit, super=256 / sub=64 (4 sub-blocks). x = d*scale*q - dmin*min.
// scales[j]: low nibble = 4-bit scale, high nibble = 4-bit min (mirrors q2_K).
// qs: 2-bit quants, sequential (elem e -> qs[e>>2] >> (2*(e&3))).
// ---------------------------------------------------------------------------
typedef struct {
    ggml_fp16_t d;            // super-block scale for quantized scales
    ggml_fp16_t dmin;         // super-block scale for quantized mins
    uint8_t scales[4];        // 4x (4-bit scale | 4-bit min)
    uint8_t qs[QK_K_64 / 4];  // 2-bit quants (64 bytes)
} block_q2_K_64;
GGML_Q64_STATIC_ASSERT(sizeof(block_q2_K_64) == 2 * sizeof(ggml_fp16_t) + 4 + QK_K_64 / 4,
               "wrong q2_K_64 block size/padding");

// ---------------------------------------------------------------------------
// Q3_K_64: 3-bit symmetric. x = d*scale*q, both q and scale two's-complement.
// HW bit-stream: scale_bits=6, W_bits=3 -> 16 + 4*(6 + 64*3) = 808 bits = 101 B (odd).
// ---------------------------------------------------------------------------
typedef struct {
    uint8_t qs[101];           // HW bit-stream (see q64 bit codec above)
} block_q3_K_64;
GGML_Q64_STATIC_ASSERT(sizeof(block_q3_K_64) == 101, "wrong q3_K_64 block size/padding");

// ---------------------------------------------------------------------------
// Q5_K_64: 5-bit, super=256 / sub=64 (4 sub-blocks). x = d*scale*q - dmin*min.
// qs: low 4 bits sequential (elem e -> qs[e>>1] >> (4*(e&1))); qh: 5th bit sequential.
// scales[6]: same 4x6-bit scale + 4x6-bit min packing as q4_K_64.
// ---------------------------------------------------------------------------
typedef struct {
    ggml_fp16_t d;             // super-block scale for quantized scales
    ggml_fp16_t dmin;          // super-block scale for quantized mins
    uint8_t scales[6];         // 4x 6-bit scale + 4x 6-bit min
    uint8_t qh[QK_K_64/8];     // 5th bit of quants (32 bytes)
    uint8_t qs[QK_K_64/2];     // low 4 bits of quants (128 bytes)
} block_q5_K_64;
GGML_Q64_STATIC_ASSERT(sizeof(block_q5_K_64) == 2 * sizeof(ggml_fp16_t) + 6 + QK_K_64/8 + QK_K_64/2,
               "wrong q5_K_64 block size/padding");

// ---------------------------------------------------------------------------
// Q6_K_64: 6-bit symmetric. x = d*scale*q, q two's-complement [-32,31], scale int8.
// HW bit-stream: scale_bits=8, W_bits=6 -> 16 + 4*(8 + 64*6) = 1584 bits = 198 B (even).
// ---------------------------------------------------------------------------
typedef struct {
    uint8_t qs[198];           // HW bit-stream (see q64 bit codec above)
} block_q6_K_64;
GGML_Q64_STATIC_ASSERT(sizeof(block_q6_K_64) == 198, "wrong q6_K_64 block size/padding");

// ---------------------------------------------------------------------------
// SYMMETRIC K-quant block-64 variants (signed sub-block scale, NO min):
//   x = d * scale * q, with BOTH q and scale stored as two's-complement SIGNED
//   integers (no zero-point / mid offset — hardware-friendly).
//   - Q2_K_64S: 2-bit, q signed [-2,1],  scale int4 signed [-8,7]
//   - Q4_K_64S: 4-bit, q signed [-8,7],  scale int6 signed [-32,31]
//   - Q5_K_64S: 5-bit, q signed [-16,15], scale int6 signed [-32,31]
// Sub-block scale bit-widths match the asymmetric counterparts (Q2_K:4, Q4/Q5_K:6).
// ---------------------------------------------------------------------------

// Q2_K_64S: 2-bit symmetric.
// HW bit-stream: scale_bits=4, W_bits=2 -> 16 + 4*(4 + 64*2) = 544 bits = 68 B (even).
typedef struct {
    uint8_t qs[68];            // HW bit-stream (see q64 bit codec above)
} block_q2_K_64S;
GGML_Q64_STATIC_ASSERT(sizeof(block_q2_K_64S) == 68, "wrong q2_K_64S block size/padding");

// Q4_K_64S: 4-bit symmetric.
// HW bit-stream: scale_bits=6, W_bits=4 -> 16 + 4*(6 + 64*4) = 1064 bits = 133 B (odd).
typedef struct {
    uint8_t qs[133];           // HW bit-stream (see q64 bit codec above)
} block_q4_K_64S;
GGML_Q64_STATIC_ASSERT(sizeof(block_q4_K_64S) == 133, "wrong q4_K_64S block size/padding");

// Q5_K_64S: 5-bit symmetric.
// HW bit-stream: scale_bits=6, W_bits=5 -> 16 + 4*(6 + 64*5) = 1320 bits = 165 B (odd).
typedef struct {
    uint8_t qs[165];           // HW bit-stream (see q64 bit codec above)
} block_q5_K_64S;
GGML_Q64_STATIC_ASSERT(sizeof(block_q5_K_64S) == 165, "wrong q5_K_64S block size/padding");

// ---------------------------------------------------------------------------
// Fixed-point Psum precision truncation (limited integer bit-width modeling).
//
// In the quantized dot product each scale-group produces an integer Psum (the
// q_w * q_a accumulator) that is later multiplied by the input/weight scales to
// become a float. This helper limits that Psum to B effective magnitude bits:
//   - B <= 0  : disabled, returns the Psum unchanged (default).
//   - |Psum| fits in B bits : unchanged.
//   - otherwise: drop just enough LSBs (s = nbits - B) to fit into B bits.
// The dropped shift s is a pure power of two and folds into the float dequant
// scale as an exponent, so clearing the low s bits in place is numerically
// identical to keeping a B-bit mantissa plus a 2^s factor. Sign is preserved;
// truncation is toward zero (magnitude domain). Pure integer; host + device.
// ---------------------------------------------------------------------------
#if defined(__CUDACC__)
__host__ __device__
#endif
static inline int32_t reex_q64_psum_trunc_b(int32_t psum, int B) {
    if (B <= 0 || psum == 0) {
        return psum;
    }
    const int neg = psum < 0;
    uint32_t mag = neg ? (uint32_t)(-(int64_t)psum) : (uint32_t)psum;
#if defined(__CUDA_ARCH__)
    const int nbits = 32 - __clz((int)mag);
#elif defined(__GNUC__) || defined(__clang__)
    const int nbits = 32 - __builtin_clz(mag);
#else
    int nbits = 0; { uint32_t t = mag; while (t) { ++nbits; t >>= 1; } }
#endif
    if (nbits <= B) {
        return psum;
    }
    const int s = nbits - B;
    mag = (mag >> s) << s;
    return neg ? -(int32_t)mag : (int32_t)mag;
}

#ifdef __cplusplus
}
#endif
