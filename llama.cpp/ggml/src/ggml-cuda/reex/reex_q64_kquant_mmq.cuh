// REEX K-quant block-64 — CUDA MMQ load_tiles (prefill / large-batch path).
//
// Phase 3 of the K-quant block-64 CUDA enablement. Same trick as the legacy
// block-64 MMQ: dequantize the K-quant weights into a standard MMQ tile and
// reuse the existing q8_0 / q8_1 MMA+dp4a vec_dot kernels — no custom vec_dot.
//
//   affine types  (Q2/Q4/Q5_K_64, w = d*sc*q - dmin*mn):
//       store the RAW quant (0..2^b-1) as int8 into the q8_1 tile, plus a
//       half2 (D, M) = (d*sc, -dmin*mn) per sub-block. vec_dot_q8_1_q8_1 then
//       computes  D*d_a*Σ(qw·qa) + M*s_a  (the activation-sum carries the min).
//   symmetric types (Q3/Q6_K_64, w = d*sc*q, q signed two's complement):
//       sign-extend the packed quant into a signed int8 in the q8_0 tile, scale
//       = d*sc. vec_dot_q8_0_q8_1 (D4 layout) needs no activation-sum.
//
// Tile geometry: MMQ iterates K in MMQ_ITER_K(=256) chunks. With qk=256 that is
// blocks_per_iter = 256/256 = 1 super-block per load_tiles call = 4 sub-blocks
// of 64 = 8 q8-style 32-subblocks, exactly the q8_0/q8_1 tile shape. Thread txi
// (0..31): sb = txi/8 (sub-block 0..3), kqsx = txi%8 (int within sub-block).
// Each thread fills the two q8-subblocks 2*sb (elems 0..31) and 2*sb+1 (32..63)
// of its sub-block; both share the sub-block scale. K-element order is natural,
// matching the q8_1 activation tile.
//
// Bit layouts MUST match ggml-reex-q64.c exactly. Must be included by mmq.cuh
// AFTER reex_q64_mmq.cuh (q8_0/q8_1 tile macros, get_int_b2 etc.) and only under
// #ifdef GGML_USE_REEX_Q64.
#pragma once

#include "reex/ggml-reex-q64-common.h"

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

// store a processed int into the q8_0 tile (symmetric) at (q8-subblock s, int j)
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
#define REEX_Q64K_QS(i, s, j)  x_qs[(i)*MMQ_MMA_TILE_X_K_Q8_0 + (s)*QI8_0 + (j)]
#define REEX_Q64K_QS1(i, s, j) x_qs[(i)*MMQ_MMA_TILE_X_K_Q8_1 + (s)*QI8_1 + (j)]
#else
#define REEX_Q64K_QS(i, s, j)  x_qs[(i)*(2*MMQ_TILE_NE_K + 1) + (s)*QI8_0 + (j)]
#define REEX_Q64K_QS1(i, s, j) x_qs[(i)*(2*MMQ_TILE_NE_K + 1) + (s)*QI8_1 + (j)]
#endif

// === scale loaders =========================================================
// affine, scales[6] (4x6-bit sc + 4x6-bit mn): Q4_K_64 / Q5_K_64.
template <int mmq_y, bool need_check, typename block_t>
static __device__ __forceinline__ void reex_q64k_load_dm_km(
        const char * __restrict__ x, half2 * __restrict__ x_dm,
        const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_tile_x_row = 8;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
    const int sb   = kbxd / 2;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) { i = min(i, i_max); }
        const block_t * bxi = (const block_t *) x + kbx0 + i*stride;
        int sc, mn;
        q64k_get_scale_min(sb, bxi->scales, sc, mn);
        const float d    = __half2float(__ushort_as_half((unsigned short) bxi->d));
        const float dmin = __half2float(__ushort_as_half((unsigned short) bxi->dmin));
        const half2 dm = __floats2half2_rn(d*sc, -dmin*mn);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*MMQ_MMA_TILE_X_K_Q8_1           + kbxd] = dm;
#else
        x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + kbxd] = dm;
#endif
    }
}

// affine, scales[4] (4-bit sc | 4-bit mn): Q2_K_64.
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void reex_q64k_load_dm_q2(
        const char * __restrict__ x, half2 * __restrict__ x_dm,
        const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_tile_x_row = 8;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
    const int sb   = kbxd / 2;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) { i = min(i, i_max); }
        const block_q2_K_64 * bxi = (const block_q2_K_64 *) x + kbx0 + i*stride;
        const uint8_t s = bxi->scales[sb];
        const int sc = s & 0xF;
        const int mn = s >> 4;
        const float d    = __half2float(__ushort_as_half((unsigned short) bxi->d));
        const float dmin = __half2float(__ushort_as_half((unsigned short) bxi->dmin));
        const half2 dm = __floats2half2_rn(d*sc, -dmin*mn);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*MMQ_MMA_TILE_X_K_Q8_1           + kbxd] = dm;
#else
        x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + kbxd] = dm;
#endif
    }
}

// symmetric, fp32 scale d*sc into the q8_0 tile. Q3: 6-bit two's-complement
// scale via q64_unpack4x6_s; Q6: int8 scales[].
template <int mmq_y, bool need_check, typename block_t, bool q3>
static __device__ __forceinline__ void reex_q64k_load_df_sym(
        const char * __restrict__ x, float * __restrict__ x_df,
        const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_tile_x_row = 8;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
    const int sb   = kbxd / 2;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) { i = min(i, i_max); }
        const block_t * bxi = (const block_t *) x + kbx0 + i*stride;
        int sc;
        if (q3) { sc = q64_unpack4x6_s(sb, (const uint8_t *) bxi->scales); }
        else    { sc = bxi->scales[sb]; }
        const float d  = __half2float(__ushort_as_half((unsigned short) bxi->d));
        const float df = d * sc;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0                 + kbxd] = df;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + kbxd] = df;
#endif
    }
}

// === Q4_K_64 (affine 4-bit) -> q8_1 tile (raw nibbles + (D,M)) =============
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K_64(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif

    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi  = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int sb   = txi / 8;
    const int kqsx = txi % 8;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) { i = min(i, i_max); }
        const block_q4_K_64 * bxi = (const block_q4_K_64 *) x + kbx0 + i*stride;
        const int vqs = get_int_b2(bxi->qs, sb*8 + kqsx);
        REEX_Q64K_QS1(i, 2*sb + 0, kqsx) = (vqs >> 0) & 0x0F0F0F0F; // elems 0..31
        REEX_Q64K_QS1(i, 2*sb + 1, kqsx) = (vqs >> 4) & 0x0F0F0F0F; // elems 32..63
    }
    reex_q64k_load_dm_km<mmq_y, need_check, block_q4_K_64>(x, x_dm, kbx0, i_max, stride);
}

// === Q5_K_64 (affine 5-bit) -> q8_1 tile (raw nibble+5th bit + (D,M)) ======
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K_64(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif

    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi  = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int sb   = txi / 8;
    const int kqsx = txi % 8;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) { i = min(i, i_max); }
        const block_q5_K_64 * bxi = (const block_q5_K_64 *) x + kbx0 + i*stride;
        const int qhl = get_int_b2(bxi->qh, sb*2 + 0); // 5th bits of sub-block elems 0..31
        const int qhh = get_int_b2(bxi->qh, sb*2 + 1); // 5th bits of sub-block elems 32..63
        const int vqs = get_int_b2(bxi->qs, sb*8 + kqsx);
        REEX_Q64K_QS1(i, 2*sb + 0, kqsx) = reex_q64_mmq_add5((vqs >> 0) & 0x0F0F0F0F, qhl >> (4*kqsx));
        REEX_Q64K_QS1(i, 2*sb + 1, kqsx) = reex_q64_mmq_add5((vqs >> 4) & 0x0F0F0F0F, qhh >> (4*kqsx));
    }
    reex_q64k_load_dm_km<mmq_y, need_check, block_q5_K_64>(x, x_dm, kbx0, i_max, stride);
}

// === Q2_K_64 (affine 2-bit) -> q8_1 tile (raw 2-bit + (D,M)) ===============
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K_64(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif

    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi  = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int sb   = txi / 8;
    const int kqsx = txi % 8;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) { i = min(i, i_max); }
        const block_q2_K_64 * bxi = (const block_q2_K_64 *) x + kbx0 + i*stride;
        const uint8_t b0 = bxi->qs[sb*16 + kqsx];     // elems 0..31  (4 per byte)
        const uint8_t b1 = bxi->qs[sb*16 + 8 + kqsx]; // elems 32..63
        const int v0 = (b0 & 3) | (((b0 >> 2) & 3) << 8) | (((b0 >> 4) & 3) << 16) | (((b0 >> 6) & 3) << 24);
        const int v1 = (b1 & 3) | (((b1 >> 2) & 3) << 8) | (((b1 >> 4) & 3) << 16) | (((b1 >> 6) & 3) << 24);
        REEX_Q64K_QS1(i, 2*sb + 0, kqsx) = v0;
        REEX_Q64K_QS1(i, 2*sb + 1, kqsx) = v1;
    }
    reex_q64k_load_dm_q2<mmq_y, need_check>(x, x_dm, kbx0, i_max, stride);
}

// === Q3_K_64 (symmetric 3-bit, signed q) -> q8_0 tile ======================
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K_64(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif

    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi  = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int sb   = txi / 8;
    const int kqsx = txi % 8;
    const int hsh  = (kqsx & 1) * 4;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) { i = min(i, i_max); }
        const block_q3_K_64 * bxi = (const block_q3_K_64 *) x + kbx0 + i*stride;
        const uint8_t lb0 = bxi->qs[sb*16 + kqsx];
        const uint8_t lb1 = bxi->qs[sb*16 + 8 + kqsx];
        const uint8_t hb0 = bxi->hmask[sb*8 + (kqsx >> 1)];
        const uint8_t hb1 = bxi->hmask[sb*8 + 4 + (kqsx >> 1)];
        int v0 = 0, v1 = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int q3a = ((lb0 >> (2*k)) & 3) | (((hb0 >> (hsh + k)) & 1) << 2);
            const int q3b = ((lb1 >> (2*k)) & 3) | (((hb1 >> (hsh + k)) & 1) << 2);
            v0 |= (((q3a ^ 0x4) - 0x4) & 0xFF) << (8*k); // sign-extend signed 3-bit
            v1 |= (((q3b ^ 0x4) - 0x4) & 0xFF) << (8*k);
        }
        REEX_Q64K_QS(i, 2*sb + 0, kqsx) = v0;
        REEX_Q64K_QS(i, 2*sb + 1, kqsx) = v1;
    }
    reex_q64k_load_df_sym<mmq_y, need_check, block_q3_K_64, true>(x, x_df, kbx0, i_max, stride);
}

// === Q6_K_64 (symmetric 6-bit, signed q) -> q8_0 tile ======================
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K_64(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif

    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi  = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int sb   = txi / 8;
    const int kqsx = txi % 8;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) { i = min(i, i_max); }
        const block_q6_K_64 * bxi = (const block_q6_K_64 *) x + kbx0 + i*stride;
        const uint8_t qhb0 = bxi->qh[sb*16 + kqsx];     // hi 2 bits, elems 0..31
        const uint8_t qhb1 = bxi->qh[sb*16 + 8 + kqsx]; // hi 2 bits, elems 32..63
        int v0 = 0, v1 = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint8_t qla = bxi->ql[sb*32      + kqsx*2 + (k >> 1)];
            const uint8_t qlb = bxi->ql[sb*32 + 16 + kqsx*2 + (k >> 1)];
            const int q6a = ((qla >> (4*(k & 1))) & 0xF) | (((qhb0 >> (2*k)) & 3) << 4);
            const int q6b = ((qlb >> (4*(k & 1))) & 0xF) | (((qhb1 >> (2*k)) & 3) << 4);
            v0 |= (((q6a ^ 0x20) - 0x20) & 0xFF) << (8*k); // sign-extend signed 6-bit
            v1 |= (((q6b ^ 0x20) - 0x20) & 0xFF) << (8*k);
        }
        REEX_Q64K_QS(i, 2*sb + 0, kqsx) = v0;
        REEX_Q64K_QS(i, 2*sb + 1, kqsx) = v1;
    }
    reex_q64k_load_df_sym<mmq_y, need_check, block_q6_K_64, false>(x, x_df, kbx0, i_max, stride);
}

#undef REEX_Q64K_QS
#undef REEX_Q64K_QS1
