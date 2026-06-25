// REEX block-64 legacy quant — CUDA MMQ load_tiles (prefill / large-batch path).
//
// Strategy (same trick Q5_0/Q1_0 already use): dequantize the block-64 weights
// into the q8_0 MMQ shared-memory tile (signed int8 quants + one fp32 scale per
// 32-element subblock), then reuse the q8_0 MMA/dp4a vec_dot (D4 activation
// layout, no activation-sum needed because the signed quants sign-extend directly
// into int8, no zero-point). The asymmetric types (Q4_1_64/Q5_1_64) instead store
// RAW nibbles + a half2 (d,m) scale into the q8_1 tile and reuse the
// vec_dot_q8_1_q8_1_{mma,dp4a} kernels (the activation-sum term carries the min).
//
// Tile geometry: the MMQ kernel iterates the K dimension in MMQ_ITER_K(=256)
// chunks. With qk=64 that is blocks_per_iter = 256/64 = 4 block-64 blocks per
// load_tiles call = 8 q8_0-style 32-subblocks, exactly the q8_0 tile shape.
// subblock s (0..7) <-> block-64 block b = s/2, half h = s%2; both halves of a
// block share the single fp16 scale. K-element order is natural (subblock s
// holds elements [s*32 .. s*32+31]), matching the q8_1 activation tile.
//
// Must be included by mmq.cuh AFTER the q8_0 tile macros / get_int_b2 etc., and
// only under #ifdef GGML_USE_REEX_Q64.
#pragma once

#include "reex/ggml-reex-q64-common.h"

// ggml_fp16_t (uint16_t bits) -> float (reinterpret bits as half, not int cast).
static __device__ __forceinline__ float reex_q64_mmq_h2f(const ggml_fp16_t h) {
    return __half2float(__ushort_as_half((unsigned short) h));
}

// Insert four 5th-bits (bits 0..3 of vh) into bit 4 of each packed byte.
static __device__ __forceinline__ int reex_q64_mmq_add5(int v_nibbles, int vh) {
    v_nibbles |= (vh <<  4) & 0x00000010;
    v_nibbles |= (vh << 11) & 0x00001000;
    v_nibbles |= (vh << 18) & 0x00100000;
    v_nibbles |= (vh << 25) & 0x10000000;
    return v_nibbles;
}

// Common scale loader: writes the (shared) fp32 scale of block-64 block (kbxd/2)
// into both 32-subblock scale slots of the q8_0 tile.
template <int mmq_y, bool need_check, typename block_t>
static __device__ __forceinline__ void reex_q64_mmq_load_scales(
        const char * __restrict__ x, float * __restrict__ x_df,
        const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr int blocks_per_tile_x_row = 8; // 8 32-subblocks per K-tile
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_t * bxi = (const block_t *) x + kbx0 + i*stride + kbxd/2;
        const float d = reex_q64_mmq_h2f(bxi->d);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0                 + kbxd] = d;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + kbxd] = d;
#endif
    }
}

// store a processed int into the q8_0 tile at (subblock s, int j)
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
#define REEX_Q64_QS(i, s, j) x_qs[(i)*MMQ_MMA_TILE_X_K_Q8_0 + (s)*QI8_0 + (j)]
#else
#define REEX_Q64_QS(i, s, j) x_qs[(i)*(2*MMQ_TILE_NE_K + 1) + (s)*QI8_0 + (j)]
#endif

// q8_1 tile variant (asymmetric types): same int positions, q8_1 MMA stride.
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
#define REEX_Q64_QS1(i, s, j) x_qs[(i)*MMQ_MMA_TILE_X_K_Q8_1 + (s)*QI8_1 + (j)]
#else
#define REEX_Q64_QS1(i, s, j) x_qs[(i)*(2*MMQ_TILE_NE_K + 1) + (s)*QI8_1 + (j)]
#endif

// Asymmetric scale loader: pack (d,m) -> half2 (low=d, high=m), shared by both
// 32-subblocks of a block-64 block, into the q8_1 tile's x_dm.
template <int mmq_y, bool need_check, typename block_t>
static __device__ __forceinline__ void reex_q64_mmq_load_dm(
        const char * __restrict__ x, half2 * __restrict__ x_dm,
        const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr int blocks_per_tile_x_row = 8;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_t * bxi = (const block_t *) x + kbx0 + i*stride + kbxd/2;
        const half2 dm = __halves2half2(__ushort_as_half((unsigned short) bxi->d),
                                        __ushort_as_half((unsigned short) bxi->m));
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*MMQ_MMA_TILE_X_K_Q8_1           + kbxd] = dm;
#else
        x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + kbxd] = dm;
#endif
    }
}

// === Q4_0_64 ===============================================================
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0_64(
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
    const int kbx  = txi / 8; // block-64 block within K-tile (0..3)
    const int kqsx = txi % 8; // qs int within block (0..7)

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_0_64 * bxi = (const block_q4_0_64 *) x + kbx0 + i*stride + kbx;
        const int vqs = get_int_b2(bxi->qs, kqsx);
        // signed 4-bit two's complement -> sign-extend per byte: (n^8)-8
        const int qlo = __vsubss4(((vqs >> 0) & 0x0F0F0F0F) ^ 0x08080808, 0x08080808);
        const int qhi = __vsubss4(((vqs >> 4) & 0x0F0F0F0F) ^ 0x08080808, 0x08080808);
        REEX_Q64_QS(i, 2*kbx + 0, kqsx) = qlo;
        REEX_Q64_QS(i, 2*kbx + 1, kqsx) = qhi;
    }

    reex_q64_mmq_load_scales<mmq_y, need_check, block_q4_0_64>(x, x_df, kbx0, i_max, stride);
}

// === Q5_0_64 ===============================================================
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0_64(
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
    const int kbx  = txi / 8;
    const int kqsx = txi % 8;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q5_0_64 * bxi = (const block_q5_0_64 *) x + kbx0 + i*stride + kbx;
        const int qhl = get_int_b2(bxi->qh, 0); // 5th bits of elements 0..31
        const int qhh = get_int_b2(bxi->qh, 1); // 5th bits of elements 32..63
        const int vqs = get_int_b2(bxi->qs, kqsx);
        int qlo = reex_q64_mmq_add5((vqs >> 0) & 0x0F0F0F0F, qhl >> (4*kqsx));
        int qhi = reex_q64_mmq_add5((vqs >> 4) & 0x0F0F0F0F, qhh >> (4*kqsx));
        // signed 5-bit two's complement -> sign-extend per byte: (n^0x10)-0x10
        qlo = __vsubss4(qlo ^ 0x10101010, 0x10101010);
        qhi = __vsubss4(qhi ^ 0x10101010, 0x10101010);
        REEX_Q64_QS(i, 2*kbx + 0, kqsx) = qlo;
        REEX_Q64_QS(i, 2*kbx + 1, kqsx) = qhi;
    }

    reex_q64_mmq_load_scales<mmq_y, need_check, block_q5_0_64>(x, x_df, kbx0, i_max, stride);
}

// === Q8_0_64 / Q8_1_64 (weight side, symmetric) ============================
// qs holds 64 int8 = 16 ints; subblock 2b uses ints 0..7, subblock 2b+1 uses
// ints 8..15. No offset (already signed). Q8_1_64's trailing s field is unused.
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0_64(
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
    const int kbx  = txi / 8;
    const int kqsx = txi % 8;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q8_0_64 * bxi = (const block_q8_0_64 *) x + kbx0 + i*stride + kbx;
        REEX_Q64_QS(i, 2*kbx + 0, kqsx) = get_int_b2(bxi->qs, kqsx + 0);
        REEX_Q64_QS(i, 2*kbx + 1, kqsx) = get_int_b2(bxi->qs, kqsx + 8);
    }

    reex_q64_mmq_load_scales<mmq_y, need_check, block_q8_0_64>(x, x_df, kbx0, i_max, stride);
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q8_1_64(
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
    const int kbx  = txi / 8;
    const int kqsx = txi % 8;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q8_1_64 * bxi = (const block_q8_1_64 *) x + kbx0 + i*stride + kbx;
        REEX_Q64_QS(i, 2*kbx + 0, kqsx) = get_int_b4(bxi->qs, kqsx + 0);
        REEX_Q64_QS(i, 2*kbx + 1, kqsx) = get_int_b4(bxi->qs, kqsx + 8);
    }

    reex_q64_mmq_load_scales<mmq_y, need_check, block_q8_1_64>(x, x_df, kbx0, i_max, stride);
}

// === Q4_1_64 (asymmetric 4-bit, w = d*q + m) ===============================
// Mirrors load_tiles_q5_1: store RAW nibbles (0..15, no offset) as int8 into the
// q8_1 MMA tile, plus a half2 (d,m) scale shared by both 32-subblocks. Reuses
// vec_dot_q8_1_q8_1_{mma,dp4a} (the activation-sum term carries the min).
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1_64(
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
    const int kbx  = txi / 8;
    const int kqsx = txi % 8;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_1_64 * bxi = (const block_q4_1_64 *) x + kbx0 + i*stride + kbx;
        const int vqs = get_int_b4(bxi->qs, kqsx);
        REEX_Q64_QS1(i, 2*kbx + 0, kqsx) = (vqs >> 0) & 0x0F0F0F0F;
        REEX_Q64_QS1(i, 2*kbx + 1, kqsx) = (vqs >> 4) & 0x0F0F0F0F;
    }

    reex_q64_mmq_load_dm<mmq_y, need_check, block_q4_1_64>(x, x_dm, kbx0, i_max, stride);
}

// === Q5_1_64 (asymmetric 5-bit, w = d*q + m) ===============================
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1_64(
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
    const int kbx  = txi / 8;
    const int kqsx = txi % 8;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q5_1_64 * bxi = (const block_q5_1_64 *) x + kbx0 + i*stride + kbx;
        const int qhl = get_int_b2(bxi->qh, 0); // 5th bits of elements 0..31
        const int qhh = get_int_b2(bxi->qh, 1); // 5th bits of elements 32..63
        const int vqs = get_int_b2(bxi->qs, kqsx);
        REEX_Q64_QS1(i, 2*kbx + 0, kqsx) = reex_q64_mmq_add5((vqs >> 0) & 0x0F0F0F0F, qhl >> (4*kqsx));
        REEX_Q64_QS1(i, 2*kbx + 1, kqsx) = reex_q64_mmq_add5((vqs >> 4) & 0x0F0F0F0F, qhh >> (4*kqsx));
    }

    reex_q64_mmq_load_dm<mmq_y, need_check, block_q5_1_64>(x, x_dm, kbx0, i_max, stride);
}

#undef REEX_Q64_QS
#undef REEX_Q64_QS1
