#include "gemm.cuh"
#include "qmac.cuh"
#include "wquant.h"

#include "reex/ggml-reex-q64-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace rgd {

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[rgd] CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    abort(); } } while (0)

// K-quant Q6_K_64 (W6): one block_q6_K_64 per super-block (256).
__global__ void kernel_gemm_q6k64(
    const block_q6_K_64 * __restrict__ w_blocks,
    const uint8_t * __restrict__ a_blocks,
    float * __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    TilingSpec ts, int psum_bits) {

    const int64_t n = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m = (int64_t) blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    const int64_t sb_per_row = K / QK_K_64;
    float acc = 0.0f;
    for (int64_t sb = 0; sb < sb_per_row; ++sb) {
        const block_q6_K_64 & w = w_blocks[weight_block_slot(n, sb, N, ts)];
        acc += rgd_q6k64_dot_superblock(w, a_blocks, m, sb, M, K, ts, psum_bits);
    }
    C[result_tiled_index(m, n, M, ts)] = acc;
}

// K-quant Q5_K_64S (symmetric W5): one block_q5_K_64S per super-block (256).
__global__ void kernel_gemm_q5k64s(
    const block_q5_K_64S * __restrict__ w_blocks,
    const uint8_t * __restrict__ a_blocks,
    float * __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    TilingSpec ts, int psum_bits) {

    const int64_t n = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m = (int64_t) blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    const int64_t sb_per_row = K / QK_K_64;
    float acc = 0.0f;
    for (int64_t sb = 0; sb < sb_per_row; ++sb) {
        const block_q5_K_64S & w = w_blocks[weight_block_slot(n, sb, N, ts)];
        acc += rgd_q5k64s_dot_superblock(w, a_blocks, m, sb, M, K, ts, psum_bits);
    }
    C[result_tiled_index(m, n, M, ts)] = acc;
}

// Legacy q8_0_64 (symmetric W8): one block_q8_0_64 per K-block (64 elements).
__global__ void kernel_gemm_q8_0_64(
    const block_q8_0_64 * __restrict__ w_blocks,
    const uint8_t * __restrict__ a_blocks,
    float * __restrict__ C,
    int64_t M, int64_t N, int64_t K,
    TilingSpec ts, int psum_bits) {

    const int64_t n = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m = (int64_t) blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    const int64_t kb_per_row = K / 64;
    float acc = 0.0f;
    for (int64_t kbw = 0; kbw < kb_per_row; ++kbw) {
        const block_q8_0_64 & w = w_blocks[weight_block_slot(n, kbw, N, ts)];
        acc += rgd_q8_0_64_dot_block(w, a_blocks, m, kbw, M, K, ts, psum_bits);
    }
    C[result_tiled_index(m, n, M, ts)] = acc;
}

void gemm_run_host(
    int wtype_id,
    const void * w_blocks,
    const uint8_t * a_blocks, size_t a_bytes,
    float * C_out,
    int64_t M, int64_t N, int64_t K,
    TilingSpec ts, int psum_bits) {

    const WQuantType & wt = wquant_get(wtype_id);
    const size_t w_bytes = wquant_blocks_bytes(wtype_id, N, K);
    const size_t c_bytes = (size_t) (M * N) * sizeof(float);

    void    * d_w = nullptr;
    uint8_t * d_a = nullptr;
    float   * d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_w, w_blocks, w_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, a_blocks, a_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((unsigned) ((N + block.x - 1) / block.x),
              (unsigned) ((M + block.y - 1) / block.y));

    if (wt.family == Family::Kquant && strcmp(wt.name, "Q6_K_64") == 0) {
        kernel_gemm_q6k64<<<grid, block>>>(
            (const block_q6_K_64 *) d_w, d_a, d_C, M, N, K, ts, psum_bits);
    } else if (wt.family == Family::Kquant && strcmp(wt.name, "Q5_K_64S") == 0) {
        kernel_gemm_q5k64s<<<grid, block>>>(
            (const block_q5_K_64S *) d_w, d_a, d_C, M, N, K, ts, psum_bits);
    } else if (wt.family == Family::Legacy) {
        kernel_gemm_q8_0_64<<<grid, block>>>(
            (const block_q8_0_64 *) d_w, d_a, d_C, M, N, K, ts, psum_bits);
    } else {
        fprintf(stderr, "[rgd] gemm_run_host: unsupported wtype %s\n", wt.name);
        abort();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C_out, d_C, c_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_w);
    cudaFree(d_a);
    cudaFree(d_C);
}

} // namespace rgd
