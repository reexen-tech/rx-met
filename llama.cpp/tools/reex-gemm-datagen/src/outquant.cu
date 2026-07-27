#include "outquant.cuh"
#include "convert.cuh"   // rgd_f32_to_f16_bits, rgd_rne_lround, rgd_store16

#include "reex/ggml-reex-q64-common.h"   // block_q8_0_64 / block_q4_0_64 sizes

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rgd {

#define OQ_CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[rgd] CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    abort(); } } while (0)

int kblock_bytes(int kbits) {
    return kbits == 4 ? (int) sizeof(block_q4_0_64) : (int) sizeof(block_q8_0_64);
}

// Quantize ONE output group (row m1, hd-group g = output cols [g*64, g*64+64))
// of the fp32 accumulator C (result tile order) into a Legacy block at dst.
// Same math as the stage-1 act quant kernel (actquant.cu), single fp32 ops only
// (no FMA), so device and host execute identically -> bit-exact cross-check.
RGD_HD inline void rgd_out_quant_group(const float * C, int64_t m1, int64_t g,
                                       int64_t M, const TilingSpec & ts,
                                       int kbits, uint8_t * dst) {
    const int     qmax = (1 << (kbits - 1)) - 1;   // INT8 -> 127, INT4 -> 7
    const int64_t n0   = g * 64;

    float amax = 0.0f;
    for (int e = 0; e < 64; ++e) {
        const float v = C[result_tiled_index(m1, n0 + e, M, ts)];
        amax = fmaxf(amax, fabsf(v));
    }
    const float scale = amax > 0.0f ? amax / (float) qmax : 1.0f;
    const float inv   = amax > 0.0f ? (float) qmax / amax : 0.0f;

    rgd_store16(dst, rgd_f32_to_f16_bits(scale));   // fp16 group scale (d)
    uint8_t * qb = dst + 2;

    if (kbits == 8) {
        for (int e = 0; e < 64; ++e) {
            long q = rgd_rne_lround(C[result_tiled_index(m1, n0 + e, M, ts)] * inv);
            if (q >  qmax) q =  qmax;
            if (q < -qmax) q = -qmax;
            qb[e] = (uint8_t) (int8_t) q;
        }
    } else {  // INT4: signed nibble, e<32 -> low nibble of qs[e], e>=32 -> high of qs[e-32]
        for (int j = 0; j < 32; ++j) qb[j] = 0;
        for (int e = 0; e < 64; ++e) {
            long q = rgd_rne_lround(C[result_tiled_index(m1, n0 + e, M, ts)] * inv);
            if (q >  qmax) q =  qmax;
            if (q < -qmax) q = -qmax;
            const uint8_t c4 = (uint8_t) ((unsigned) q & 0xF);   // two's complement low 4 bits
            if (e < 32) qb[e]      = (uint8_t) (qb[e]      | c4);
            else        qb[e - 32] = (uint8_t) (qb[e - 32] | (uint8_t) (c4 << 4));
        }
    }
}

// One thread per (m1, hd-group).
__global__ void kernel_out_quant(const float * __restrict__ C, uint8_t * __restrict__ kb,
                                 int64_t M, int64_t N, TilingSpec ts, int kbits,
                                 int block_bytes) {
    const int64_t ng = N / 64;
    const int64_t g  = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m  = (int64_t) blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || g >= ng) return;
    rgd_out_quant_group(C, m, g, M, ts, kbits,
                        kb + (size_t) (m * ng + g) * block_bytes);
}

void out_quant_run_host(const float * C_tiled, int64_t M, int64_t N,
                        TilingSpec ts, int kbits,
                        std::vector<uint8_t> & k_blocks) {
    const int     bb = kblock_bytes(kbits);
    const int64_t ng = N / 64;
    k_blocks.assign((size_t) (M * ng) * bb, 0);

    const size_t c_bytes = (size_t) (M * N) * sizeof(float);
    float   * d_C  = nullptr;
    uint8_t * d_kb = nullptr;
    OQ_CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    OQ_CUDA_CHECK(cudaMalloc(&d_kb, k_blocks.size()));
    OQ_CUDA_CHECK(cudaMemcpy(d_C, C_tiled, c_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((unsigned) ((ng + block.x - 1) / block.x),
              (unsigned) ((M  + block.y - 1) / block.y));
    kernel_out_quant<<<grid, block>>>(d_C, d_kb, M, N, ts, kbits, bb);
    OQ_CUDA_CHECK(cudaGetLastError());
    OQ_CUDA_CHECK(cudaDeviceSynchronize());
    OQ_CUDA_CHECK(cudaMemcpy(k_blocks.data(), d_kb, k_blocks.size(), cudaMemcpyDeviceToHost));

    cudaFree(d_C);
    cudaFree(d_kb);
}

void out_quant_cpu(const float * C_tiled, int64_t M, int64_t N,
                   const TilingSpec & ts, int kbits,
                   std::vector<uint8_t> & k_blocks) {
    const int     bb = kblock_bytes(kbits);
    const int64_t ng = N / 64;
    k_blocks.assign((size_t) (M * ng) * bb, 0);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t g = 0; g < ng; ++g)
            rgd_out_quant_group(C_tiled, m, g, M, ts, kbits,
                                k_blocks.data() + (size_t) (m * ng + g) * bb);
}

void kblocks_grid_transpose(const std::vector<uint8_t> & pre,
                            std::vector<uint8_t> & post,
                            int64_t num_keys, int64_t n_hd_groups,
                            int block_bytes) {
    post.assign(pre.size(), 0);
    for (int64_t key = 0; key < num_keys; ++key)
        for (int64_t g = 0; g < n_hd_groups; ++g)
            std::memcpy(post.data() + (size_t) (g * num_keys + key) * block_bytes,
                        pre.data()  + (size_t) (key * n_hd_groups + g) * block_bytes,
                        (size_t) block_bytes);
}

} // namespace rgd
