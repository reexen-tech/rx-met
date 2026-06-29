#include "actquant.cuh"
#include "convert.cuh"   // rgd_f32_to_f16_bits, rgd_rne_lround

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace rgd {

#define AQ_CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[rgd] CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    abort(); } } while (0)

// One thread per (m, kg) activation group.
__global__ void kernel_act_quant(const float * __restrict__ A, uint8_t * __restrict__ ab,
                                 int64_t M, int64_t K, TilingSpec ts, int A_bits) {
    const int64_t ngK = K / ts.agroup;
    const int64_t kg  = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m   = (int64_t) blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || kg >= ngK) return;

    const int agroup = ts.agroup;
    const int ebytes = act_elem_bytes(A_bits);          // 1 (A<=8) or 2 (A16)
    const int stride = act_block_bytes(ts, A_bits);
    const int qmax   = (1 << (A_bits - 1)) - 1;          // A8->127, A16->32767, A4->7

    const float * row = A + (size_t) (m * K + kg * agroup);
    float amax = 0.0f;
    for (int e = 0; e < agroup; ++e) amax = fmaxf(amax, fabsf(row[e]));

    const float scale = amax > 0.0f ? amax / (float) qmax : 1.0f;
    const float inv   = amax > 0.0f ? (float) qmax / amax : 0.0f;

    uint8_t * blk = ab + (size_t) act_group_slot(m, kg, M, K, ts) * stride;
    rgd_store16(blk, rgd_f32_to_f16_bits(scale));        // fp16 group scale
    uint8_t * qb = blk + 2;
    for (int e = 0; e < agroup; ++e) {
        long q = rgd_rne_lround(row[e] * inv);          // round-half-to-even
        if (q >  qmax) q =  qmax;
        if (q < -qmax) q = -qmax;                        // symmetric clamp
        if (ebytes == 2) { int16_t v16 = (int16_t) q; qb[(size_t) e * 2] = (uint8_t) (v16 & 0xff); qb[(size_t) e * 2 + 1] = (uint8_t) ((v16 >> 8) & 0xff); }
        else             { qb[e] = (uint8_t) (int8_t) q; }
    }
}

void act_quant_run_host(const float * A, int64_t M, int64_t K,
                        TilingSpec ts, int A_bits,
                        std::vector<uint8_t> & a_blocks) {
    const int     stride = act_block_bytes(ts, A_bits);
    const int64_t ngrp   = M * K / ts.agroup;
    a_blocks.assign((size_t) ngrp * stride, 0);

    const size_t a_bytes  = (size_t) (M * K) * sizeof(float);
    const size_t ab_bytes = a_blocks.size();

    float   * d_A  = nullptr;
    uint8_t * d_ab = nullptr;
    AQ_CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    AQ_CUDA_CHECK(cudaMalloc(&d_ab, ab_bytes));
    AQ_CUDA_CHECK(cudaMemcpy(d_A, A, a_bytes, cudaMemcpyHostToDevice));

    const int64_t ngK = K / ts.agroup;
    dim3 block(16, 16);
    dim3 grid((unsigned) ((ngK + block.x - 1) / block.x),
              (unsigned) ((M   + block.y - 1) / block.y));
    kernel_act_quant<<<grid, block>>>(d_A, d_ab, M, K, ts, A_bits);
    AQ_CUDA_CHECK(cudaGetLastError());
    AQ_CUDA_CHECK(cudaDeviceSynchronize());
    AQ_CUDA_CHECK(cudaMemcpy(a_blocks.data(), d_ab, ab_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_ab);
}

} // namespace rgd
