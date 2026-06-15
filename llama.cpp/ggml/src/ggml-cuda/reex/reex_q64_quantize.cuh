// REEX block-64 — CPU-aligned activation quantization for CUDA MMVQ.
//
// The standard MMVQ pipeline quantizes activations to block_q8_1 with ONE scale
// per 32 elements (QK8_1), so the GPU integer Psum is limited to 32 elements.
// The CPU paths instead use a coarser activation scale:
//   - legacy : block_q8_0_64 / block_q8_1_64 -> ONE scale per 64 elements
//   - K-quant: block_q8_K                     -> ONE scale per 256 super-block
// To mirror the CPU 64-element integer Psum (and make the fixed-point Psum
// truncation consistent across CPU/GPU) we quantize activations with a single
// scale shared over GROUP elements (64 or 256) while KEEPING the block_q8_1
// memory layout, so the rest of the MMVQ framework (strides, kernel) is
// unchanged. All q8_1 sub-blocks of a GROUP then share the same scale, letting
// the vec_dot add their partial sums into one GROUP-wide integer Psum.
//
// ds.y stores d*sum(quantized) per 32-block, matching the CPU min-correction
// term (q8_1_64.s / q8_K bsums*y.d).
//
// Only included by mmvq.cu, under #ifdef GGML_USE_REEX_Q64.
#pragma once

#include "../common.cuh"

// One CUDA block (GROUP threads) quantizes one GROUP-element group of one
// activation row, writing into the flat block_q8_1 layout (quantize_q8_1).
template <int GROUP>
static __global__ void reex_q64_quantize_q8_1_grp(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    constexpr int NW = GROUP / 32; // warps per group (2 or 8)

    const int     tid = threadIdx.x;           // 0..GROUP-1 within the group
    const int64_t gb  = blockIdx.x;            // group index along the row
    const int64_t i1  = blockIdx.y;            // row (ne1)
    const int64_t i3  = fastdiv((uint32_t) blockIdx.z, ne2);
    const int64_t i2  = blockIdx.z - i3*ne2.z;

    const int64_t i0 = gb*GROUP + tid;         // column within the row
    const float   xi = i0 < ne00 ? x[i3*s03 + i2*s02 + i1*s01 + i0] : 0.0f;

    // amax over the full GROUP: warp-reduce then cross-warp via shared memory.
    __shared__ float s_amax[NW];
    float amax = fabsf(xi);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o, 32));
    if ((tid & 31) == 0) s_amax[tid >> 5] = amax;
    __syncthreads();
    if (tid < NW) {
        float a = s_amax[tid];
#pragma unroll
        for (int o = NW/2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync((1u << NW) - 1, a, o, 32));
        s_amax[tid] = a;
    }
    __syncthreads();

    const float   d = s_amax[0] / 127.0f;
    const int8_t  q = s_amax[0] == 0.0f ? 0 : (int8_t) roundf(xi / d);

    // d * sum(quantized) per 32-block -> CPU min term (q8_1_64.s / y.d*bsums).
    float qsum = (float) q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qsum += __shfl_xor_sync(0xFFFFFFFF, qsum, o, 32);

    block_q8_1 * y = (block_q8_1 *) vy;
    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;
    const int64_t ib  = i_cont / QK8_1;
    const int64_t iqs = i_cont % QK8_1;
    y[ib].qs[iqs] = q;
    if ((tid & 31) == 0) {
        y[ib].ds = make_half2(d, d * qsum);
    }
}

template <int GROUP>
static void reex_q64_quantize_q8_1_grp_cuda(
        const float * x, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne0 % GROUP == 0); // MATRIX_ROW_PADDING (512) guarantees this
    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);
    const dim3 num_blocks(ne0 / GROUP, ne1, ne2*ne3);
    reex_q64_quantize_q8_1_grp<GROUP><<<num_blocks, GROUP, 0, stream>>>(
        x, vy, ne00, s01, s02, s03, ne0, (uint32_t) ne1, ne2_fastdiv);
}
