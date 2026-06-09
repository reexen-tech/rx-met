/*
 * REEX GEMM CUDA — W4×INT16 量化矩阵乘法 CUDA 实现
 *
 * W4×INT8 使用主路径：quantize_row_q8_1_cuda + ggml_cuda_mul_mat_vec_q (Q4_K×Q8_1)。
 * 本文件仅实现 W4×INT16：
 *   1. F32 激活 → block_q16_K (per-block QK_K=256, int16 量化)
 *   2. Q4_K × Q16 MMVQ kernel (每 warp 处理 1 行权重，int16×int4 整数乘累加)
 *   3. 顶层 dispatch: ggml_cuda_mul_mat_reex_q16
 */

#include "reex_gemm_cuda.cuh"
#include "reex_q16_trace.cuh"

/* ================================================================
 *  环境变量控制
 * ================================================================ */

bool ggml_reex_cuda_should_use(void) {
#if defined(GGML_USE_REEX_GEMM_CUDA)
    return true;
#else
    return false;
#endif
}

/* ================================================================
 *  Kernel 1: F32 → block_q16_K 量化
 *
 *  每个 CUDA block 处理 1 个 QK_K=256 元素的量化块。
 *  Phase 1: warp reduce 求 amax
 *  Phase 2: 计算 scale，量化为 int16
 * ================================================================ */

static __global__ void kernel_quantize_q16_K(
    const float * __restrict__ x,
    block_q16_K_cuda * __restrict__ y,
    const int64_t ne00,
    const int64_t stride_row,
    const int64_t nrows)
{
    const int64_t row = blockIdx.y;
    const int64_t blk = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= nrows) return;

    const int64_t nb = ne00 / REEX_QK_K;
    if (blk >= nb) return;

    const float * xb = x + row * stride_row + blk * REEX_QK_K;
    block_q16_K_cuda * yb = y + row * nb + blk;

    __shared__ int s_amax_int;

    if (tid == 0) s_amax_int = 0;
    __syncthreads();

    float local_amax = 0.0f;
    for (int j = tid; j < REEX_QK_K; j += blockDim.x) {
        float v = fabsf(xb[j]);
        if (v > local_amax) local_amax = v;
    }

    local_amax = warp_reduce_max<WARP_SIZE>(local_amax);

    if (tid % WARP_SIZE == 0) {
        atomicMax(&s_amax_int, __float_as_int(local_amax));
    }
    __syncthreads();

    float amax = __int_as_float(s_amax_int);
    if (amax < 1e-9f) amax = 1e-9f;

    const float scale_inv = REEX_Q16_SCALE / amax;
    const float scale     = amax / REEX_Q16_SCALE;

    for (int j = tid; j < REEX_QK_K; j += blockDim.x) {
        yb->qs[j] = (int16_t)rintf(xb[j] * scale_inv);
    }

    if (tid == 0) {
        yb->d = scale;
    }
}

void quantize_row_q16_K_cuda(
    const float * x, void * vy,
    int64_t ne00, int64_t stride_row,
    int64_t nrows, cudaStream_t stream)
{
    GGML_ASSERT(ne00 % REEX_QK_K == 0);
    const int64_t nb = ne00 / REEX_QK_K;

    const int threads = 256;
    dim3 blocks(nb, nrows);

    kernel_quantize_q16_K<<<blocks, threads, 0, stream>>>(
        x, (block_q16_K_cuda *)vy, ne00, stride_row, nrows);
}

/* ================================================================
 *  Device function: Q4_K × Q16 vec_dot
 *
 *  参考 CPU ggml_vec_dot_q4_K_q16_reex 的整数域实现。
 *  Q4_K: d, dmin, scales[12] packed, qs[QK_K/2]
 *  Q16:  d (float), qs[QK_K] (int16_t)
 *
 *  核心公式：
 *    dot = q16_scale * [ d * Σ(scale_s * q4[i] * q16[i])
 *                       - dmin * Σ(min_s * q16_bsum[j]) ]
 *
 *  与 Q8_1 的 vec_dot 差异：
 *    - int16 × int8(q4) = int32，需 int64 累加
 *    - 不使用 dp4a，直接标量乘累加
 * ================================================================ */

static __device__ __forceinline__ float vec_dot_q4_K_q16_reex_device(
    const void * __restrict__ vbq4, const block_q16_K_cuda * __restrict__ bq16,
    const int kbx, const int64_t ne00)
{
    const block_q4_K * bq4 = (const block_q4_K *)vbq4 + kbx;

    const uint8_t * q4 = bq4->qs;
    const float y_scale = bq16->d;
    const int16_t * q16 = bq16->qs;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    memcpy(utmp, bq4->scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    const uint8_t * scales = (const uint8_t *)&utmp[0];
    const uint8_t * mins   = (const uint8_t *)&utmp[2];

    int8_t aux8[REEX_QK_K];
    int8_t * a = aux8;
    for (int j = 0; j < REEX_QK_K / 64; ++j) {
        for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
        a += 32;
        for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
        a += 32; q4 += 32;
    }

    int64_t sumi = 0;
    for (int j = 0; j < REEX_QK_K / 16; ++j) {
        int32_t bsum = 0;
        for (int l = 0; l < 16; ++l) bsum += q16[j * 16 + l];
        sumi += (int64_t)bsum * mins[j / 2];
    }

    int64_t aux64[8] = {0};
    a = aux8;
    const int16_t * q16p = q16;
    int is = 0;
    for (int j = 0; j < REEX_QK_K / 32; ++j) {
        int32_t scale = scales[is++];
        for (int l = 0; l < 8; ++l) aux64[l] += (int64_t)scale * ((int32_t)q16p[l] * a[l]);
        q16p += 8; a += 8;
        for (int l = 0; l < 8; ++l) aux64[l] += (int64_t)scale * ((int32_t)q16p[l] * a[l]);
        q16p += 8; a += 8;
        for (int l = 0; l < 8; ++l) aux64[l] += (int64_t)scale * ((int32_t)q16p[l] * a[l]);
        q16p += 8; a += 8;
        for (int l = 0; l < 8; ++l) aux64[l] += (int64_t)scale * ((int32_t)q16p[l] * a[l]);
        q16p += 8; a += 8;
    }

    const float d = __half2float(bq4->dm.x) * y_scale;
    float sumf = 0.0f;
    for (int l = 0; l < 8; ++l) sumf += d * (float)aux64[l];
    const float dmin = __half2float(bq4->dm.y) * y_scale;
    sumf -= dmin * (float)sumi;

    return sumf;
}

/* ================================================================
 *  Kernel 2: MMVQ — Q4_K × Q16 Matrix-Vector Multiply
 *
 *  每行权重由一个 warp 处理，每个线程处理一部分 blocks。
 *  适用于 batch size <= 8 的推理场景。
 * ================================================================ */

/*
 * MMVQ kernel: 每个 CUDA block 处理一行权重。
 * 128 threads，每个线程独立处理不同的 Q4_K/Q16 blocks，
 * 然后通过 shared memory + warp reduce 归约。
 */
template <int ncols_dst>
__launch_bounds__(128, 1)
static __global__ void kernel_mul_mat_vec_q4_K_q16_reex(
    const void * __restrict__ vx,
    const block_q16_K_cuda * __restrict__ vy,
    float * __restrict__ dst,
    const int64_t ncols_x,
    const int64_t nrows_x,
    const int64_t stride_row_x,
    const int64_t stride_col_y,
    const int64_t stride_col_dst)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= nrows_x) return;

    const int64_t nb = ncols_x / REEX_QK_K;

    float tmp[ncols_dst] = {0.0f};

    for (int64_t blk = tid; blk < nb; blk += blockDim.x) {
        const int kbx = row * stride_row_x + blk;

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const block_q16_K_cuda * bq16 = vy + j * stride_col_y + blk;
            tmp[j] += vec_dot_q4_K_q16_reex_device(vx, bq16, kbx, ncols_x);
        }
    }

    /*
     * Block-level reduction: 128 threads → 4 warps → final sum in warp 0.
     * Each thread holds partial sums; first do warp reduce, then use shared
     * memory to combine across warps.
     */
    constexpr int nwarps = 128 / WARP_SIZE; // 4

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        tmp[j] = warp_reduce_sum<WARP_SIZE>(tmp[j]);
    }

    __shared__ float s_partial[nwarps][ncols_dst];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            s_partial[warp_id][j] = tmp[j];
        }
    }
    __syncthreads();

    if (tid == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            float sum = 0.0f;
            for (int w = 0; w < nwarps; ++w) {
                sum += s_partial[w][j];
            }
            dst[j * stride_col_dst + row] = sum;
        }
    }
}

/* ================================================================
 *  Top-level dispatch: ggml_cuda_mul_mat_reex_q16
 *
 *  1. 将 F32 src1 量化为 Q16
 *  2. 调用 MMVQ kernel 做 Q4_K × Q16 矩阵乘 / 专家路由矩阵乘
 *
 *  当前支持：
 *    - MUL_MAT:    小 batch / MMVQ (M <= 8)
 *    - MUL_MAT_ID: 单 sample 专家路由 (src1->ne[3] == 1)
 *
 *  更大 MUL_MAT batch 仍由调度层回退到现有 CUDA 主路径。
 * ================================================================ */

__launch_bounds__(128, 1)
static __global__ void kernel_mul_mat_vec_q4_K_q16_reex_mmid(
    const void * __restrict__ vx,
    const block_q16_K_cuda * __restrict__ vy,
    const int32_t * __restrict__ ids,
    float * __restrict__ dst,
    const int64_t ncols_x,
    const int64_t nrows_x,
    const int64_t nchannels_y,
    const int64_t stride_row_x,
    const int64_t stride_channel_x,
    const int64_t stride_channel_y,
    const int64_t stride_col_y,
    const int64_t stride_channel_dst,
    const int64_t stride_col_dst,
    const int64_t ids_stride)
{
    const int row = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int token_idx = blockIdx.z;
    const int tid = threadIdx.x;

    if (row >= nrows_x) {
        return;
    }

    const int channel_x = ids[channel_dst + token_idx * ids_stride];
    const int channel_y = nchannels_y > 1 ? channel_dst % nchannels_y : 0;

    const int64_t nb = ncols_x / REEX_QK_K;
    float tmp = 0.0f;

    for (int64_t blk = tid; blk < nb; blk += blockDim.x) {
        const int64_t kbx = channel_x * stride_channel_x + row * stride_row_x + blk;
        const block_q16_K_cuda * bq16 = vy + token_idx * stride_col_y + channel_y * stride_channel_y + blk;
        tmp += vec_dot_q4_K_q16_reex_device(vx, bq16, kbx, ncols_x);
    }

    tmp = warp_reduce_sum<WARP_SIZE>(tmp);

    constexpr int nwarps = 128 / WARP_SIZE;
    __shared__ float s_partial[nwarps];

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        s_partial[warp_id] = tmp;
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int w = 0; w < nwarps; ++w) {
            sum += s_partial[w];
        }
        dst[token_idx * stride_col_dst + channel_dst * stride_channel_dst + row] = sum;
    }
}

void ggml_cuda_mul_mat_reex_q16(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1,
    const ggml_tensor * ids, ggml_tensor * dst)
{
    GGML_ASSERT(src0->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // N / rows per expert
    const int64_t ne02 = src0->ne[2]; // experts
    const int64_t ne10 = src1->ne[0]; // K
    const int64_t ne11 = src1->ne[1]; // activation channels / n_used or batch
    const int64_t ne12 = src1->ne[2]; // tokens or batch cols
    const int64_t ne13 = src1->ne[3]; // samples

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ne00 % REEX_QK_K == 0);

    cudaStream_t stream = ctx.stream();

    const int64_t nb = ne00 / REEX_QK_K;
    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  = dst->nb[1]  / ts_dst;
    const int64_t s2  = dst->nb[2]  / ts_dst;
    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    const int64_t nrows_q16 = ne11 * ne12 * ne13;
    const size_t q16_row_bytes = nb * sizeof(block_q16_K_cuda);
    ggml_cuda_pool_alloc<char> src1_q16(ctx.pool(), nrows_q16 * q16_row_bytes);

    const float * src1_f32 = (const float *)src1->data;
    quantize_row_q16_K_cuda(
        src1_f32, src1_q16.get(),
        ne00, s11,
        nrows_q16, stream);
    CUDA_CHECK(cudaGetLastError());

    float * dst_f32 = (float *)dst->data;
    const int threads = 128;

    if (ids) {
        GGML_ASSERT(ids->type == GGML_TYPE_I32);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(dst->ne[3] == 1);
        GGML_ASSERT(ne02 > 0);

        ggml_reex_cuda_note_q16_mul_mat_id_hit();

        kernel_mul_mat_vec_q4_K_q16_reex_mmid<<<dim3(ne01, dst->ne[1], ne12), threads, 0, stream>>>(
            src0->data,
            (const block_q16_K_cuda *)src1_q16.get(),
            (const int32_t *)ids->data,
            dst_f32,
            ne00,
            ne01,
            ne11,
            s01,
            s02,
            nb,
            ne11 * nb,
            s1,
            s2,
            ids_stride);
    } else {
        ggml_reex_cuda_note_q16_mul_mat_hit();

        const int64_t stride_row_x = nb;
        const int64_t stride_col_y = nb;
        const int64_t stride_col_dst = ne01;

        switch (ne11) {
            case 1: {
                kernel_mul_mat_vec_q4_K_q16_reex<1><<<ne01, threads, 0, stream>>>(
                    src0->data, (const block_q16_K_cuda *)src1_q16.get(), dst_f32,
                    ne00, ne01, stride_row_x, stride_col_y, stride_col_dst);
                break;
            }
            case 2: {
                kernel_mul_mat_vec_q4_K_q16_reex<2><<<ne01, threads, 0, stream>>>(
                    src0->data, (const block_q16_K_cuda *)src1_q16.get(), dst_f32,
                    ne00, ne01, stride_row_x, stride_col_y, stride_col_dst);
                break;
            }
            case 3: {
                kernel_mul_mat_vec_q4_K_q16_reex<3><<<ne01, threads, 0, stream>>>(
                    src0->data, (const block_q16_K_cuda *)src1_q16.get(), dst_f32,
                    ne00, ne01, stride_row_x, stride_col_y, stride_col_dst);
                break;
            }
            case 4: {
                kernel_mul_mat_vec_q4_K_q16_reex<4><<<ne01, threads, 0, stream>>>(
                    src0->data, (const block_q16_K_cuda *)src1_q16.get(), dst_f32,
                    ne00, ne01, stride_row_x, stride_col_y, stride_col_dst);
                break;
            }
            default: {
                for (int64_t col = 0; col < ne11; ++col) {
                    const block_q16_K_cuda * q16_col = (const block_q16_K_cuda *)src1_q16.get() + col * nb;
                    kernel_mul_mat_vec_q4_K_q16_reex<1><<<ne01, threads, 0, stream>>>(
                        src0->data, q16_col, dst_f32 + col * ne01,
                        ne00, ne01, stride_row_x, 0, 0);
                }
                break;
            }
        }
    }

    CUDA_CHECK(cudaGetLastError());
}
