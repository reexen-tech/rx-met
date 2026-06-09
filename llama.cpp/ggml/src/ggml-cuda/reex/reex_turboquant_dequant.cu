// SPDX-License-Identifier: MIT
//
// reex_turboquant_dequant.cu — REEX_TURBOQUANT only.
//
// CUDA kernels and launchers for dequantizing GGML_TYPE_TQ_K3 / TQ_V2 / TQ_V4.
// Bit-identical to ggml/src/reex_turboquant_quants_ref.c::dequantize_row_*.

#ifdef REEX_TURBOQUANT

#include "reex_turboquant_dequant.cuh"

// ---------------------------------------------------------------------------
// TQ_K3 codebook constants (mirror of
// ggml/src/ggml-cpu/reex/reex_turboquant_codebook_tq_k3.inc).  These live in
// __constant__ memory so they cost a single broadcast load per warp.
// Keep in sync with the CPU .inc file — both are produced by the same
// generator (tools/reex/reex_turboquant_gen_constants).
//
// P4 A2.2 K3 P3 upgrade (TheTom turbo3_0): 8 centroids, full 3-bit PolarQuant,
// no residual term (idx is reconstructed from qs low-2-bits + signs high-bit
// and value = centroids[idx] * corrected_norm — see reex_turboquant_dequant.cuh).
// ---------------------------------------------------------------------------

static __constant__ float reex_tq_k3_centroids_d[8] = {
    -0.1861817092f, -0.1154012680f, -0.0635419339f, -0.0184134189f,
     0.0247995369f,  0.0696418509f,  0.1208978295f,  0.1906363368f,
};

// ---------------------------------------------------------------------------
// TQ_V_POLAR2 / TQ_V_POLAR4 codebook constants (mirror of
// ggml/src/ggml-cpu/reex/reex_turboquant_codebook_tq_v_polar.inc).  Keep in
// sync with the CPU .inc file — both produced by
// tools/reex/reex_turboquant_gen_constants_v_polar.
//
// P4 A2.5 V-β (TheTom turbo2_0 / turbo4_0): 4-/16-centroid Lloyd-Max +
// norm correction (no signs / no residual).  Decode: value =
// centroids[idx] * corrected_norm — see reex_turboquant_dequant.cuh.
// ---------------------------------------------------------------------------

static __constant__ float reex_tq_v_polar2_centroids_d[4] = {
    -0.1323675364f, -0.0391626097f, 0.0408212505f, 0.1337173581f,
};

static __constant__ float reex_tq_v_polar4_centroids_d[16] = {
    -0.2315807939f, -0.1735078096f, -0.1336077154f, -0.1014359593f,
    -0.0734821036f, -0.0480686277f, -0.0242136829f, -0.0012462543f,
     0.0213646460f,  0.0441083387f,  0.0675085932f,  0.0922231078f,
     0.1192075014f,  0.1500890255f,  0.1882706583f,  0.2438714951f,
};

// ---------------------------------------------------------------------------
// Block-wide dequantization kernels (used by ggml_get_to_*_cuda).
// ---------------------------------------------------------------------------

template <typename dst_t>
static __global__ void reex_kernel_to_t_tq_k3(
        const block_tq_k3 * __restrict__ x, dst_t * __restrict__ y, int64_t nb) {
    const int64_t ib = blockIdx.x;
    if (ib >= nb) return;
    const int j = threadIdx.x;
    if (j >= QK_TQ_K3) return;

    const float v = reex_dequant_tq_k3_elt(x + ib, j, reex_tq_k3_centroids_d);
    y[ib * QK_TQ_K3 + j] = ggml_cuda_cast<dst_t>(v);
}

template <typename dst_t>
static __global__ void reex_kernel_to_t_tq_v2(
        const block_tq_v2 * __restrict__ x, dst_t * __restrict__ y, int64_t nb) {
    const int     local_block = threadIdx.x / QK_TQ_V2;
    const int     j           = threadIdx.x % QK_TQ_V2;
    const int     blocks_per_grid_block = blockDim.x / QK_TQ_V2;
    const int64_t ib          = (int64_t) blockIdx.x * blocks_per_grid_block + local_block;
    if (ib >= nb) return;

    const float v = reex_dequant_tq_v2_elt(x + ib, j);
    y[ib * QK_TQ_V2 + j] = ggml_cuda_cast<dst_t>(v);
}

template <typename dst_t>
static __global__ void reex_kernel_to_t_tq_v4(
        const block_tq_v4 * __restrict__ x, dst_t * __restrict__ y, int64_t nb) {
    const int     local_block = threadIdx.x / QK_TQ_V4;
    const int     j           = threadIdx.x % QK_TQ_V4;
    const int     blocks_per_grid_block = blockDim.x / QK_TQ_V4;
    const int64_t ib          = (int64_t) blockIdx.x * blocks_per_grid_block + local_block;
    if (ib >= nb) return;

    const float v = reex_dequant_tq_v4_elt(x + ib, j);
    y[ib * QK_TQ_V4 + j] = ggml_cuda_cast<dst_t>(v);
}

template <typename dst_t>
static __global__ void reex_kernel_to_t_tq_v_polar2(
        const block_tq_v_polar2 * __restrict__ x, dst_t * __restrict__ y, int64_t nb) {
    const int64_t ib = blockIdx.x;
    if (ib >= nb) return;
    const int j = threadIdx.x;
    if (j >= QK_TQ_V_POLAR2) return;

    const float v = reex_dequant_tq_v_polar2_elt(x + ib, j, reex_tq_v_polar2_centroids_d);
    y[ib * QK_TQ_V_POLAR2 + j] = ggml_cuda_cast<dst_t>(v);
}

template <typename dst_t>
static __global__ void reex_kernel_to_t_tq_v_polar4(
        const block_tq_v_polar4 * __restrict__ x, dst_t * __restrict__ y, int64_t nb) {
    const int64_t ib = blockIdx.x;
    if (ib >= nb) return;
    const int j = threadIdx.x;
    if (j >= QK_TQ_V_POLAR4) return;

    const float v = reex_dequant_tq_v_polar4_elt(x + ib, j, reex_tq_v_polar4_centroids_d);
    y[ib * QK_TQ_V_POLAR4 + j] = ggml_cuda_cast<dst_t>(v);
}

template <typename dst_t>
void reex_turboquant_to_t_tq_k3_cuda(
        const void * __restrict__ src, dst_t * __restrict__ dst,
        int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TQ_K3 == 0);
    const int64_t nb = k / QK_TQ_K3;
    if (nb == 0) return;
    const dim3 block_dims(QK_TQ_K3, 1, 1);
    const dim3 block_nums((unsigned int) nb, 1, 1);
    reex_kernel_to_t_tq_k3<dst_t><<<block_nums, block_dims, 0, stream>>>(
        (const block_tq_k3 *) src, dst, nb);
}

template <typename dst_t>
void reex_turboquant_to_t_tq_v2_cuda(
        const void * __restrict__ src, dst_t * __restrict__ dst,
        int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TQ_V2 == 0);
    const int64_t nb = k / QK_TQ_V2;
    if (nb == 0) return;
    constexpr int blocks_per_cuda_block = 4;  // 4 * 32 = 128 threads
    const dim3    block_dims(blocks_per_cuda_block * QK_TQ_V2, 1, 1);
    const int64_t cuda_blocks = (nb + blocks_per_cuda_block - 1) / blocks_per_cuda_block;
    const dim3    block_nums((unsigned int) cuda_blocks, 1, 1);
    reex_kernel_to_t_tq_v2<dst_t><<<block_nums, block_dims, 0, stream>>>(
        (const block_tq_v2 *) src, dst, nb);
}

template <typename dst_t>
void reex_turboquant_to_t_tq_v4_cuda(
        const void * __restrict__ src, dst_t * __restrict__ dst,
        int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TQ_V4 == 0);
    const int64_t nb = k / QK_TQ_V4;
    if (nb == 0) return;
    constexpr int blocks_per_cuda_block = 4;
    const dim3    block_dims(blocks_per_cuda_block * QK_TQ_V4, 1, 1);
    const int64_t cuda_blocks = (nb + blocks_per_cuda_block - 1) / blocks_per_cuda_block;
    const dim3    block_nums((unsigned int) cuda_blocks, 1, 1);
    reex_kernel_to_t_tq_v4<dst_t><<<block_nums, block_dims, 0, stream>>>(
        (const block_tq_v4 *) src, dst, nb);
}

template void reex_turboquant_to_t_tq_k3_cuda<float>      (const void *, float       *, int64_t, cudaStream_t);
template void reex_turboquant_to_t_tq_k3_cuda<half>       (const void *, half        *, int64_t, cudaStream_t);
template void reex_turboquant_to_t_tq_k3_cuda<nv_bfloat16>(const void *, nv_bfloat16 *, int64_t, cudaStream_t);

template void reex_turboquant_to_t_tq_v2_cuda<float>      (const void *, float       *, int64_t, cudaStream_t);
template void reex_turboquant_to_t_tq_v2_cuda<half>       (const void *, half        *, int64_t, cudaStream_t);
template void reex_turboquant_to_t_tq_v2_cuda<nv_bfloat16>(const void *, nv_bfloat16 *, int64_t, cudaStream_t);

template void reex_turboquant_to_t_tq_v4_cuda<float>      (const void *, float       *, int64_t, cudaStream_t);
template void reex_turboquant_to_t_tq_v4_cuda<half>       (const void *, half        *, int64_t, cudaStream_t);
template void reex_turboquant_to_t_tq_v4_cuda<nv_bfloat16>(const void *, nv_bfloat16 *, int64_t, cudaStream_t);

template <typename dst_t>
void reex_turboquant_to_t_tq_v_polar2_cuda(
        const void * __restrict__ src, dst_t * __restrict__ dst,
        int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TQ_V_POLAR2 == 0);
    const int64_t nb = k / QK_TQ_V_POLAR2;
    if (nb == 0) return;
    const dim3 block_dims(QK_TQ_V_POLAR2, 1, 1);
    const dim3 block_nums((unsigned int) nb, 1, 1);
    reex_kernel_to_t_tq_v_polar2<dst_t><<<block_nums, block_dims, 0, stream>>>(
        (const block_tq_v_polar2 *) src, dst, nb);
}

template <typename dst_t>
void reex_turboquant_to_t_tq_v_polar4_cuda(
        const void * __restrict__ src, dst_t * __restrict__ dst,
        int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TQ_V_POLAR4 == 0);
    const int64_t nb = k / QK_TQ_V_POLAR4;
    if (nb == 0) return;
    const dim3 block_dims(QK_TQ_V_POLAR4, 1, 1);
    const dim3 block_nums((unsigned int) nb, 1, 1);
    reex_kernel_to_t_tq_v_polar4<dst_t><<<block_nums, block_dims, 0, stream>>>(
        (const block_tq_v_polar4 *) src, dst, nb);
}

template void reex_turboquant_to_t_tq_v_polar2_cuda<float>      (const void *, float       *, int64_t, cudaStream_t);
template void reex_turboquant_to_t_tq_v_polar2_cuda<half>       (const void *, half        *, int64_t, cudaStream_t);
template void reex_turboquant_to_t_tq_v_polar2_cuda<nv_bfloat16>(const void *, nv_bfloat16 *, int64_t, cudaStream_t);

template void reex_turboquant_to_t_tq_v_polar4_cuda<float>      (const void *, float       *, int64_t, cudaStream_t);
template void reex_turboquant_to_t_tq_v_polar4_cuda<half>       (const void *, half        *, int64_t, cudaStream_t);
template void reex_turboquant_to_t_tq_v_polar4_cuda<nv_bfloat16>(const void *, nv_bfloat16 *, int64_t, cudaStream_t);

// ---------------------------------------------------------------------------
// GET_ROWS launchers.
//
// One CUDA block per output row. Each block has `threads` threads which
// cooperatively decode the row's `ne00` elements (must be a multiple of QK_*).
// Layout mirrors getrows.cu's k_get_rows: blockIdx.x = row in src1,
// blockIdx.z packs (i11, i12) so the strides are 1D.
// ---------------------------------------------------------------------------

template <typename dst_t, int QK, int LAYOUT_KIND>
static __global__ void reex_kernel_get_rows_tq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        int64_t ne00,
        int64_t ne12,
        size_t  s1, size_t s2, size_t s3,
        size_t  nb01, size_t nb02, size_t nb03,
        size_t  s10, size_t s11, size_t s12) {
    const int64_t z   = blockIdx.z;
    const int     i11 = (int)(z / ne12);
    const int     i12 = (int)(z % ne12);
    const int     i10 = blockIdx.x;

    const int     i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t          * dst_row  = dst  + i10*s1   + i11*s2   + i12*s3;
    const char     * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

    for (int64_t j = threadIdx.x; j < ne00; j += blockDim.x) {
        const int ib_in_row = (int)(j / QK);
        const int j_in_blk  = (int)(j % QK);
        float v = 0.0f;
        if constexpr (LAYOUT_KIND == 0) {
            const block_tq_k3 * b = (const block_tq_k3 *) src0_row + ib_in_row;
            v = reex_dequant_tq_k3_elt(b, j_in_blk, reex_tq_k3_centroids_d);
        } else if constexpr (LAYOUT_KIND == 1) {
            const block_tq_v2 * b = (const block_tq_v2 *) src0_row + ib_in_row;
            v = reex_dequant_tq_v2_elt(b, j_in_blk);
        } else if constexpr (LAYOUT_KIND == 2) {
            const block_tq_v4 * b = (const block_tq_v4 *) src0_row + ib_in_row;
            v = reex_dequant_tq_v4_elt(b, j_in_blk);
        } else if constexpr (LAYOUT_KIND == 3) {
            const block_tq_v_polar2 * b = (const block_tq_v_polar2 *) src0_row + ib_in_row;
            v = reex_dequant_tq_v_polar2_elt(b, j_in_blk, reex_tq_v_polar2_centroids_d);
        } else {
            const block_tq_v_polar4 * b = (const block_tq_v_polar4 *) src0_row + ib_in_row;
            v = reex_dequant_tq_v_polar4_elt(b, j_in_blk, reex_tq_v_polar4_centroids_d);
        }
        dst_row[j] = ggml_cuda_cast<dst_t>(v);
    }
}

template <typename dst_t, int QK, int LAYOUT_KIND>
static void reex_get_rows_launch(
        const void *  src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00,  size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10,  int64_t ne11, int64_t ne12,
        size_t  nb10,  size_t nb11, size_t nb12,
        size_t  nb1,   size_t nb2,  size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK == 0);
    GGML_UNUSED(nb10);
    const size_t s1  = nb1  / sizeof(dst_t);
    const size_t s2  = nb2  / sizeof(dst_t);
    const size_t s3  = nb3  / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    const int    threads = 128;
    const dim3   block_dims(threads, 1, 1);
    const dim3   block_nums((unsigned int) ne10, 1, (unsigned int) (ne11 * ne12));
    reex_kernel_get_rows_tq<dst_t, QK, LAYOUT_KIND><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne12,
        s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

template <typename dst_t>
void reex_turboquant_get_rows_tq_k3_cuda(
        const void *  src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00,  size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10,  int64_t ne11, int64_t ne12,
        size_t  nb10,  size_t nb11, size_t nb12,
        size_t  nb1,   size_t nb2,  size_t nb3,
        cudaStream_t stream) {
    reex_get_rows_launch<dst_t, QK_TQ_K3, /*kind=*/0>(
        src0_d, src1_d, dst_d,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12,
        nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

template <typename dst_t>
void reex_turboquant_get_rows_tq_v2_cuda(
        const void *  src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00,  size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10,  int64_t ne11, int64_t ne12,
        size_t  nb10,  size_t nb11, size_t nb12,
        size_t  nb1,   size_t nb2,  size_t nb3,
        cudaStream_t stream) {
    reex_get_rows_launch<dst_t, QK_TQ_V2, /*kind=*/1>(
        src0_d, src1_d, dst_d,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12,
        nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

template <typename dst_t>
void reex_turboquant_get_rows_tq_v4_cuda(
        const void *  src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00,  size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10,  int64_t ne11, int64_t ne12,
        size_t  nb10,  size_t nb11, size_t nb12,
        size_t  nb1,   size_t nb2,  size_t nb3,
        cudaStream_t stream) {
    reex_get_rows_launch<dst_t, QK_TQ_V4, /*kind=*/2>(
        src0_d, src1_d, dst_d,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12,
        nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

template <typename dst_t>
void reex_turboquant_get_rows_tq_v_polar2_cuda(
        const void *  src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00,  size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10,  int64_t ne11, int64_t ne12,
        size_t  nb10,  size_t nb11, size_t nb12,
        size_t  nb1,   size_t nb2,  size_t nb3,
        cudaStream_t stream) {
    reex_get_rows_launch<dst_t, QK_TQ_V_POLAR2, /*kind=*/3>(
        src0_d, src1_d, dst_d,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12,
        nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

template <typename dst_t>
void reex_turboquant_get_rows_tq_v_polar4_cuda(
        const void *  src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00,  size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10,  int64_t ne11, int64_t ne12,
        size_t  nb10,  size_t nb11, size_t nb12,
        size_t  nb1,   size_t nb2,  size_t nb3,
        cudaStream_t stream) {
    reex_get_rows_launch<dst_t, QK_TQ_V_POLAR4, /*kind=*/4>(
        src0_d, src1_d, dst_d,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12,
        nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

template void reex_turboquant_get_rows_tq_k3_cuda<float>      (const void *, const int32_t *, float       *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void reex_turboquant_get_rows_tq_k3_cuda<half>       (const void *, const int32_t *, half        *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void reex_turboquant_get_rows_tq_k3_cuda<nv_bfloat16>(const void *, const int32_t *, nv_bfloat16 *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);

template void reex_turboquant_get_rows_tq_v2_cuda<float>      (const void *, const int32_t *, float       *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void reex_turboquant_get_rows_tq_v2_cuda<half>       (const void *, const int32_t *, half        *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void reex_turboquant_get_rows_tq_v2_cuda<nv_bfloat16>(const void *, const int32_t *, nv_bfloat16 *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);

template void reex_turboquant_get_rows_tq_v4_cuda<float>      (const void *, const int32_t *, float       *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void reex_turboquant_get_rows_tq_v4_cuda<half>       (const void *, const int32_t *, half        *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void reex_turboquant_get_rows_tq_v4_cuda<nv_bfloat16>(const void *, const int32_t *, nv_bfloat16 *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);

template void reex_turboquant_get_rows_tq_v_polar2_cuda<float>      (const void *, const int32_t *, float       *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void reex_turboquant_get_rows_tq_v_polar2_cuda<half>       (const void *, const int32_t *, half        *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void reex_turboquant_get_rows_tq_v_polar2_cuda<nv_bfloat16>(const void *, const int32_t *, nv_bfloat16 *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);

template void reex_turboquant_get_rows_tq_v_polar4_cuda<float>      (const void *, const int32_t *, float       *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void reex_turboquant_get_rows_tq_v_polar4_cuda<half>       (const void *, const int32_t *, half        *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void reex_turboquant_get_rows_tq_v_polar4_cuda<nv_bfloat16>(const void *, const int32_t *, nv_bfloat16 *, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);

// I32 dst is unreachable in practice (TQ -> int32 has no meaning) but
// getrows.cu's `ggml_cuda_get_rows_switch_src0_type<int32_t>` template still
// instantiates the call site, so we provide an aborting implementation to
// satisfy the linker without silently producing garbage.
template <typename dst_t>
static void reex_turboquant_get_rows_unsupported_dst(
        const void *, const int32_t *, dst_t *,
        int64_t, size_t, size_t, size_t,
        int64_t, int64_t, int64_t,
        size_t, size_t, size_t,
        size_t, size_t, size_t,
        cudaStream_t) {
    GGML_ABORT("REEX_TURBOQUANT GET_ROWS into int32_t dst is not supported\n");
}

template <>
void reex_turboquant_get_rows_tq_k3_cuda<int32_t>(
        const void * src0_d, const int32_t * src1_d, int32_t * dst_d,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t  nb10, size_t nb11, size_t nb12,
        size_t  nb1,  size_t nb2,  size_t nb3,
        cudaStream_t stream) {
    reex_turboquant_get_rows_unsupported_dst<int32_t>(
        src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

template <>
void reex_turboquant_get_rows_tq_v2_cuda<int32_t>(
        const void * src0_d, const int32_t * src1_d, int32_t * dst_d,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t  nb10, size_t nb11, size_t nb12,
        size_t  nb1,  size_t nb2,  size_t nb3,
        cudaStream_t stream) {
    reex_turboquant_get_rows_unsupported_dst<int32_t>(
        src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

template <>
void reex_turboquant_get_rows_tq_v4_cuda<int32_t>(
        const void * src0_d, const int32_t * src1_d, int32_t * dst_d,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t  nb10, size_t nb11, size_t nb12,
        size_t  nb1,  size_t nb2,  size_t nb3,
        cudaStream_t stream) {
    reex_turboquant_get_rows_unsupported_dst<int32_t>(
        src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

template <>
void reex_turboquant_get_rows_tq_v_polar2_cuda<int32_t>(
        const void * src0_d, const int32_t * src1_d, int32_t * dst_d,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t  nb10, size_t nb11, size_t nb12,
        size_t  nb1,  size_t nb2,  size_t nb3,
        cudaStream_t stream) {
    reex_turboquant_get_rows_unsupported_dst<int32_t>(
        src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

template <>
void reex_turboquant_get_rows_tq_v_polar4_cuda<int32_t>(
        const void * src0_d, const int32_t * src1_d, int32_t * dst_d,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t  nb10, size_t nb11, size_t nb12,
        size_t  nb1,  size_t nb2,  size_t nb3,
        cudaStream_t stream) {
    reex_turboquant_get_rows_unsupported_dst<int32_t>(
        src0_d, src1_d, dst_d, ne00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

#endif  // REEX_TURBOQUANT
