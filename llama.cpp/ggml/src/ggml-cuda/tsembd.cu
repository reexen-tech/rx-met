#include "tsembd.cuh"
#ifdef GGML_USE_REEX
#include "reex/reex_lut.cuh"
#endif

static __global__ void timestep_embedding_f32(const float * timesteps, float * dst, const int nb1, const int dim, const int max_period) {
    // blockIDx.y: idx of timesteps->ne[0]
    // blockIDx.x: idx of ((dim + 1) / 2) / BLOCK_SIZE
    int i = blockIdx.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    float * embed_data = (float *)((char *)dst +  i*nb1);

    int half = dim / 2;
    if (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    if (j >= half) {
        return;
    }

    float timestep = timesteps[i];
#ifdef GGML_USE_REEX
    float freq = (float)ggml_cuda_exp_lut_mixed_fp16_reex(-ggml_cuda_log_lut_mixed_fp16_reex((float)max_period) * j / half);
    float arg = timestep * freq;
    embed_data[j] = ggml_cuda_cos_lut_mixed_fp16_reex(arg);
    embed_data[j + half] = ggml_cuda_sin_lut_mixed_fp16_reex(arg);
#else
    float freq = (float)expf(-logf(max_period) * j / half);
    float arg = timestep * freq;
    embed_data[j] = cosf(arg);
    embed_data[j + half] = sinf(arg);
#endif
}

static void timestep_embedding_f32_cuda(const float * x, float * dst, const int ne00, const int nb1,
                                        const int dim, const int max_period, cudaStream_t stream) {
    int half_ceil = (dim + 1) / 2;
    int num_blocks = (half_ceil + CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne00, 1);
    timestep_embedding_f32<<<gridDim, CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE, 0, stream>>>(x, dst, nb1, dim, max_period);
}

void ggml_cuda_op_timestep_embedding(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    const int max_period = dst->op_params[1];

    timestep_embedding_f32_cuda(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
