#pragma once

#include "reex_cuda_trace.cuh"
#include "reex_gemm_cuda.cuh"
#include "reex_q16_trace.cuh"

// Target-layer filtering lives outside CUDA codegen, but is only used by REEX dispatch.
extern "C" int ggml_reex_get_target_layer(void);
extern "C" int ggml_reex_tensor_layer(const char * name);

static inline bool ggml_reex_cuda_q4_k_layer_allowed(const ggml_tensor * src0, const ggml_tensor * dst) {
    if (src0 == nullptr || dst == nullptr || src0->type != GGML_TYPE_Q4_K) {
        return true;
    }

    const int target = ggml_reex_get_target_layer();
    return target < 0 || ggml_reex_tensor_layer(dst->name) == target;
}

static inline bool ggml_reex_cuda_is_q16_mul_mat_candidate(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        bool split,
        bool bad_padding_clear) {
    return ggml_reex_cuda_q4_k_layer_allowed(src0, dst)
        && !split
        && !bad_padding_clear
        && ggml_reex_cuda_should_use()
        && src0->type == GGML_TYPE_Q4_K
        && src1->type == GGML_TYPE_F32
        && dst->type == GGML_TYPE_F32;
}

static inline bool ggml_reex_cuda_is_q16_mul_mat_id_candidate(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
    return ggml_reex_cuda_should_use()
        && ggml_reex_cuda_q4_k_layer_allowed(src0, dst)
        && src0->type == GGML_TYPE_Q4_K
        && src1->ne[3] == 1
        && dst->ne[3] == 1;
}

static inline bool ggml_reex_cuda_should_note_q16_mul_mat_id_unsupported(
        const ggml_tensor * src0,
        const ggml_tensor * dst) {
    return ggml_reex_cuda_should_use()
        && ggml_reex_cuda_q4_k_layer_allowed(src0, dst)
        && src0->type == GGML_TYPE_Q4_K;
}

static inline bool ggml_reex_cuda_is_q16_mul_mat_id_graph_candidate(const ggml_tensor * node) {
    return node != nullptr
        && node->op == GGML_OP_MUL_MAT_ID
        && node->src[0] != nullptr
        && node->src[0]->type == GGML_TYPE_Q4_K
        && node->src[1] != nullptr
        && node->src[1]->type == GGML_TYPE_F32
        && node->src[1]->ne[3] == 1
        && node->ne[3] == 1;
}

static inline bool ggml_reex_cuda_should_disable_cuda_graph_for_q16_mul_mat_id(const ggml_tensor * node) {
    return ggml_reex_cuda_is_q16_mul_mat_id_graph_candidate(node) && !ggml_reex_cuda_should_use();
}

static inline bool ggml_reex_cuda_should_use_q16_mul_mat(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        bool split,
        bool bad_padding_clear) {
    const bool q16_candidate = ggml_reex_cuda_is_q16_mul_mat_candidate(src0, src1, dst, split, bad_padding_clear);
    if (q16_candidate && src1->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        ggml_reex_cuda_note_q16_batch_fallback();
    }

    return q16_candidate && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
}

static inline bool ggml_reex_cuda_try_mul_mat_id_q16(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst) {
    if (ggml_reex_cuda_is_q16_mul_mat_id_candidate(src0, src1, dst)) {
        ggml_cuda_mul_mat_reex_q16(ctx, src0, src1, ids, dst);
        return true;
    }

    if (ggml_reex_cuda_should_note_q16_mul_mat_id_unsupported(src0, dst)) {
        ggml_reex_cuda_note_q16_mul_mat_id_unsupported();
    }

    return false;
}
