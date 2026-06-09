#pragma once

#include <cstdint>

constexpr int GGML_CUDA_FATTN_KQ_STRIDE_VALUE = 256;

inline bool ggml_cuda_reex_can_use_fattn_vector_kernel(
        bool same_kv_type,
        int64_t q_width,
        int64_t k_stride_width) {
    return same_kv_type &&
           q_width <= 256 &&
           q_width % 64 == 0 &&
           k_stride_width % GGML_CUDA_FATTN_KQ_STRIDE_VALUE == 0;
}
