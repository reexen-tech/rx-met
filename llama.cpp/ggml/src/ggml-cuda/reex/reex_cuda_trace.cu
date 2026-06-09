#include "reex_cuda_trace.cuh"

#include <atomic>

namespace {
std::atomic<int> g_reex_cuda_q8_mul_mat_hits{0};
std::atomic<int> g_reex_cuda_q8_mul_mat_id_hits{0};
}

extern "C" void ggml_reex_cuda_reset_q8_stats(void) {
    g_reex_cuda_q8_mul_mat_hits.store(0, std::memory_order_relaxed);
    g_reex_cuda_q8_mul_mat_id_hits.store(0, std::memory_order_relaxed);
}

extern "C" int ggml_reex_cuda_get_q8_mul_mat_hits(void) {
    return g_reex_cuda_q8_mul_mat_hits.load(std::memory_order_relaxed);
}

extern "C" int ggml_reex_cuda_get_q8_mul_mat_id_hits(void) {
    return g_reex_cuda_q8_mul_mat_id_hits.load(std::memory_order_relaxed);
}

void ggml_reex_cuda_note_q8_mul_mat_hit(const ggml_tensor * src0) {
    if (src0 && src0->type == GGML_TYPE_Q4_K) {
        g_reex_cuda_q8_mul_mat_hits.fetch_add(1, std::memory_order_relaxed);
    }
}

void ggml_reex_cuda_note_q8_mul_mat_id_hit(const ggml_tensor * src0) {
    if (src0 && src0->type == GGML_TYPE_Q4_K) {
        g_reex_cuda_q8_mul_mat_id_hits.fetch_add(1, std::memory_order_relaxed);
    }
}
