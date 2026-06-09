#include "reex_q16_trace.cuh"

#include <atomic>

namespace {
std::atomic<int> g_q16_mul_mat_hits{0};
std::atomic<int> g_q16_batch_fallbacks{0};
std::atomic<int> g_q16_mul_mat_id_hits{0};
std::atomic<int> g_q16_mul_mat_id_unsupported{0};
}

extern "C" void ggml_reex_cuda_reset_stats(void) {
    g_q16_mul_mat_hits.store(0, std::memory_order_relaxed);
    g_q16_batch_fallbacks.store(0, std::memory_order_relaxed);
    g_q16_mul_mat_id_hits.store(0, std::memory_order_relaxed);
    g_q16_mul_mat_id_unsupported.store(0, std::memory_order_relaxed);
}

extern "C" int ggml_reex_cuda_get_q16_mul_mat_hits(void) {
    return g_q16_mul_mat_hits.load(std::memory_order_relaxed);
}

extern "C" int ggml_reex_cuda_get_q16_batch_fallbacks(void) {
    return g_q16_batch_fallbacks.load(std::memory_order_relaxed);
}

extern "C" int ggml_reex_cuda_get_q16_mul_mat_id_hits(void) {
    return g_q16_mul_mat_id_hits.load(std::memory_order_relaxed);
}

extern "C" int ggml_reex_cuda_get_q16_mul_mat_id_unsupported(void) {
    return g_q16_mul_mat_id_unsupported.load(std::memory_order_relaxed);
}

void ggml_reex_cuda_note_q16_mul_mat_hit(void) {
    g_q16_mul_mat_hits.fetch_add(1, std::memory_order_relaxed);
}

void ggml_reex_cuda_note_q16_batch_fallback(void) {
    g_q16_batch_fallbacks.fetch_add(1, std::memory_order_relaxed);
}

void ggml_reex_cuda_note_q16_mul_mat_id_hit(void) {
    g_q16_mul_mat_id_hits.fetch_add(1, std::memory_order_relaxed);
}

void ggml_reex_cuda_note_q16_mul_mat_id_unsupported(void) {
    g_q16_mul_mat_id_unsupported.fetch_add(1, std::memory_order_relaxed);
}
