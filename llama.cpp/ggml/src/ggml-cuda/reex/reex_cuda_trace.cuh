#pragma once

#include "../common.cuh"

extern "C" void ggml_reex_cuda_reset_q8_stats(void);
extern "C" int  ggml_reex_cuda_get_q8_mul_mat_hits(void);
extern "C" int  ggml_reex_cuda_get_q8_mul_mat_id_hits(void);

void ggml_reex_cuda_note_q8_mul_mat_hit(const ggml_tensor * src0);
void ggml_reex_cuda_note_q8_mul_mat_id_hit(const ggml_tensor * src0);

static inline void ggml_reex_cuda_note_q8_mul_mat_dispatch(const ggml_tensor * src0) {
    ggml_reex_cuda_note_q8_mul_mat_hit(src0);
}

static inline void ggml_reex_cuda_note_q8_mul_mat_id_dispatch(const ggml_tensor * src0) {
    ggml_reex_cuda_note_q8_mul_mat_id_hit(src0);
}
