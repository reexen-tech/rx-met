#pragma once

extern "C" void ggml_reex_cuda_reset_stats(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_hits(void);
extern "C" int  ggml_reex_cuda_get_q16_batch_fallbacks(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_id_hits(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_id_unsupported(void);

void ggml_reex_cuda_note_q16_mul_mat_hit(void);
void ggml_reex_cuda_note_q16_batch_fallback(void);
void ggml_reex_cuda_note_q16_mul_mat_id_hit(void);
void ggml_reex_cuda_note_q16_mul_mat_id_unsupported(void);
