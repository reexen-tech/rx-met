/**
 * REEX block-64 legacy quantization — base-side reference (de)quantizers.
 *
 * These are referenced by ggml.c's type-traits table (to_float / from_float_ref)
 * and ggml_quantize_chunk(), so they live in libggml-base alongside the native
 * ggml-quants.c reference kernels. CPU vec_dot / activation quantization lives
 * separately in ggml-cpu/reex/reex_q64_quants.*.
 *
 * Only compiled/declared when GGML_USE_REEX_Q64 is defined.
 */
#pragma once

#ifdef GGML_USE_REEX_Q64

#include "reex/ggml-reex-q64-common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Target effective bit-width B for fixed-point Psum truncation, read once from
// the REEX_Q64_PSUM_BITS environment variable (cached). Returns <= 0 when unset
// or invalid, meaning truncation is disabled (default). Pair with
// reex_q64_psum_trunc_b() from ggml-reex-q64-common.h.
int reex_q64_psum_bits(void);

#include "reex/ggml-reex-q64-hw-dump.h"

// row reference (de)quantizers
void quantize_row_q4_0_64_ref(const float * GGML_RESTRICT x, block_q4_0_64 * GGML_RESTRICT y, int64_t k);
void dequantize_row_q4_0_64(const block_q4_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q8_0_64_ref(const float * GGML_RESTRICT x, block_q8_0_64 * GGML_RESTRICT y, int64_t k);
void dequantize_row_q8_0_64(const block_q8_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q4_1_64_ref(const float * GGML_RESTRICT x, block_q4_1_64 * GGML_RESTRICT y, int64_t k);
void dequantize_row_q4_1_64(const block_q4_1_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q5_0_64_ref(const float * GGML_RESTRICT x, block_q5_0_64 * GGML_RESTRICT y, int64_t k);
void dequantize_row_q5_0_64(const block_q5_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q5_1_64_ref(const float * GGML_RESTRICT x, block_q5_1_64 * GGML_RESTRICT y, int64_t k);
void dequantize_row_q5_1_64(const block_q5_1_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q8_1_64_ref(const float * GGML_RESTRICT x, block_q8_1_64 * GGML_RESTRICT y, int64_t k);
void dequantize_row_q8_1_64(const block_q8_1_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// K-quant block-64 (super-block 256, sub-block 64)
void quantize_row_q4_K_64_ref(const float * GGML_RESTRICT x, block_q4_K_64 * GGML_RESTRICT y, int64_t k);
void dequantize_row_q4_K_64(const block_q4_K_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q2_K_64_ref(const float * GGML_RESTRICT x, block_q2_K_64 * GGML_RESTRICT y, int64_t k);
void dequantize_row_q2_K_64(const block_q2_K_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q3_K_64_ref(const float * GGML_RESTRICT x, block_q3_K_64 * GGML_RESTRICT y, int64_t k);
void dequantize_row_q3_K_64(const block_q3_K_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q5_K_64_ref(const float * GGML_RESTRICT x, block_q5_K_64 * GGML_RESTRICT y, int64_t k);
void dequantize_row_q5_K_64(const block_q5_K_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q6_K_64_ref(const float * GGML_RESTRICT x, block_q6_K_64 * GGML_RESTRICT y, int64_t k);
void dequantize_row_q6_K_64(const block_q6_K_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q5_K_64S_ref(const float * GGML_RESTRICT x, block_q5_K_64S * GGML_RESTRICT y, int64_t k);
void dequantize_row_q5_K_64S(const block_q5_K_64S * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q4_K_64S_ref(const float * GGML_RESTRICT x, block_q4_K_64S * GGML_RESTRICT y, int64_t k);
void dequantize_row_q4_K_64S(const block_q4_K_64S * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_q2_K_64S_ref(const float * GGML_RESTRICT x, block_q2_K_64S * GGML_RESTRICT y, int64_t k);
void dequantize_row_q2_K_64S(const block_q2_K_64S * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// chunk quantizers for ggml_quantize_chunk() (imatrix ignored: legacy round-to-nearest)
size_t quantize_q4_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q8_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q4_1_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q5_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q5_1_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q8_1_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q4_K_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q2_K_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q3_K_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q5_K_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q6_K_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q5_K_64S(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q4_K_64S(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);
size_t quantize_q2_K_64S(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrow, int64_t n_per_row, const float * quant_weights);

#ifdef __cplusplus
}
#endif

#endif /* GGML_USE_REEX_Q64 */
