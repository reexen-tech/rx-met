// CPU golden references + comparison.
#pragma once

#include "case.h"

#include <cstdint>
#include <vector>

namespace rgd {

struct CaseError { double max_abs; double max_rel; double mse; };

// Element-wise error of a vs b over n elements.
CaseError compare(const float * a, const float * b, int64_t n);

// Integer-MAC golden (shares qmac.cuh with the GPU kernel; supports psum trunc).
// a_blocks: raw contiguous activation group buffer (act_group_slot order).
// Writes C_ref in result tile order (length M*N).
void golden_cpu_symmetric(int wtype_id,
                          const void * w_blocks,
                          const std::vector<uint8_t> & a_blocks,
                          int64_t M, int64_t N, int64_t K,
                          const TilingSpec & ts, int A_bits, int psum_bits,
                          std::vector<float> & C_ref);

// Independent float golden via reex dequantize_row_* + float matmul (psum=0 only).
// Checks the first `rows` rows of C_gpu (result-tiled). Returns max abs diff.
struct DequantCheck { double max_abs; int64_t rows; };
DequantCheck golden_dequant_check(int wtype_id,
                                  const void * w_blocks,
                                  const std::vector<uint8_t> & a_blocks,
                                  const float * C_gpu_tiled,
                                  int64_t M, int64_t N, int64_t K,
                                  const TilingSpec & ts, int A_bits, int64_t rows);

} // namespace rgd
