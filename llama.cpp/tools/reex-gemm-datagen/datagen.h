// Input generation + activation quantization (-> block_q8_1 in §4.1 order).
#pragma once

#include "case.h"

#include <cstdint>
#include <vector>

namespace rgd {

// Fill A[M*K] and W[N*K] with deterministic pseudo-random floats (case.seed).
void datagen_fill(std::vector<float> & A, std::vector<float> & W, const GemmCase & c);

// Quantize activations A[M,K] into contiguous group blocks { fp16 d; int8 qs[agroup] },
// one scale per agroup, stored in §4.1 act-block tile order. Output is a raw byte
// buffer with stride act_block_bytes(ts); use act_group_slot() to address blocks.
void quantize_act(const std::vector<float> & A, int64_t M, int64_t K,
                  const TilingSpec & ts, int A_bits, ActDType act_in,
                  std::vector<uint8_t> & a_blocks);

} // namespace rgd
