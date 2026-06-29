// stage-1: in-chip activation quantization (device kernel).
//   A[M,K] row-major fp32 (already rounded to act_in) -> act_blocks
//   { fp16 d; intX qs[agroup] } in act_group_slot order (§4.1).
//   per group: amax(fp32) -> scale=amax/qmax (stored fp16) -> round-half-to-even
//              -> clamp(±qmax). Container int16 (A16) / int8 (A8/A4).
#pragma once

#include "case.h"

#include <cstdint>
#include <vector>

namespace rgd {

void act_quant_run_host(const float * A, int64_t M, int64_t K,
                        TilingSpec ts, int A_bits,
                        std::vector<uint8_t> & a_blocks);

} // namespace rgd
