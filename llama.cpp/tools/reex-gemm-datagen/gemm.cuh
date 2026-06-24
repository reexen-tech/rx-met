// GPU GEMM host wrapper — dispatches to the right kernel by weight qtype.
#pragma once

#include "case.h"

#include <cstddef>
#include <cstdint>

namespace rgd {

// C[M,N] = A @ W^T on the GPU, reading packed weight blocks (§4.1 order) and
// contiguous activation group blocks (raw byte buffer, act_group_slot order);
// writes C_out in result tile order (length M*N).
void gemm_run_host(
    int wtype_id,
    const void * w_blocks,
    const uint8_t * a_blocks, size_t a_bytes,
    float * C_out,
    int64_t M, int64_t N, int64_t K,
    TilingSpec ts, int psum_bits);

} // namespace rgd
