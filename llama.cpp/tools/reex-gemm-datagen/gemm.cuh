// GPU GEMM host wrapper — dispatches to the right kernel by weight qtype.
#pragma once

#include "case.h"

#include <cstddef>
#include <cstdint>

namespace rgd {

// C[M,N] = A @ W^T on the GPU, reading packed weight blocks (§4.1 order) and
// contiguous activation group blocks (raw byte buffer, act_group_slot order).
// Writes C_out (fp32 acc, result tile order, length M*N) AND, via in-chip OutConv,
// the 6 dtype outputs into out_bufs[0..5] (order: F16,BF16,I16,I8,I6,I4; each a
// host buffer of M*N*out_specs()[s].bytes). out_bufs may be null to skip.
void gemm_run_host(
    int wtype_id,
    const void * w_blocks,
    const uint8_t * a_blocks, size_t a_bytes,
    float * C_out,
    uint8_t * const out_bufs[6],
    int64_t M, int64_t N, int64_t K,
    TilingSpec ts, int A_bits, int psum_bits);

} // namespace rgd
