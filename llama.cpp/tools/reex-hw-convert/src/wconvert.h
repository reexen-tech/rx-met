// Offline single-tensor weight converter: native (GGUF / reex row-major struct)
// -> hardware weight_blocks layout (§4.1 tile order). CPU-only, no CUDA.
//
// Reuses the datagen-verified wquant primitives (reorder / repack) so there is a
// single implementation shared with reex-gemm-datagen (no drift).
//
// Phase 1 scope: Legacy block-64 quant (q8_0_64 / q8_1_64s / q4_0_64 / q4_1_64 /
// q5_0_64 / q5_1_64) — a pure whole-block reorder into §4.1 tile order (block
// bytes are opaque). The machinery is generic: K-quant would additionally hit the
// HW bitstream repack automatically (hw_block_bytes != 0), but the CLI gates to
// Legacy for now.
#pragma once

#include "wquant.h"
#include "reex_layout.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rgd {

// Result of converting one weight tensor into the HW weight_blocks layout.
struct WConvResult {
    std::vector<uint8_t> bytes;        // weight_blocks payload, §4.1 tile order
    size_t               block_bytes = 0;
    int64_t              n_blocks    = 0;
    bool                 repacked    = false; // true when K-quant HW bitstream
    std::string          layout_desc;
};

// Encode fp32 W[N,K] (row-major, K contiguous) into native reex blocks using the
// registry encoder for `wtype_id`. `native` is sized to wquant_blocks_bytes.
void wconvert_encode_native(int wtype_id, const float * W, int64_t N, int64_t K,
                            std::vector<uint8_t> & native);

// Fill W[N,K] with deterministic pseudo-random fp16-rounded floats (same recipe
// as datagen_fill's weight path) for the --random input mode.
void wconvert_random_weights(std::vector<float> & W, int64_t N, int64_t K,
                             uint64_t seed);

// Convert native (row-major reex blocks) -> HW weight_blocks layout.
//   Legacy : §4.1 whole-block reorder (reex struct passthrough).
//   K-quant: reorder + HW bitstream repack (auto when hw_block_bytes != 0).
// `native_bytes` must equal wquant_blocks_bytes(wtype_id, N, K).
WConvResult wconvert_weight(int wtype_id, const void * native, size_t native_bytes,
                            int64_t N, int64_t K);

// Self-describing meta.json (as a string) for a converted tensor.
std::string wconvert_meta_json(int wtype_id, int64_t N, int64_t K,
                               const WConvResult & r);

} // namespace rgd
