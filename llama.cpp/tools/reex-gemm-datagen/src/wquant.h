// Weight quantization type registry — one entry per weight qtype, so new quant
// strategies plug in by adding a row (encoder + block metadata).
#pragma once

#include "case.h"

#include <cstddef>
#include <cstdint>

namespace rgd {

struct WQuantType {
    const char * name;
    Family       family;
    int          W_bits;
    bool         has_min;
    size_t       block_bytes;       // sizeof(reex block struct)
    int          elems_per_block;   // K-elements per block (== TilingSpec.Kt)
    int          scale_bits;        // K-quant HW sub_scale bit-width (Q6:8, Q5S/Q4S:6, Q2S:4)
    void (*encode)(const float * W, void * blocks, int64_t N, int64_t K);
};

// Registry lookup.
int                 wquant_find(const char * name);   // -1 if not found
const WQuantType &  wquant_get(int id);
int                 wquant_count();

// Total bytes of the packed weight buffer for a [N,K] matrix.
size_t wquant_blocks_bytes(int id, int64_t N, int64_t K);

// Copy packed blocks from native row-major order into §4.1 block-tile order
// (weight_block_slot). Both buffers hold wquant_blocks_bytes() bytes.
void wquant_reorder_to_tiled(int id, const void * native, void * tiled,
                             int64_t N, int64_t K, const TilingSpec & ts);

// HW dump layout for K-quant: re-pack each reex super-block into a continuous
// LSB-first bitstream
//   glb_scale(fp16,16b) + 4 x [ sub_scale(scale_bits, signed) + 64 codes @ W bits ]
// (scale-first, sub-block interleaved; whole block bit-packed). Returns 0 for
// families that need no repack (Legacy is already scale-first as reex {d; qs}).
size_t wquant_hw_block_bytes(int id);
void   wquant_repack_hw(int id, const void * reex_tiled, void * hw_tiled,
                        int64_t N, int64_t K);

} // namespace rgd
