// Fused MMA1 output-stage quantization (chain --chain qkt, docs/chain-qkt.md §3-4).
//   Reads the MMA1 fp32 accumulator C (result tile order) and emits Legacy
//   quant blocks directly — one block per (output row m1, head-dim group g):
//     kbits=8 -> block_q8_0_64 { f16 d; int8 qs[64] }          (66 B)
//     kbits=4 -> block_q4_0_64 { f16 d; nibble qs[64] }        (34 B, e<32 low / e>=32 high)
//   Quant algorithm mirrors stage-1 act quant: amax -> scale=amax/qmax (fp16)
//   -> round-half-to-even -> symmetric clamp(+/-qmax). Source is the RAW fp32
//   accumulator (no fp16 round-trip). Blocks are written in PRE-TRANSPOSE order
//   (token-major: slot = m1 * n_hd_groups + g).
// The per-group routine is RGD_HD and shared by the device kernel and the CPU
// recompute, so the GPU/CPU cross-check is bit-exact by construction.
#pragma once

#include "reex_layout.h"

#include <cstdint>
#include <vector>

namespace rgd {

// Bytes of one K block for the given kbits (66 for INT8, 34 for INT4).
int kblock_bytes(int kbits);

// GPU fused output quantization. C_tiled: host fp32 MMA1 result, result tile
// order (length M*N, tiling ts). Emits M*(N/64) blocks in pre-transpose
// (token-major) order into k_blocks.
void out_quant_run_host(const float * C_tiled, int64_t M, int64_t N,
                        TilingSpec ts, int kbits,
                        std::vector<uint8_t> & k_blocks);

// CPU recompute of the same fused quantization (same RGD_HD routine, same fp32
// input) — must match out_quant_run_host byte-for-byte.
void out_quant_cpu(const float * C_tiled, int64_t M, int64_t N,
                   const TilingSpec & ts, int kbits,
                   std::vector<uint8_t> & k_blocks);

// Block-atomic grid transpose (models the HW block-transpose unit): whole
// blocks move, group contents untouched.
//   pre  : [num_keys x n_hd_groups] token-major   (slot = key*n_hd_groups + g)
//   post : [n_hd_groups x num_keys] hd-major      (slot = g*num_keys + key)
// The post order equals MMA2's weight_block_slot order for Nt=64.
void kblocks_grid_transpose(const std::vector<uint8_t> & pre,
                            std::vector<uint8_t> & post,
                            int64_t num_keys, int64_t n_hd_groups,
                            int block_bytes);

} // namespace rgd
