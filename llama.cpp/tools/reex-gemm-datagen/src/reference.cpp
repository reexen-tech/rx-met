#include "reference.h"
#include "qmac.cuh"                       // shared MACs (mirror reex vec_dot)
#include "wquant.h"                       // wquant_get / WQuantType

#include "ggml.h"                         // ggml_fp16_to_fp32
#include "reex/ggml-reex-q64-common.h"   // block_q6_K_64, block_q8_0_64, block_q5_K_64S
#include "reex/ggml-reex-q64.h"          // dequantize_row_* (reex reference)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace rgd {

CaseError compare(const float * a, const float * b, int64_t n) {
    double max_abs = 0.0, max_rel = 0.0, se = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        max_abs = std::max(max_abs, d);
        const double den = std::max(std::fabs((double) b[i]), 1e-6);
        max_rel = std::max(max_rel, d / den);
        se += d * d;
    }
    return CaseError{ max_abs, max_rel, se / (double) (n > 0 ? n : 1) };
}

void golden_cpu_symmetric(int wtype_id,
                          const void * w_blocks,
                          const std::vector<uint8_t> & a_blocks,
                          int64_t M, int64_t N, int64_t K,
                          const TilingSpec & ts, int A_bits, int psum_bits,
                          std::vector<float> & C_ref_tiled) {
    const WQuantType & wt = wquant_get(wtype_id);
    C_ref_tiled.assign((size_t) (M * N), 0.0f);

    if (wt.family == Family::Kquant && std::strcmp(wt.name, "Q6_K_64") == 0) {
        const block_q6_K_64 * wb = (const block_q6_K_64 *) w_blocks;
        const int64_t sb_per_row = K / QK_K_64;
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                    const block_q6_K_64 & w = wb[weight_block_slot(n, sb, N, ts)];
                    acc += (double) rgd_q6k64_dot_superblock(w, a_blocks.data(), m, sb, M, K, ts, A_bits, psum_bits);
                }
                C_ref_tiled[result_tiled_index(m, n, M, ts)] = (float) acc;
            }
    } else if (wt.family == Family::Kquant && std::strcmp(wt.name, "Q5_K_64S") == 0) {
        const block_q5_K_64S * wb = (const block_q5_K_64S *) w_blocks;
        const int64_t sb_per_row = K / QK_K_64;
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                    const block_q5_K_64S & w = wb[weight_block_slot(n, sb, N, ts)];
                    acc += (double) rgd_q5k64s_dot_superblock(w, a_blocks.data(), m, sb, M, K, ts, A_bits, psum_bits);
                }
                C_ref_tiled[result_tiled_index(m, n, M, ts)] = (float) acc;
            }
    } else if (wt.family == Family::Kquant && std::strcmp(wt.name, "Q4_K_64S") == 0) {
        const block_q4_K_64S * wb = (const block_q4_K_64S *) w_blocks;
        const int64_t sb_per_row = K / QK_K_64;
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                    const block_q4_K_64S & w = wb[weight_block_slot(n, sb, N, ts)];
                    acc += (double) rgd_q4k64s_dot_superblock(w, a_blocks.data(), m, sb, M, K, ts, A_bits, psum_bits);
                }
                C_ref_tiled[result_tiled_index(m, n, M, ts)] = (float) acc;
            }
    } else if (wt.family == Family::Kquant && std::strcmp(wt.name, "Q3_K_64") == 0) {
        const block_q3_K_64 * wb = (const block_q3_K_64 *) w_blocks;
        const int64_t sb_per_row = K / QK_K_64;
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                    const block_q3_K_64 & w = wb[weight_block_slot(n, sb, N, ts)];
                    acc += (double) rgd_q3k64_dot_superblock(w, a_blocks.data(), m, sb, M, K, ts, A_bits, psum_bits);
                }
                C_ref_tiled[result_tiled_index(m, n, M, ts)] = (float) acc;
            }
    } else if (wt.family == Family::Kquant && std::strcmp(wt.name, "Q2_K_64S") == 0) {
        const block_q2_K_64S * wb = (const block_q2_K_64S *) w_blocks;
        const int64_t sb_per_row = K / QK_K_64;
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                    const block_q2_K_64S & w = wb[weight_block_slot(n, sb, N, ts)];
                    acc += (double) rgd_q2k64s_dot_superblock(w, a_blocks.data(), m, sb, M, K, ts, A_bits, psum_bits);
                }
                C_ref_tiled[result_tiled_index(m, n, M, ts)] = (float) acc;
            }
    } else if (wt.family == Family::Legacy && std::strcmp(wt.name, "q4_0_64") == 0) {
        const block_q4_0_64 * wb = (const block_q4_0_64 *) w_blocks;
        const int64_t kb_per_row = K / 64;
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int64_t kbw = 0; kbw < kb_per_row; ++kbw) {
                    const block_q4_0_64 & w = wb[weight_block_slot(n, kbw, N, ts)];
                    acc += (double) rgd_q4_0_64_dot_block(w, a_blocks.data(), m, kbw, M, K, ts, A_bits, psum_bits);
                }
                C_ref_tiled[result_tiled_index(m, n, M, ts)] = (float) acc;
            }
    } else if (wt.family == Family::Legacy && std::strcmp(wt.name, "q8_1_64s") == 0) {
        const block_q8_1_64 * wb = (const block_q8_1_64 *) w_blocks;
        const int64_t kb_per_row = K / 64;
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int64_t kbw = 0; kbw < kb_per_row; ++kbw) {
                    const block_q8_1_64 & w = wb[weight_block_slot(n, kbw, N, ts)];
                    acc += (double) rgd_q8_1_64s_dot_block(w, a_blocks.data(), m, kbw, M, K, ts, A_bits, psum_bits);
                }
                C_ref_tiled[result_tiled_index(m, n, M, ts)] = (float) acc;
            }
    } else { // Legacy q8_0_64
        const block_q8_0_64 * wb = (const block_q8_0_64 *) w_blocks;
        const int64_t kb_per_row = K / 64;
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int64_t kbw = 0; kbw < kb_per_row; ++kbw) {
                    const block_q8_0_64 & w = wb[weight_block_slot(n, kbw, N, ts)];
                    acc += (double) rgd_q8_0_64_dot_block(w, a_blocks.data(), m, kbw, M, K, ts, A_bits, psum_bits);
                }
                C_ref_tiled[result_tiled_index(m, n, M, ts)] = (float) acc;
            }
    }
}

// Dequantize the full weight matrix back to row-major Wf[N*K] via reex.
static void dequant_weight_full(int wtype_id, const void * w_blocks,
                                int64_t N, int64_t K, const TilingSpec & ts,
                                std::vector<float> & Wf) {
    const WQuantType & wt = wquant_get(wtype_id);
    Wf.resize((size_t) N * K);
    if (wt.family == Family::Kquant && std::strcmp(wt.name, "Q6_K_64") == 0) {
        const block_q6_K_64 * wb = (const block_q6_K_64 *) w_blocks;
        const int64_t sb_per_row = K / QK_K_64;
        std::vector<float> tmp(QK_K_64);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                dequantize_row_q6_K_64(&wb[weight_block_slot(n, sb, N, ts)], tmp.data(), QK_K_64);
                std::copy(tmp.begin(), tmp.end(), Wf.begin() + (size_t) (n * K + sb * QK_K_64));
            }
    } else if (wt.family == Family::Kquant && std::strcmp(wt.name, "Q5_K_64S") == 0) {
        const block_q5_K_64S * wb = (const block_q5_K_64S *) w_blocks;
        const int64_t sb_per_row = K / QK_K_64;
        std::vector<float> tmp(QK_K_64);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                dequantize_row_q5_K_64S(&wb[weight_block_slot(n, sb, N, ts)], tmp.data(), QK_K_64);
                std::copy(tmp.begin(), tmp.end(), Wf.begin() + (size_t) (n * K + sb * QK_K_64));
            }
    } else if (wt.family == Family::Kquant && std::strcmp(wt.name, "Q4_K_64S") == 0) {
        const block_q4_K_64S * wb = (const block_q4_K_64S *) w_blocks;
        const int64_t sb_per_row = K / QK_K_64;
        std::vector<float> tmp(QK_K_64);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                dequantize_row_q4_K_64S(&wb[weight_block_slot(n, sb, N, ts)], tmp.data(), QK_K_64);
                std::copy(tmp.begin(), tmp.end(), Wf.begin() + (size_t) (n * K + sb * QK_K_64));
            }
    } else if (wt.family == Family::Kquant && std::strcmp(wt.name, "Q3_K_64") == 0) {
        const block_q3_K_64 * wb = (const block_q3_K_64 *) w_blocks;
        const int64_t sb_per_row = K / QK_K_64;
        std::vector<float> tmp(QK_K_64);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                dequantize_row_q3_K_64(&wb[weight_block_slot(n, sb, N, ts)], tmp.data(), QK_K_64);
                std::copy(tmp.begin(), tmp.end(), Wf.begin() + (size_t) (n * K + sb * QK_K_64));
            }
    } else if (wt.family == Family::Kquant && std::strcmp(wt.name, "Q2_K_64S") == 0) {
        const block_q2_K_64S * wb = (const block_q2_K_64S *) w_blocks;
        const int64_t sb_per_row = K / QK_K_64;
        std::vector<float> tmp(QK_K_64);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t sb = 0; sb < sb_per_row; ++sb) {
                dequantize_row_q2_K_64S(&wb[weight_block_slot(n, sb, N, ts)], tmp.data(), QK_K_64);
                std::copy(tmp.begin(), tmp.end(), Wf.begin() + (size_t) (n * K + sb * QK_K_64));
            }
    } else if (wt.family == Family::Legacy && std::strcmp(wt.name, "q4_0_64") == 0) {
        const block_q4_0_64 * wb = (const block_q4_0_64 *) w_blocks;
        const int64_t kb_per_row = K / 64;
        std::vector<float> tmp(64);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t kbw = 0; kbw < kb_per_row; ++kbw) {
                dequantize_row_q4_0_64(&wb[weight_block_slot(n, kbw, N, ts)], tmp.data(), 64);
                std::copy(tmp.begin(), tmp.end(), Wf.begin() + (size_t) (n * K + kbw * 64));
            }
    } else if (wt.family == Family::Legacy && std::strcmp(wt.name, "q8_1_64s") == 0) {
        const block_q8_1_64 * wb = (const block_q8_1_64 *) w_blocks;
        const int64_t kb_per_row = K / 64;
        std::vector<float> tmp(64);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t kbw = 0; kbw < kb_per_row; ++kbw) {
                dequantize_row_q8_1_64(&wb[weight_block_slot(n, kbw, N, ts)], tmp.data(), 64);
                std::copy(tmp.begin(), tmp.end(), Wf.begin() + (size_t) (n * K + kbw * 64));
            }
    } else { // Legacy q8_0_64
        const block_q8_0_64 * wb = (const block_q8_0_64 *) w_blocks;
        const int64_t kb_per_row = K / 64;
        std::vector<float> tmp(64);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t kbw = 0; kbw < kb_per_row; ++kbw) {
                dequantize_row_q8_0_64(&wb[weight_block_slot(n, kbw, N, ts)], tmp.data(), 64);
                std::copy(tmp.begin(), tmp.end(), Wf.begin() + (size_t) (n * K + kbw * 64));
            }
    }
}

DequantCheck golden_dequant_check(int wtype_id,
                                  const void * w_blocks,
                                  const std::vector<uint8_t> & a_blocks,
                                  const float * C_gpu_tiled,
                                  int64_t M, int64_t N, int64_t K,
                                  const TilingSpec & ts, int A_bits, int64_t rows) {
    rows = std::min(rows, M);
    std::vector<float> Wf;
    dequant_weight_full(wtype_id, w_blocks, N, K, ts, Wf);

    const int     agroup   = ts.agroup;
    const int     stride   = act_block_bytes(ts, A_bits);
    const int64_t kg_per_row = K / agroup;
    double max_abs = 0.0;
    std::vector<float> Arow((size_t) K);
    for (int64_t m = 0; m < rows; ++m) {
        for (int64_t kg = 0; kg < kg_per_row; ++kg) {
            const uint8_t * blk = a_blocks.data() + (size_t) act_group_slot(m, kg, M, K, ts) * stride;
            ggml_fp16_t dh; std::memcpy(&dh, blk, 2);
            const float ad = rgd_h2f(dh);
            const uint8_t * qs = blk + 2;
            for (int j = 0; j < agroup; ++j)
                Arow[(size_t) (kg * agroup + j)] = ad * (float) rgd_act_at(qs, j, A_bits);
        }
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            const float * wf = &Wf[(size_t) n * K];
            for (int64_t k = 0; k < K; ++k) acc += (double) Arow[(size_t) k] * (double) wf[k];
            const double diff = std::fabs(acc - (double) C_gpu_tiled[result_tiled_index(m, n, M, ts)]);
            max_abs = std::max(max_abs, diff);
        }
    }
    return DequantCheck{ max_abs, rows };
}

} // namespace rgd
