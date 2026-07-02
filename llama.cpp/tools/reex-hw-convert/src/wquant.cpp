#include "wquant.h"

#include "ggml.h"
#include "reex/ggml-reex-q64-common.h"
#include "reex/ggml-reex-q64.h"

#include <cstring>

namespace rgd {

// ---------------------------------------------------------------------------
// Encoders — reuse reex reference quantizers (row-major, native packing).
// ---------------------------------------------------------------------------
static void q6k64_encode(const float * W, void * blocks, int64_t N, int64_t K) {
    quantize_row_q6_K_64_ref(W, (block_q6_K_64 *) blocks, N * K);
}

static void q8_0_64_encode(const float * W, void * blocks, int64_t N, int64_t K) {
    quantize_row_q8_0_64_ref(W, (block_q8_0_64 *) blocks, N * K);
}

// q4_0_64 — symmetric W4: w = d*q, q signed 4-bit [-8,7] two's complement,
//   nibble-interleaved (elem e<32 -> qs[e]&0xF, e>=32 -> qs[e-32]>>4). d = -max/8.
static void q4_0_64_encode(const float * W, void * blocks, int64_t N, int64_t K) {
    quantize_row_q4_0_64_ref(W, (block_q4_0_64 *) blocks, N * K);
}

// q8_1_64s — symmetric W8 carrying a running sum s = d*sum(qs) (HW format field;
//   not used by the symmetric dot). qs sequential int8, w = d*q.
static void q8_1_64s_encode(const float * W, void * blocks, int64_t N, int64_t K) {
    quantize_row_q8_1_64_ref(W, (block_q8_1_64 *) blocks, N * K);
}

// q4_1_64 — asymmetric W4 (delta + min): w = d*q + m, q unsigned 4-bit nibble.
static void q4_1_64_encode(const float * W, void * blocks, int64_t N, int64_t K) {
    quantize_row_q4_1_64_ref(W, (block_q4_1_64 *) blocks, N * K);
}

// q5_0_64 — symmetric W5: w = d*q, q signed 5-bit [-16,15]; low4 in qs, 5th bit in qh.
static void q5_0_64_encode(const float * W, void * blocks, int64_t N, int64_t K) {
    quantize_row_q5_0_64_ref(W, (block_q5_0_64 *) blocks, N * K);
}

// q5_1_64 — asymmetric W5 (delta + min): w = d*q + m, q unsigned 5-bit (low4 + qh).
static void q5_1_64_encode(const float * W, void * blocks, int64_t N, int64_t K) {
    quantize_row_q5_1_64_ref(W, (block_q5_1_64 *) blocks, N * K);
}

// Q5_K_64S — symmetric K-quant W5. block_q5_K_64S = { d; scales[4]; qh[32]; qs[128] }.
//   sc = unpack4x6_s(scales,j),  w = d * sc * q5,  q5/sc signed two's complement.
static void q5k64s_encode(const float * W, void * blocks, int64_t N, int64_t K) {
    quantize_row_q5_K_64S_ref(W, (block_q5_K_64S *) blocks, N * K);
}

// Q2_K_64S — symmetric K-quant W2. block_q2_K_64S = { d; scales[2]; qs[64] }.
//   sc = unpack4x4_s(scales,j),  w = d * sc * q2,  q2 signed 2-bit / sc signed 4-bit.
static void q2k64s_encode(const float * W, void * blocks, int64_t N, int64_t K) {
    quantize_row_q2_K_64S_ref(W, (block_q2_K_64S *) blocks, N * K);
}

// Q3_K_64 — signed K-quant W3. block_q3_K_64 = { d; hmask[32]; qs[64]; scales[4] }.
//   sc = unpack4x6_s(scales,j),  w = d * sc * q3,  q3 signed 3-bit (low2 in qs,
//   sign bit in hmask) / sc signed 6-bit, all two's complement.
static void q3k64_encode(const float * W, void * blocks, int64_t N, int64_t K) {
    quantize_row_q3_K_64_ref(W, (block_q3_K_64 *) blocks, N * K);
}

// Q4_K_64S — symmetric K-quant W4. block_q4_K_64S = { d; scales[4]; qs[128] }.
//   sc = unpack4x6_s(scales,j),  w = d * sc * q4,  q4 signed 4-bit / sc signed 6-bit.
static void q4k64s_encode(const float * W, void * blocks, int64_t N, int64_t K) {
    quantize_row_q4_K_64S_ref(W, (block_q4_K_64S *) blocks, N * K);
}

static const WQuantType g_registry[] = {
    //  name        family          W_bits has_min block_bytes              Kt       scale_bits encode
    { "Q6_K_64",  Family::Kquant, 6, false, sizeof(block_q6_K_64),  QK_K_64, 8, q6k64_encode },
    { "Q5_K_64S", Family::Kquant, 5, false, sizeof(block_q5_K_64S), QK_K_64, 6, q5k64s_encode },
    { "Q4_K_64S", Family::Kquant, 4, false, sizeof(block_q4_K_64S), QK_K_64, 6, q4k64s_encode },
    { "Q3_K_64",  Family::Kquant, 3, false, sizeof(block_q3_K_64),  QK_K_64, 6, q3k64_encode },
    { "Q2_K_64S", Family::Kquant, 2, false, sizeof(block_q2_K_64S), QK_K_64, 4, q2k64s_encode },
    { "q8_0_64",  Family::Legacy, 8, false, sizeof(block_q8_0_64),  64,      0, q8_0_64_encode },
    { "q8_1_64s", Family::Legacy, 8, false, sizeof(block_q8_1_64),  64,      0, q8_1_64s_encode },
    { "q4_0_64",  Family::Legacy, 4, false, sizeof(block_q4_0_64),  64,      0, q4_0_64_encode },
    { "q4_1_64",  Family::Legacy, 4, true,  sizeof(block_q4_1_64),  64,      0, q4_1_64_encode },
    { "q5_0_64",  Family::Legacy, 5, false, sizeof(block_q5_0_64),  64,      0, q5_0_64_encode },
    { "q5_1_64",  Family::Legacy, 5, true,  sizeof(block_q5_1_64),  64,      0, q5_1_64_encode },
    // IntBlock — pure integer GEMM, NO scale; bit-width & signedness come from CLI
    // (--wbits/--asign/--wsign), bit-packed [16x16] blocks. Handled by intgemm.cu.
    { "INT",      Family::IntBlock, 8, false, 0,                    16,      0, nullptr },
    { "W8_16",    Family::IntBlock, 8, false, 256,                  16,      0, nullptr },
};
static const int g_registry_n = (int) (sizeof(g_registry) / sizeof(g_registry[0]));

int wquant_find(const char * name) {
    for (int i = 0; i < g_registry_n; ++i)
        if (std::strcmp(g_registry[i].name, name) == 0) return i;
    return -1;
}
const WQuantType & wquant_get(int id) { return g_registry[id]; }
int                wquant_count()     { return g_registry_n; }

size_t wquant_blocks_bytes(int id, int64_t N, int64_t K) {
    const WQuantType & t = g_registry[id];
    return (size_t) (N * K / t.elems_per_block) * t.block_bytes;
}

// ---- K-quant HW repack: glb_scale + 4*[sub_scale + 64 codes @ W bits] -------

size_t wquant_hw_block_bytes(int id) {
    const WQuantType & t = wquant_get(id);
    if (t.family != Family::Kquant) return 0;            // legacy already scale-first
    // continuous bitstream: glb(16) + 4*(scale_bits + 64*W_bits), rounded up to bytes
    const int64_t bits = 16 + 4 * (int64_t)(t.scale_bits + 64 * t.W_bits);
    return (size_t) ((bits + 7) / 8);
}

// Extract one reex super-block into raw codes[256] (signed two's-complement bit
// patterns, W_bits wide) + signed sub_scale[4] + glb scale d. The raw code bits are
// passed straight through to the HW bitstream; bit layouts mirror the reex vec_dots.
static void kq_extract_block(const WQuantType & t, const void * blk,
                             uint8_t codes[256], int8_t subsc[4], ggml_fp16_t & d) {
    if (std::strcmp(t.name, "Q6_K_64") == 0) {
        const block_q6_K_64 * b = (const block_q6_K_64 *) blk;
        d = b->d;
        for (int s = 0; s < 4; ++s) {
            subsc[s] = b->scales[s];
            const uint8_t * ql = b->ql + s * 32;
            const uint8_t * qh = b->qh + s * 16;
            for (int e = 0; e < 64; ++e) {
                const int low4 = (ql[e >> 1] >> (4 * (e & 1))) & 0xF;
                const int hi2  = (qh[e >> 2] >> (2 * (e & 3))) & 3;
                codes[s * 64 + e] = (uint8_t) (low4 | (hi2 << 4));   // signed 6-bit (two's complement)
            }
        }
    } else if (std::strcmp(t.name, "Q5_K_64S") == 0) {
        const block_q5_K_64S * b = (const block_q5_K_64S *) blk;
        d = b->d;
        for (int s = 0; s < 4; ++s) {
            subsc[s] = (int8_t) q64_unpack4x6_s(s, b->scales); // signed 6-bit two's complement
            const uint8_t * q  = b->qs + s * 32;
            const uint8_t * qh = b->qh + s * 8;
            for (int l = 0; l < 32; ++l) {
                const int hb0 = (qh[l >> 3] >> (l & 7)) & 1;
                codes[s * 64 + l]      = (uint8_t) ((q[l] & 0xF) | (hb0 << 4));  // signed 5-bit (two's complement)
                const int e   = l + 32;
                const int hb1 = (qh[e >> 3] >> (e & 7)) & 1;
                codes[s * 64 + e]      = (uint8_t) ((q[l] >> 4) | (hb1 << 4));
            }
        }
    } else if (std::strcmp(t.name, "Q4_K_64S") == 0) {
        const block_q4_K_64S * b = (const block_q4_K_64S *) blk;
        d = b->d;
        for (int s = 0; s < 4; ++s) {
            subsc[s] = (int8_t) q64_unpack4x6_s(s, b->scales); // signed 6-bit two's complement
            const uint8_t * q = b->qs + s * 32;
            for (int l = 0; l < 32; ++l) {
                codes[s * 64 + l]      = (uint8_t) (q[l] & 0xF);  // signed 4-bit (two's complement)
                codes[s * 64 + l + 32] = (uint8_t) (q[l] >> 4);
            }
        }
    } else if (std::strcmp(t.name, "Q3_K_64") == 0) {
        const block_q3_K_64 * b = (const block_q3_K_64 *) blk;
        d = b->d;
        for (int s = 0; s < 4; ++s) {
            subsc[s] = (int8_t) q64_unpack4x6_s(s, b->scales); // signed 6-bit two's complement
            for (int ii = 0; ii < 64; ++ii) {
                const int e    = s * 64 + ii;
                const int low2 = (b->qs[e >> 2] >> (2 * (e & 3))) & 3;
                const int hbit = (b->hmask[e >> 3] >> (e & 7)) & 1;
                codes[e] = (uint8_t) (low2 | (hbit << 2));  // signed 3-bit (two's complement)
            }
        }
    } else if (std::strcmp(t.name, "Q2_K_64S") == 0) {
        const block_q2_K_64S * b = (const block_q2_K_64S *) blk;
        d = b->d;
        for (int s = 0; s < 4; ++s) {
            subsc[s] = (int8_t) q64_unpack4x4_s(s, b->scales); // signed 4-bit two's complement
            for (int ii = 0; ii < 64; ++ii) {
                const int e = s * 64 + ii;
                codes[e] = (uint8_t) ((b->qs[e >> 2] >> (2 * (e & 3))) & 3);  // signed 2-bit (two's complement)
            }
        }
    }
}

// LSB-first continuous bit writer into a pre-zeroed buffer.
struct BitWriter {
    uint8_t * buf;
    int64_t   pos = 0;
    void put(uint32_t val, int bits) {
        for (int b = 0; b < bits; ++b, ++pos)
            if ((val >> b) & 1u) buf[pos >> 3] |= (uint8_t) (1u << (pos & 7));
    }
};

void wquant_repack_hw(int id, const void * reex_tiled, void * hw_tiled,
                      int64_t N, int64_t K) {
    const WQuantType & t = wquant_get(id);
    const size_t hw_bb = wquant_hw_block_bytes(id);
    if (hw_bb == 0) return;
    const uint32_t scale_mask = (t.scale_bits >= 32) ? 0xFFFFFFFFu : ((1u << t.scale_bits) - 1u);
    const uint32_t code_mask  = (1u << t.W_bits) - 1u;
    const int64_t nblk = N * K / t.elems_per_block;
    const uint8_t * src = (const uint8_t *) reex_tiled;
    uint8_t       * dst = (uint8_t *)       hw_tiled;

    for (int64_t i = 0; i < nblk; ++i) {
        uint8_t codes[256]; int8_t subsc[4]; ggml_fp16_t d;
        kq_extract_block(t, src + i * t.block_bytes, codes, subsc, d);

        uint8_t * o = dst + i * hw_bb;
        std::memset(o, 0, hw_bb);
        BitWriter bw{ o, 0 };

        uint16_t draw; std::memcpy(&draw, &d, 2);
        bw.put(draw, 16);                                       // glb_scale (fp16)
        for (int s = 0; s < 4; ++s) {
            bw.put((uint32_t) subsc[s] & scale_mask, t.scale_bits);   // signed sub_scale, two's complement
            for (int e = 0; e < 64; ++e)
                bw.put((uint32_t) codes[s * 64 + e] & code_mask, t.W_bits);  // 64 codes @ W bits
        }
    }
}

void wquant_reorder_to_tiled(int id, const void * native, void * tiled,
                             int64_t N, int64_t K, const TilingSpec & ts) {
    const WQuantType & t  = g_registry[id];
    const int64_t sb_per_row = K / t.elems_per_block;       // K-blocks per weight row
    const size_t  bb = t.block_bytes;
    const uint8_t * src = (const uint8_t *) native;          // [n][sb] row-major
    uint8_t       * dst = (uint8_t *)       tiled;

    for (int64_t n = 0; n < N; ++n)
        for (int64_t sb = 0; sb < sb_per_row; ++sb) {
            const int64_t s_idx = n * sb_per_row + sb;       // native row-major slot
            const int64_t d_idx = weight_block_slot(n, sb, N, ts);
            std::memcpy(dst + d_idx * bb, src + s_idx * bb, bb);
        }
}

} // namespace rgd
