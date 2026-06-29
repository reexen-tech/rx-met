#include "datagen.h"
#include "fp8.h"

#include "ggml.h"

#include <cmath>
#include <cstring>
#include <random>

namespace rgd {

// Round a float through the activation input dtype to simulate the source format.
static inline float round_to_act(float v, ActDType d) {
    switch (d) {
        case ActDType::F16:  return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
        case ActDType::BF16: return ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));
        case ActDType::E5M2: return fp8_round(v, fp8_e5m2());
        case ActDType::E4M3: return fp8_round(v, fp8_e4m3());
        case ActDType::F32:  default: return v;
    }
}

void datagen_fill(std::vector<float> & A, std::vector<float> & W, const GemmCase & c) {
    A.resize((size_t) (c.M * c.K));
    W.resize((size_t) (c.N * c.K));
    // A source is rounded through the activation input dtype (c.act_in); W source is
    // kept as fp16 (weight_src is fp16, then re-quantized by the reex encoder).
    std::mt19937_64 rng(c.seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & x : A) x = round_to_act(dist(rng), c.act_in);
    std::mt19937_64 rngw(c.seed ^ 0x9E3779B97F4A7C15ULL);
    std::normal_distribution<float> distw(0.0f, 0.5f);
    for (auto & x : W) x = round_to_act(distw(rngw), ActDType::F16);
}

void quantize_act(const std::vector<float> & A, int64_t M, int64_t K,
                  const TilingSpec & ts, int A_bits, ActDType act_in,
                  std::vector<uint8_t> & a_blocks) {
    const int     agroup = ts.agroup;
    const int     ebytes = act_elem_bytes(A_bits);     // 1 (A<=8) or 2 (A16)
    const int     stride = act_block_bytes(ts, A_bits);
    const int64_t ngrp   = M * K / agroup;
    a_blocks.assign((size_t) ngrp * stride, 0);

    const int qmax = (1 << (A_bits - 1)) - 1;          // A8->127, A16->32767, A4->7

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t kg = 0; kg < K / agroup; ++kg) {
            const int64_t g0 = kg * agroup;
            float amax = 0.0f;
            for (int e = 0; e < agroup; ++e) {
                const float v = round_to_act(A[(size_t) (m * K + g0 + e)], act_in);
                amax = std::fmax(amax, std::fabs(v));
            }
            const float scale = amax > 0.0f ? amax / (float) qmax : 1.0f;
            const float inv   = amax > 0.0f ? (float) qmax / amax : 0.0f;

            uint8_t * blk = a_blocks.data() + (size_t) act_group_slot(m, kg, M, K, ts) * stride;
            const ggml_fp16_t dh = ggml_fp32_to_fp16(scale);
            std::memcpy(blk, &dh, 2);
            uint8_t * qbase = blk + 2;
            for (int e = 0; e < agroup; ++e) {
                const float v = round_to_act(A[(size_t) (m * K + g0 + e)], act_in);
                int q = (int) std::lrintf(v * inv);
                if (q >  qmax) q =  qmax;
                if (q < -qmax) q = -qmax;
                if (ebytes == 2) { int16_t v16 = (int16_t) q; std::memcpy(qbase + (size_t) e * 2, &v16, 2); }
                else             { int8_t  v8  = (int8_t)  q; qbase[e] = (uint8_t) v8; }
            }
        }
    }
}

} // namespace rgd
