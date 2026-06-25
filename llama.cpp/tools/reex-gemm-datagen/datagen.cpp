#include "datagen.h"

#include "ggml.h"

#include <cmath>
#include <cstring>
#include <random>

namespace rgd {

// Round a float through the activation input dtype to simulate FP16/BF16 inputs.
static inline float round_to_act(float v, ActDType d) {
    switch (d) {
        case ActDType::F16:  return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
        case ActDType::BF16: return ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));
        case ActDType::F32:  default: return v;
    }
}

void datagen_fill(std::vector<float> & A, std::vector<float> & W, const GemmCase & c) {
    A.resize((size_t) (c.M * c.K));
    W.resize((size_t) (c.N * c.K));
    // Inputs are FP16 source data: round every generated value through fp16 so the
    // float buffers hold exactly representable fp16 values (act_in == F16 here).
    std::mt19937_64 rng(c.seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & x : A) x = round_to_act(dist(rng), ActDType::F16);
    std::mt19937_64 rngw(c.seed ^ 0x9E3779B97F4A7C15ULL);
    std::normal_distribution<float> distw(0.0f, 0.5f);
    for (auto & x : W) x = round_to_act(distw(rngw), ActDType::F16);
}

void quantize_act(const std::vector<float> & A, int64_t M, int64_t K,
                  const TilingSpec & ts, int A_bits, ActDType act_in,
                  std::vector<uint8_t> & a_blocks) {
    const int     agroup = ts.agroup;
    const int     stride = act_block_bytes(ts);       // 2 + agroup
    const int64_t ngrp   = M * K / agroup;
    a_blocks.assign((size_t) ngrp * stride, 0);

    const int qmax = (1 << (A_bits - 1)) - 1;          // A_bits=8 -> 127

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
            int8_t * qs = (int8_t *) (blk + 2);
            for (int e = 0; e < agroup; ++e) {
                const float v = round_to_act(A[(size_t) (m * K + g0 + e)], act_in);
                int q = (int) std::lrintf(v * inv);
                if (q >  qmax) q =  qmax;
                if (q < -qmax) q = -qmax;
                qs[e] = (int8_t) q;
            }
        }
    }
}

} // namespace rgd
