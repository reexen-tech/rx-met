// SPDX-License-Identifier: MIT
//
// TurboQuant algorithm reference implementation (oracle).
//
// This file is the authoritative bit-level implementation of the TurboQuant
// quantization algorithm used as `golden` in tests under `tests/reex/`. It is
// extracted from the original `src/reex/reex_turboquant_core.{h,cpp}` of the
// pre-v2 prototype branch, with all runtime/integration glue
// (`CompressedKVStore`, `KVCaptureEngine`, `reex_turboquant_attention_compute`,
// arena management, ggml backend buft hooks) stripped away.
//
// Constraints:
//   - **Test-only**.  This translation unit is compiled exclusively by
//     `tests/oracle/CMakeLists.txt` (when REEX_TURBOQUANT=ON and tests are
//     enabled) and must NEVER be linked into `libllama` / `libggml` or any
//     other production target.
//   - **Pure host C++17**.  No dependency on ggml, ggml-cuda, or any backend.
//   - **Bit-level matched** to the Python reference at
//     `/home/zcx/CLionProjects/turboquant`.  Any change here must come with
//     a corresponding bump of golden fixtures and an explicit note in
//     `docs/turboquant/01_设计与实现计划.md` §6.
//
// Namespace:
//   The oracle lives in `reex_turboquant_oracle::` (not `reex_turboquant::`)
//   so that production code in `src/reex/` and `ggml/src/ggml-*/reex/` can
//   freely use the shorter `reex_turboquant::` namespace without symbol
//   collisions when both are linked into the same test binary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace reex_turboquant_oracle {

struct LayerConfig {
    int32_t head_dim = 128;
    int32_t num_kv_heads = 1;
    int32_t num_query_heads = 1;
    int32_t key_bits = 3;
    int32_t value_bits = 2;
    int32_t value_group_size = 32;
    int32_t layer_idx = 0;
    int32_t seed = 42;
};

struct Codebook {
    int32_t dim = 0;
    int32_t bits = 0;
    std::vector<float> centroids;
    std::vector<float> boundaries;
    std::vector<float> decision_boundaries;
};

struct QuantizerResources {
    Codebook mse_codebook;
    std::vector<float> rotation;
    std::vector<float> qjl;
};

struct MSEQuantized {
    std::vector<uint8_t> indices;
    std::vector<float>   norms;
    int32_t bits   = 0;
    int32_t dim    = 0;
    int32_t batch  = 0;
    int32_t heads  = 0;
    int32_t tokens = 0;
};

// View types pointing into externally-managed memory.  Layout convention is
// head-major: row index r = h * row_stride + t, where `row_stride` is the
// stride between adjacent heads measured in tokens.  For owning containers
// produced by `quantize()` below, `row_stride == tokens` (compact); the
// `row_stride > tokens` case (arena views over fixed-capacity stores) is
// kept for parity with the pre-v2 `CompressedKVStore` byte layout that some
// existing fixtures still encode.
struct ProdQuantizedView {
    const uint8_t * mse_indices    = nullptr;
    const uint8_t * qjl_signs      = nullptr;
    const float *   residual_norms = nullptr;
    const float *   norms          = nullptr;
    int32_t mse_bits   = 0;
    int32_t dim        = 0;
    int32_t batch      = 0;
    int32_t heads      = 0;
    int32_t tokens     = 0;
    int32_t row_stride = 0;
};

struct ValueQuantizedView {
    const uint8_t * data   = nullptr;
    const float *   scales = nullptr;
    const float *   zeros  = nullptr;
    int32_t bits       = 2;
    int32_t group_size = 32;
    int32_t dim        = 0;
    int32_t batch      = 0;
    int32_t heads      = 0;
    int32_t tokens     = 0;
    int32_t row_stride = 0;
};

struct ProdQuantized {
    std::vector<uint8_t> mse_indices;
    std::vector<uint8_t> qjl_signs;
    std::vector<float>   residual_norms;
    std::vector<float>   norms;
    int32_t mse_bits = 0;
    int32_t dim      = 0;
    int32_t batch    = 0;
    int32_t heads    = 0;
    int32_t tokens   = 0;

    ProdQuantizedView view() const {
        ProdQuantizedView v;
        v.mse_indices    = mse_indices.data();
        v.qjl_signs      = qjl_signs.data();
        v.residual_norms = residual_norms.data();
        v.norms          = norms.data();
        v.mse_bits   = mse_bits;
        v.dim        = dim;
        v.batch      = batch;
        v.heads      = heads;
        v.tokens     = tokens;
        v.row_stride = tokens;
        return v;
    }
};

struct ValueQuantized {
    std::vector<uint8_t> data;
    std::vector<float>   scales;
    std::vector<float>   zeros;
    int32_t bits       = 2;
    int32_t group_size = 32;
    int32_t dim        = 0;
    int32_t batch      = 0;
    int32_t heads      = 0;
    int32_t tokens     = 0;

    ValueQuantizedView view() const {
        ValueQuantizedView v;
        v.data       = data.data();
        v.scales     = scales.data();
        v.zeros      = zeros.data();
        v.bits       = bits;
        v.group_size = group_size;
        v.dim        = dim;
        v.batch      = batch;
        v.heads      = heads;
        v.tokens     = tokens;
        v.row_stride = tokens;
        return v;
    }
};

class TurboQuantMSE {
public:
    TurboQuantMSE(int32_t dim, int32_t bits, int32_t seed = 42);
    TurboQuantMSE(int32_t dim, int32_t bits, const Codebook & codebook, const std::vector<float> & rotation);

    MSEQuantized       quantize  (const float * x, int32_t batch, int32_t heads, int32_t tokens) const;
    std::vector<float> dequantize(const MSEQuantized & q) const;

    int32_t                    dim()       const { return dim_; }
    int32_t                    bits()      const { return bits_; }
    const Codebook &           codebook()  const { return codebook_; }
    const std::vector<float> & rotation()  const { return pi_; }

private:
    int32_t            dim_;
    int32_t            bits_;
    Codebook           codebook_;
    std::vector<float> pi_;
};

// TurboQuant — Algorithm 1 (PolarQuant only) with **norm correction**.
// This is the algorithm landed by upstream `llama-cpp-turboquant`'s
// `turbo3_0` path (ggml/src/ggml-turbo-quant.c::quantize_row_turbo3_0_ref);
// see docs/turboquant/p4.a22_k3_p3_upgrade_plan.md §0 for the choice of
// this path over Algorithm 2 (Prod + QJL) for the K3 P3 upgrade.
//
// Difference from `TurboQuantMSE`: at quantize time the per-row `norms`
// field stores `corrected = ‖x‖ / ‖recon_unit‖` (rather than raw ‖x‖)
// so dequantize output ‖y‖ ≡ original ‖x‖ — strict idempotency.
//
// Reuses `MSEQuantized` as the on-disk record because its layout
// (uint8 indices + per-row scalar `norms`) already matches what the
// production `block_tq_k3` carries; the only semantic change is the
// `norms[r]` field's interpretation (raw → corrected).
class TurboQuantPolar {
public:
    TurboQuantPolar(int32_t dim, int32_t bits, int32_t seed = 42);
    TurboQuantPolar(int32_t dim, int32_t bits, const Codebook & codebook, const std::vector<float> & rotation);

    MSEQuantized       quantize  (const float * x, int32_t batch, int32_t heads, int32_t tokens) const;
    std::vector<float> dequantize(const MSEQuantized & q) const;

    int32_t                    dim()       const { return dim_; }
    int32_t                    bits()      const { return bits_; }
    const Codebook &           codebook()  const { return codebook_; }
    const std::vector<float> & rotation()  const { return pi_; }

private:
    int32_t            dim_;
    int32_t            bits_;
    Codebook           codebook_;
    std::vector<float> pi_;
};

class TurboQuantProd {
public:
    TurboQuantProd(int32_t dim, int32_t bits, int32_t seed = 42);
    TurboQuantProd(int32_t dim, int32_t bits, const QuantizerResources & resources);

    ProdQuantized      quantize  (const float * x, int32_t batch, int32_t heads, int32_t tokens) const;
    std::vector<float> dequantize(const ProdQuantizedView & v) const;
    std::vector<float> dequantize(const ProdQuantized & q) const { return dequantize(q.view()); }

    // QK^T dot products (no softmax / scale / mask), shape:
    // (batch, q_heads, q_tokens, kv_tokens) with GQA expansion.  Pure
    // reference; tests pair this with their own softmax for comparing against
    // ggml flash-attention output.
    std::vector<float> attention_score(
        const float * query,
        int32_t batch,
        int32_t q_heads,
        int32_t q_tokens,
        const ProdQuantizedView & key_v,
        int32_t kv_heads
    ) const;
    std::vector<float> attention_score(
        const float * query,
        int32_t batch,
        int32_t q_heads,
        int32_t q_tokens,
        const ProdQuantized & key_q,
        int32_t kv_heads
    ) const {
        return attention_score(query, batch, q_heads, q_tokens, key_q.view(), kv_heads);
    }

    int32_t dim()  const { return dim_; }
    int32_t bits() const { return bits_; }

private:
    int32_t            dim_;
    int32_t            bits_;
    TurboQuantMSE      mse_;
    std::vector<float> s_;
    float              qjl_scale_;
};

// Free-standing accessors (cached internally with std::map + std::mutex so
// repeated calls are cheap).
Codebook              reex_turboquant_oracle_codebook_get(int32_t dim, int32_t bits);
std::vector<float>    reex_turboquant_oracle_rotation_get(int32_t dim, int32_t seed, int32_t layer_idx);
std::vector<float>    reex_turboquant_oracle_qjl_get     (int32_t dim, int32_t seed, int32_t layer_idx);

// Per-channel min/max scalar quantization of V tensors.  `dim % group_size`
// must be 0; `levels = (1 << bits) - 1`.
ValueQuantized        reex_turboquant_oracle_value_quantize(
    const float * v,
    int32_t batch,
    int32_t heads,
    int32_t tokens,
    int32_t dim,
    int32_t bits,
    int32_t group_size);

std::vector<float>    reex_turboquant_oracle_value_dequantize(const ValueQuantizedView & vq);
inline std::vector<float> reex_turboquant_oracle_value_dequantize(const ValueQuantized & vq) {
    return reex_turboquant_oracle_value_dequantize(vq.view());
}

// Materialise an owning ProdQuantized / ValueQuantized that contains the
// head-major slice [heads × tokens] in a compact (row_stride == tokens)
// layout.  Useful for tests that need to compare against pre-baked golden
// fixtures or for serialisation paths that expect contiguous vectors.
ProdQuantized         reex_turboquant_oracle_compact(const ProdQuantizedView  & v);
ValueQuantized        reex_turboquant_oracle_compact(const ValueQuantizedView & v);

}  // namespace reex_turboquant_oracle
