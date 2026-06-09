// Mini "KV-cache-shaped" smoke test for TurboQuant types.
//
// We don't load a real llama_model here; full end-to-end exercise of
// llama_kv_cache is part of the Phase 2 exit acceptance (CPU llama-cli +
// Qwen3-1.7B).  Instead, we allocate ggml tensors at Qwen3-1.7B-shaped
// dimensions in TQ_K3 / TQ_V2 / TQ_V4 and put them through the same
// graph ops that llama_kv_cache uses internally:
//
//   * `ggml_cpy` of an F32 source into a 3-D TQ tensor (mimics
//     llama_kv_cache::cpy_k / cpy_v "write into cell").
//   * `ggml_get_rows` from a 2-D TQ tensor (mimics
//     llama_kv_cache::get_k / get_v "view into cells [head, head+n_tokens)").
//
// We then dequantize back to F32 on CPU and check the round-trip is
// reasonable.  The thresholds were tightened in P4 A2.1 (Apr 2026, see
// `docs/turboquant/p4.a21_stage0_algorithm_diagnostics_report.md` §5.1)
// to match the upstream `scos-lab/turboquant` reference and stay aligned
// with `test-reex-turboquant-block-roundtrip`:
//   - TQ_K3 ≥ 0.99 (upstream 1.000000 with full QJL)
//   - TQ_V2 ≥ 0.93 (upstream 0.940 paper-marginal tier)
//   - TQ_V4 ≥ 0.99 (upstream 0.997 paper-neutrality tier)
// CPY → dequantize is exactly the same algorithm path as block-roundtrip,
// so the gates must agree row-for-row.  The previous laxer 0.85/0.93 gates
// were intentionally keeping ~0.07 of headroom that re-hid the K3 P1
// simplification quality gap; that headroom is now removed.
//
// Why this is enough for P2.3:
//   - resolve_layer_spec correctness: covered by test-reex-turboquant-layer-spec.
//   - TQ kernel correctness:           covered by test-reex-turboquant-vs-oracle.
//   - WHT graph op:                    covered by test-reex-turboquant-backend-op-wht.
//   - llama_kv_cache::seq_rm/cp/keep:  pure metadata on llama_kv_cells, type-agnostic.
// What remained was the missing smoke-test that the *3-D layouts*
// llama_kv_cache asks ggml to allocate (n_embd_gqa × kv_size × 1, etc.)
// are actually buildable for TQ_K3/TQ_V2/TQ_V4 and survive a write + read
// cycle through `ggml_cpy` and `ggml_get_rows` on the CPU backend.  That
// is what this file checks.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Reproducible random F32 matrix
// ---------------------------------------------------------------------------

std::vector<float> rand_f32(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

float cosine(const std::vector<float> & a, const std::vector<float> & b) {
    assert(a.size() == b.size());
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na  += (double) a[i] * a[i];
        nb  += (double) b[i] * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0f;
    return (float) (dot / (std::sqrt(na) * std::sqrt(nb)));
}

// ---------------------------------------------------------------------------
// Test parameters chosen to mimic Qwen3-1.7B (head_dim=128, n_head_kv=8,
// kv_size=64 for the test).  TQ_K3 needs row_size % 128 == 0 (head_dim *
// some n_head_kv works), TQ_V2/V4 need row_size % 32 == 0.
// ---------------------------------------------------------------------------

struct shape {
    int head_dim;
    int n_head_kv;
    int kv_size;
};

const shape S = { /*head_dim=*/128, /*n_head_kv=*/8, /*kv_size=*/64 };

// ---------------------------------------------------------------------------
// CPY round-trip: allocate dst tensor of the given TQ type and shape
// [row_dim, kv_size, 1] (matches K-cache layout), copy an F32 source into it
// via ggml_cpy on CPU backend, then dequantize back to F32 and compare.
// ---------------------------------------------------------------------------

bool test_cpy_round_trip(ggml_type tq_type, int row_dim, int kv_size, const char * tag) {
    const int n_elem = row_dim * kv_size;
    const auto src_data = rand_f32(n_elem, /*seed=*/0xC0FFEEu ^ (uint32_t) tq_type);

    // ggml_backend_alloc_ctx_tensors() requires no_alloc=true so it can manage
    // backend buffer placement itself; otherwise tensor data has already been
    // allocated in the context's own internal buffer and the assert fires.
    const size_t mem_size = (size_t) 64*1024*1024;
    ggml_init_params params{ mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx);

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row_dim, kv_size);
    ggml_tensor * dst = ggml_new_tensor_3d(ctx, tq_type,        row_dim, kv_size, 1);
    ggml_set_name(src, "src");
    ggml_set_name(dst, "dst");

    // dst is a 3-D tensor so we view it as 2-D for the cpy
    ggml_tensor * dst_v = ggml_view_2d(ctx, dst, row_dim, kv_size, dst->nb[1], 0);
    ggml_tensor * cpy   = ggml_cpy(ctx, src, dst_v);
    ggml_set_name(cpy, "cpy");

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, cpy);

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf);

    ggml_backend_tensor_set(src, src_data.data(), 0, ggml_nbytes(src));

    {
        enum ggml_status st = ggml_backend_graph_compute(backend, graph);
        assert(st == GGML_STATUS_SUCCESS);
        (void) st;
    }

    // Read dst back as raw bytes, then dequantize via type traits.
    std::vector<uint8_t> dst_bytes(ggml_nbytes(dst));
    ggml_backend_tensor_get(dst, dst_bytes.data(), 0, ggml_nbytes(dst));

    const ggml_type_traits * tt = ggml_get_type_traits(tq_type);
    assert(tt && tt->to_float);

    std::vector<float> dq(n_elem);
    for (int row = 0; row < kv_size; ++row) {
        const uint8_t * row_ptr = dst_bytes.data() + (size_t) row * dst->nb[1];
        tt->to_float(row_ptr, dq.data() + (size_t) row * row_dim, row_dim);
    }

    const float c = cosine(src_data, dq);
    fprintf(stdout, "[kv-types/%s/cpy] cos=%.5f\n", tag, c);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    // CPY uses the reference quantizer; gates match
    // test-reex-turboquant-block-roundtrip after the P4 A2.2 K3 P3 upgrade:
    //   TQ_K3 ≥ 0.97 (theoretical limit of 3-bit Lloyd-Max-Gaussian
    //                  PolarQuant at head_dim=128, see block-roundtrip test
    //                  header for the derivation; TheTom's turbo3_0 — the
    //                  same algorithm — uses NMSE ≤ 0.05 ≈ cos_sim ≥ 0.975)
    //   TQ_V2 ≥ 0.93 (paper-marginal tier)
    //   TQ_V4 ≥ 0.99 (paper-neutrality tier)
    float threshold = 0.93f;
    if (tq_type == GGML_TYPE_TQ_K3) threshold = 0.97f;
    if (tq_type == GGML_TYPE_TQ_V4) threshold = 0.99f;
    if (c < threshold) {
        fprintf(stderr, "[FAIL] kv-types/%s/cpy: cos=%.5f < %.2f\n", tag, c, threshold);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// GET_ROWS round-trip: allocate a 2-D TQ src [row_dim, kv_size] populated by
// quantizing an F32 buffer ourselves, then build a tiny graph that performs
// `ggml_get_rows(src, idx)` on the CPU backend.  This mimics the read-side
// of llama_kv_cache::get_k where a contiguous slice of cells is materialised
// into the attention compute path.
// ---------------------------------------------------------------------------

bool test_get_rows_round_trip(ggml_type tq_type, int row_dim, int kv_size, const char * tag) {
    const int n_pick = std::min(8, kv_size);
    const auto src_data = rand_f32((size_t) row_dim * kv_size, /*seed=*/0xBADF00Du ^ (uint32_t) tq_type);

    const size_t mem_size = (size_t) 64*1024*1024;
    ggml_init_params params{ mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx);

    ggml_tensor * src_q = ggml_new_tensor_2d(ctx, tq_type, row_dim, kv_size);
    ggml_tensor * idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pick);
    ggml_set_name(src_q, "src_q");
    ggml_set_name(idx,   "idx");

    ggml_tensor * gathered = ggml_get_rows(ctx, src_q, idx);
    ggml_set_name(gathered, "gathered");

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, gathered);

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf);

    // Quantize the F32 src row-by-row into src_q via type traits.
    const ggml_type_traits * tt = ggml_get_type_traits(tq_type);
    assert(tt && tt->from_float_ref);

    std::vector<uint8_t> src_q_bytes(ggml_nbytes(src_q));
    for (int row = 0; row < kv_size; ++row) {
        tt->from_float_ref(src_data.data() + (size_t) row * row_dim,
                           src_q_bytes.data() + (size_t) row * src_q->nb[1],
                           row_dim);
    }
    ggml_backend_tensor_set(src_q, src_q_bytes.data(), 0, ggml_nbytes(src_q));

    std::vector<int32_t> idx_data(n_pick);
    for (int i = 0; i < n_pick; ++i) idx_data[i] = i * (kv_size / n_pick);
    ggml_backend_tensor_set(idx, idx_data.data(), 0, ggml_nbytes(idx));

    {
        enum ggml_status st = ggml_backend_graph_compute(backend, graph);
        assert(st == GGML_STATUS_SUCCESS);
        (void) st;
    }

    std::vector<float> got((size_t) row_dim * n_pick);
    ggml_backend_tensor_get(gathered, got.data(), 0, ggml_nbytes(gathered));

    // Compare each fetched row against the dequantized original at that row.
    std::vector<float> ref_dq((size_t) row_dim * n_pick);
    for (int i = 0; i < n_pick; ++i) {
        tt->to_float(src_q_bytes.data() + (size_t) idx_data[i] * src_q->nb[1],
                     ref_dq.data() + (size_t) i * row_dim,
                     row_dim);
    }

    const float c = cosine(ref_dq, got);
    fprintf(stdout, "[kv-types/%s/get_rows] cos=%.5f\n", tag, c);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (c < 0.99999f) {
        // get_rows on CPU is a strict bitwise gather of already-quantized
        // bytes followed by dequant; we expect numerically identical output.
        fprintf(stderr, "[FAIL] kv-types/%s/get_rows: cos=%.5f < 0.99999\n", tag, c);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const int row_k = S.head_dim * S.n_head_kv;   // 1024 — divisible by 128 (TQ_K3 block)
    const int row_v = S.head_dim * S.n_head_kv;   // 1024 — divisible by  32 (TQ_V2/V4 block)

    fprintf(stdout, "[kv-types] shape head_dim=%d n_head_kv=%d kv_size=%d row_k=%d row_v=%d\n",
            S.head_dim, S.n_head_kv, S.kv_size, row_k, row_v);

    bool ok = true;

    ok = test_cpy_round_trip       (GGML_TYPE_TQ_K3, row_k, S.kv_size, "TQ_K3") && ok;
    ok = test_cpy_round_trip       (GGML_TYPE_TQ_V2, row_v, S.kv_size, "TQ_V2") && ok;
    ok = test_cpy_round_trip       (GGML_TYPE_TQ_V4, row_v, S.kv_size, "TQ_V4") && ok;

    ok = test_get_rows_round_trip  (GGML_TYPE_TQ_K3, row_k, S.kv_size, "TQ_K3") && ok;
    ok = test_get_rows_round_trip  (GGML_TYPE_TQ_V2, row_v, S.kv_size, "TQ_V2") && ok;
    ok = test_get_rows_round_trip  (GGML_TYPE_TQ_V4, row_v, S.kv_size, "TQ_V4") && ok;

    if (!ok) {
        fprintf(stderr, "[kv-types] FAIL\n");
        return 1;
    }
    fprintf(stdout, "[kv-types] PASS\n");
    return 0;
}
