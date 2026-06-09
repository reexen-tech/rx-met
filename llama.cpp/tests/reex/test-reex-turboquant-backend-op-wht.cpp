// SPDX-License-Identifier: MIT
//
// test-reex-turboquant-backend-op-wht
//
// REEX_TURBOQUANT P1.3 acceptance test (c/3): exercise `GGML_OP_REEX_WHT`
// through the standard ggml graph-compute path and assert it bit-equally
// matches the scalar `reex_turboquant_cpu_wht_apply_rows` reference.
//
// Why a custom test instead of slotting into `tests/test-backend-ops.cpp`:
//   The upstream backend-ops harness pulls in a global op registry whose
//   regenerator we don't want to touch from inside our `REEX_TURBOQUANT`
//   gate (it would require adding cases to upstream files that have nothing
//   to do with REEX otherwise).  This standalone test is comparable in
//   coverage — it goes through ggml_graph_compute, op_params encode/decode,
//   thread dispatch, and contiguity checks — without touching upstream.
//
// What we cover:
//   1. Forward direction with no scale_inv,                  group_size 64 / 128
//   2. Inverse direction with no scale_inv,                  group_size 64 / 128
//   3. Forward direction with F32 scale_inv (uniform 0.7),   group_size 128
//   4. Forward direction with F16 scale_inv (uniform 1.3),   group_size 128
//   5. Multi-threaded compute (n_threads = 4) matches single-threaded.
//   6. (P4 A2.4 Stage 2a) sq_accum accumulator: ggml_graph reads the same
//      F64 per-channel post-WHT variance as the scalar reference.
//   7. (P4 A2.4 Stage 2a) InnerQ inner-product invariance: for any per-
//      channel scale s, `<WHT(Q ⊙ s_inv), WHT(K ⊙ s)> ≈ <Q, K>` within FP
//      tolerance.  This is the math that justifies pre-WHT scale_inv
//      placement; if it ever fails, the InnerQ EMA assumption is broken.
//
// Acceptance: max-abs(graph_out - scalar_out) ≤ 1e-6 on every fixture.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu/reex/reex_turboquant_wht.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr float kAbsTol = 1e-6f;

void fill_gaussian(float * out, std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) out[i] = nd(rng);
}

double max_abs_diff(const float * a, const float * b, std::size_t n) {
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::abs(static_cast<double>(a[i]) - b[i]);
        if (d > m) m = d;
    }
    return m;
}

void compute_with_graph(std::vector<float> & buffer_for_work,
                        struct ggml_context * ctx,
                        struct ggml_tensor  * leaf,
                        int                   n_threads) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, leaf);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
    if (plan.work_size > 0) {
        buffer_for_work.resize(plan.work_size);
        plan.work_data = reinterpret_cast<uint8_t *>(buffer_for_work.data());
    }
    ggml_graph_compute(gf, &plan);
}

struct Case {
    const char * name;
    int dim;
    int rows;
    int direction;
    enum scale_kind { NONE, F32, F16 } scale;
    int n_threads;
};

bool run_case(const Case & c, std::uint32_t seed) {
    struct ggml_init_params ip = {
        /* .mem_size   = */ 64 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed for case %s\n", c.name);
        return false;
    }

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.dim, c.rows);
    fill_gaussian(static_cast<float *>(x->data),
                  static_cast<std::size_t>(c.dim) * c.rows, seed);

    struct ggml_tensor * scale = nullptr;
    std::vector<float> scale_f32_for_kernel(static_cast<std::size_t>(c.dim));

    if (c.scale == Case::F32) {
        scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.dim);
        float * sd = static_cast<float *>(scale->data);
        for (int i = 0; i < c.dim; ++i) sd[i] = 0.7f + 0.001f * i;
        std::memcpy(scale_f32_for_kernel.data(), sd, sizeof(float) * c.dim);
    } else if (c.scale == Case::F16) {
        scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, c.dim);
        ggml_fp16_t * sd = static_cast<ggml_fp16_t *>(scale->data);
        for (int i = 0; i < c.dim; ++i) {
            const float v = 1.3f - 0.0005f * i;
            sd[i] = ggml_fp32_to_fp16(v);
            scale_f32_for_kernel[i] = ggml_fp16_to_fp32(sd[i]);
        }
    }

    struct ggml_tensor * y = ggml_reex_wht(ctx, x, c.direction, c.dim, scale, /*sq_accum=*/nullptr);

    std::vector<float> work;
    compute_with_graph(work, ctx, y, c.n_threads);

    std::vector<float> ref(static_cast<std::size_t>(c.dim) * c.rows);
    reex_turboquant_cpu_wht_apply_rows(
        static_cast<const float *>(x->data),
        ref.data(),
        c.rows,
        c.dim,
        c.direction,
        c.scale == Case::NONE ? nullptr : scale_f32_for_kernel.data(),
        /*sq_accum=*/nullptr);

    const double err = max_abs_diff(static_cast<const float *>(y->data),
                                    ref.data(),
                                    ref.size());
    const bool ok = err <= kAbsTol;

    std::fprintf(stdout,
        "[%-32s d=%-3d rows=%-3d dir=%d scale=%-4s nth=%d] max|graph - scalar|=%.2e%s\n",
        c.name, c.dim, c.rows, c.direction,
        c.scale == Case::NONE ? "none" : c.scale == Case::F32 ? "f32" : "f16",
        c.n_threads, err, ok ? "" : "  <-- FAIL");

    ggml_free(ctx);
    return ok;
}

// ---------------------------------------------------------------------------
// (P4 A2.4 Stage 2a) sq_accum accumulator parity test.
// ---------------------------------------------------------------------------
// Verify that the GGML_OP_REEX_WHT path's per-channel F64 sq_accum exactly
// matches the scalar reference's accumulator over the same inputs.  Also
// verifies that the rotated output `y` is unchanged regardless of whether
// sq_accum is present.
bool run_sq_accum_case(int dim, int rows, int direction, int n_threads, std::uint32_t seed) {
    struct ggml_init_params ip = {
        /* .mem_size   = */ 64 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed for sq_accum case\n");
        return false;
    }

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, rows);
    fill_gaussian(static_cast<float *>(x->data),
                  static_cast<std::size_t>(dim) * rows, seed);

    struct ggml_tensor * accum = ggml_new_tensor_1d(ctx, GGML_TYPE_F64, dim);
    std::memset(accum->data, 0, static_cast<std::size_t>(dim) * sizeof(double));

    struct ggml_tensor * y = ggml_reex_wht(ctx, x, direction, dim, /*scale_inv=*/nullptr, accum);

    std::vector<float> work;
    compute_with_graph(work, ctx, y, n_threads);

    // Scalar reference: same inputs, fresh accumulator.
    std::vector<float>  ref(static_cast<std::size_t>(dim) * rows);
    std::vector<double> ref_accum(dim, 0.0);
    reex_turboquant_cpu_wht_apply_rows(
        static_cast<const float *>(x->data),
        ref.data(),
        rows, dim, direction,
        /*scale_inv=*/nullptr,
        ref_accum.data());

    const double err_y = max_abs_diff(static_cast<const float *>(y->data),
                                      ref.data(), ref.size());
    double err_accum = 0.0;
    const double * graph_accum = static_cast<const double *>(accum->data);
    for (int i = 0; i < dim; ++i) {
        const double d = std::abs(graph_accum[i] - ref_accum[i]);
        if (d > err_accum) err_accum = d;
    }
    const bool ok = err_y <= kAbsTol && err_accum <= 1e-9 * static_cast<double>(rows);

    std::fprintf(stdout,
        "[sq_accum-d=%-3d-rows=%-3d-dir=%d-nth=%d] "
        "max|graph y - ref y|=%.2e  max|graph accum - ref accum|=%.2e%s\n",
        dim, rows, direction, n_threads, err_y, err_accum, ok ? "" : "  <-- FAIL");

    ggml_free(ctx);
    return ok;
}

// ---------------------------------------------------------------------------
// (P4 A2.4 Stage 2a) InnerQ inner-product invariance.
// ---------------------------------------------------------------------------
// For any per-channel rescaling vector s (with s_inv = 1/s):
//
//      <WHT_fwd(q ⊙ s_inv), WHT_fwd(k ⊙ s)>  ≈  <q, k>
//
// This is the math InnerQ EMA depends on.  H_d is orthogonal so
// `<WHT(a), WHT(b)> = <a, b>`; the diagonal scaling cancels:
// `<a ⊙ s_inv, b ⊙ s> = sum_i a_i * s_inv_i * b_i * s_i = sum_i a_i * b_i`.
//
// We test it through the GGML graph (not just the scalar reference) so any
// regression in the dispatch / scale_inv plumbing surfaces here.
bool run_innerq_invariance_case(int dim, int direction, std::uint32_t seed) {
    constexpr int rows = 4;  // multiple rows make sure scale is broadcast per-row.
    struct ggml_init_params ip = {
        /* .mem_size   = */ 64 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed for innerq invariance case\n");
        return false;
    }

    struct ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, rows);
    struct ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, rows);
    fill_gaussian(static_cast<float *>(q->data), static_cast<std::size_t>(dim) * rows, seed);
    fill_gaussian(static_cast<float *>(k->data), static_cast<std::size_t>(dim) * rows, seed ^ 0xdeadu);

    // Construct s in [0.5, 2.0] (matches InnerQ::compute_scale clamp range)
    // and its reciprocal.  Use a deterministic ramp so a regression in the
    // scale-application order is reproducible.
    struct ggml_tensor * s     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    struct ggml_tensor * s_inv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    float * sd  = static_cast<float *>(s->data);
    float * sid = static_cast<float *>(s_inv->data);
    for (int i = 0; i < dim; ++i) {
        sd[i]  = 0.5f + 1.5f * (float) i / (float) (dim - 1);  // 0.5..2.0
        sid[i] = 1.0f / sd[i];
    }

    // q_rot = WHT(q ⊙ s_inv);  k_rot = WHT(k ⊙ s).
    struct ggml_tensor * q_rot = ggml_reex_wht(ctx, q, direction, dim, s_inv, /*sq_accum=*/nullptr);
    struct ggml_tensor * k_rot = ggml_reex_wht(ctx, k, direction, dim, s,     /*sq_accum=*/nullptr);

    std::vector<float> work;
    // We could use one graph, but separate ones make the failure mode
    // localisable per tensor.
    compute_with_graph(work, ctx, q_rot, /*n_threads=*/2);
    compute_with_graph(work, ctx, k_rot, /*n_threads=*/2);

    // Dot per row, before vs after.  Tolerance accounts for FP cancellation
    // through the WHT butterfly + sign + 1/sqrt(d) chain (worst case ~1e-4
    // relative on dim=128 random Gaussians).
    const float * qd  = static_cast<const float *>(q->data);
    const float * kd  = static_cast<const float *>(k->data);
    const float * qrd = static_cast<const float *>(q_rot->data);
    const float * krd = static_cast<const float *>(k_rot->data);

    double max_rel = 0.0, max_abs = 0.0;
    for (int r = 0; r < rows; ++r) {
        double dot_orig = 0.0;
        double dot_rot  = 0.0;
        for (int i = 0; i < dim; ++i) {
            dot_orig += static_cast<double>(qd [r * dim + i]) * kd [r * dim + i];
            dot_rot  += static_cast<double>(qrd[r * dim + i]) * krd[r * dim + i];
        }
        const double diff = std::abs(dot_rot - dot_orig);
        const double base = std::max(1e-6, std::abs(dot_orig));
        const double rel  = diff / base;
        if (diff > max_abs) max_abs = diff;
        if (rel  > max_rel) max_rel = rel;
    }

    // 5e-4 relative tolerance is generous but well below the systematic
    // error we'd expect from a wrong-side scale_inv application (which
    // would skew the dot by O(1) since H_d does not commute with Diag).
    const bool ok = max_rel < 5e-4;
    std::fprintf(stdout,
        "[innerq-invariance-d=%-3d-dir=%d] max|dot_orig - dot_rot|=%.3e (rel %.3e)%s\n",
        dim, direction, max_abs, max_rel, ok ? "" : "  <-- FAIL");

    ggml_free(ctx);
    return ok;
}

}  // namespace

int main() {
    static const Case cases[] = {
        { "fwd-d128-noscale-1t",     128, 32, 0, Case::NONE, 1 },
        { "fwd-d128-noscale-4t",     128, 32, 0, Case::NONE, 4 },
        { "inv-d128-noscale-1t",     128, 32, 1, Case::NONE, 1 },
        { "inv-d128-noscale-4t",     128, 32, 1, Case::NONE, 4 },
        { "fwd-d64-noscale-1t",       64, 32, 0, Case::NONE, 1 },
        { "fwd-d64-noscale-4t",       64, 32, 0, Case::NONE, 4 },
        { "inv-d64-noscale-1t",       64, 32, 1, Case::NONE, 1 },
        { "fwd-d128-scaleF32-2t",    128, 16, 0, Case::F32,  2 },
        { "fwd-d128-scaleF16-3t",    128, 16, 0, Case::F16,  3 },
        { "fwd-d128-scaleF32-rows1",128,  1, 0, Case::F32,  1 },
        { "fwd-d128-scaleF16-rows1",128,  1, 0, Case::F16,  1 },
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        if (!run_case(c, /*seed=*/0xc0ffeeu ^ static_cast<uint32_t>(c.direction * 17 + c.dim))) {
            all_ok = false;
        }
    }

    // (6) sq_accum accumulator parity (single + multi-thread caller; the op
    // itself pins to 1 task when sq_accum is non-NULL, so multi-thread
    // callers should still see correct values).
    all_ok &= run_sq_accum_case(/*dim=*/128, /*rows=*/64, /*dir=*/0, /*nth=*/1, 0xacc01u);
    all_ok &= run_sq_accum_case(/*dim=*/128, /*rows=*/64, /*dir=*/0, /*nth=*/4, 0xacc02u);
    all_ok &= run_sq_accum_case(/*dim=*/64,  /*rows=*/32, /*dir=*/0, /*nth=*/2, 0xacc03u);
    all_ok &= run_sq_accum_case(/*dim=*/128, /*rows=*/1,  /*dir=*/0, /*nth=*/1, 0xacc04u);

    // (7) InnerQ inner-product invariance — the load-bearing math test.
    all_ok &= run_innerq_invariance_case(/*dim=*/128, /*dir=*/0, 0x10ad01u);
    all_ok &= run_innerq_invariance_case(/*dim=*/128, /*dir=*/1, 0x10ad02u);
    all_ok &= run_innerq_invariance_case(/*dim=*/64,  /*dir=*/0, 0x10ad03u);

    std::fprintf(stdout, "test-reex-turboquant-backend-op-wht: %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
