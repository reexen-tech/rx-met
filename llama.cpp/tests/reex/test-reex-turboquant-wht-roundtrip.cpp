// SPDX-License-Identifier: MIT
//
// test-reex-turboquant-wht-roundtrip
//
// REEX_TURBOQUANT P1.3 acceptance test (a/3): the CPU WHT kernel
// `reex_turboquant_cpu_wht_apply_row` must satisfy
//
//     inverse(forward(x)) == x   and   forward(inverse(x)) == x
//
// up to fp32 numerical noise (max-abs ≤ 5e-6 per element across the random
// fixtures we sample).  This is the strictest of the three WHT tests: any
// single-bit error in the sign tables, normalisation, or butterfly indexing
// blows the contraction far past the threshold immediately.
//
// Scope: pure scalar reference path — does not exercise the ggml graph or
// `GGML_OP_REEX_WHT` dispatch (those land in
// `test-reex-turboquant-backend-op-wht.cpp`).

#include "ggml-cpu/reex/reex_turboquant_wht.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr float kAbsTol = 5e-6f;

void fill_gaussian(float * out, std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = nd(rng);
    }
}

double max_abs_diff(const float * a, const float * b, std::size_t n) {
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::abs(static_cast<double>(a[i]) - b[i]);
        if (d > m) m = d;
    }
    return m;
}

double l2_norm(const float * a, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * a[i];
    return std::sqrt(s);
}

bool run_one(int64_t dim, int blocks, std::uint32_t seed) {
    const std::size_t n = static_cast<std::size_t>(blocks) * dim;
    std::vector<float> x(n), y(n), z(n);
    fill_gaussian(x.data(), n, seed);

    reex_turboquant_cpu_wht_apply_rows(
        x.data(), y.data(), blocks, dim, /*direction=*/0, /*scale_inv=*/nullptr, /*sq_accum=*/nullptr);
    reex_turboquant_cpu_wht_apply_rows(
        y.data(), z.data(), blocks, dim, /*direction=*/1, /*scale_inv=*/nullptr, /*sq_accum=*/nullptr);

    const double err_fwd_inv = max_abs_diff(x.data(), z.data(), n);

    std::vector<float> u(n), v(n), w(n);
    fill_gaussian(u.data(), n, seed ^ 0xa5a5);
    reex_turboquant_cpu_wht_apply_rows(
        u.data(), v.data(), blocks, dim, /*direction=*/1, /*scale_inv=*/nullptr, /*sq_accum=*/nullptr);
    reex_turboquant_cpu_wht_apply_rows(
        v.data(), w.data(), blocks, dim, /*direction=*/0, /*scale_inv=*/nullptr, /*sq_accum=*/nullptr);
    const double err_inv_fwd = max_abs_diff(u.data(), w.data(), n);

    // Sanity: forward should preserve the L2 norm (orthogonal transform).
    const double in_norm  = l2_norm(x.data(), n);
    const double out_norm = l2_norm(y.data(), n);
    const double rel_norm = in_norm > 1e-12 ? std::abs(in_norm - out_norm) / in_norm : 0.0;

    const bool ok_fwd_inv = err_fwd_inv <= kAbsTol;
    const bool ok_inv_fwd = err_inv_fwd <= kAbsTol;
    const bool ok_norm    = rel_norm    <= 1e-5;
    const bool ok = ok_fwd_inv && ok_inv_fwd && ok_norm;

    std::fprintf(stdout,
        "[d=%lld blocks=%-3d seed=0x%08x] "
        "max|inv∘fwd - I|=%.2e  max|fwd∘inv - I|=%.2e  |Δ‖x‖|/‖x‖=%.2e%s\n",
        (long long) dim, blocks, seed,
        err_fwd_inv, err_inv_fwd, rel_norm,
        ok ? "" : "  <-- FAIL");
    return ok;
}

}  // namespace

int main() {
    bool all_ok = true;

    const std::uint32_t seeds[]   = { 0xc0ffeeu, 0xdeadu, 0xbeefu, 0x12345u };
    const int           blocks[]  = { 1, 4, 17 };
    const int64_t       dims[]    = { 64, 128 };

    for (auto d : dims) {
        for (auto b : blocks) {
            for (auto s : seeds) {
                if (!run_one(d, b, s)) all_ok = false;
            }
        }
    }

    std::fprintf(stdout, "test-reex-turboquant-wht-roundtrip: %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
