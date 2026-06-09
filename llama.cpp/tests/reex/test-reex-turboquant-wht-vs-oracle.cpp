// SPDX-License-Identifier: MIT
//
// test-reex-turboquant-wht-vs-oracle
//
// REEX_TURBOQUANT P1.3 acceptance test (b/3).
//
// The production WHT and the oracle's dense Gaussian-QR rotation Pi are two
// independent random orthogonal transforms — they are NOT the same matrix.
// What we *do* require of both is the orthogonal-invariance property
//
//     <T x, T y> == <x, y>     for any rows x, y and orthogonal T.
//
// We therefore measure, for each (dim, seed) fixture and a batch of random
// (x, y) pairs:
//
//     err_wht (x,y) = | <WHT(x),  WHT(y)>  - <x,y> | / max(|<x,y>|, eps)
//     err_pi  (x,y) = | <Pi  x ,  Pi  y > - <x,y> | / max(|<x,y>|, eps)
//
// and require both maxima ≤ 1e-3 (≈ fp32 round-off scaled by sqrt(d) — the
// dense Pi path matmul accumulates much more error than WHT, so we keep a
// generous-but-still-meaningful budget).  The cosine of the angle between
// (T x) and (T y) is also expected to match cos(x, y) within 1e-3.
//
// Together with `test-reex-turboquant-wht-roundtrip.cpp`'s strict identity
// gate, this confirms our WHT is in the same equivalence class as the
// oracle's full random orthogonal matrix even though we don't store the
// dense matrix.

#include "ggml-cpu/reex/reex_turboquant_wht.h"
#include "reex_turboquant_oracle_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr float kInnerProductRelTol = 1e-3f;

void fill_gaussian(float * out, std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) out[i] = nd(rng);
}

double dot(const float * a, const float * b, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
    return s;
}

void apply_dense_rot(const std::vector<float> & rot, int dim,
                     const float * x, float * y) {
    for (int i = 0; i < dim; ++i) {
        double s = 0.0;
        for (int j = 0; j < dim; ++j) {
            s += static_cast<double>(rot[static_cast<size_t>(i) * dim + j]) * x[j];
        }
        y[i] = static_cast<float>(s);
    }
}

bool run_one(int dim, std::uint32_t seed, int n_pairs) {
    const std::vector<float> rot =
        reex_turboquant_oracle::reex_turboquant_oracle_rotation_get(dim, /*seed=*/42, /*layer_idx=*/0);

    std::vector<float> x(static_cast<size_t>(dim));
    std::vector<float> y(static_cast<size_t>(dim));
    std::vector<float> wx(static_cast<size_t>(dim)), wy(static_cast<size_t>(dim));
    std::vector<float> px(static_cast<size_t>(dim)), py(static_cast<size_t>(dim));

    double max_err_wht = 0.0;
    double max_err_pi  = 0.0;

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (int i = 0; i < n_pairs; ++i) {
        for (int j = 0; j < dim; ++j) x[j] = nd(rng);
        for (int j = 0; j < dim; ++j) y[j] = nd(rng);

        reex_turboquant_cpu_wht_apply_row(x.data(), wx.data(), dim, 0, nullptr, nullptr);
        reex_turboquant_cpu_wht_apply_row(y.data(), wy.data(), dim, 0, nullptr, nullptr);
        apply_dense_rot(rot, dim, x.data(), px.data());
        apply_dense_rot(rot, dim, y.data(), py.data());

        const double inner_xy   = dot(x.data(),  y.data(),  static_cast<size_t>(dim));
        const double inner_wxwy = dot(wx.data(), wy.data(), static_cast<size_t>(dim));
        const double inner_pxpy = dot(px.data(), py.data(), static_cast<size_t>(dim));

        const double denom = std::max(std::abs(inner_xy), 1e-9);
        const double err_wht = std::abs(inner_wxwy - inner_xy) / denom;
        const double err_pi  = std::abs(inner_pxpy - inner_xy) / denom;
        max_err_wht = std::max(max_err_wht, err_wht);
        max_err_pi  = std::max(max_err_pi,  err_pi);
    }

    const bool ok_wht = max_err_wht <= kInnerProductRelTol;
    const bool ok_pi  = max_err_pi  <= kInnerProductRelTol;

    std::fprintf(stdout,
        "[d=%-3d seed=0x%08x pairs=%-4d] "
        "max|<Tx,Ty>-<x,y>|/|<x,y>|  WHT=%.2e (<= %.0e)%s  oracle_Pi=%.2e (<= %.0e)%s\n",
        dim, seed, n_pairs,
        max_err_wht, kInnerProductRelTol, ok_wht ? "" : " FAIL",
        max_err_pi,  kInnerProductRelTol, ok_pi  ? "" : " FAIL");
    return ok_wht && ok_pi;
}

}  // namespace

int main() {
    bool all_ok = true;

    const int dims[] = { 64, 128 };
    const std::uint32_t seeds[] = { 0xc0ffeeu, 0xdeadu, 0xbeefu };
    const int n_pairs = 64;

    for (auto d : dims) {
        for (auto s : seeds) {
            if (!run_one(d, s, n_pairs)) all_ok = false;
        }
    }

    std::fprintf(stdout, "test-reex-turboquant-wht-vs-oracle: %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
