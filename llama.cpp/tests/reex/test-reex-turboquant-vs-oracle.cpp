// SPDX-License-Identifier: MIT
//
// test-reex-turboquant-vs-oracle
//
// REEX_TURBOQUANT P1.4 acceptance test: oracle parity for the production
// quantize/dequantize paths reachable via `ggml_get_type_traits()`.
//
// What's exercised
// ----------------
//   Section A — V cache (TQ_V2 / TQ_V4)
//     Both ggml's `type_traits.from_float / to_float` and oracle's
//     `reex_turboquant_oracle_value_quantize / value_dequantize` implement
//     the same per-block-32 min/max scalar quantization rule, so the
//     reconstructions must match up to FP16 round-trip noise in the per-
//     block scale/zero.  Gate: per-row cos_sim ≥ 0.99999.
//
//   Section B — K cache (TQ_K3) full path vs TurboQuantPolar
//     Apples-to-apples comparison against `TurboQuantPolar` (oracle, bits=3,
//     codebook from `reex_turboquant_oracle_codebook_get(128, 3)`, identity
//     rotation, norm-correction trick).  Our production
//     `quantize_row_tq_k3 + dequantize_row_tq_k3` runs the full TheTom
//     turbo3_0 path (3-bit PolarQuant + norm correction); with identity
//     Π on both sides the bit decoding lines up element-by-element so the
//     reconstructions must match up to FP16 round-trip noise in the
//     per-block `corrected_norm`.  Gate: per-row cos_sim ≥ 0.99999.
//
//     This section subsumes the old "Section C — full Phase 1 chain"
//     (P1 stub had centroid + 1-bit residual sign decoded with an
//     empirical magnitude; that gate was expected-FAIL at 0.99 because
//     the residual magnitude was off by ~0.023 cos_sim).  K3 P3 closes
//     that gap by switching to a full 3-bit codebook decoder, so we
//     fold both checks into a single strict (≥ 0.99999) comparison.
//
//   Section C — V cache β (TQ_V_POLAR2 / TQ_V_POLAR4) full path vs TurboQuantPolar
//     P4 A2.5 V-β.  Same `TurboQuantPolar` machinery as Section B but
//     dim=128 / bits∈{2,4}, mirroring TheTom turbo2_0 / turbo4_0 on the
//     value side.  Production path:
//       quantize_row_tq_v_polar{2,4}  +  dequantize_row_tq_v_polar{2,4}
//     wired via `block_tq_v_polar{2,4}` (34 / 68 B, group=128, norm in
//     FP16) → both decoders consume the same Lloyd-Max centroid table
//     emitted by `reex_turboquant_gen_constants_v_polar.cpp`, which is
//     mathematically the same Beta_d(x) PDF + Lloyd-Max iteration as
//     `reex_turboquant_oracle_codebook_get(128, bits)`.  With identity
//     rotation on the oracle (production WHT lives in `GGML_OP_REEX_WHT`,
//     not in `quantize_row_tq_v_polar*_ref`) and the same norm-correction
//     trick on both sides, the reconstructions must match element-by-
//     element up to FP16 round-trip noise on `corrected_norm`.  Gate:
//     per-row cos_sim ≥ 0.99999 (identical to Section B).
//
// Routing
// -------
// We deliberately go through `ggml_get_type_traits()` (libggml-base) and
// `ggml_get_type_traits_cpu()` (libggml-cpu), rather than calling our
// `quantize_row_tq_*` / `dequantize_row_tq_*` directly.  This catches any
// type-traits wiring regression (missing `from_float`, wrong block_size,
// etc.) as part of the same acceptance test.

#include "ggml.h"
#include "ggml-cpu.h"
// `reex_turboquant_quants.h` brings in `ggml-common.h` with the right
// GGML_COMMON_DECL guard, so we get `block_tq_k3 / QK_TQ_K3 / QK_TQ_V2`
// and friends without re-doing the macro dance here.
#include "reex_turboquant_quants.h"

#include "reex_turboquant_oracle_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace tq = reex_turboquant_oracle;

namespace {

constexpr int kKDim       = QK_TQ_K3;          // 128
constexpr int kVGroupSize = QK_TQ_V2;          // 32 (== QK_TQ_V4)
constexpr int kKBits      = 3;                 // P4 A2.2: full 3-bit PolarQuant
constexpr int kVPolarDim  = QK_TQ_V_POLAR2;    // 128 (== QK_TQ_V_POLAR4)
constexpr int kVPolar2Bits = 2;                // P4 A2.5: V-β 2-bit PolarQuant
constexpr int kVPolar4Bits = 4;                // P4 A2.5: V-β 4-bit PolarQuant

constexpr float kStrictCosSim = 0.99999f;

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

double cos_sim(const float * a, const float * b, std::size_t n) {
    const double na = std::sqrt(dot(a, a, n));
    const double nb = std::sqrt(dot(b, b, n));
    if (na < 1e-30 || nb < 1e-30) return 1.0;
    return dot(a, b, n) / (na * nb);
}

double row_min_cos_sim(const float * a, const float * b, int rows, int dim) {
    double m = 1.0;
    for (int r = 0; r < rows; ++r) {
        const double c = cos_sim(a + r * dim, b + r * dim, static_cast<std::size_t>(dim));
        if (c < m) m = c;
    }
    return m;
}

double row_max_abs_diff(const float * a, const float * b, int rows, int dim) {
    double mx = 0.0;
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < dim; ++j) {
            const double d = std::abs(static_cast<double>(a[r * dim + j]) - b[r * dim + j]);
            if (d > mx) mx = d;
        }
    }
    return mx;
}

// type_traits round-trip: float[rows*dim] → from_float(_ref or production) →
// to_float → float[rows*dim].  We exercise the production `from_float` (via
// ggml-cpu type_traits) so any vec_dot / SIMD wiring change shows up here.
void quant_dequant_via_type_traits(enum ggml_type type,
                                   const float * src,
                                   int rows,
                                   int dim,
                                   float * dst) {
    const auto * t_base = ggml_get_type_traits(type);
    const auto * t_cpu  = ggml_get_type_traits_cpu(type);
    if (!t_base || !t_cpu) {
        std::fprintf(stderr, "type_traits not registered for type %d\n", static_cast<int>(type));
        std::abort();
    }
    const std::size_t row_bytes_q = ggml_row_size(type, dim);
    std::vector<uint8_t> blocks(static_cast<std::size_t>(rows) * row_bytes_q);
    for (int r = 0; r < rows; ++r) {
        t_cpu->from_float(src + r * dim, blocks.data() + r * row_bytes_q, dim);
    }
    for (int r = 0; r < rows; ++r) {
        t_base->to_float(blocks.data() + r * row_bytes_q, dst + r * dim, dim);
    }
}

// Helper: run oracle TurboQuantPolar with identity rotation (matches our
// "already-rotated input" convention — production WHT lives in the graph
// op `GGML_OP_REEX_WHT`, not inside `quantize_row_tq_k3_ref`).  bits=3
// gives the same 8-centroid Lloyd-Max codebook + 7 decision boundaries
// as the production .inc file, and the `corrected_norm = ‖x‖/‖recon‖`
// trick on the oracle side mirrors `block_tq_k3::norm` so the resulting
// reconstructions can be compared element-by-element.
std::vector<float> oracle_polar_full_reconstruct(const float * src, int rows, int dim) {
    const tq::Codebook cb = tq::reex_turboquant_oracle_codebook_get(dim, kKBits);
    std::vector<float> identity(static_cast<std::size_t>(dim) * dim, 0.0f);
    for (int i = 0; i < dim; ++i) identity[static_cast<std::size_t>(i) * dim + i] = 1.0f;

    tq::TurboQuantPolar polar(dim, kKBits, cb, identity);
    tq::MSEQuantized q = polar.quantize(src, /*batch=*/1, /*heads=*/1, /*tokens=*/rows);
    return polar.dequantize(q);
}

// V-β oracle reconstruction.  Same `TurboQuantPolar` Algorithm 1 + norm
// correction as `oracle_polar_full_reconstruct`, but parameterised on
// `bits` (2 or 4) for `TQ_V_POLAR2 / TQ_V_POLAR4` instead of the
// fixed bits=3 of TQ_K3.  Identity rotation: production WHT lives in
// `GGML_OP_REEX_WHT` (graph-side, group=128 V-shift), not inside
// `quantize_row_tq_v_polar*_ref`.
std::vector<float> oracle_v_polar_full_reconstruct(const float * src, int rows, int dim, int bits) {
    const tq::Codebook cb = tq::reex_turboquant_oracle_codebook_get(dim, bits);
    std::vector<float> identity(static_cast<std::size_t>(dim) * dim, 0.0f);
    for (int i = 0; i < dim; ++i) identity[static_cast<std::size_t>(i) * dim + i] = 1.0f;

    tq::TurboQuantPolar polar(dim, bits, cb, identity);
    tq::MSEQuantized q = polar.quantize(src, /*batch=*/1, /*heads=*/1, /*tokens=*/rows);
    return polar.dequantize(q);
}

bool run_v_section(enum ggml_type type, const char * name, int rows, int dim,
                   int bits, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(rows) * dim);
    fill_gaussian(v.data(), v.size(), seed);

    std::vector<float> v_ours(v.size(), 0.0f);
    quant_dequant_via_type_traits(type, v.data(), rows, dim, v_ours.data());

    tq::ValueQuantized vq = tq::reex_turboquant_oracle_value_quantize(
        v.data(), /*batch=*/1, /*heads=*/1, /*tokens=*/rows, dim, bits, kVGroupSize);
    std::vector<float> v_oracle = tq::reex_turboquant_oracle_value_dequantize(vq);

    const double cs   = row_min_cos_sim(v_ours.data(), v_oracle.data(), rows, dim);
    const double mxad = row_max_abs_diff(v_ours.data(), v_oracle.data(), rows, dim);

    const bool ok = cs >= kStrictCosSim;
    std::fprintf(stdout,
        "[V  %-5s rows=%-4d dim=%-3d seed=0x%08x] "
        "min row cos_sim(ours, oracle)=%.7f  max|Δ|=%.2e%s\n",
        name, rows, dim, seed, cs, mxad, ok ? "" : "  <-- FAIL");
    return ok;
}

bool run_k_full_section(int rows, std::uint32_t seed) {
    const int dim = kKDim;
    std::vector<float> u(static_cast<std::size_t>(rows) * dim);
    fill_gaussian(u.data(), u.size(), seed);

    std::vector<float> u_ours_full(u.size(), 0.0f);
    quant_dequant_via_type_traits(GGML_TYPE_TQ_K3, u.data(), rows, dim, u_ours_full.data());

    const std::vector<float> u_oracle_polar = oracle_polar_full_reconstruct(u.data(), rows, dim);

    const double cs   = row_min_cos_sim(u_ours_full.data(), u_oracle_polar.data(), rows, dim);
    const double mxad = row_max_abs_diff(u_ours_full.data(), u_oracle_polar.data(), rows, dim);

    const bool ok = cs >= kStrictCosSim;
    std::fprintf(stdout,
        "[K  TQ_K3 (full P3)  rows=%-4d dim=%-3d seed=0x%08x] "
        "min row cos_sim(ours, oracle_polar)=%.7f  max|Δ|=%.2e%s\n",
        rows, dim, seed, cs, mxad, ok ? "" : "  <-- FAIL");
    return ok;
}

bool run_v_polar_full_section(enum ggml_type type, const char * name, int bits,
                              int rows, std::uint32_t seed) {
    const int dim = kVPolarDim;
    std::vector<float> u(static_cast<std::size_t>(rows) * dim);
    fill_gaussian(u.data(), u.size(), seed);

    std::vector<float> u_ours_full(u.size(), 0.0f);
    quant_dequant_via_type_traits(type, u.data(), rows, dim, u_ours_full.data());

    const std::vector<float> u_oracle_polar = oracle_v_polar_full_reconstruct(u.data(), rows, dim, bits);

    const double cs   = row_min_cos_sim(u_ours_full.data(), u_oracle_polar.data(), rows, dim);
    const double mxad = row_max_abs_diff(u_ours_full.data(), u_oracle_polar.data(), rows, dim);

    const bool ok = cs >= kStrictCosSim;
    std::fprintf(stdout,
        "[V  %-11s rows=%-4d dim=%-3d seed=0x%08x] "
        "min row cos_sim(ours, oracle_polar)=%.7f  max|Δ|=%.2e%s\n",
        name, rows, dim, seed, cs, mxad, ok ? "" : "  <-- FAIL");
    return ok;
}

}  // namespace

int main() {
    bool all_ok = true;

    const std::uint32_t seeds[] = { 0xc0ffeeu, 0xdeadu, 0xbeefu };
    const int v_dims[]    = { 32, 64, 128 };
    const int v_rows[]    = { 1, 8, 33 };
    const int k_rows[]    = { 1, 8, 33 };

    std::fprintf(stdout, "=== Section A: V cache (TQ_V2 / TQ_V4) vs oracle value_quantize ===\n");
    for (auto s : seeds) {
        for (auto rows : v_rows) {
            for (auto dim : v_dims) {
                if (!run_v_section(GGML_TYPE_TQ_V2, "TQ_V2", rows, dim, /*bits=*/2, s ^ 0x12345u)) all_ok = false;
                if (!run_v_section(GGML_TYPE_TQ_V4, "TQ_V4", rows, dim, /*bits=*/4, s ^ 0x67890u)) all_ok = false;
            }
        }
    }

    std::fprintf(stdout, "\n=== Section B: K cache (TQ_K3) full path vs oracle TurboQuantPolar (identity Pi, bits=3) ===\n");
    for (auto s : seeds) {
        for (auto rows : k_rows) {
            if (!run_k_full_section(rows, s)) all_ok = false;
        }
    }

    std::fprintf(stdout, "\n=== Section C: V cache β (TQ_V_POLAR2 / TQ_V_POLAR4) full path vs oracle TurboQuantPolar (identity Pi) ===\n");
    for (auto s : seeds) {
        for (auto rows : k_rows) {
            if (!run_v_polar_full_section(GGML_TYPE_TQ_V_POLAR2, "TQ_V_POLAR2", kVPolar2Bits, rows, s ^ 0xa5a5u)) all_ok = false;
            if (!run_v_polar_full_section(GGML_TYPE_TQ_V_POLAR4, "TQ_V_POLAR4", kVPolar4Bits, rows, s ^ 0x5a5au)) all_ok = false;
        }
    }

    std::fprintf(stdout, "\ntest-reex-turboquant-vs-oracle: %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
