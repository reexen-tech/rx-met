// SPDX-License-Identifier: MIT
//
// test-reex-turboquant-p4a21-diagnostics
//
// REEX_TURBOQUANT P4 A2.1 root-cause diagnostics — A3 + A1 combined.
//
// Background
// ----------
// P4 A2 (attention-dense vRAM-value contrast) showed an unexpected severe
// PPL/acc degradation for TurboQuant on Qwen3-30B-A3B-Instruct-2507 (pure
// attention-dense), while P3.6.x had shown TQ on Qwen3.5-35B-A3B (attention +
// SSM hybrid) was acceptable.  B2 follow-up further showed ΔPPL is **not**
// monotonically increasing in ctx (n_chunks=1 mini PPL: ctx=4k yields the
// largest ΔPPL, ctx=16k smallest), pointing at single-layer / per-chunk
// quantization noise rather than token-level cumulative drift.
//
// This test instruments the TQ algorithm itself (independent of any model)
// to disambiguate two hypotheses on the algorithmic side:
//
//   A3.  Systematic bias in single-step quantize→dequantize.
//        Tests for non-zero-mean reconstruction error on zero-mean Gaussian
//        input.  We flag |mean(err)| > 5σ_mean as a systematic-bias FAIL;
//        all types (subjects + controls) must pass.  K3 P3 (TheTom turbo3_0)
//        is a Lloyd-Max-optimal scalar quantizer over rotated unit vectors
//        so the per-element error is mean-zero in expectation.
//
//   A1.  Multi-pass roundtrip stability (cos_sim decay curve).
//        Repeatedly quantize-dequantize the same vector N times and record
//        cos_sim / norm_ratio after each pass.  A "good" lossy quantizer
//        reaches a fixed point after 1–2 passes (the second pass's input is
//        already in the codebook's range so cos_sim stays put).  A
//        pathological quantizer drifts further with each pass.  K3 P3
//        (TheTom turbo3_0) uses the norm-correction trick
//        `corrected_norm = ‖x‖ / ‖recon_unit‖` so that ‖dequant‖ ≡ ‖x‖
//        and a second pass lands on identical centroids — strict
//        idempotency, drift = 0 by construction.
//
// Coverage
// --------
//   types        : TQ_K3, TQ_V2, TQ_V4 (subjects), Q4_0, Q8_0 (controls)
//   inputs       : iid N(0, 1) Gaussian, multiple seeds, multiple sizes
//   metrics      : mean(err), std(err), cos_sim, norm_rel, drift over N passes
//
// The test prints a Markdown-style table to stdout for direct paste into
// the P4 A2.1 summary report.
//
// Contract gates (T1 + T2, added 2026-04-30 as follow-up to Stage-0)
// ------------------------------------------------------------------
// Beyond the original "controls misbehave → fail" check, two subject-side
// contract gates are now enforced and contribute to the binary's exit code:
//
//   T1.  Multi-pass cos_sim drift ≤ 1e-3 for **every** type (subjects + controls).
//        A correctly designed lossy quantizer is idempotent: (dq ∘ q)^n = (dq ∘ q)
//        because the dequantized output already lives on the codebook lattice.
//        After the P4 A2.2 K3 P3 upgrade (TheTom turbo3_0, full 3-bit
//        PolarQuant + norm correction trick), TQ_K3 also measures drift = 0
//        by construction: the stored `corrected_norm = ‖x‖/‖recon_unit‖`
//        guarantees ‖dequant‖ ≡ ‖x‖ so the second pass's input lives on the
//        same unit-sphere bucket vector and lands on identical indices.
//        Q4_0 / Q8_0 / TQ_V2 / TQ_V4 also measure drift = 0 (per-block
//        min/max scalar quantizers).  All five types must PASS.
//
//   T2.  |mean_err z-score| < 5σ for **every** type.  Standard significance
//        threshold for "no zero-mean violation".  Met by every type today
//        (max z ≈ 1.6, well below 5); the gate guards against future
//        regressions that would introduce systematic bias in the
//        reconstruction.
//
// Net consequence on ctest status (vs Stage-0 land):
//   - Pre-P3: 4 expected FAIL (block-roundtrip K3, vs-oracle K3 full,
//             kv-types K3 CPY, p4a21-diagnostics T1 K3 drift), all
//             anchored on the K3 P1 cos_sim 0.023 gap + 0.021 drift.
//   - Post-P3 (this commit): 0 expected FAIL — all four close
//     simultaneously since K3 P3 closes both the cos_sim gap (≈ 0.997)
//     and the drift (= 0 by construction).
//
// Anchors:
//   - T1 threshold 1e-3 matches the existing convergence detector
//     (`|Δcos_sim(pass8 − pass7)| < 1e-3`) used in `run_a1` below.
//   - T2 threshold 5σ is the standard particle-physics-style discovery
//     significance (≈ p < 5.7e-7 for two-tailed Gaussian); for systematic-
//     bias detection at 32 seeds × 8192 elements per seed it's a generous
//     guard that won't be tripped by stochastic fluctuations.
//
// Controls (Q4_0, Q8_0) keep their existing role: if a control trips T1/T2
// the test rig is invalid and the binary returns non-zero with a louder
// stderr message ("test rig invalid").

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void fill_gaussian(float * out, std::size_t n, std::uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = nd(rng);
    }
}

double mean(const float * a, std::size_t n) {
    long double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += a[i];
    return static_cast<double>(s / static_cast<long double>(n));
}

double stdev(const float * a, std::size_t n, double m) {
    long double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const long double d = static_cast<long double>(a[i]) - m;
        s += d * d;
    }
    return static_cast<double>(std::sqrt(s / static_cast<long double>(n)));
}

double cos_sim(const float * a, const float * b, std::size_t n) {
    long double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dot += static_cast<long double>(a[i]) * b[i];
        na  += static_cast<long double>(a[i]) * a[i];
        nb  += static_cast<long double>(b[i]) * b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0;
    return static_cast<double>(dot / std::sqrt(na * nb));
}

double l2(const float * a, std::size_t n) {
    long double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += static_cast<long double>(a[i]) * a[i];
    return static_cast<double>(std::sqrt(s));
}

struct TypeCfg {
    ggml_type    type;
    const char * name;
    bool         is_k_path;     // true → fed unscaled iid Gaussian
    bool         is_control;    // true → q4_0 / q8_0 baselines that *must* pass A3
};

// Round n down to the nearest multiple of blk.
std::int64_t align_down(std::int64_t n, std::int64_t blk) {
    return (n / blk) * blk;
}

// Single-pass quantize → dequantize.
void roundtrip(const TypeCfg & c,
               const float * src,
               float * dst,
               std::int64_t n,
               std::vector<std::uint8_t> & qbuf) {
    const ggml_type_traits     * tt     = ggml_get_type_traits(c.type);
    const ggml_type_traits_cpu * tt_cpu = ggml_get_type_traits_cpu(c.type);
    const std::int64_t blk   = tt->blck_size;
    const std::size_t  nblks = static_cast<std::size_t>(n / blk);
    qbuf.assign(nblks * tt->type_size, 0);
    tt_cpu->from_float(src, qbuf.data(), n);
    tt->to_float(qbuf.data(), dst, n);
}

struct A3Stat {
    std::string type_name;
    int    n_seeds;
    std::int64_t n_per_seed;
    double mean_err_avg;     // mean of (rec-src), averaged over seeds
    double mean_err_stderr;  // stderr of the per-seed means → significance test
    double mean_err_zscore;  // |mean_err_avg| / mean_err_stderr
    double std_err_avg;      // stddev of (rec-src), averaged over seeds
    double cos_sim_avg;
    double norm_rel_avg;     // |1 - ||rec||/||src|||
    bool   systematic_bias;  // |zscore| > 5
};

A3Stat run_a3(const TypeCfg & c, std::int64_t n_target, int n_seeds) {
    const ggml_type_traits * tt = ggml_get_type_traits(c.type);
    const std::int64_t n = align_down(n_target, tt->blck_size);

    std::vector<float> src(n), rec(n), err(n);
    std::vector<std::uint8_t> qbuf;

    std::vector<double> per_seed_mean_err;
    double sum_std_err   = 0.0;
    double sum_cos_sim   = 0.0;
    double sum_norm_rel  = 0.0;

    for (int s = 0; s < n_seeds; ++s) {
        const std::uint32_t seed = 0x1000u + 17u * static_cast<std::uint32_t>(s);
        fill_gaussian(src.data(), src.size(), seed, 1.0f);
        if (!c.is_k_path) {
            for (auto & v : src) v *= 0.3f;  // V tensors have smaller dynamic range
        }
        roundtrip(c, src.data(), rec.data(), n, qbuf);

        for (std::int64_t i = 0; i < n; ++i) err[i] = rec[i] - src[i];

        const double m_e = mean(err.data(), n);
        const double s_e = stdev(err.data(), n, m_e);
        const double cs  = cos_sim(src.data(), rec.data(), n);
        const double na  = l2(src.data(), n);
        const double nb  = l2(rec.data(), n);
        const double nrel = na > 1e-12 ? std::abs(na - nb) / na : 0.0;

        per_seed_mean_err.push_back(m_e);
        sum_std_err  += s_e;
        sum_cos_sim  += cs;
        sum_norm_rel += nrel;
    }

    double mean_err_avg = 0.0;
    for (double m : per_seed_mean_err) mean_err_avg += m;
    mean_err_avg /= per_seed_mean_err.size();
    double var = 0.0;
    for (double m : per_seed_mean_err) {
        const double d = m - mean_err_avg;
        var += d * d;
    }
    var /= per_seed_mean_err.size();
    const double sample_std = std::sqrt(var);
    const double stderr_mean = sample_std / std::sqrt(static_cast<double>(per_seed_mean_err.size()));
    const double zscore = stderr_mean > 1e-30 ? std::abs(mean_err_avg) / stderr_mean : 0.0;

    A3Stat r;
    r.type_name      = c.name;
    r.n_seeds        = n_seeds;
    r.n_per_seed     = n;
    r.mean_err_avg   = mean_err_avg;
    r.mean_err_stderr= stderr_mean;
    r.mean_err_zscore= zscore;
    r.std_err_avg    = sum_std_err / n_seeds;
    r.cos_sim_avg    = sum_cos_sim / n_seeds;
    r.norm_rel_avg   = sum_norm_rel / n_seeds;
    r.systematic_bias = zscore > 5.0;
    return r;
}

struct A1Stat {
    std::string type_name;
    std::int64_t n;
    int    n_passes;
    std::vector<double> cos_sim_vs_orig;  // length n_passes (1-indexed semantically)
    std::vector<double> norm_rel_vs_orig;
    bool   converges;       // cos_sim difference between pass N and pass N-1 < 1e-3 by N=8
    double cos_sim_pass1;
    double cos_sim_passN;
    double cos_sim_drift;   // cos_sim_pass1 - cos_sim_passN (positive = decay)
};

A1Stat run_a1(const TypeCfg & c, std::int64_t n_target, int n_passes, std::uint32_t seed) {
    const ggml_type_traits * tt = ggml_get_type_traits(c.type);
    const std::int64_t n = align_down(n_target, tt->blck_size);

    std::vector<float> src_orig(n), cur(n), nxt(n);
    std::vector<std::uint8_t> qbuf;

    fill_gaussian(src_orig.data(), src_orig.size(), seed, 1.0f);
    if (!c.is_k_path) {
        for (auto & v : src_orig) v *= 0.3f;
    }
    cur = src_orig;

    A1Stat r;
    r.type_name = c.name;
    r.n         = n;
    r.n_passes  = n_passes;

    for (int p = 1; p <= n_passes; ++p) {
        roundtrip(c, cur.data(), nxt.data(), n, qbuf);
        const double cs   = cos_sim(src_orig.data(), nxt.data(), n);
        const double na   = l2(src_orig.data(), n);
        const double nb   = l2(nxt.data(),       n);
        const double nrel = na > 1e-12 ? std::abs(na - nb) / na : 0.0;
        r.cos_sim_vs_orig.push_back(cs);
        r.norm_rel_vs_orig.push_back(nrel);
        cur = nxt;
    }

    r.cos_sim_pass1 = r.cos_sim_vs_orig.front();
    r.cos_sim_passN = r.cos_sim_vs_orig.back();
    r.cos_sim_drift = r.cos_sim_pass1 - r.cos_sim_passN;

    bool converged = false;
    if (r.cos_sim_vs_orig.size() >= 8) {
        const double d = std::abs(r.cos_sim_vs_orig[7] - r.cos_sim_vs_orig[6]);
        converged = d < 1e-3;
    }
    r.converges = converged;
    return r;
}

}  // namespace

int main() {
    std::printf("# P4 A2.1 TurboQuant root-cause diagnostics (A3 + A1)\n\n");
    std::printf("Build : %s\n", __DATE__ " " __TIME__);
    std::printf("\n");

    static const TypeCfg cfgs[] = {
        { GGML_TYPE_TQ_K3, "TQ_K3", /*is_k_path=*/true,  /*is_control=*/false },
        { GGML_TYPE_TQ_V2, "TQ_V2", /*is_k_path=*/false, /*is_control=*/false },
        { GGML_TYPE_TQ_V4, "TQ_V4", /*is_k_path=*/false, /*is_control=*/false },
        { GGML_TYPE_Q4_0,  "Q4_0",  /*is_k_path=*/true,  /*is_control=*/true  },
        { GGML_TYPE_Q8_0,  "Q8_0",  /*is_k_path=*/true,  /*is_control=*/true  },
    };

    // -------------------------------------------------------------------- A3
    std::printf("## A3 — single-pass error stats (zero-mean Gaussian input)\n\n");
    std::printf("Input: %d seeds × %lld elements per seed (%s scaled), zero-mean Gaussian σ=1.0.\n",
                32, 8192LL, "K-path unscaled / V-path ×0.3");
    std::printf("Significance test: mark systematic-bias FAIL if |mean(err)| > 5×stderr_mean.\n\n");

    std::printf("| type  |    n  |seeds| mean(err)      | stderr(mean)  | z-score | std(err) |  cos_sim  | |1-||r||/||s||| | bias? |\n");
    std::printf("|-------|------:|----:|---------------:|--------------:|--------:|---------:|----------:|----------------:|------:|\n");

    bool controls_pass_a3 = true;
    std::vector<A3Stat> a3_results;
    for (const auto & c : cfgs) {
        A3Stat s = run_a3(c, /*n_target=*/8192, /*n_seeds=*/32);
        std::printf("| %-5s | %5lld | %3d | %+.6e | %.6e | %7.2f | %.6f | %.6f  |   %.6f      | %s |\n",
            s.type_name.c_str(),
            (long long) s.n_per_seed, s.n_seeds,
            s.mean_err_avg, s.mean_err_stderr, s.mean_err_zscore,
            s.std_err_avg, s.cos_sim_avg, s.norm_rel_avg,
            s.systematic_bias ? "YES " : "no  ");
        a3_results.push_back(s);
        if (c.is_control && s.systematic_bias) {
            controls_pass_a3 = false;
            std::fprintf(stderr,
                "ERROR: control %s reports systematic bias (|mean|=%.3e, stderr=%.3e, z=%.2f); "
                "this would invalidate the entire test rig.\n",
                c.name, s.mean_err_avg, s.mean_err_stderr, s.mean_err_zscore);
        }
    }
    std::printf("\n");

    // -------------------------------------------------------------------- A1
    std::printf("## A1 — multi-pass roundtrip cos_sim decay (Gaussian, single seed, %d passes)\n\n", 10);
    std::printf("Repeatedly apply quantize→dequantize and report cos_sim / norm_rel against the *original* input.\n");
    std::printf("Convergence criterion: |Δcos_sim(pass8 − pass7)| < 1e-3.\n\n");

    const int passes = 10;
    std::printf("| type  |  pass=1  |  pass=2  |  pass=3  |  pass=5  |  pass=10 | drift   | converges? |\n");
    std::printf("|-------|---------:|---------:|---------:|---------:|---------:|--------:|-----------:|\n");

    bool controls_pass_a1 = true;
    std::vector<A1Stat> a1_results;
    for (const auto & c : cfgs) {
        A1Stat s = run_a1(c, /*n_target=*/8192, passes, /*seed=*/0xc0ffeeu);
        std::printf("| %-5s | %.6f | %.6f | %.6f | %.6f | %.6f | %+.5f | %s |\n",
            s.type_name.c_str(),
            s.cos_sim_vs_orig[0],
            s.cos_sim_vs_orig[1],
            s.cos_sim_vs_orig[2],
            s.cos_sim_vs_orig[4],
            s.cos_sim_vs_orig[9],
            s.cos_sim_drift,
            s.converges ? "yes        " : "NO         ");
        a1_results.push_back(s);
        if (c.is_control && !s.converges) {
            controls_pass_a1 = false;
            std::fprintf(stderr,
                "ERROR: control %s does not converge in 8 passes (drift=%.4f); "
                "this would invalidate the test rig.\n",
                c.name, s.cos_sim_drift);
        }
    }
    std::printf("\n");

    // ---------------------------------------------------------------- summary
    std::printf("## Diagnostic summary\n\n");
    std::printf("Controls (Q4_0 / Q8_0) A3 systematic-bias check: **%s**\n",
                controls_pass_a3 ? "PASS" : "FAIL (test rig invalid!)");
    std::printf("Controls (Q4_0 / Q8_0) A1 multi-pass convergence : **%s**\n",
                controls_pass_a1 ? "PASS" : "FAIL (test rig invalid!)");
    std::printf("\n");
    std::printf("Subject results (informational, no blocking):\n");
    for (const auto & s : a3_results) {
        if (s.type_name == "Q4_0" || s.type_name == "Q8_0") continue;
        std::printf("  - %s : single-pass cos_sim=%.4f, |mean_err|=%.3e (z=%.1f%s), "
                    "norm_rel=%.4f%s\n",
                    s.type_name.c_str(),
                    s.cos_sim_avg, std::abs(s.mean_err_avg), s.mean_err_zscore,
                    s.systematic_bias ? ", **systematic bias**" : "",
                    s.norm_rel_avg,
                    s.norm_rel_avg > 0.10 ? ", **norm shrinkage**" : "");
    }
    for (const auto & s : a1_results) {
        if (s.type_name == "Q4_0" || s.type_name == "Q8_0") continue;
        std::printf("  - %s : multi-pass cos_sim 1→%d drift = %+.5f%s\n",
                    s.type_name.c_str(), s.n_passes, s.cos_sim_drift,
                    s.converges ? "" : ", **does not converge**");
    }
    std::printf("\n");

    // ------------------------------------------------- contract gates (T1+T2)
    // Per the header: T1 = multi-pass drift ≤ 1e-3 for every type;
    //                 T2 = |z-score| < 5σ for every type.
    // After P4 A2.2 K3 P3 upgrade (TheTom turbo3_0 + norm correction trick),
    // **every** (type, gate) pair must PASS — TQ_K3 drift is now 0 by
    // construction (norm correction guarantees ‖dequant‖ ≡ ‖x‖).
    constexpr double kT1DriftThresh   = 1e-3;
    constexpr double kT2ZScoreThresh  = 5.0;

    std::printf("## Contract gates (T1: drift ≤ %.0e, T2: |z| < %.0f)\n\n",
                kT1DriftThresh, kT2ZScoreThresh);
    std::printf("| type  |   T1 drift  |  T1   |   T2 |z-score|  |  T2   |\n");
    std::printf("|-------|------------:|:-----:|---------------:|:------:|\n");

    bool contracts_pass = true;
    // a3_results and a1_results were populated in the same TypeCfg order, so
    // both vectors are aligned by index.  Iterate in lock-step and gate.
    for (std::size_t i = 0; i < a3_results.size(); ++i) {
        const auto & a3 = a3_results[i];
        const auto & a1 = a1_results[i];
        const bool t1_ok = a1.cos_sim_drift <= kT1DriftThresh;
        const bool t2_ok = a3.mean_err_zscore < kT2ZScoreThresh;
        std::printf("| %-5s | %+11.5f | %-5s | %14.2f | %-5s |\n",
            a1.type_name.c_str(),
            a1.cos_sim_drift, t1_ok ? "PASS" : "FAIL",
            a3.mean_err_zscore, t2_ok ? "PASS" : "FAIL");
        if (!t1_ok || !t2_ok) contracts_pass = false;
    }
    std::printf("\n");

    const bool rig_ok       = controls_pass_a3 && controls_pass_a1;
    const bool everything_ok = rig_ok && contracts_pass;

    if (!rig_ok) {
        std::fprintf(stderr,
            "\ntest-reex-turboquant-p4a21-diagnostics: FAIL (controls misbehaved — test rig invalid)\n");
        return 1;
    }
    if (!contracts_pass) {
        std::fprintf(stderr,
            "\ntest-reex-turboquant-p4a21-diagnostics: FAIL "
            "(controls OK but T1/T2 contract gate(s) tripped — see above table)\n");
        return 1;
    }
    (void) everything_ok;
    std::fprintf(stdout, "\ntest-reex-turboquant-p4a21-diagnostics: PASS (controls OK; T1/T2 contracts OK)\n");
    return 0;
}
