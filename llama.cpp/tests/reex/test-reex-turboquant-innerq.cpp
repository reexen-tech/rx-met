// Tests for `reex_turboquant::InnerQ` and the underlying `innerq_algo` pure
// host functions.
//
// Stage-1 scope: algorithm correctness only.  KV cache integration, ggml_tensor
// scale_inv plumbing, and the runtime cparams flag are validated in later
// stages with end-to-end PPL/HellaSwag eval.
//
// What we verify here:
//   * `accumulate` is associative across multiple chunks (streaming use case).
//   * `compute_scale` early-returns for `count == 0`, all-zero data, and
//     well-balanced channels (max_ratio < skip_threshold).
//   * `compute_scale` produces scales in [clamp_lo, clamp_hi] with
//     scale * scale_inv == 1.0.
//   * Outlier channels (high RMS) get scale < 1; quiet channels get scale > 1.
//   * Per-channel RMS spread *after* applying scale is strictly smaller than
//     before — i.e. equalization actually equalizes.
//   * `InnerQ` lifecycle: init → accumulate → freeze flips frozen/active
//     correctly; reset clears state.
//
// Pure-CPU, fast, no I/O.

#include "reex_turboquant_innerq.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace tq = reex_turboquant;

namespace {

// Synthetic K with linear per-channel RMS ramp.
//
// channel_rms[j] = base + (peak - base) * j / (channel_dim - 1)
//   so for base=1.0, peak=5.0 the max/min ratio is 5x.
//
// Each element ~ N(0, channel_rms[j]^2).  Deterministic via fixed seed.
std::vector<float> make_synthetic_k_ramp(uint32_t n_tokens,
                                         uint32_t channel_dim,
                                         float    base,
                                         float    peak,
                                         uint32_t seed = 42) {
    std::mt19937                          rng(seed);
    std::normal_distribution<float>       dist(0.0f, 1.0f);
    std::vector<float>                    k((size_t) n_tokens * channel_dim);
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t j = 0; j < channel_dim; ++j) {
            const float rms = base + (peak - base) * (float) j / (float) (channel_dim - 1);
            k[(size_t) t * channel_dim + j] = rms * dist(rng);
        }
    }
    return k;
}

// Per-channel RMS of K, [n_tokens, channel_dim] row-major.
std::vector<double> per_channel_rms(const std::vector<float> & k,
                                    uint32_t                   n_tokens,
                                    uint32_t                   channel_dim) {
    std::vector<double> sq(channel_dim, 0.0);
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t j = 0; j < channel_dim; ++j) {
            const double v = (double) k[(size_t) t * channel_dim + j];
            sq[j] += v * v;
        }
    }
    std::vector<double> rms(channel_dim);
    for (uint32_t j = 0; j < channel_dim; ++j) {
        rms[j] = std::sqrt(sq[j] / (double) n_tokens);
    }
    return rms;
}

double rms_max_over_min(const std::vector<double> & rms) {
    double mx = 0.0, mn = 1e30;
    for (double r : rms) {
        if (r > mx) mx = r;
        if (r > 0 && r < mn) mn = r;
    }
    return (mn > 0.0) ? (mx / mn) : 0.0;
}

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s:%d  %s\n", __FILE__, __LINE__, msg); \
        return false; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// 1) accumulate is associative across chunks.
// ---------------------------------------------------------------------------
bool test_accumulate_streaming() {
    const uint32_t channel_dim = 16;
    const uint32_t n1 = 100, n2 = 250;
    const auto k = make_synthetic_k_ramp(n1 + n2, channel_dim, 1.0f, 3.0f);

    std::vector<double> sq_one(channel_dim, 0.0);
    uint64_t            count_one = 0;
    tq::innerq_algo::accumulate(sq_one.data(), &count_one,
                                k.data(), n1 + n2, channel_dim);

    std::vector<double> sq_two(channel_dim, 0.0);
    uint64_t            count_two = 0;
    tq::innerq_algo::accumulate(sq_two.data(), &count_two,
                                k.data(), n1, channel_dim);
    tq::innerq_algo::accumulate(sq_two.data(), &count_two,
                                k.data() + (size_t) n1 * channel_dim, n2, channel_dim);

    EXPECT(count_one == count_two, "count mismatch across chunks");
    for (uint32_t j = 0; j < channel_dim; ++j) {
        const double diff = std::fabs(sq_one[j] - sq_two[j]);
        EXPECT(diff < 1e-9 * (1.0 + std::fabs(sq_one[j])),
               "sq_accum mismatch across chunks");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 2) compute_scale early-return semantics.
// ---------------------------------------------------------------------------
bool test_compute_scale_early_returns() {
    const uint32_t channel_dim = 16;
    std::vector<float>  scale(channel_dim, 0.0f);
    std::vector<float>  scale_inv(channel_dim, 0.0f);
    std::vector<double> sq_zero(channel_dim, 0.0);

    // (a) count == 0 → identity, returns false.
    bool nontrivial = tq::innerq_algo::compute_scale(
        sq_zero.data(), 0, scale.data(), scale_inv.data(), channel_dim);
    EXPECT(!nontrivial, "count==0 should be no-op");
    for (uint32_t j = 0; j < channel_dim; ++j) {
        EXPECT(scale[j] == 1.0f && scale_inv[j] == 1.0f,
               "count==0 must fill identity");
    }

    // (b) all-zero data → identity, returns false.
    nontrivial = tq::innerq_algo::compute_scale(
        sq_zero.data(), 1024, scale.data(), scale_inv.data(), channel_dim);
    EXPECT(!nontrivial, "all-zero data should be no-op");
    for (uint32_t j = 0; j < channel_dim; ++j) {
        EXPECT(scale[j] == 1.0f && scale_inv[j] == 1.0f,
               "all-zero data must fill identity");
    }

    // (c) all-equal channels → identity, returns false (skip threshold 1.2).
    std::vector<double> sq_equal(channel_dim, 100.0);
    nontrivial = tq::innerq_algo::compute_scale(
        sq_equal.data(), 100, scale.data(), scale_inv.data(), channel_dim);
    EXPECT(!nontrivial, "balanced channels should be no-op");
    for (uint32_t j = 0; j < channel_dim; ++j) {
        EXPECT(scale[j] == 1.0f && scale_inv[j] == 1.0f,
               "balanced channels must fill identity");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 3) compute_scale produces correctly-shaped scales for outlier inputs.
// ---------------------------------------------------------------------------
bool test_compute_scale_outlier_shape() {
    const uint32_t channel_dim = 32;
    std::vector<double> sq(channel_dim, 1.0 * 1024.0);  // base RMS = 1.0
    sq[0]  = 25.0 * 1024.0;  // RMS = 5.0  (outlier)
    sq[31] = 0.04 * 1024.0;  // RMS = 0.2  (quiet)

    std::vector<float> scale(channel_dim), scale_inv(channel_dim);
    bool nontrivial = tq::innerq_algo::compute_scale(
        sq.data(), 1024, scale.data(), scale_inv.data(), channel_dim,
        /*strength=*/0.5f, /*lo=*/0.5f, /*hi=*/2.0f);

    EXPECT(nontrivial, "outlier input must trigger non-trivial scale");

    // Outlier channel: RMS=5.0, mean_rms ≈ 1.0 → ratio<1 → scale<1.
    EXPECT(scale[0] < 1.0f, "outlier channel must be suppressed (scale<1)");
    EXPECT(scale[0] >= 0.5f, "outlier channel scale must clamp at 0.5");

    // Quiet channel: RMS=0.2 → ratio>1 → scale>1.
    EXPECT(scale[31] > 1.0f, "quiet channel must be boosted (scale>1)");
    EXPECT(scale[31] <= 2.0f, "quiet channel scale must clamp at 2.0");

    // Clamp range + roundtrip.
    for (uint32_t j = 0; j < channel_dim; ++j) {
        EXPECT(scale[j] >= 0.5f && scale[j] <= 2.0f, "scale out of clamp range");
        const float prod = scale[j] * scale_inv[j];
        EXPECT(std::fabs(prod - 1.0f) < 1e-5f, "scale * scale_inv != 1");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 4) Equalization: post-scale RMS spread is strictly smaller.
// ---------------------------------------------------------------------------
bool test_equalization_reduces_spread() {
    const uint32_t channel_dim = 64;
    const uint32_t n_tokens    = 2048;
    auto k = make_synthetic_k_ramp(n_tokens, channel_dim,
                                   /*base=*/1.0f, /*peak=*/5.0f);

    // Before: per-channel RMS spans 1.0..5.0 → ratio ≈ 5x.
    auto rms_pre        = per_channel_rms(k, n_tokens, channel_dim);
    const double spread_pre = rms_max_over_min(rms_pre);
    EXPECT(spread_pre > 4.0, "synthetic K should have wide RMS spread");

    // Calibrate.
    std::vector<double> sq(channel_dim, 0.0);
    uint64_t            count = 0;
    tq::innerq_algo::accumulate(sq.data(), &count,
                                k.data(), n_tokens, channel_dim);
    std::vector<float> scale(channel_dim), scale_inv(channel_dim);
    bool nontrivial = tq::innerq_algo::compute_scale(
        sq.data(), count, scale.data(), scale_inv.data(), channel_dim);
    EXPECT(nontrivial, "ramp K must trigger non-trivial scale");

    // After: scale K and recompute spread.
    std::vector<float> k_scaled(k.size());
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t j = 0; j < channel_dim; ++j) {
            k_scaled[(size_t) t * channel_dim + j] =
                k[(size_t) t * channel_dim + j] * scale[j];
        }
    }
    auto rms_post = per_channel_rms(k_scaled, n_tokens, channel_dim);
    const double spread_post = rms_max_over_min(rms_post);

    fprintf(stdout, "[INFO] equalization: spread %.3f → %.3f (×%.2f better)\n",
            spread_pre, spread_post, spread_pre / spread_post);
    EXPECT(spread_post < spread_pre, "equalization must reduce RMS spread");
    // Math: with strength=0.5 (sqrt), scale[outlier]=sqrt(mean/peak)=sqrt(0.3)
    // ≈0.548 (~clamps at 0.5); scale[quiet]=sqrt(mean/base)=sqrt(1.5)≈1.22.
    // For a linear 1..5 ramp, post-spread is ≈ (5×0.5)/(1×1.22) ≈ 2.05x —
    // i.e. ~2.4x improvement, not full equalization (intentional, prevents
    // over-correction amplifying quantization noise).  Require ≥50% reduction.
    EXPECT(spread_post * 2.0 < spread_pre,
           "equalization must cut RMS spread at least in half");
    return true;
}

// ---------------------------------------------------------------------------
// 5) InnerQ class lifecycle: init / accumulate / freeze / reset.
// ---------------------------------------------------------------------------
bool test_innerq_lifecycle() {
    const uint32_t n_layer_kv = 4;
    const uint32_t channel_dim = 16;
    const uint32_t n_tokens   = 512;

    tq::InnerQ iq;
    iq.init(n_layer_kv, channel_dim, /*target_tokens=*/512);

    EXPECT(iq.n_layer_kv() == n_layer_kv, "n_layer_kv mismatch");
    EXPECT(iq.channel_dim() == channel_dim, "channel_dim mismatch");
    for (uint32_t il = 0; il < n_layer_kv; ++il) {
        EXPECT(!iq.is_frozen(il), "layer should start unfrozen");
        EXPECT(!iq.is_active(il), "layer should start inactive");
        EXPECT(iq.accumulated_count(il) == 0, "count should start zero");
    }

    // Layer 0: outlier ramp → freeze → must be active.
    auto k_ramp = make_synthetic_k_ramp(n_tokens, channel_dim, 1.0f, 5.0f);
    iq.accumulate_host(0, k_ramp.data(), n_tokens);
    EXPECT(iq.accumulated_count(0) == n_tokens, "count after accumulate");
    iq.freeze(0);
    EXPECT(iq.is_frozen(0), "layer 0 should be frozen");
    EXPECT(iq.is_active(0), "layer 0 with outliers should be active");

    // Post-freeze accumulate must be a no-op.
    iq.accumulate_host(0, k_ramp.data(), n_tokens);
    EXPECT(iq.accumulated_count(0) == n_tokens, "post-freeze accumulate leaked");

    // Layer 1: balanced data → freeze → must be inactive (skip threshold).
    std::mt19937                    rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float>              k_balanced((size_t) n_tokens * channel_dim);
    for (auto & x : k_balanced) x = dist(rng);
    iq.accumulate_host(1, k_balanced.data(), n_tokens);
    iq.freeze(1);
    EXPECT(iq.is_frozen(1), "layer 1 should be frozen");
    EXPECT(!iq.is_active(1), "layer 1 (balanced) should be inactive");

    // Read back scale; layer 0 must be non-identity, layer 1 must be identity.
    std::vector<float> scale0(channel_dim), scale1(channel_dim);
    EXPECT(iq.get_scale_host(0, scale0.data()), "frozen layer must report frozen=true");
    EXPECT(iq.get_scale_host(1, scale1.data()), "frozen layer must report frozen=true");
    bool any_non_id_0 = false;
    for (float s : scale0) if (std::fabs(s - 1.0f) > 1e-5f) { any_non_id_0 = true; break; }
    EXPECT(any_non_id_0, "active layer must have non-identity scale");
    for (float s : scale1) {
        EXPECT(std::fabs(s - 1.0f) < 1e-5f, "inactive layer must have identity scale");
    }

    // freeze_all should be idempotent for already-frozen layers.
    iq.freeze_all();
    EXPECT(iq.is_frozen(0) && iq.is_frozen(1), "layers 0/1 still frozen");
    EXPECT(iq.is_frozen(2) && iq.is_frozen(3), "layers 2/3 frozen via freeze_all");
    EXPECT(!iq.is_active(2), "layer 2 (no data) cannot be active");

    // reset wipes everything.
    iq.reset();
    for (uint32_t il = 0; il < n_layer_kv; ++il) {
        EXPECT(!iq.is_frozen(il), "reset should clear frozen");
        EXPECT(iq.accumulated_count(il) == 0, "reset should clear count");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 6) Out-of-range il is gracefully handled.
// ---------------------------------------------------------------------------
bool test_out_of_range_layer() {
    tq::InnerQ iq;
    iq.init(2, 16);

    // None of these should crash.
    iq.accumulate_host(99, nullptr, 0);
    iq.freeze(99);
    EXPECT(!iq.is_frozen(99), "out-of-range il must report unfrozen");
    EXPECT(!iq.is_active(99), "out-of-range il must report inactive");
    EXPECT(iq.accumulated_count(99) == 0, "out-of-range il count must be 0");

    // get_scale_host on out-of-range fills identity and returns false.
    std::vector<float> scale(16, 999.0f);
    EXPECT(!iq.get_scale_host(99, scale.data()), "OOR get_scale_host must return false");
    for (float s : scale) {
        EXPECT(s == 1.0f, "OOR get_scale_host must fill identity");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 7) Default-constructed instance (init never called) is a complete no-op.
// ---------------------------------------------------------------------------
bool test_uninitialized_noop() {
    tq::InnerQ iq;
    EXPECT(iq.n_layer_kv() == 0, "default n_layer_kv is 0");
    EXPECT(iq.channel_dim() == 0, "default channel_dim is 0");

    iq.accumulate_host(0, nullptr, 0);
    iq.freeze(0);
    iq.freeze_all();
    iq.reset();
    EXPECT(!iq.is_frozen(0), "uninitialized must report unfrozen");
    return true;
}

// ===========================================================================
// Stage 2b — ggml-aware interface tests.
// ===========================================================================
//
// These verify the ggml_tensor lifecycle gating (sq_accum nullable post
// freeze, scale nullable pre-freeze / inactive), the host pending counter
// triggering finalize at the right time, and that finalize_freeze_pending
// correctly re-uploads the computed scale to the backend (round-trip via
// ggml_backend_tensor_get).  We use the CPU backend so the test runs
// everywhere — Stage 4 will add a CUDA-specific port.

bool test_ggml_init_rejects_invalid_args() {
    tq::InnerQ iq;
    // nullptr buft is invalid (no backend memory to allocate into).
    EXPECT(!iq.init_ggml(nullptr, /*n_layer_kv=*/4, /*channel_dim=*/16),
           "init_ggml(nullptr) must return false");
    EXPECT(!iq.ggml_ready(), "ggml_ready false on null buft");
    // Stage 4 (2026-05-09) — non-host buft is now LEGAL: CUDA and other
    // device bufts are accepted because the WHT op natively atomicAdds
    // into sq_accum and finalize_one's tensor_get/set transparently
    // bridges device <-> host.  We don't try a CUDA buft here because
    // this unit test must run on machines without CUDA — the ggml-aware
    // CUDA parity test in tests/test-backend-ops covers that path.
    return true;
}

bool test_ggml_init_basic() {
    auto buft = ggml_backend_cpu_buffer_type();
    tq::InnerQ iq;
    EXPECT(iq.init_ggml(buft, 4, 16),
           "init_ggml on CPU host buft must succeed");
    EXPECT(iq.ggml_ready(), "ggml_ready true after init_ggml");

    // Pre-freeze: sq_accum non-null, scale/scale_inv null (no calibration done).
    for (uint32_t il = 0; il < 4; ++il) {
        EXPECT(iq.get_sq_accum(il) != nullptr,
               "sq_accum must be non-null pre-freeze");
        EXPECT(iq.get_scale_for_k_write(il) == nullptr,
               "scale must be null pre-freeze");
        EXPECT(iq.get_scale_inv_for_q_read(il) == nullptr,
               "scale_inv must be null pre-freeze");
        EXPECT(!iq.is_frozen(il), "default unfrozen");
        EXPECT(!iq.should_finalize(il), "default no finalize trigger");
    }
    return true;
}

bool test_notify_threshold_crossing() {
    auto buft = ggml_backend_cpu_buffer_type();
    tq::InnerQ iq;
    iq.init_ggml(buft, /*n_layer_kv=*/3, /*channel_dim=*/16,
                 /*target_tokens=*/100);

    // Sub-threshold → no trigger.
    EXPECT(!iq.notify_tokens_written(0, 50), "first 50 tokens, no trigger");
    EXPECT(!iq.should_finalize(0), "should_finalize false at 50");
    EXPECT(!iq.notify_tokens_written(0, 49), "49 more, still under, no trigger");
    EXPECT(!iq.should_finalize(0), "should_finalize false at 99");

    // Cross threshold → trigger ONCE.
    EXPECT(iq.notify_tokens_written(0, 1), "100th token must trigger");
    EXPECT(iq.should_finalize(0), "should_finalize true after crossing");

    // Subsequent calls past threshold do NOT trigger again.
    EXPECT(!iq.notify_tokens_written(0, 50), "post-threshold notify must not retrigger");
    EXPECT(iq.should_finalize(0), "should_finalize stays true until freeze");

    // Frozen layer: notify is a no-op (returns false).
    auto backend = ggml_backend_cpu_init();
    ggml_backend_synchronize(backend);  // sq_accum_t is host buffer; no-op here
    iq.finalize_freeze_pending();
    EXPECT(iq.is_frozen(0), "finalize_freeze_pending must freeze layer 0");
    EXPECT(!iq.notify_tokens_written(0, 1000), "frozen notify must be no-op");
    ggml_backend_free(backend);
    return true;
}

bool test_finalize_pending_roundtrip() {
    auto buft    = ggml_backend_cpu_buffer_type();
    auto backend = ggml_backend_cpu_init();

    const uint32_t n_layer    = 3;
    const uint32_t channel    = 32;
    const uint32_t target     = 1024;

    tq::InnerQ iq;
    iq.init_ggml(buft, n_layer, channel, target);

    // Manually inject sq_accum_t (simulating what ggml_reex_wht's
    // src[2] accumulation would do over the calibration window).  Layer 0
    // gets an outlier ramp (1..5), layer 1 gets balanced (all 1.0),
    // layer 2 is left untouched.
    auto inject_sq = [&](uint32_t il, double base, double peak) {
        std::vector<double> sq(channel);
        for (uint32_t j = 0; j < channel; ++j) {
            const double rms = base + (peak - base) * (double) j / (double) (channel - 1);
            sq[j] = rms * rms * (double) target;
        }
        ggml_tensor * t = iq.get_sq_accum(il);
        EXPECT(t != nullptr, "sq_accum must be exposable for injection");
        ggml_backend_tensor_set(t, sq.data(), 0, sq.size() * sizeof(double));
        // Tell InnerQ "we wrote `target` rows" so should_finalize trips.
        iq.notify_tokens_written(il, target);
        return true;
    };
    EXPECT(inject_sq(0, 1.0, 5.0), "inject layer 0");
    EXPECT(inject_sq(1, 1.0, 1.0), "inject layer 1");

    EXPECT(iq.should_finalize(0), "layer 0 must trigger");
    EXPECT(iq.should_finalize(1), "layer 1 must trigger");
    EXPECT(!iq.should_finalize(2), "layer 2 has no notify, must not trigger");

    ggml_backend_synchronize(backend);
    const uint32_t n_frozen = iq.finalize_freeze_pending();
    EXPECT(n_frozen == 2, "must freeze layers 0 and 1");

    // Aggregate counts (Stage-5 follow-up: auto-skip visibility):
    // - layer 0 (outlier) → frozen + active
    // - layer 1 (balanced) → frozen + auto-skipped (inactive)
    // - layer 2 (untouched) → still calibrating
    // So expect frozen=2, active=1.
    EXPECT(iq.frozen_layer_count() == 2,
           "post-finalize_pending: 2 layers frozen (0 active outlier + 1 auto-skip)");
    EXPECT(iq.active_layer_count() == 1,
           "post-finalize_pending: only the outlier layer is active");

    // Layer 0: outlier → active → scale tensors must round-trip non-identity.
    EXPECT(iq.is_frozen(0) && iq.is_active(0), "layer 0 frozen+active");
    {
        ggml_tensor * st = iq.get_scale_for_k_write(0);
        ggml_tensor * sit = iq.get_scale_inv_for_q_read(0);
        EXPECT(st != nullptr && sit != nullptr,
               "active layer must expose scale tensors");

        std::vector<float> s(channel), si(channel);
        ggml_backend_tensor_get(st,  s.data(),  0, s.size()  * sizeof(float));
        ggml_backend_tensor_get(sit, si.data(), 0, si.size() * sizeof(float));

        bool any_non_id = false;
        for (uint32_t j = 0; j < channel; ++j) {
            if (std::fabs(s[j] - 1.0f) > 1e-5f) { any_non_id = true; break; }
            EXPECT(std::fabs(s[j] * si[j] - 1.0f) < 1e-5f,
                   "scale * scale_inv must roundtrip to 1");
        }
        EXPECT(any_non_id, "active layer scale must be non-identity");
    }

    // Layer 1: balanced → frozen but inactive → scale getters return null
    // (op should take no-scale fast path).
    EXPECT(iq.is_frozen(1), "layer 1 frozen");
    EXPECT(!iq.is_active(1), "layer 1 (balanced) inactive");
    EXPECT(iq.get_scale_for_k_write(1) == nullptr,
           "inactive layer must not expose scale_t");
    EXPECT(iq.get_scale_inv_for_q_read(1) == nullptr,
           "inactive layer must not expose scale_inv_t");

    // Layer 2: untouched → still calibrating.
    EXPECT(!iq.is_frozen(2), "layer 2 still calibrating");
    EXPECT(iq.get_sq_accum(2) != nullptr, "layer 2 sq_accum still available");

    // Reset clears everything (host + backend).  Re-publish identity to
    // backend means the scale tensor must read back as 1.0.
    iq.reset();
    EXPECT(!iq.is_frozen(0), "reset must unfreeze");
    {
        // After reset layer 0 is no longer active → scale getter returns
        // null.  But the underlying tensor still exists; we can verify
        // identity by going through finalize_freeze_all which re-reads
        // sq_accum_t (now zeroed by reset → freeze becomes identity).
        ggml_backend_synchronize(backend);
        const uint32_t n2 = iq.finalize_freeze_all();
        EXPECT(n2 == n_layer, "freeze_all after reset must freeze all layers");
        for (uint32_t il = 0; il < n_layer; ++il) {
            EXPECT(iq.is_frozen(il), "all frozen post freeze_all");
            EXPECT(!iq.is_active(il), "all inactive (zero sq) post reset+freeze_all");
            EXPECT(iq.get_scale_for_k_write(il) == nullptr,
                   "inactive scale must be null");
        }
        // Aggregate counts must agree: all frozen, none active.
        EXPECT(iq.frozen_layer_count() == n_layer,
               "all layers frozen by reset + freeze_all");
        EXPECT(iq.active_layer_count() == 0,
               "no layer can be active after reset (sq_accum is zero)");
    }

    ggml_backend_free(backend);
    return true;
}

bool test_finalize_freeze_all_partial() {
    auto buft    = ggml_backend_cpu_buffer_type();
    auto backend = ggml_backend_cpu_init();

    tq::InnerQ iq;
    iq.init_ggml(buft, /*n_layer=*/3, /*channel=*/16, /*target=*/1024);

    // Only layer 0 hits target; layer 1 partial; layer 2 untouched.
    iq.notify_tokens_written(0, 1024);
    iq.notify_tokens_written(1, 500);

    // finalize_freeze_pending: only freezes layer 0.
    ggml_backend_synchronize(backend);
    EXPECT(iq.finalize_freeze_pending() == 1,
           "only layer 0 should be pending");
    EXPECT(iq.is_frozen(0), "layer 0 frozen");
    EXPECT(!iq.is_frozen(1), "layer 1 still calibrating");
    EXPECT(!iq.is_frozen(2), "layer 2 still calibrating");

    // finalize_freeze_all: forces freeze on layers 1 + 2.
    ggml_backend_synchronize(backend);
    EXPECT(iq.finalize_freeze_all() == 2,
           "freeze_all must finish remaining 2 layers");
    EXPECT(iq.is_frozen(1) && iq.is_frozen(2), "all frozen");

    ggml_backend_free(backend);
    return true;
}

bool test_aggregate_count_api() {
    // Pure host-shadow path (no ggml).  Verify that frozen_layer_count
    // and active_layer_count both report 0 before any freeze, count
    // frozen layers including auto-skipped ones after a partial freeze,
    // and reset cleanly back to 0.  Mirrors the diagnostic surface that
    // llama_innerq_{frozen,active}_layer_count exposes via the C API.
    const uint32_t n_layer = 4;
    const uint32_t channel = 16;

    tq::InnerQ iq;
    iq.init(n_layer, channel);

    EXPECT(iq.frozen_layer_count() == 0, "starts zero frozen");
    EXPECT(iq.active_layer_count() == 0, "starts zero active");

    // Layer 0: heavy outlier on channel 5 → active after freeze.
    {
        std::vector<float> k(64 * channel, 0.05f);
        for (uint64_t t = 0; t < 64; ++t) {
            k[t * channel + 5] = 1.5f;  // outlier channel
        }
        iq.accumulate_host(0, k.data(), 64);
        iq.freeze(0);
    }
    EXPECT(iq.is_active(0), "layer 0 outlier is active");
    EXPECT(iq.frozen_layer_count() == 1, "1 frozen after layer 0");
    EXPECT(iq.active_layer_count() == 1, "1 active after layer 0");

    // Layer 1: balanced (all channels equal) → frozen but auto-skipped.
    {
        std::vector<float> k(64 * channel, 0.5f);
        iq.accumulate_host(1, k.data(), 64);
        iq.freeze(1);
    }
    EXPECT(!iq.is_active(1), "layer 1 balanced is auto-skipped");
    EXPECT(iq.frozen_layer_count() == 2, "2 frozen total (1 active + 1 skip)");
    EXPECT(iq.active_layer_count() == 1, "active count unchanged by auto-skip");

    // freeze_all on the remaining 2 layers (no data) → frozen-but-inactive.
    iq.freeze_all();
    EXPECT(iq.frozen_layer_count() == n_layer, "all layers frozen post freeze_all");
    EXPECT(iq.active_layer_count() == 1, "still only 1 active (layer 0 outlier)");

    iq.reset();
    EXPECT(iq.frozen_layer_count() == 0, "reset clears frozen count");
    EXPECT(iq.active_layer_count() == 0, "reset clears active count");
    return true;
}

bool test_ggml_aware_when_not_initialized() {
    // ggml-aware methods must be safe no-ops when init_ggml was never called.
    tq::InnerQ iq;
    EXPECT(!iq.ggml_ready(), "default not ggml-ready");
    EXPECT(iq.get_sq_accum(0) == nullptr, "no sq_accum without init_ggml");
    EXPECT(iq.get_scale_for_k_write(0) == nullptr, "no scale without init_ggml");
    EXPECT(iq.get_scale_inv_for_q_read(0) == nullptr, "no scale_inv without init_ggml");
    EXPECT(!iq.notify_tokens_written(0, 100), "notify on uninit must be no-op");
    EXPECT(!iq.should_finalize(0), "should_finalize false on uninit");
    EXPECT(!iq.finalize_freeze_layer(0), "finalize_freeze_layer no-op");
    EXPECT(iq.finalize_freeze_pending() == 0, "finalize_freeze_pending no-op");
    EXPECT(iq.finalize_freeze_all() == 0, "finalize_freeze_all no-op");
    return true;
}

}  // namespace

int main() {
    bool ok = true;
    ok &= test_accumulate_streaming();
    ok &= test_compute_scale_early_returns();
    ok &= test_compute_scale_outlier_shape();
    ok &= test_equalization_reduces_spread();
    ok &= test_innerq_lifecycle();
    ok &= test_out_of_range_layer();
    ok &= test_uninitialized_noop();
    ok &= test_aggregate_count_api();
    ok &= test_ggml_init_rejects_invalid_args();
    ok &= test_ggml_init_basic();
    ok &= test_notify_threshold_crossing();
    ok &= test_finalize_pending_roundtrip();
    ok &= test_finalize_freeze_all_partial();
    ok &= test_ggml_aware_when_not_initialized();

    if (!ok) {
        fprintf(stderr, "[FAIL] test-reex-turboquant-innerq\n");
        return 1;
    }
    fprintf(stdout, "[PASS] test-reex-turboquant-innerq\n");
    return 0;
}
