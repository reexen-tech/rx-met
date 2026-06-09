#pragma once

#ifdef REEX_TURBOQUANT

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <vector>

struct llama_hparams;
struct llama_cparams;

namespace reex_turboquant {

// ============================================================================
// InnerQ EMA — per-channel pre-WHT equalization for K cache.
//
// Goal: K channels often have wildly different RMS (some channels are "outlier
// channels" with 5-10x the mean RMS).  Lloyd-Max codebooks for TQ_K3 assume a
// roughly N(0,1) input distribution after WHT; outlier channels make the
// post-WHT distribution heavy-tailed and waste bits at the codebook edges.
//
// Algorithm (TheTom `turbo-innerq`-style, calibrate-then-freeze):
//
//   1. CALIBRATE: for the first `target_tokens` prefill K writes, accumulate
//      per-channel x[j] * x[j] into sq_accum[j] (atomic on CUDA, scalar on CPU).
//   2. FREEZE: when `count >= target_tokens`, compute
//          channel_rms[j] = sqrt(sq_accum[j] / count)
//          mean_rms       = mean(channel_rms)
//          ratio[j]       = mean_rms / channel_rms[j]
//          scale[j]       = clamp(ratio[j]^strength, [clamp_lo, clamp_hi])
//          scale_inv[j]   = 1 / scale[j]
//   3. APPLY: for all subsequent K writes, do x[j] *= scale[j] before WHT.
//      For dot products to remain invariant, Q reads must do x[j] *= scale_inv[j]
//      before WHT (handled by ggml_reex_wht's optional `scale_inv` input).
//
// Math invariance: <Q × scale_inv, K × scale> = <Q, K>, regardless of WHT
// (which is a linear orthogonal transform that preserves inner products).
//
// "EMA" in the name is historical: TheTom's reference implementation uses a
// fixed-window cumulative average rather than a true exponential moving
// average, but the two are mathematically equivalent under fixed-window
// equal-weight settings (α = 1/N).  See docs/turboquant for a future
// streaming-EMA variant if calibration needs to track distribution drift.
//
// Auto-skip (TheTom-aligned): `compute_scale` short-circuits to identity
// (scale = scale_inv = 1) when the per-channel RMS imbalance ratio
// max(max_ratio, 1/min_ratio) < `skip_threshold` (default 1.2).  In that
// case the layer is marked `frozen && !active`, and `get_scale_for_k_write` /
// `get_scale_inv_for_q_read` return nullptr — `ggml_reex_wht` then takes its
// no-scale fast path, so an auto-skipped layer pays *zero* steady-state
// overhead beyond the (~50 KB / layer) sq_accum/scale storage.  This makes
// "InnerQ default ON" cheap on already-balanced models (e.g. Qwen3.5-35B-A3B,
// where the +0.01 PPL gain is dominated by per-chunk calibration noise) while
// preserving the large gain on outlier-prone models (e.g. Qwen3-30B-A3B-
// Instruct, where Stage-5 first-pass measured −0.15 PPL ≈ 3σ̂).  See
// docs/turboquant/p4.a24_innerq_stage5_qwen35_35b/README.md.
// ============================================================================

// ---------------------------------------------------------------------------
// Pure host algorithm (no ggml dependency).  Splitting these out as free
// functions makes Stage-1 unit tests simple and keeps the InnerQ class focused
// on storage + lifecycle.
// ---------------------------------------------------------------------------
namespace innerq_algo {

// Accumulate K^2 per channel.  `k` is laid out as [n_tokens, channel_dim] in
// row-major order (channel_dim contiguous along the inner axis).  Adds onto
// existing sq_accum / count so callers can stream multiple chunks.
//
// Thread-safety: caller's responsibility (this is a serial reference; CUDA
// path uses atomicAdd directly inside the K cpy kernel).
void accumulate(double * sq_accum,
                uint64_t * count,
                const float * k,
                uint64_t n_tokens,
                uint32_t channel_dim);

// From accumulator state, compute per-channel scale + scale_inv vectors.
//
// Behaviour follows TheTom's reference:
//   * If `count == 0`, sets scale = scale_inv = 1.0 and returns false (no-op).
//   * If `max(ratio) < skip_threshold` (default 1.2), the channels are already
//     balanced enough; sets scale = scale_inv = 1.0 and returns false.
//   * Otherwise computes scale[j] = clamp(ratio[j]^strength, [clamp_lo, clamp_hi]).
//
// Returns true iff equalization is non-trivial (i.e. any scale != 1.0).
bool compute_scale(const double * sq_accum,
                   uint64_t count,
                   float * scale,
                   float * scale_inv,
                   uint32_t channel_dim,
                   float strength       = 0.5f,
                   float clamp_lo       = 0.5f,
                   float clamp_hi       = 2.0f,
                   float skip_threshold = 1.2f);

}  // namespace innerq_algo

// ---------------------------------------------------------------------------
// InnerQ class — owns per-layer accumulator + scale state.
//
// Lifetime model:
//   * One instance lives next to `llama_kv_cache` (Stage-3 will wire this).
//   * `init(n_layer_kv, channel_dim)` configures the internal layer table.
//   * Caller streams K data into `accumulate_host(il, k, n_tokens)` during
//     prefill; calls `freeze(il)` (or `freeze_all()`) once enough tokens have
//     been seen.
//   * After freeze, `get_scale_host(il, out)` / `get_scale_inv_host(il, out)`
//     return the per-channel vectors that should be multiplied into K (write
//     side) and Q (read side).
//   * `get_scale_inv(il)` returns a `ggml_tensor *` view (Stage-3+); Stage-1
//     stub still returns nullptr because we haven't allocated the ggml
//     storage context yet.
//
// Default-constructed instance has `n_layer_kv() == 0` and is a complete
// no-op (all accumulate/freeze calls early-return).  Call `init(...)` to
// activate.
// ---------------------------------------------------------------------------
class InnerQ {
public:
    InnerQ();
    ~InnerQ();

    InnerQ(const InnerQ &)            = delete;
    InnerQ & operator=(const InnerQ &) = delete;

    // Stage-1 setup: allocate per-layer accumulator + scale storage.  Safe to
    // call multiple times (resets state).  channel_dim must equal QK_TQ_K3
    // (= 128) in production; tests may pass smaller values.
    void init(uint32_t n_layer_kv,
              uint32_t channel_dim,
              uint32_t target_tokens = 1024,
              float    strength      = 0.5f,
              float    clamp_lo      = 0.5f,
              float    clamp_hi      = 2.0f);

    // ---- query ----
    uint32_t n_layer_kv()  const { return n_layer_kv_; }
    uint32_t channel_dim() const;
    uint64_t accumulated_count(uint32_t il) const;
    bool     is_frozen(uint32_t il) const;
    bool     is_active(uint32_t il) const;  // true iff frozen AND scale is non-identity

    // ---- aggregate counts (cheap O(n_layer_kv) host scans) ----
    // Number of layers currently in `frozen == true` state.  Includes both
    // active and auto-skipped (well-balanced or zero-sample) layers.
    uint32_t frozen_layer_count() const;
    // Number of layers currently in `frozen && active` state — i.e. the
    // layers whose finalize produced a non-identity scale and that are
    // therefore actually paying any per-K-write / per-Q-read InnerQ cost.
    // For a well-behaved model this can be 0 even when frozen_layer_count
    // == n_layer_kv (TheTom-style auto-skip; see compute_scale's
    // `skip_threshold` parameter).
    uint32_t active_layer_count() const;

    // ---- calibration ----
    // Add `n_tokens` rows of K (laid out [n_tokens, channel_dim]) into the
    // per-channel accumulator for layer `il`.  No-op after freeze.
    void accumulate_host(uint32_t il, const float * k, uint64_t n_tokens);

    // Compute scale / scale_inv from current accumulator and mark layer
    // frozen.  After freeze, `accumulate_host` becomes a no-op for this layer.
    // `freeze_all()` is a convenience that freezes every initialized layer.
    void freeze(uint32_t il);
    void freeze_all();

    // ---- scale readout (host) ----
    // Copies the per-channel scale (or its inverse) into the caller-supplied
    // buffer of size `channel_dim`.  Returns false if the layer is not
    // frozen yet (in which case `out` is filled with 1.0).
    bool get_scale_host(uint32_t il, float * out) const;
    bool get_scale_inv_host(uint32_t il, float * out) const;

    // ---- legacy / forward-compat hooks ----
    // Reset all accumulators and unfreeze every layer.  When `init_ggml`
    // had been called this also re-uploads identity scale tensors to the
    // backend.  Used at the start of a fresh prefill (e.g. when the user
    // starts a new conversation) or by `llama_kv_cache::clear`.
    void reset();

    // ========================================================================
    // ggml-aware interface (P4 A2.4 Stage 2b).
    // ========================================================================
    //
    // Wraps the Stage-1 host algorithm in a ggml + ggml-backend layer so
    // InnerQ can plug into the KV cache cpy_k path and the
    // build_attn_mha Q-side WHT call.  The host-side algorithm is the
    // single source of truth — `finalize_freeze_layer` does its compute
    // step on the host and merely uses the backend for buffer transfer.
    //
    // Calibration → freeze flow (executed entirely from the KV cache
    // host code, never from inside a kernel):
    //
    //   cpy_k build phase (every TQ-K layer, every ubatch):
    //     wt = innerq.get_sq_accum(il);              // non-null while calibrating
    //     sw = innerq.get_scale_for_k_write(il);     // null while calibrating
    //     ggml_reex_wht(ctx, k, /*direction=*/0, /*group_size=*/128, sw, wt);
    //     innerq.notify_tokens_written(il, n_tokens);
    //
    //   After graph_compute (KV cache `apply_ubatch` post-hook):
    //     innerq.finalize_freeze_pending(backend);
    //   internally:
    //     - ggml_backend_synchronize(backend)
    //     - ggml_backend_tensor_get(sq_accum_t[il], host_buf, ...)
    //     - innerq_algo::compute_scale on host
    //     - ggml_backend_tensor_set(scale_t[il]) and (scale_inv_t[il])
    //     - mark frozen + active
    //
    // Buffer placement: `init_ggml` requires a HOST buffer type so the
    // finalize step can host-side memcpy without a D2H stall.  When the
    // model runs on a non-host backend (CUDA), ggml_backend_sched
    // transparently transfers the small (≤ ~64 KB total) scale tensors
    // to the device on first use; this is amortised over the whole
    // prefill so the per-graph cost is negligible.

    // Allocate per-layer ggml tensors on the supplied HOST buffer type.
    // `buft` must satisfy `ggml_backend_buft_is_host(buft) == true`; if
    // not, init_ggml is a no-op and returns false (defensive — the KV
    // cache filters this already, but we don't want to silently
    // mis-allocate to GPU memory and break finalize's host-side compute).
    //
    // `n_layer_kv` is the *KV-cache layer count* (i.e. layers actually
    // present in `llama_kv_cache::layers`, not the model's full layer
    // count).  Callers index via the same KV-cache layer index they'd
    // pass to `cpy_k`.  `channel_dim` must be QK_TQ_K3 == 128 in
    // production; tests may pass a smaller value.
    //
    // Idempotent: a second call frees and re-allocates everything, then
    // re-runs the Stage-1 host init so the host algorithm side stays in
    // sync.  Returns true iff allocation succeeded.
    bool init_ggml(ggml_backend_buffer_type_t buft,
                   uint32_t                   n_layer_kv,
                   uint32_t                   channel_dim,
                   uint32_t                   target_tokens = 1024,
                   float                      strength      = 0.5f,
                   float                      clamp_lo      = 0.5f,
                   float                      clamp_hi      = 2.0f);

    // True iff `init_ggml` was called and allocation succeeded — i.e.
    // get_*(il) may return non-null tensors.  Default-constructed
    // instances and instances initialised only via host-side `init` are
    // not ggml-ready and report false.
    bool ggml_ready() const;

    // ggml_tensor accessors (see lifecycle comments above).  Return
    // nullptr when not ggml_ready, when `il >= n_layer_kv`, or when the
    // lifecycle gate forbids exposure (sq_accum after freeze, scale
    // before freeze, scale on an inactive-after-freeze layer).
    ggml_tensor * get_sq_accum(uint32_t il)             const;
    ggml_tensor * get_scale_for_k_write(uint32_t il)    const;
    ggml_tensor * get_scale_inv_for_q_read(uint32_t il) const;

    // Add `n_tokens` to the host-side pending counter for layer `il`.
    // Returns true iff this call crossed the `target_tokens` threshold
    // for the FIRST time — only one notify call per layer per
    // calibration session ever returns true.  Touches no device state,
    // no backend handle.
    //
    // CALLERS MUST ENSURE THIS RUNS ONLY ON ACTUAL POST-COMPUTE EVENTS,
    // not on graph-build / graph-reserve passes.  llama.cpp's
    // graph_reserve invokes the same build_graph code path with dummy
    // worst-case-shape ubatches, which would overshoot the counter by
    // ~1024 tokens before the first real prefill if naively called from
    // cpy_k.  The KV-cache integration uses notify_ubatch_committed()
    // from `apply()` instead — see that method's contract below.
    //
    // No-op (returns false) when not ggml_ready, when il is out of range,
    // or when the layer is already frozen.
    bool notify_tokens_written(uint32_t il, uint64_t n_tokens);

    // Convenience: add `n_tokens` to every non-frozen layer's pending
    // counter at once.  Used by the KV cache post-compute hook (which
    // doesn't know which layers were actually TQ-K3 — non-TQ-K3 layers
    // simply accumulate phantom counts but their sq_accum_t stays zero,
    // so finalize will turn them into frozen-but-inactive identity
    // layers, harmless beyond ~64KB of unused state).
    //
    // Returns the number of layers that crossed target_tokens for the
    // first time on this call.
    uint32_t notify_ubatch_committed(uint64_t n_tokens);

    // True iff layer `il` accumulated >= target_tokens K writes and is
    // not yet frozen.  Cheap host-only check; safe in the build phase.
    bool should_finalize(uint32_t il) const;

    // Finalize layer `il` against the host shadow:
    //   - ggml_backend_tensor_get(sq_accum_t[il], host_buf, ...)
    //   - innerq_algo::compute_scale on host
    //   - ggml_backend_tensor_set(scale_t[il]) + (scale_inv_t[il])
    //   - Mark frozen + (active iff scale is non-identity).
    //
    // CONTRACT: caller MUST ensure that any backend compute that has been
    // queued to write into sq_accum_t has flushed before calling.  In
    // production this means `ggml_backend_sched_synchronize(sched)` on
    // the sched that owns the K-cpy graph.  Unit tests using a single
    // backend can do `ggml_backend_synchronize(backend)` themselves.
    // We do NOT synchronize internally because the InnerQ class has no
    // visibility into the multi-backend sched used by llama_context.
    //
    // No-op (returns false) when not ggml_ready, when il is out of range,
    // or when the layer is already frozen.  Returns true iff a fresh
    // freeze actually happened.
    bool finalize_freeze_layer(uint32_t il);

    // Iterate every layer that returns `should_finalize(il) == true` and
    // finalize it.  Returns the number of layers freshly frozen.
    // Same flush contract as `finalize_freeze_layer` above.
    uint32_t finalize_freeze_pending();

    // Force-freeze every layer that's still in calibration mode, even if
    // target_tokens was not reached.  Layers with no accumulated samples
    // (e.g. layers that haven't seen a single K write yet) are marked
    // frozen-but-inactive (identity scale).  Returns the number of
    // layers freshly frozen.  Used by the explicit user-facing
    // `llama_innerq_finalize_now` API and by shutdown paths.  Same flush
    // contract as `finalize_freeze_layer`.
    uint32_t finalize_freeze_all();

private:
    uint32_t n_layer_kv_ = 0;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// V-side InnerQ (rc_d4) — mirror K3 InnerQ scale into V WHT path.
//
// Background: TheTom's set_rows_turbo{2,4} kernels apply InnerQ scale to V
// inputs too (V *= d_innerq_scale before WHT + quantize), and undo it at the
// inverse WHT on the attention output (cur *= d_innerq_scale_inv).  Our V-β
// land (P4 A2.5) wired the V WHT op without scale_inv, so V channels keep
// their raw per-channel variance and the post-WHT marginal stays farther
// from Gaussian than necessary — Lloyd-Max quantisation pays the resulting
// MSE penalty.  See docs/turboquant/archive/runs/p4.a25_v_beta_design_diff_audit/
// for the audit that surfaced this gap.
//
// Wiring (reuses the K3 scale_t / scale_inv_t tensors, NO new state):
//   * cpy_v (forward V WHT): scale_inv parameter <- scale_for_k_write(il).
//     ggml_reex_wht then pre-multiplies V[j] *= scale[j] before the rotation.
//   * build_attn_mha (inverse V WHT post-mul): cur *= scale_inv_for_q_read(il)
//     via ggml_mul broadcast.  This is OUTSIDE ggml_reex_wht because the op's
//     scale_inv input is always pre-multiplied, but the V-side InnerQ math
//     requires post-multiplication on the un-rotated output (Diag does not
//     commute with H_d).
//
// Off by default — opt-in via env var REEX_TQ_VSIDE_INNERQ=1.  Off-path is
// bitwise-identical to the pre-rc_d4 code (env check folds to a constant on
// the first call thanks to static-local init).
//
// Returns true iff env var REEX_TQ_VSIDE_INNERQ parses to an "enabled" flag
// (1 / true / on / yes — same parser as REEX_CUSTOM_UNIFIED_SPLIT).
bool reex_tq_vside_innerq_enabled();

// rc_d4b (path 1) — V-side InnerQ with K+V MIXED sq_accum calibration.
//
// Background: rc_d4 option A (`REEX_TQ_VSIDE_INNERQ=1`) only applies the
// K-only-calibrated scale to V, without letting V participate in the
// calibration itself.  On 30B-Instruct Q4_0 this regressed perplexity by
// +0.0185 PPL because K-only outlier patterns do not generalise to V's
// channel distribution.  TheTom's `llama-cpp-turboquant` reference
// (`set-rows.cu::g_innerq_sq_accum_host`, atomicAdd-from-both-K-and-V
// kernels) computes a K+V joint scale instead, which by construction is
// "safe" for both stores — it under-corrects K outliers slightly but
// never distorts V.
//
// Path 1 wiring (V_POLAR* only — V2/V4 bitwise-identical to baseline):
//   * cpy_v calibration phase  (layer NOT frozen):
//       sq_accum_v = tq_innerq_sq_accum(il)         (SAME buffer as K)
//       scale_v    = nullptr
//     ggml_reex_wht then accumulates V's post-WHT x² into the SAME
//     sq_accum_t tensor that K writes to.  CPU dispatch serialises the
//     K-WHT-with-sq_accum and V-WHT-with-sq_accum ops (both off-loaded
//     to CPU backend when sq_accum != NULL), so there is no race on
//     the shared tensor.
//   * cpy_v freeze phase (layer frozen):
//       sq_accum_v = nullptr
//       scale_v    = tq_innerq_scale_for_k_write(il)   (joint-calibrated)
//     V is multiplied by the K+V joint scale before WHT, exactly as
//     option A does — the only difference is *what* scale_t contains.
//   * build_attn_mha (inverse V WHT post-mul): same wire as option A.
//     The post-mul fires whenever EITHER `REEX_TQ_VSIDE_INNERQ` OR
//     `REEX_TQ_VSIDE_INNERQ_MIX` is enabled, because both modes have V
//     pre-multiplied by a per-channel scale in cpy_v.
//
// Off by default — opt-in via env var REEX_TQ_VSIDE_INNERQ_MIX=1.  When
// `MIX=1` is set, it SUPERSEDES `REEX_TQ_VSIDE_INNERQ=1` (option A and
// path 1 are mutually exclusive — option A is K-only scale-reuse, path 1
// is K+V mixed calibration).  Both off → bitwise-identical to baseline.
//
// Returns true iff env var REEX_TQ_VSIDE_INNERQ_MIX parses to an "enabled"
// flag (1 / true / on / yes — same parser as REEX_CUSTOM_UNIFIED_SPLIT).
bool reex_tq_vside_innerq_mix_enabled();

}  // namespace reex_turboquant

#endif  // REEX_TURBOQUANT
