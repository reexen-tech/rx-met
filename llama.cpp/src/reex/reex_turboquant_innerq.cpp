#ifdef REEX_TURBOQUANT

#include "reex_turboquant_innerq.h"

#include "ggml-backend.h"
#include "llama-reex-custom.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace reex_turboquant {

// ============================================================================
// innerq_algo --- pure host algorithm
// ============================================================================
namespace innerq_algo {

void accumulate(double *      sq_accum,
                uint64_t *    count,
                const float * k,
                uint64_t      n_tokens,
                uint32_t      channel_dim) {
    if (sq_accum == nullptr || count == nullptr || k == nullptr || channel_dim == 0) {
        return;
    }
    for (uint64_t t = 0; t < n_tokens; ++t) {
        const float * row = k + t * (uint64_t) channel_dim;
        for (uint32_t j = 0; j < channel_dim; ++j) {
            const double v = (double) row[j];
            sq_accum[j] += v * v;
        }
    }
    *count += n_tokens;
}

bool compute_scale(const double * sq_accum,
                   uint64_t       count,
                   float *        scale,
                   float *        scale_inv,
                   uint32_t       channel_dim,
                   float          strength,
                   float          clamp_lo,
                   float          clamp_hi,
                   float          skip_threshold) {
    if (scale == nullptr || scale_inv == nullptr || channel_dim == 0) {
        return false;
    }

    auto fill_identity = [&]() {
        for (uint32_t j = 0; j < channel_dim; ++j) {
            scale[j]     = 1.0f;
            scale_inv[j] = 1.0f;
        }
    };

    if (sq_accum == nullptr || count == 0) {
        fill_identity();
        return false;
    }

    double mean_rms = 0.0;
    double rms[256];  // QK_TQ_K3 = 128 in production; bound generously for tests.
    assert(channel_dim <= sizeof(rms) / sizeof(rms[0]));
    for (uint32_t j = 0; j < channel_dim; ++j) {
        rms[j]   = std::sqrt(sq_accum[j] / (double) count);
        mean_rms += rms[j];
    }
    mean_rms /= (double) channel_dim;

    if (mean_rms <= 1e-12) {
        fill_identity();
        return false;
    }

    double max_ratio = 0.0;
    double min_ratio = 1e30;
    double ratio[256];
    for (uint32_t j = 0; j < channel_dim; ++j) {
        const double r = (rms[j] > 1e-12) ? (mean_rms / rms[j]) : 1.0;
        ratio[j]  = r;
        max_ratio = std::max(max_ratio, r);
        min_ratio = std::min(min_ratio, r);
    }

    // Skip if channels are already well-balanced.  TheTom uses max_ratio < 1.2;
    // we treat both directions (max > 1.2 OR min < 1/1.2) symmetrically.
    const double imbalance = std::max(max_ratio, 1.0 / std::max(min_ratio, 1e-12));
    if (imbalance < (double) skip_threshold) {
        fill_identity();
        return false;
    }

    for (uint32_t j = 0; j < channel_dim; ++j) {
        double s = std::pow(ratio[j], (double) strength);
        if (s < (double) clamp_lo) s = (double) clamp_lo;
        if (s > (double) clamp_hi) s = (double) clamp_hi;
        scale[j]     = (float) s;
        scale_inv[j] = (float) (1.0 / s);
    }
    return true;
}

}  // namespace innerq_algo

// ============================================================================
// InnerQ::Impl --- per-layer storage (host shadow + ggml-aware tier)
// ============================================================================
//
// Two storage tiers:
//
//   1. Host shadow ("Stage-1"): plain std::vector buffers used by the host
//      algorithm tests + as the source-of-truth for scale values after
//      compute_scale.  Always present once `init` was called.
//
//   2. ggml-aware tier ("Stage 2b"): per-layer ggml_tensor triples
//      { sq_accum_t (F64), scale_t (F32), scale_inv_t (F32) } allocated in
//      a private ggml_context backed by `host_buft`.  Only present after
//      `init_ggml(...)` succeeded.
//
// The host pending counter (LayerStats::pending_writes) is the source of
// truth for "how many K writes have been queued onto the graph since the
// start of calibration".  It is the trigger for freeze; the K^2 sum that
// flows into compute_scale lives in sq_accum_t (F64) and is only valid
// AFTER ggml_backend_synchronize.
struct InnerQ::Impl {
    struct LayerStats {
        // ---- host shadow (always present after init) ----
        std::vector<double> sq_accum;
        uint64_t            count    = 0;
        std::vector<float>  scale;
        std::vector<float>  scale_inv;
        bool                frozen   = false;
        bool                active   = false;

        // ---- ggml-aware (present iff Impl::ggml_initialized) ----
        ggml_tensor * sq_accum_t   = nullptr;
        ggml_tensor * scale_t      = nullptr;
        ggml_tensor * scale_inv_t  = nullptr;

        // # K rows queued since calibration started; freeze trigger only.
        uint64_t pending_writes = 0;
    };

    std::vector<LayerStats> layers;
    uint32_t                channel_dim   = 0;
    uint32_t                target_tokens = 1024;
    float                   strength      = 0.5f;
    float                   clamp_lo      = 0.5f;
    float                   clamp_hi      = 2.0f;

    // ---- ggml-aware state ----
    bool                        ggml_initialized = false;
    ggml_backend_buffer_type_t  host_buft        = nullptr;
    ggml_context *              ctx              = nullptr;
    ggml_backend_buffer_t       buf              = nullptr;

    void init(uint32_t n_layer_kv, uint32_t cd, uint32_t tt, float st, float lo, float hi) {
        channel_dim   = cd;
        target_tokens = tt;
        strength      = st;
        clamp_lo      = lo;
        clamp_hi      = hi;
        layers.assign(n_layer_kv, LayerStats{});
        for (auto & l : layers) {
            l.sq_accum.assign(cd, 0.0);
            l.scale.assign(cd, 1.0f);
            l.scale_inv.assign(cd, 1.0f);
        }
    }

    void reset_host() {
        for (auto & l : layers) {
            std::fill(l.sq_accum.begin(),  l.sq_accum.end(),  0.0);
            std::fill(l.scale.begin(),     l.scale.end(),     1.0f);
            std::fill(l.scale_inv.begin(), l.scale_inv.end(), 1.0f);
            l.count          = 0;
            l.frozen         = false;
            l.active         = false;
            l.pending_writes = 0;
        }
    }

    // Free the ggml_context + backend buffer; tensor pointers become
    // dangling and are nulled.  Host shadow is preserved.
    void teardown_ggml() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        for (auto & l : layers) {
            l.sq_accum_t  = nullptr;
            l.scale_t     = nullptr;
            l.scale_inv_t = nullptr;
        }
        host_buft        = nullptr;
        ggml_initialized = false;
    }

    // Commit per-layer state into a freeze:
    //   1. Pull sq_accum_t into the host shadow.
    //   2. compute_scale on the host shadow.
    //   3. Mark frozen + active.
    //   4. Push scale_t / scale_inv_t back to the backend.
    // Caller is responsible for ggml_backend_synchronize beforehand so the
    // sq_accum_t read sees a stable value.
    void finalize_one(uint32_t il) {
        auto & layer = layers[il];
        ggml_backend_tensor_get(layer.sq_accum_t,
                                layer.sq_accum.data(),
                                0,
                                (size_t) channel_dim * sizeof(double));
        layer.count = layer.pending_writes;
        const bool nontrivial = innerq_algo::compute_scale(
            layer.sq_accum.data(),
            layer.count,
            layer.scale.data(),
            layer.scale_inv.data(),
            channel_dim,
            strength,
            clamp_lo,
            clamp_hi);
        layer.frozen = true;
        layer.active = nontrivial;
        ggml_backend_tensor_set(layer.scale_t,
                                layer.scale.data(),
                                0,
                                (size_t) channel_dim * sizeof(float));
        ggml_backend_tensor_set(layer.scale_inv_t,
                                layer.scale_inv.data(),
                                0,
                                (size_t) channel_dim * sizeof(float));
    }

    // Push host shadow (identity scale + zero sq_accum) to backend.
    void upload_identity_all() {
        if (!ggml_initialized) return;
        std::vector<double> zeros_d(channel_dim, 0.0);
        std::vector<float>  ones_f (channel_dim, 1.0f);
        for (auto & l : layers) {
            ggml_backend_tensor_set(l.sq_accum_t,  zeros_d.data(), 0,
                                    (size_t) channel_dim * sizeof(double));
            ggml_backend_tensor_set(l.scale_t,     ones_f.data(),  0,
                                    (size_t) channel_dim * sizeof(float));
            ggml_backend_tensor_set(l.scale_inv_t, ones_f.data(),  0,
                                    (size_t) channel_dim * sizeof(float));
        }
    }

    ~Impl() {
        teardown_ggml();
    }
};

// ============================================================================
// InnerQ --- public API
// ============================================================================
InnerQ::InnerQ() : impl_(std::make_unique<Impl>()) {}
InnerQ::~InnerQ() = default;

void InnerQ::init(uint32_t n_layer_kv,
                  uint32_t channel_dim,
                  uint32_t target_tokens,
                  float    strength,
                  float    clamp_lo,
                  float    clamp_hi) {
    n_layer_kv_ = n_layer_kv;
    impl_->init(n_layer_kv, channel_dim, target_tokens, strength, clamp_lo, clamp_hi);
}

uint32_t InnerQ::channel_dim() const {
    return impl_->channel_dim;
}

uint64_t InnerQ::accumulated_count(uint32_t il) const {
    if (il >= impl_->layers.size()) return 0;
    return impl_->layers[il].count;
}

bool InnerQ::is_frozen(uint32_t il) const {
    if (il >= impl_->layers.size()) return false;
    return impl_->layers[il].frozen;
}

bool InnerQ::is_active(uint32_t il) const {
    if (il >= impl_->layers.size()) return false;
    return impl_->layers[il].active;
}

uint32_t InnerQ::frozen_layer_count() const {
    uint32_t n = 0;
    for (const auto & layer : impl_->layers) {
        if (layer.frozen) ++n;
    }
    return n;
}

uint32_t InnerQ::active_layer_count() const {
    uint32_t n = 0;
    for (const auto & layer : impl_->layers) {
        if (layer.frozen && layer.active) ++n;
    }
    return n;
}

void InnerQ::accumulate_host(uint32_t il, const float * k, uint64_t n_tokens) {
    if (il >= impl_->layers.size() || k == nullptr || n_tokens == 0) {
        return;
    }
    auto & layer = impl_->layers[il];
    if (layer.frozen) {
        return;
    }
    innerq_algo::accumulate(layer.sq_accum.data(),
                            &layer.count,
                            k,
                            n_tokens,
                            impl_->channel_dim);
}

void InnerQ::freeze(uint32_t il) {
    if (il >= impl_->layers.size()) return;
    auto & layer = impl_->layers[il];
    if (layer.frozen) return;

    const bool nontrivial = innerq_algo::compute_scale(
        layer.sq_accum.data(),
        layer.count,
        layer.scale.data(),
        layer.scale_inv.data(),
        impl_->channel_dim,
        impl_->strength,
        impl_->clamp_lo,
        impl_->clamp_hi);

    layer.frozen = true;
    layer.active = nontrivial;
}

void InnerQ::freeze_all() {
    for (uint32_t il = 0; il < (uint32_t) impl_->layers.size(); ++il) {
        freeze(il);
    }
}

bool InnerQ::get_scale_host(uint32_t il, float * out) const {
    if (out == nullptr) return false;
    if (il >= impl_->layers.size()) {
        for (uint32_t j = 0; j < impl_->channel_dim; ++j) out[j] = 1.0f;
        return false;
    }
    const auto & layer = impl_->layers[il];
    std::memcpy(out, layer.scale.data(), (size_t) impl_->channel_dim * sizeof(float));
    return layer.frozen;
}

bool InnerQ::get_scale_inv_host(uint32_t il, float * out) const {
    if (out == nullptr) return false;
    if (il >= impl_->layers.size()) {
        for (uint32_t j = 0; j < impl_->channel_dim; ++j) out[j] = 1.0f;
        return false;
    }
    const auto & layer = impl_->layers[il];
    std::memcpy(out, layer.scale_inv.data(), (size_t) impl_->channel_dim * sizeof(float));
    return layer.frozen;
}

void InnerQ::reset() {
    impl_->reset_host();
    // Re-publish identity to backend buffers so device-side reads see
    // cleared state.  Cheap (3 small tensors per layer) and avoids
    // stale "frozen scale" values lingering on device when the user
    // restarts a conversation via llama_kv_cache::clear.
    impl_->upload_identity_all();
}

// ============================================================================
// Stage 2b --- ggml-aware interface
// ============================================================================

bool InnerQ::init_ggml(ggml_backend_buffer_type_t buft,
                       uint32_t                   n_layer_kv,
                       uint32_t                   channel_dim_arg,
                       uint32_t                   target_tokens,
                       float                      strength,
                       float                      clamp_lo,
                       float                      clamp_hi) {
    if (buft == nullptr) {
        return false;
    }
    if (n_layer_kv == 0 || channel_dim_arg == 0) {
        return false;
    }
    // P4 A2.4 Stage 4: any backend whose buft supports F64 and the
    // GGML_OP_REEX_WHT op is acceptable.  This used to be host-only
    // because pre-Stage-4 the CUDA WHT kernel could not consume
    // sq_accum (and the scheduler offloaded the entire op to CPU);
    // post-Stage-4 the CUDA path has a native F64 atomicAdd accumulator
    // and we want sq_accum to live on the same device as the K cache so
    // there's no D2H stall on every K write.  finalize_one's
    // ggml_backend_tensor_get / set already handles the (one-shot,
    // per-layer, ≤ 64 KB) D2H + H2D for the actual scale computation.

    impl_->teardown_ggml();
    init(n_layer_kv, channel_dim_arg, target_tokens, strength, clamp_lo, clamp_hi);

    const size_t obj_overhead = ggml_tensor_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ obj_overhead * (size_t) n_layer_kv * 3 + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    impl_->ctx = ggml_init(params);
    if (impl_->ctx == nullptr) {
        return false;
    }

    char name_buf[64];
    for (uint32_t il = 0; il < n_layer_kv; ++il) {
        auto & l = impl_->layers[il];

        l.sq_accum_t = ggml_new_tensor_1d(impl_->ctx, GGML_TYPE_F64, channel_dim_arg);
        std::snprintf(name_buf, sizeof(name_buf), "innerq_sq_accum_%u", il);
        ggml_set_name(l.sq_accum_t, name_buf);

        l.scale_t = ggml_new_tensor_1d(impl_->ctx, GGML_TYPE_F32, channel_dim_arg);
        std::snprintf(name_buf, sizeof(name_buf), "innerq_scale_%u", il);
        ggml_set_name(l.scale_t, name_buf);

        l.scale_inv_t = ggml_new_tensor_1d(impl_->ctx, GGML_TYPE_F32, channel_dim_arg);
        std::snprintf(name_buf, sizeof(name_buf), "innerq_scale_inv_%u", il);
        ggml_set_name(l.scale_inv_t, name_buf);
    }

    impl_->buf = ggml_backend_alloc_ctx_tensors_from_buft(impl_->ctx, buft);
    if (impl_->buf == nullptr) {
        ggml_free(impl_->ctx);
        impl_->ctx = nullptr;
        return false;
    }
    impl_->host_buft        = buft;
    impl_->ggml_initialized = true;

    impl_->upload_identity_all();
    return true;
}

bool InnerQ::ggml_ready() const {
    return impl_->ggml_initialized;
}

ggml_tensor * InnerQ::get_sq_accum(uint32_t il) const {
    if (!impl_->ggml_initialized || il >= impl_->layers.size()) {
        return nullptr;
    }
    const auto & layer = impl_->layers[il];
    if (layer.frozen) {
        return nullptr;
    }
    return layer.sq_accum_t;
}

ggml_tensor * InnerQ::get_scale_for_k_write(uint32_t il) const {
    if (!impl_->ggml_initialized || il >= impl_->layers.size()) {
        return nullptr;
    }
    const auto & layer = impl_->layers[il];
    if (!layer.frozen || !layer.active) {
        return nullptr;
    }
    return layer.scale_t;
}

ggml_tensor * InnerQ::get_scale_inv_for_q_read(uint32_t il) const {
    if (!impl_->ggml_initialized || il >= impl_->layers.size()) {
        return nullptr;
    }
    const auto & layer = impl_->layers[il];
    if (!layer.frozen || !layer.active) {
        return nullptr;
    }
    return layer.scale_inv_t;
}

bool InnerQ::notify_tokens_written(uint32_t il, uint64_t n_tokens) {
    if (!impl_->ggml_initialized || il >= impl_->layers.size() || n_tokens == 0) {
        return false;
    }
    auto & layer = impl_->layers[il];
    if (layer.frozen) {
        return false;
    }
    const uint64_t before = layer.pending_writes;
    layer.pending_writes  = before + n_tokens;
    const uint64_t target = (uint64_t) impl_->target_tokens;
    return before < target && layer.pending_writes >= target;
}

uint32_t InnerQ::notify_ubatch_committed(uint64_t n_tokens) {
    if (!impl_->ggml_initialized || n_tokens == 0) {
        return 0;
    }
    const uint64_t target = (uint64_t) impl_->target_tokens;
    uint32_t n_crossed    = 0;
    for (auto & layer : impl_->layers) {
        if (layer.frozen) continue;
        const uint64_t before = layer.pending_writes;
        layer.pending_writes  = before + n_tokens;
        if (before < target && layer.pending_writes >= target) {
            ++n_crossed;
        }
    }
    return n_crossed;
}

bool InnerQ::should_finalize(uint32_t il) const {
    if (!impl_->ggml_initialized || il >= impl_->layers.size()) {
        return false;
    }
    const auto & layer = impl_->layers[il];
    if (layer.frozen) {
        return false;
    }
    return layer.pending_writes >= (uint64_t) impl_->target_tokens;
}

bool InnerQ::finalize_freeze_layer(uint32_t il) {
    if (!impl_->ggml_initialized || il >= impl_->layers.size()) {
        return false;
    }
    if (impl_->layers[il].frozen) {
        return false;
    }
    impl_->finalize_one(il);
    return true;
}

uint32_t InnerQ::finalize_freeze_pending() {
    if (!impl_->ggml_initialized) {
        return 0;
    }

    uint32_t n_frozen = 0;
    for (uint32_t il = 0; il < (uint32_t) impl_->layers.size(); ++il) {
        if (!should_finalize(il)) {
            continue;
        }
        impl_->finalize_one(il);
        ++n_frozen;
    }
    return n_frozen;
}

uint32_t InnerQ::finalize_freeze_all() {
    if (!impl_->ggml_initialized) {
        return 0;
    }

    uint32_t n_frozen = 0;
    for (uint32_t il = 0; il < (uint32_t) impl_->layers.size(); ++il) {
        if (impl_->layers[il].frozen) continue;
        impl_->finalize_one(il);
        ++n_frozen;
    }
    return n_frozen;
}

// ============================================================================
// rc_d4 — V-side InnerQ env-gated toggle.
// ============================================================================
bool reex_tq_vside_innerq_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("REEX_TQ_VSIDE_INNERQ");
        return llama_reex_parse_env_flag(env) == llama_reex_env_flag::enabled;
    }();
    return enabled;
}

// ============================================================================
// rc_d4b (path 1) — V-side InnerQ with K+V mixed sq_accum calibration.
// ============================================================================
bool reex_tq_vside_innerq_mix_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("REEX_TQ_VSIDE_INNERQ_MIX");
        return llama_reex_parse_env_flag(env) == llama_reex_env_flag::enabled;
    }();
    return enabled;
}

}  // namespace reex_turboquant

#endif  // REEX_TURBOQUANT
