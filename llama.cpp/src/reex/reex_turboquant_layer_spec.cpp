#ifdef REEX_TURBOQUANT

#include "reex_turboquant_layer_spec.h"

#include "llama-hparams.h"
#include "llama-reex-custom.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace reex_turboquant {

namespace {

bool is_tq_k_type(ggml_type t) {
    return t == GGML_TYPE_TQ_K3;
}

bool is_tq_v_type(ggml_type t, int32_t & v_bits_out) {
    if (t == GGML_TYPE_TQ_V2) { v_bits_out = 2; return true; }
    if (t == GGML_TYPE_TQ_V4) { v_bits_out = 4; return true; }
    return false;
}

bool is_tq_v_polar_type(ggml_type t) {
    return t == GGML_TYPE_TQ_V_POLAR2 || t == GGML_TYPE_TQ_V_POLAR4;
}

}  // namespace

ggml_type compute_boundary_v_override(uint32_t  il,
                                      uint32_t  n_layer,
                                      ggml_type type_v,
                                      bool      boundary_enabled,
                                      uint32_t  n_boundary) {
    // Off-path: any one of these short-circuits to "no override" and is
    // bitwise-identical to pre-D13 behaviour.
    if (!boundary_enabled)              { return GGML_TYPE_COUNT; }
    if (!is_tq_v_polar_type(type_v))    { return GGML_TYPE_COUNT; }
    if (n_boundary == 0)                { return GGML_TYPE_COUNT; }
    // Mode-7 requires the model to be large enough that boundary layers are
    // a minority.  TheTom reference uses n_layer >= 8 (so 4 boundary + at
    // least 4 middle).  Tiny models (e.g. Gemma-3-1B has 26 attention layers
    // and clearly satisfies this) gain nothing if boundary >= middle.
    if (n_layer < 8)                    { return GGML_TYPE_COUNT; }
    // Sanity: refuse to make every layer a boundary layer (would just be a
    // uniform-Q8_0 V cache, defeating the purpose of TQ_V_POLAR).
    if (2 * n_boundary >= n_layer)      { return GGML_TYPE_COUNT; }

    // first N or last N attention layers.
    if (il < n_boundary || il + n_boundary >= n_layer) {
        return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_COUNT;
}

bool reex_tq_boundary_v_q8_0_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("REEX_TQ_BOUNDARY_V_Q8_0");
        return llama_reex_parse_env_flag(env) == llama_reex_env_flag::enabled;
    }();
    return enabled;
}

uint32_t reex_tq_boundary_v_layers() {
    static const uint32_t n = []() -> uint32_t {
        const char * env = std::getenv("REEX_TQ_BOUNDARY_V_LAYERS");
        if (env == nullptr || env[0] == '\0') { return 2u; }
        char * end = nullptr;
        const unsigned long v = std::strtoul(env, &end, 10);
        // Reject unparseable / negative / unreasonably large values; fall back
        // to default 2 (matches TheTom mode-7 boundary count).
        if (end == env || *end != '\0' || v == 0 || v > 1024) { return 2u; }
        return static_cast<uint32_t>(v);
    }();
    return n;
}

LayerSpec resolve_layer_spec(const llama_hparams & hparams,
                             ggml_type             type_k,
                             ggml_type             type_v,
                             uint32_t              il) {
    LayerSpec spec;

    if (!hparams.has_kv(il)) {
        return spec;
    }

    const bool want_tq_k = is_tq_k_type(type_k);
    int32_t    v_bits    = 0;
    const bool want_tq_v = is_tq_v_type(type_v, v_bits);

    // WHT block size constraint: the CPU kernel rotates 128 elements at a time
    // (group_size = QK_TQ_K3 = 128), and we drive the rotation along ne[0] =
    // n_embd_head_{k,v} of the per-token K/V tensors.
    //
    // Production callers (llama_kv_cache constructor) pre-validate this at
    // context init time and throw if the model is incompatible — they never
    // reach this function with bad head_dim.  This guard exists for the
    // standalone unit test (test-reex-turboquant-layer-spec.cpp) to stay
    // a pure-function check; it returns "no override" in that case so the
    // test can assert spec.use_tq_{k,v} == false.
    const uint32_t head_k = hparams.n_embd_head_k(il);
    const uint32_t head_v = hparams.n_embd_head_v(il);
    const bool wht_dim_ok_k = (head_k > 0) && (head_k % 128 == 0);
    const bool wht_dim_ok_v = (head_v > 0) && (head_v % 128 == 0);

    if (want_tq_k && wht_dim_ok_k && wht_dim_ok_v) {
        spec.use_tq_k = true;
        spec.type_k_override = GGML_TYPE_TQ_K3;
    }
    if (want_tq_v && wht_dim_ok_k && wht_dim_ok_v) {
        spec.use_tq_v = true;
        spec.v_bits   = v_bits;
        spec.type_v_override = type_v;
    }

    // D13 boundary-V (env-gated).  Computed after the K3/V2/V4 path so that
    // off-path (env unset) the spec is bitwise-identical to pre-D13.  Only
    // applies when global type_v is TQ_V_POLAR{2,4} (V-β family) — neither
    // V2/V4 nor non-TQ V types participate, matching the mode-7 reference
    // semantics where boundary-Q8_0 is paired with turbo2 in the middle.
    if (wht_dim_ok_k && wht_dim_ok_v) {
        const ggml_type boundary = compute_boundary_v_override(
                il, hparams.n_layer, type_v,
                reex_tq_boundary_v_q8_0_enabled(),
                reex_tq_boundary_v_layers());
        if (boundary != GGML_TYPE_COUNT) {
            spec.boundary_v_active = true;
            spec.boundary_v_type   = boundary;
            // The boundary layer's V is not a TQ type, so clear use_tq_v /
            // type_v_override / v_bits to keep the spec self-consistent.
            // (use_tq_v was already false for V_POLAR since is_tq_v_type only
            // matches V2/V4; this is defensive for future extension.)
            spec.use_tq_v        = false;
            spec.v_bits          = 0;
            spec.type_v_override = GGML_TYPE_COUNT;
        }
    }

    return spec;
}

}  // namespace reex_turboquant

#endif  // REEX_TURBOQUANT
