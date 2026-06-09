#pragma once

#ifdef REEX_TURBOQUANT

#include "ggml.h"

#include <cstdint>

struct llama_hparams;

namespace reex_turboquant {

// Per-layer decision on whether/how the K/V cache for this layer should be
// quantized via TurboQuant.  Pure-data (no ggml/llama tensors) so the function
// is unit-testable in isolation.
//
// Decision is driven by the user-supplied global `type_k` / `type_v`
// (i.e. `-ctk` / `-ctv`) plus the env-gated D13 boundary-V override.  The
// resolver applies on top of the global type:
//   1. WHT block divisibility guard: layers whose `head_dim_{k,v}` is not a
//      multiple of 128 silently fall back to the upstream cache type
//      (because TQ_K3 / TQ_V* quantize_row asserts on `nb01 % QK_TQ_K3 == 0`
//      and we'd otherwise crash).
//   2. D13 boundary-V mode-7 (env `REEX_TQ_BOUNDARY_V_Q8_0=1`): when global
//      type_v is TQ_V_POLAR{2,4}, force the first N and last N attention
//      layers to GGML_TYPE_Q8_0 (V cache only), mirroring TheTom's
//      TURBO_LAYER_ADAPTIVE=7 mode.  Off by default.  See docs/turboquant/
//      archive/runs/p4.a25_v_beta_design_diff_audit/ for the audit that
//      identified D13 as a hidden problem of uniform V_POLAR deployment.
struct LayerSpec {
    bool      use_tq_k = false;   // true iff this layer's K cache is TQ_K3
    bool      use_tq_v = false;   // true iff this layer's V cache is TQ_V2/V4
    int32_t   v_bits   = 0;       // 2 or 4 (only meaningful when use_tq_v == true)
    ggml_type type_k_override = GGML_TYPE_COUNT;  // GGML_TYPE_COUNT = "no override"
    ggml_type type_v_override = GGML_TYPE_COUNT;  // GGML_TYPE_COUNT = "no override"

    // D13 boundary-V: when set, the V cache for this layer is forced to a
    // non-TQ ggml_type (typically Q8_0) regardless of the user's global
    // `-ctv` choice.  Takes precedence over use_tq_v / type_v_override in
    // the llama_kv_cache constructor.  v_bits is cleared.  Off-path (env
    // unset / non-V_POLAR global type / n_layer < 8): always false.
    bool      boundary_v_active = false;
    ggml_type boundary_v_type   = GGML_TYPE_COUNT;
};

// Decide whether TurboQuant types apply to layer `il` (0-indexed) given the
// user's global `-ctk` / `-ctv` choice.
//
// Behaviour:
//   * Layer has no KV (`hparams.has_kv(il) == false`)              -> no override.
//   * Either head_dim_{k,v} is not a positive multiple of 128      -> no override.
//   * `type_k == GGML_TYPE_TQ_K3`                                  -> use_tq_k = true.
//   * `type_v == GGML_TYPE_TQ_V2 / TQ_V4`                          -> use_tq_v = true.
//   * `type_v == GGML_TYPE_TQ_V_POLAR{2,4}` + D13 env on + boundary
//                                                                  -> boundary_v_active = true.
//   * Otherwise the corresponding side stays on the global type    (no override).
//
// The function reads env for D13 boundary policy via the helpers below.
// Outside of D13 the function is pure / deterministic / side-effect free.
LayerSpec resolve_layer_spec(const llama_hparams & hparams,
                             ggml_type             type_k,
                             ggml_type             type_v,
                             uint32_t              il);

// ---------------------------------------------------------------------------
// D13 boundary-V (env-gated, opt-in).
//
// Mirrors TheTom's `TURBO_LAYER_ADAPTIVE=7` mode: first N + last N attention
// layers use Q8_0 for V cache (preserving sensitive boundary numerics), rest
// use the user-selected TQ_V_POLAR{2,4}.  Default N = 2.  Requires n_layer >= 8
// to enable (boundary should be a minority of layers, otherwise the mode-7
// premise that "boundary V is hardest" breaks down — small models do not gain).
//
// Env vars (parsed once, cached in static-local; env changes mid-process are
// ignored, same behaviour as REEX_TQ_VSIDE_INNERQ / REEX_CUSTOM_UNIFIED_SPLIT):
//   * REEX_TQ_BOUNDARY_V_Q8_0=1/true/on/yes -> enable boundary mode (default off)
//   * REEX_TQ_BOUNDARY_V_LAYERS=N           -> set N (default 2, valid range [1, 1024])
//
// Off-path: enabled() returns false, the boundary branch in resolve_layer_spec
// is skipped, and the resulting LayerSpec is bitwise-identical to pre-D13.
// ---------------------------------------------------------------------------
bool     reex_tq_boundary_v_q8_0_enabled();
uint32_t reex_tq_boundary_v_layers();

// Pure helper — given the boundary policy parameters explicitly (no env
// reads), return the boundary V override type for layer `il`, or
// GGML_TYPE_COUNT if no boundary override applies.  Extracted for direct
// unit-testing without env mocking (the env-reading helpers above cache in
// static-local so cannot easily be toggled mid-process).
//
// Returns GGML_TYPE_Q8_0 when:
//   * boundary_enabled == true, AND
//   * type_v is GGML_TYPE_TQ_V_POLAR2 or TQ_V_POLAR4, AND
//   * n_layer >= 8, AND
//   * n_boundary > 0, AND
//   * il < n_boundary || il + n_boundary >= n_layer (first N or last N layer).
// Returns GGML_TYPE_COUNT otherwise (no override).
ggml_type compute_boundary_v_override(uint32_t  il,
                                      uint32_t  n_layer,
                                      ggml_type type_v,
                                      bool      boundary_enabled,
                                      uint32_t  n_boundary);

}  // namespace reex_turboquant

#endif  // REEX_TURBOQUANT
