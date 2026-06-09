# tests/reex/ — REEX_TURBOQUANT only.
#
# Build is gated at the parent (tests/CMakeLists.txt) via add_subdirectory()
# under `if (REEX_TURBOQUANT)`. We early-return defensively here as well so
# the subdirectory remains safe to add unconditionally if that ever changes.
#
# Each test follows the existing tests/ helpers (llama_build_and_test) so it
# auto-registers with ctest under LABEL=main.

if (NOT REEX_TURBOQUANT)
    return()
endif()

# P1.2 acceptance: quantize_row → dequantize_row contraction for TQ_K3 / V2 / V4.
#
# We use llama_build + llama_test (rather than llama_build_and_test) so we
# don't drag in tests/get-model.cpp from the parent directory — these tests
# are self-contained and don't load gguf models.
llama_build(test-reex-turboquant-block-roundtrip.cpp)
target_compile_definitions(test-reex-turboquant-block-roundtrip PRIVATE REEX_TURBOQUANT)
llama_test(test-reex-turboquant-block-roundtrip)

# P1.3 acceptance: WHT kernel.  All three tests need access to the private
# header `ggml/src/ggml-cpu/reex/reex_turboquant_wht.h`, so we add the
# repository's `ggml/src/` to the include path explicitly.  The vs-oracle
# test additionally links against the test-only `reex_turboquant_oracle`
# static library.
set(REEX_TQ_GGML_SRC "${PROJECT_SOURCE_DIR}/ggml/src")

# (a) forward / inverse strict identity contraction.
llama_build(test-reex-turboquant-wht-roundtrip.cpp)
target_compile_definitions(test-reex-turboquant-wht-roundtrip PRIVATE REEX_TURBOQUANT)
target_include_directories(test-reex-turboquant-wht-roundtrip PRIVATE ${REEX_TQ_GGML_SRC})
llama_test(test-reex-turboquant-wht-roundtrip)

# (b) inner-product invariance vs. the oracle's dense Pi rotation.
llama_build(test-reex-turboquant-wht-vs-oracle.cpp)
target_compile_definitions(test-reex-turboquant-wht-vs-oracle PRIVATE REEX_TURBOQUANT)
target_include_directories(test-reex-turboquant-wht-vs-oracle PRIVATE ${REEX_TQ_GGML_SRC})
target_link_libraries     (test-reex-turboquant-wht-vs-oracle PRIVATE reex_turboquant_oracle)
llama_test(test-reex-turboquant-wht-vs-oracle)

# (c) ggml_reex_wht through ggml_graph_compute matches the scalar reference.
llama_build(test-reex-turboquant-backend-op-wht.cpp)
target_compile_definitions(test-reex-turboquant-backend-op-wht PRIVATE REEX_TURBOQUANT)
target_include_directories(test-reex-turboquant-backend-op-wht PRIVATE ${REEX_TQ_GGML_SRC})
llama_test(test-reex-turboquant-backend-op-wht)

# P1.4 acceptance: oracle parity for ggml type_traits paths (TQ_V2/V4 strict,
# TQ_K3 split into MSE-only strict and full Phase 1 chain non-catastrophic).
# Needs `ggml-common.h` (private) and the oracle static library.
llama_build(test-reex-turboquant-vs-oracle.cpp)
target_compile_definitions(test-reex-turboquant-vs-oracle PRIVATE REEX_TURBOQUANT)
target_include_directories(test-reex-turboquant-vs-oracle PRIVATE ${REEX_TQ_GGML_SRC})
target_link_libraries     (test-reex-turboquant-vs-oracle PRIVATE reex_turboquant_oracle)
llama_test(test-reex-turboquant-vs-oracle)

# P2.2 acceptance: pure-function determinism of resolve_layer_spec across
# every preset mode (0/1/2/5/7) and unknown mode fallback.  Links against
# libllama (provides reex_turboquant_layer_spec.cpp + llama_hparams symbols).
set(REEX_TQ_LLAMA_SRC "${PROJECT_SOURCE_DIR}/src")
llama_build(test-reex-turboquant-layer-spec.cpp)
target_compile_definitions(test-reex-turboquant-layer-spec PRIVATE REEX_TURBOQUANT)
target_include_directories(test-reex-turboquant-layer-spec PRIVATE
    ${REEX_TQ_LLAMA_SRC}
    ${REEX_TQ_LLAMA_SRC}/reex
)
llama_test(test-reex-turboquant-layer-spec)

# P2.3 acceptance: 3-D KV-cache-shaped tensors of TQ_K3/TQ_V2/TQ_V4 survive
# CPY (write) + GET_ROWS (read) round trips on the CPU backend.  This
# covers the wiring in ggml_quantize_chunk + ggml_compute_forward_get_rows
# at shapes representative of llama_kv_cache.  No model is loaded.
llama_build(test-reex-turboquant-kv-types.cpp)
target_compile_definitions(test-reex-turboquant-kv-types PRIVATE REEX_TURBOQUANT)
llama_test(test-reex-turboquant-kv-types)

# P4 A2.1 root-cause diagnostics: A3 systematic-bias detector + A1 multi-pass
# cos_sim decay curve on TQ_K3 / TQ_V2 / TQ_V4 with Q4_0 / Q8_0 as controls.
# Diagnostic only — controls block the build, subjects are reported informationally.
llama_build(test-reex-turboquant-p4a21-diagnostics.cpp)
target_compile_definitions(test-reex-turboquant-p4a21-diagnostics PRIVATE REEX_TURBOQUANT)
llama_test(test-reex-turboquant-p4a21-diagnostics)

# P4 A2.5 V-β — CPU smoke test (sub-item b acceptance gate).
# Covers type_traits wiring + round-trip cos_sim + strict norm preservation +
# codebook coverage + quasi-idempotency for GGML_TYPE_TQ_V_POLAR2/V_POLAR4.
# Oracle bit-identical parity lands in sub-item e (separate test file).
llama_build(test-reex-turboquant-v-polar-smoke.cpp)
target_compile_definitions(test-reex-turboquant-v-polar-smoke PRIVATE REEX_TURBOQUANT)
llama_test(test-reex-turboquant-v-polar-smoke)

# P4 A2.4 InnerQ EMA — Stage-1 (algorithm) + Stage 2b (ggml-aware
# interface) unit tests.  Stage-1 covers innerq_algo::accumulate /
# compute_scale + InnerQ class lifecycle.  Stage 2b adds: init_ggml on
# CPU buft, ggml_tensor lifecycle gating, host pending-counter trigger,
# and finalize_freeze_pending round-trip via ggml_backend_tensor_get/set.
# KV cache + graph integration land in Stages 2b-b/c.
llama_build(test-reex-turboquant-innerq.cpp)
target_compile_definitions(test-reex-turboquant-innerq PRIVATE REEX_TURBOQUANT)
target_include_directories(test-reex-turboquant-innerq PRIVATE
    ${REEX_TQ_LLAMA_SRC}
    ${REEX_TQ_LLAMA_SRC}/reex
)
llama_test(test-reex-turboquant-innerq)
