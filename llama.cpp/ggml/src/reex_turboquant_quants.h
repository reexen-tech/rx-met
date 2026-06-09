// SPDX-License-Identifier: MIT
//
// reex_turboquant_quants.h — REEX_TURBOQUANT only.
//
// Public per-row quantize / dequantize entry points for GGML_TYPE_TQ_K3 /
// TQ_V2 / TQ_V4. Used by:
//   - `type_traits[]` in ggml.c    (the `_ref` and `dequantize_*` half;
//                                    implemented in reex_turboquant_quants_ref.c
//                                    inside libggml-base)
//   - `type_traits_cpu[]` in ggml-cpu.c (the production `quantize_row_*` half;
//                                        implemented in
//                                        ggml-cpu/reex/reex_turboquant_quants.c
//                                        inside libggml-cpu)
//
// References:
//   - tests/oracle/reex_turboquant_oracle_core.{h,cpp} — bit-level oracle
//   - docs/turboquant/02_data_layout.md                — block bit layout
//   - docs/turboquant/03_graph_integration.md          — caller responsibility
//
// IMPORTANT (TQ_K3): the K vector is expected to be ALREADY WHT-rotated by
// the caller via GGML_OP_REEX_WHT before quantize_row_tq_k3_* runs, and the
// dequantized output is still in the WHT-rotated frame. The graph is
// responsible for inserting an inverse WHT after the dequant. This keeps the
// rotation observable at the graph level and lets the backend WHT kernel
// fuse with surrounding ops.
//
// IMPORTANT (TQ_V2 / TQ_V4, factually corrected 2026-05-11): the V cache IS
// rotated via V WHT — same group=128 WHT op as K cache, applied in
// src/llama-kv-cache.cpp::cpy_v under tq_v_active before quantize, with the
// inverse WHT applied at attention output (build_attn_mha) to restore the
// un-rotated tensor before o_proj. The earlier "V cache is NOT rotated"
// comment was a doc/code drift bug from P2 phase docs; V WHT has been
// present since P2 phase land. The quantize_row_tq_v{2,4}_ref kernels
// themselves are still straight per-block-32 min/max scalar quantization
// (matching the oracle's `reex_turboquant_oracle_value_quantize` path) —
// the WHT happens at the graph layer, not inside the kernel. See
// docs/turboquant/02_data_layout.md §1 末 "V WHT 状态（事实勘正，2026-05-11）".
//
// IMPORTANT (TQ_V_POLAR2 / TQ_V_POLAR4, P4 A2.5 V-β 2026-05-11+): the V
// cache is rotated via V WHT (same as above; tq_v_active predicate is
// extended to include V_POLAR2/V_POLAR4 with no other graph-layer change).
// Quantization is PolarQuant: each block_size=128 input is normalized to a
// unit vector, each element is matched to the nearest Lloyd-Max centroid
// (4 centroids @ 2-bit / 16 centroids @ 4-bit, N(0,1)-optimal),
// and `norm = corrected = ‖x‖ / ‖recon_unit‖` is stored so that
// ‖dequant(q)‖ ≡ ‖x‖ (strict idempotency). Centroid table is shared with
// TheTom turbo2_0/turbo4_0 (llama-cpp-turboquant) for bit-identical
// interop. See docs/turboquant/p4.a25_v_polar_design.md §5 for the full
// quantize/dequant pseudocode.

#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef REEX_TURBOQUANT

// === TQ_K3 (K cache, 3 bits per element = full PolarQuant, no QJL split) ===
// `k` must be a multiple of QK_TQ_K3 (= 128).
//
// P4 A2.2 K3 P3 upgrade (TheTom turbo3_0 path, 2026-04-30):
// `signs[]` byte stream now packs the HIGH 1 bit of the 3-bit centroid index
// (8 centroids per Lloyd-Max codebook), not a QJL sign.  The `norm` field
// stores `corrected = ‖x‖ / ‖recon_unit‖` so dequant output ‖y‖ ≡ ‖x‖
// (idempotency guarantee, see ggml/src/reex_turboquant_quants_ref.c).
// On-disk byte layout sizeof unchanged from P1 (50 B / 128 elt) but the
// byte-stream interpretation is **not bitwise compatible** with P1 K cache
// GGUFs — re-quantize required.  See
// docs/turboquant/p4.a22_k3_p3_upgrade_plan.md §0 for full rationale.
GGML_API void quantize_row_tq_k3_ref(const float       * GGML_RESTRICT x,
                                            block_tq_k3 * GGML_RESTRICT y,
                                            int64_t                     k);
GGML_API void quantize_row_tq_k3   (const float       * GGML_RESTRICT x,
                                            void        * GGML_RESTRICT y,
                                            int64_t                     k);
GGML_API void dequantize_row_tq_k3 (const block_tq_k3 * GGML_RESTRICT x,
                                            float       * GGML_RESTRICT y,
                                            int64_t                     k);

// === TQ_V2 (V cache, 2-bit per-block-32 min-max scalar) ===
// `k` must be a multiple of QK_TQ_V2 (= 32).
GGML_API void quantize_row_tq_v2_ref(const float       * GGML_RESTRICT x,
                                            block_tq_v2 * GGML_RESTRICT y,
                                            int64_t                     k);
GGML_API void quantize_row_tq_v2   (const float       * GGML_RESTRICT x,
                                            void        * GGML_RESTRICT y,
                                            int64_t                     k);
GGML_API void dequantize_row_tq_v2 (const block_tq_v2 * GGML_RESTRICT x,
                                            float       * GGML_RESTRICT y,
                                            int64_t                     k);

// === TQ_V4 (V cache, 4-bit per-block-32 min-max scalar) ===
// `k` must be a multiple of QK_TQ_V4 (= 32).
GGML_API void quantize_row_tq_v4_ref(const float       * GGML_RESTRICT x,
                                            block_tq_v4 * GGML_RESTRICT y,
                                            int64_t                     k);
GGML_API void quantize_row_tq_v4   (const float       * GGML_RESTRICT x,
                                            void        * GGML_RESTRICT y,
                                            int64_t                     k);
GGML_API void dequantize_row_tq_v4 (const block_tq_v4 * GGML_RESTRICT x,
                                            float       * GGML_RESTRICT y,
                                            int64_t                     k);

// === TQ_V_POLAR2 (V cache, P4 A2.5 V-β, 2-bit PolarQuant + Lloyd-Max codebook) ===
// `k` must be a multiple of QK_TQ_V_POLAR2 (= 128). Input is expected to be
// in the WHT-rotated frame (caller responsibility, via GGML_OP_REEX_WHT in
// cpy_v under tq_v_active, same as legacy tq_v2/tq_v4). Dequantized output
// is still in the WHT-rotated frame — the graph inserts inverse WHT at
// attention output.
GGML_API void quantize_row_tq_v_polar2_ref(const float             * GGML_RESTRICT x,
                                                  block_tq_v_polar2 * GGML_RESTRICT y,
                                                  int64_t                           k);
GGML_API void quantize_row_tq_v_polar2   (const float             * GGML_RESTRICT x,
                                                  void              * GGML_RESTRICT y,
                                                  int64_t                           k);
GGML_API void dequantize_row_tq_v_polar2 (const block_tq_v_polar2 * GGML_RESTRICT x,
                                                  float             * GGML_RESTRICT y,
                                                  int64_t                           k);

// === TQ_V_POLAR4 (V cache, P4 A2.5 V-β, 4-bit PolarQuant + Lloyd-Max codebook) ===
// `k` must be a multiple of QK_TQ_V_POLAR4 (= 128).
GGML_API void quantize_row_tq_v_polar4_ref(const float             * GGML_RESTRICT x,
                                                  block_tq_v_polar4 * GGML_RESTRICT y,
                                                  int64_t                           k);
GGML_API void quantize_row_tq_v_polar4   (const float             * GGML_RESTRICT x,
                                                  void              * GGML_RESTRICT y,
                                                  int64_t                           k);
GGML_API void dequantize_row_tq_v_polar4 (const block_tq_v_polar4 * GGML_RESTRICT x,
                                                  float             * GGML_RESTRICT y,
                                                  int64_t                           k);

// === Whole-row "chunk" entry points used by ggml_quantize_chunk ===
// Signature mirrors quantize_q4_0(src, dst, nrow, n_per_row, imatrix); the
// imatrix argument is currently ignored (TurboQuant does not use it in
// Phase 1).  Each implementation forwards to the corresponding
// quantize_row_*_ref.  Returns the number of bytes written.
GGML_API size_t quantize_tq_k3(const float * GGML_RESTRICT src,
                                     void  * GGML_RESTRICT dst,
                                     int64_t                nrow,
                                     int64_t                n_per_row,
                                     const float * GGML_RESTRICT quant_weights);
GGML_API size_t quantize_tq_v2(const float * GGML_RESTRICT src,
                                     void  * GGML_RESTRICT dst,
                                     int64_t                nrow,
                                     int64_t                n_per_row,
                                     const float * GGML_RESTRICT quant_weights);
GGML_API size_t quantize_tq_v4(const float * GGML_RESTRICT src,
                                     void  * GGML_RESTRICT dst,
                                     int64_t                nrow,
                                     int64_t                n_per_row,
                                     const float * GGML_RESTRICT quant_weights);
GGML_API size_t quantize_tq_v_polar2(const float * GGML_RESTRICT src,
                                           void  * GGML_RESTRICT dst,
                                           int64_t                nrow,
                                           int64_t                n_per_row,
                                           const float * GGML_RESTRICT quant_weights);
GGML_API size_t quantize_tq_v_polar4(const float * GGML_RESTRICT src,
                                           void  * GGML_RESTRICT dst,
                                           int64_t                nrow,
                                           int64_t                n_per_row,
                                           const float * GGML_RESTRICT quant_weights);

// === vec_dot entry points (CPU only, used in `type_traits_cpu[].vec_dot`) ===
// Signature mirrors ggml_vec_dot_q4_0_q8_0 — vy is an F32 vector of length n
// because we set vec_dot_type = GGML_TYPE_F32 for the TQ types in
// ggml-cpu.c.  Phase 2 implementation: scalar dequantize-then-dot, no SIMD.
// SIMD variants will be revisited if perf becomes a bottleneck.
GGML_API void ggml_vec_dot_tq_k3_f32(int n, float * GGML_RESTRICT s, size_t bs,
                                     const void * GGML_RESTRICT vx, size_t bx,
                                     const void * GGML_RESTRICT vy, size_t by,
                                     int nrc);
GGML_API void ggml_vec_dot_tq_v2_f32(int n, float * GGML_RESTRICT s, size_t bs,
                                     const void * GGML_RESTRICT vx, size_t bx,
                                     const void * GGML_RESTRICT vy, size_t by,
                                     int nrc);
GGML_API void ggml_vec_dot_tq_v4_f32(int n, float * GGML_RESTRICT s, size_t bs,
                                     const void * GGML_RESTRICT vx, size_t bx,
                                     const void * GGML_RESTRICT vy, size_t by,
                                     int nrc);
GGML_API void ggml_vec_dot_tq_v_polar2_f32(int n, float * GGML_RESTRICT s, size_t bs,
                                           const void * GGML_RESTRICT vx, size_t bx,
                                           const void * GGML_RESTRICT vy, size_t by,
                                           int nrc);
GGML_API void ggml_vec_dot_tq_v_polar4_f32(int n, float * GGML_RESTRICT s, size_t bs,
                                           const void * GGML_RESTRICT vx, size_t bx,
                                           const void * GGML_RESTRICT vy, size_t by,
                                           int nrc);

#endif // REEX_TURBOQUANT

#ifdef __cplusplus
}
#endif
