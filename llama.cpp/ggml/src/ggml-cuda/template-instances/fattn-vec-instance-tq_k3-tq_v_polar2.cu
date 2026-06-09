// SPDX-License-Identifier: MIT
//
// REEX_TURBOQUANT FA-vec instance: K = TQ_K3, V = TQ_V_POLAR2.
//
// P4 A2.5 V-β (TheTom turbo3_0 K + turbo2_0 V):
//   K reaches `vec_dot_fattn_vec_KQ_tq_k3<D, nthreads>` (D % QK_TQ_K3 == 0)
//   V reaches `dequantize_V_tq_v_polar2<>` (group=128, Lloyd-Max codebook +
//   norm correction).
//
// Same D ladder ({128, 256}) as the legacy (K3, V2)/(K3, V4) siblings so
// hd=128 (Llama-3 / Qwen2-7B) and hd=256 (Qwen3 series) main-line models
// pick up V_POLAR via -ctv tq_v_polar2 once tq_v_active is extended.

#ifdef REEX_TURBOQUANT

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V_POLAR2);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V_POLAR2);

#endif  // REEX_TURBOQUANT
