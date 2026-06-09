// SPDX-License-Identifier: MIT
//
// REEX_TURBOQUANT FA-vec instance: K = TQ_K3, V = TQ_V_POLAR4.
//
// P4 A2.5 V-β (TheTom turbo3_0 K + turbo4_0 V):
//   K reaches `vec_dot_fattn_vec_KQ_tq_k3<D, nthreads>` (D % QK_TQ_K3 == 0)
//   V reaches `dequantize_V_tq_v_polar4<>` (group=128, 16-centroid Lloyd-Max
//   codebook + norm correction).
//
// High-quality 4-bit V variant. Same D ladder ({128, 256}) as the TQ_V_POLAR2
// sibling.

#ifdef REEX_TURBOQUANT

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V_POLAR4);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V_POLAR4);

#endif  // REEX_TURBOQUANT
