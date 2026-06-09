// SPDX-License-Identifier: MIT
//
// REEX_TURBOQUANT FA-vec instance: K = TQ_K3, V = TQ_V4.
//
// High-quality V variant (4-bit per element, group=32). Reaches
// `vec_dot_fattn_vec_KQ_tq_k3<D, nthreads>` (D % QK_TQ_K3 == 0) and
// `dequantize_V_tq_v4<>`. Same D ladder as the TQ_V2 sibling.

#ifdef REEX_TURBOQUANT

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V4);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V4);

#endif  // REEX_TURBOQUANT
