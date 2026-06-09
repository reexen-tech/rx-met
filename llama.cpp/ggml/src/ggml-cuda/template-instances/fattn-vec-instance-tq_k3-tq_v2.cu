// SPDX-License-Identifier: MIT
//
// REEX_TURBOQUANT FA-vec instance: K = TQ_K3, V = TQ_V2.
//
// Reaches `vec_dot_fattn_vec_KQ_tq_k3<D, nthreads>` via
// fattn-common.cuh::get_vec_dot_KQ<>. The kernel enforces
// `D % QK_TQ_K3 == 0` (one head_dim is a string of N x 128-elt TQ_K3 blocks,
// `ib = elem0 / QK_TQ_K3` walks them). We instantiate {128, 256} to cover
// hd=128 (Llama-3 / Qwen2-7B) and hd=256 (Qwen3 series) main-line models.
// V side reaches `dequantize_V_tq_v2<>` (group=32, D-agnostic).

#ifdef REEX_TURBOQUANT

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V2);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ_K3, GGML_TYPE_TQ_V2);

#endif  // REEX_TURBOQUANT
