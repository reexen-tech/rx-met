// SPDX-License-Identifier: MIT
//
// REEX_TURBOQUANT FA-vec instance: K = F16, V = TQ_V2.
//
// Reaches `dequantize_V_tq_v2<>` via fattn-common.cuh::get_dequantize_V<>;
// K side keeps the upstream `vec_dot_fattn_vec_KQ_f16` path.
//
// V is group=32 (`QK_TQ_V2 = 32`) and does not impose any constraint on the
// head-dim D, so we instantiate the same {64, 128, 256} ladder upstream uses
// for the vanilla quant cases. K-side TQ_K3 instances only ship D=128 in
// `fattn-vec-instance-tq_k3-tq_v*.cu`.

#ifdef REEX_TURBOQUANT

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_TQ_V2);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_TQ_V2);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_TQ_V2);

#endif  // REEX_TURBOQUANT
