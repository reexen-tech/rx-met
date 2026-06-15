// REEX K-quant block-64 MMQ instance (dequant -> q8_0/q8_1 tile, reuse vec_dot).
#include "../mmq.cuh"

#ifdef GGML_USE_REEX_Q64
DECL_MMQ_CASE(GGML_TYPE_Q4_K_64);
#endif
