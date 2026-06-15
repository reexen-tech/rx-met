// REEX block-64 MMQ instance (asymmetric type, reuses q8_1 tile/vec_dot).
#include "../mmq.cuh"

#ifdef GGML_USE_REEX_Q64
DECL_MMQ_CASE(GGML_TYPE_Q4_1_64);
#endif
