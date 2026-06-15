// REEX block-64 MMQ instance (symmetric type, reuses q8_0 tile/vec_dot).
#include "../mmq.cuh"

#ifdef GGML_USE_REEX_Q64
DECL_MMQ_CASE(GGML_TYPE_Q5_0_64);
#endif
