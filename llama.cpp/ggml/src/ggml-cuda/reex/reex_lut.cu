#ifdef GGML_USE_REEX

/* LUT __constant__ and ggml_cuda_init_trigonometric_lut_REEX are defined in unary.cu
 * (same TU as kernel so device link sees the constant). This file is kept for any
 * future REEX CUDA LUT code that does not need to share constant memory with unary.cu.
 */

#endif /* GGML_USE_REEX */
