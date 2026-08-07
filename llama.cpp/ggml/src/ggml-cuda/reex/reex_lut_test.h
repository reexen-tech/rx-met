#pragma once

#include "ggml-backend.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_cuda_reex_lut_test_op {
    GGML_CUDA_REEX_LUT_TEST_RECIPROCAL = 0,
    GGML_CUDA_REEX_LUT_TEST_RSQRT = 1,
};

GGML_BACKEND_API bool ggml_cuda_reex_lut_test_launch(
    int device, enum ggml_cuda_reex_lut_test_op op,
    const float * input, float * output, size_t count);

#ifdef __cplusplus
}
#endif
