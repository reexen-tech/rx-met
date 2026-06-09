#include "../ggml/src/ggml-cuda/fattn-policy.h"

#include <cassert>

int main() {
    assert(ggml_cuda_reex_can_use_fattn_vector_kernel(true, 64, 256));
    assert(ggml_cuda_reex_can_use_fattn_vector_kernel(true, 256, 512));

    assert(!ggml_cuda_reex_can_use_fattn_vector_kernel(false, 64, 256));
    assert(!ggml_cuda_reex_can_use_fattn_vector_kernel(true, 32, 256));
    assert(!ggml_cuda_reex_can_use_fattn_vector_kernel(true, 96, 255));
    assert(!ggml_cuda_reex_can_use_fattn_vector_kernel(true, 320, 256));

    return 0;
}
