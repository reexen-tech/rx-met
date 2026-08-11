#include "reex/reex_lut.h"

#ifdef GGML_CUDA
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#ifndef GGML_REEX_SIN_COS_NO_LUT
#error "test-reex-sin-cos-no-lut requires GGML_REEX_SIN_COS_NO_LUT"
#endif

static uint32_t float_bits(float value) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool same_result(float actual, float expected) {
    return (std::isnan(actual) && std::isnan(expected)) ||
           float_bits(actual) == float_bits(expected);
}

#ifdef GGML_CUDA
static std::vector<float> eval_cuda(ggml_backend_t backend, const std::vector<float> & inputs, bool cosine) {
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * context = ggml_init(params);
    if (context == nullptr) {
        return {};
    }

    ggml_tensor * input = ggml_new_tensor_1d(context, GGML_TYPE_F32, inputs.size());
    ggml_tensor * output = cosine ? ggml_cos(context, input) : ggml_sin(context, input);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        return {};
    }

    ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
    std::vector<float> results(inputs.size());
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, results.data(), 0, results.size() * sizeof(float));
    } else {
        results.clear();
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return results;
}
#endif

int main() {
    const std::vector<float> inputs = {
        -10000.0f,
        -10.0f,
        -3.1415927f,
        -1.0f,
        -0.0f,
        0.0f,
        0.125f,
        0.5f,
        1.0f,
        3.1415927f,
        10.0f,
        10000.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };

    for (float input : inputs) {
        const float actual_sin = ggml_sin_lut_mixed_fp16_f32_REEX(input);
        const float actual_cos = ggml_cos_lut_mixed_fp16_f32_REEX(input);
        const float native_sin = std::sin(input);
        const float native_cos = std::cos(input);
        if (!same_result(actual_sin, native_sin) || !same_result(actual_cos, native_cos)) {
            fprintf(stderr,
                    "native sin/cos mismatch: input=%08x sin=%08x/%08x cos=%08x/%08x\n",
                    float_bits(input), float_bits(actual_sin), float_bits(native_sin),
                    float_bits(actual_cos), float_bits(native_cos));
            return 1;
        }
    }

#ifdef GGML_CUDA
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr || !ggml_backend_is_cuda(backend)) {
        fprintf(stderr, "CUDA backend unavailable; fallback is forbidden\n");
        return 1;
    }
    const std::vector<float> cuda_sin = eval_cuda(backend, inputs, false);
    const std::vector<float> cuda_cos = eval_cuda(backend, inputs, true);
    ggml_backend_free(backend);
    if (cuda_sin.size() != inputs.size() || cuda_cos.size() != inputs.size()) {
        fprintf(stderr, "CUDA sin/cos graph execution failed\n");
        return 1;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        const float input = inputs[i];
        const float native_sin = std::sin(input);
        const float native_cos = std::cos(input);
        if (std::isfinite(input) && std::abs(input) <= 10.0f) {
            if (std::abs(cuda_sin[i] - native_sin) > 2.0e-6f ||
                std::abs(cuda_cos[i] - native_cos) > 2.0e-6f) {
                fprintf(stderr,
                        "CUDA native sin/cos mismatch: input=%08x sin=%08x/%08x cos=%08x/%08x\n",
                        float_bits(input), float_bits(cuda_sin[i]), float_bits(native_sin),
                        float_bits(cuda_cos[i]), float_bits(native_cos));
                return 1;
            }
        } else if (!std::isfinite(input) &&
                   (!std::isnan(cuda_sin[i]) || !std::isnan(cuda_cos[i]))) {
            fprintf(stderr, "CUDA native sin/cos special-value mismatch: input=%08x\n", float_bits(input));
            return 1;
        }
    }

    const size_t one_index = 8;
    if ((float_bits(cuda_sin[one_index]) & 0x1fffU) == 0 ||
        (float_bits(cuda_cos[one_index]) & 0x1fffU) == 0) {
        fprintf(stderr, "CUDA sin/cos output at x=1 appears FP16 LUT-rounded\n");
        return 1;
    }
#endif

    printf("REEX sin/cos native path: PASS (%zu inputs", inputs.size());
#ifdef GGML_CUDA
    printf(", CPU + CUDA");
#else
    printf(", CPU");
#endif
    printf(")\n");
    return 0;
}
