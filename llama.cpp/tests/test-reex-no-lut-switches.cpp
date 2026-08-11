#include "reex/reex_lut_direct.h"
#include "reex/reex_lut_normalized.h"

#ifdef GGML_CUDA
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "reex_lut_test.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if !defined(GGML_REEX_EXP_NO_LUT) && !defined(GGML_REEX_LOG_NO_LUT) && \
    !defined(GGML_REEX_RECIPROCAL_NO_LUT) && !defined(GGML_REEX_RSQRT_NO_LUT) && \
    !defined(GGML_REEX_SQRT_NO_LUT)
#error "test-reex-no-lut-switches requires at least one NO_LUT switch"
#endif

enum class test_op {
    exp,
    log,
    reciprocal,
    rsqrt,
    sqrt,
};

static const char * op_name(test_op op) {
    switch (op) {
        case test_op::exp:        return "exp";
        case test_op::log:        return "log";
        case test_op::reciprocal: return "reciprocal";
        case test_op::rsqrt:      return "rsqrt";
        case test_op::sqrt:       return "sqrt";
    }
    return "unknown";
}

static std::vector<test_op> active_ops() {
    std::vector<test_op> result;
#ifdef GGML_REEX_EXP_NO_LUT
    result.push_back(test_op::exp);
#endif
#ifdef GGML_REEX_LOG_NO_LUT
    result.push_back(test_op::log);
#endif
#ifdef GGML_REEX_RECIPROCAL_NO_LUT
    result.push_back(test_op::reciprocal);
#endif
#ifdef GGML_REEX_RSQRT_NO_LUT
    result.push_back(test_op::rsqrt);
#endif
#ifdef GGML_REEX_SQRT_NO_LUT
    result.push_back(test_op::sqrt);
#endif
    return result;
}

static float eval_cpu(test_op op, float x) {
    switch (op) {
        case test_op::exp:        return ggml_exp_lut_mixed_fp16_f32_REEX(x);
        case test_op::log:        return ggml_log_lut_mixed_fp16_f32_REEX(x);
        case test_op::reciprocal: return ggml_reciprocal_lut_mixed_fp16_f32_REEX(x);
        case test_op::rsqrt:      return ggml_rsqrt_lut_mixed_fp16_f32_REEX(x);
        case test_op::sqrt:       return ggml_sqrt_lut_mixed_fp16_f32_REEX(x);
    }
    return std::numeric_limits<float>::quiet_NaN();
}

static float native_reference(test_op op, float x) {
    switch (op) {
        case test_op::exp:        return std::exp(x);
        case test_op::log:        return std::log(x);
        case test_op::reciprocal: return 1.0f / x;
        case test_op::rsqrt:      return 1.0f / std::sqrt(x);
        case test_op::sqrt:       return std::sqrt(x);
    }
    return std::numeric_limits<float>::quiet_NaN();
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool same_cpu(float actual, float expected) {
    return (std::isnan(actual) && std::isnan(expected)) ||
           float_bits(actual) == float_bits(expected);
}

static bool close_cuda(float actual, float expected) {
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    if (std::isinf(expected) || expected == 0.0f) {
        return float_bits(actual) == float_bits(expected);
    }
    const float tolerance = 8.0e-6f * std::max(1.0f, std::abs(expected));
    return std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
}

#ifdef GGML_CUDA
static ggml_tensor * build_cuda_graph_op(
        ggml_context * context, ggml_tensor * input, test_op op) {
    switch (op) {
        case test_op::exp:     return ggml_exp(context, input);
        case test_op::log:     return ggml_log(context, input);
        case test_op::sqrt:    return ggml_sqrt(context, input);
        case test_op::reciprocal:
        case test_op::rsqrt:
            break;
    }
    throw std::runtime_error("no CUDA graph op");
}

static std::vector<float> eval_cuda_graph(
        ggml_backend_t backend, test_op op, const std::vector<float> & inputs) {
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * context = ggml_init(params);
    if (context == nullptr) {
        throw std::runtime_error("cannot create CUDA graph context");
    }

    ggml_tensor * input = ggml_new_tensor_1d(context, GGML_TYPE_F32, inputs.size());
    ggml_tensor * output = build_cuda_graph_op(context, input, op);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        throw std::runtime_error("cannot allocate CUDA graph tensors");
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

static std::vector<float> eval_cuda_direct(test_op op, const std::vector<float> & inputs) {
    const ggml_cuda_reex_lut_test_op adapter_op =
        op == test_op::reciprocal ? GGML_CUDA_REEX_LUT_TEST_RECIPROCAL
                                  : GGML_CUDA_REEX_LUT_TEST_RSQRT;
    std::vector<float> results(inputs.size());
    if (!ggml_cuda_reex_lut_test_launch(
            0, adapter_op, inputs.data(), results.data(), inputs.size())) {
        results.clear();
    }
    return results;
}
#endif

int main() {
    const std::vector<float> inputs = {
        -10.0f,
        -3.0f,
        -1.0f,
        -0.0f,
        0.0f,
        0.125f,
        0.3f,
        0.5f,
        0.7f,
        1.0f,
        1.3f,
        2.0f,
        3.0f,
        10.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    const std::vector<test_op> ops = active_ops();

    for (test_op op : ops) {
        for (float input : inputs) {
            const float actual = eval_cpu(op, input);
            const float expected = native_reference(op, input);
            if (!same_cpu(actual, expected)) {
                fprintf(stderr,
                        "CPU native mismatch: op=%s input=%08x actual=%08x expected=%08x\n",
                        op_name(op), float_bits(input), float_bits(actual), float_bits(expected));
                return 1;
            }
        }
    }

#ifdef GGML_CUDA
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr || !ggml_backend_is_cuda(backend)) {
        fprintf(stderr, "CUDA backend unavailable; fallback is forbidden\n");
        return 1;
    }
    for (test_op op : ops) {
        const std::vector<float> actual =
            op == test_op::reciprocal || op == test_op::rsqrt
                ? eval_cuda_direct(op, inputs)
                : eval_cuda_graph(backend, op, inputs);
        if (actual.size() != inputs.size()) {
            fprintf(stderr, "CUDA execution failed: op=%s\n", op_name(op));
            ggml_backend_free(backend);
            return 1;
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            const float expected = native_reference(op, inputs[i]);
            if (!close_cuda(actual[i], expected)) {
                fprintf(stderr,
                        "CUDA native mismatch: op=%s input=%08x actual=%08x expected=%08x\n",
                        op_name(op), float_bits(inputs[i]), float_bits(actual[i]), float_bits(expected));
                ggml_backend_free(backend);
                return 1;
            }
        }
    }
    ggml_backend_free(backend);
#endif

    printf("REEX native helper switches: PASS (");
    for (size_t i = 0; i < ops.size(); ++i) {
        printf("%s%s", i == 0 ? "" : ",", op_name(ops[i]));
    }
#ifdef GGML_CUDA
    printf("; CPU + CUDA");
#else
    printf("; CPU");
#endif
    printf(")\n");
    return 0;
}
