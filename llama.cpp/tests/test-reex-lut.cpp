/**
 * ADA300 LUT bit-level conformance test.
 *
 * Python is the only input/golden producer. This executable consumes the exact
 * input bits, invokes production CPU/CUDA interfaces, and writes result TSVs.
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "reex/ggml-reex-lut-config.h"
#ifdef GGML_USE_REEX
#include "reex/reex_lut.h"
#include "reex/reex_lut_direct.h"
#include "reex/reex_lut_normalized.h"
#endif

#ifdef GGML_CUDA
#include "ggml-cuda.h"
#include "reex_lut_test.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string tsv_header(const std::string & precision) {
    return "# reex-lut-bit-v1 segments=" + std::to_string(GGML_LUT_NUM_SEGMENTS_REEX) +
           " precision=" + precision;
}

struct test_case {
    std::string op;
    std::string case_id;
    uint32_t input_bits;
    uint32_t expected_bits;
    std::string expected_class;
};

struct op_stats {
    uint64_t total = 0;
    uint64_t exact = 0;
    uint64_t failures = 0;
    uint64_t max_ulp = 0;
};

static uint32_t f32_bits(float value) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_f32(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static std::string classify(float value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return std::signbit(value) ? "-inf" : "+inf";
    }
    if (value == 0.0f) {
        return std::signbit(value) ? "-zero" : "+zero";
    }
    return "finite";
}

static uint64_t ordered_bits(uint32_t bits) {
    return (bits & 0x80000000U) ? static_cast<uint64_t>(~bits)
                                : static_cast<uint64_t>(bits | 0x80000000U);
}

static uint64_t ulp_distance(uint32_t expected_bits, const std::string & expected_class,
                             uint32_t actual_bits, const std::string & actual_class) {
    if (expected_class == "finite" && actual_class == "finite") {
        const uint64_t a = ordered_bits(expected_bits);
        const uint64_t b = ordered_bits(actual_bits);
        return a > b ? a - b : b - a;
    }
    return expected_class == actual_class ? 0 : std::numeric_limits<uint32_t>::max();
}

static uint32_t parse_hex32(const std::string & text, size_t line_number) {
    if (text.size() != 8) {
        throw std::runtime_error("line " + std::to_string(line_number) + ": invalid hex width");
    }
    uint32_t value = 0;
    for (char ch : text) {
        value <<= 4;
        if (ch >= '0' && ch <= '9') {
            value |= static_cast<uint32_t>(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
            value |= static_cast<uint32_t>(ch - 'a' + 10);
        } else {
            throw std::runtime_error("line " + std::to_string(line_number) + ": invalid lowercase hex");
        }
    }
    return value;
}

static std::vector<test_case> load_cases(const std::string & path, const std::string & precision) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open golden TSV: " + path);
    }
    std::string line;
    if (!std::getline(input, line) || line != tsv_header(precision)) {
        throw std::runtime_error("invalid or missing TSV header: " + path);
    }

    std::vector<test_case> cases;
    size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream row(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(row, field, '\t')) {
            fields.push_back(field);
        }
        if (fields.size() != 5) {
            throw std::runtime_error("line " + std::to_string(line_number) + ": expected five TSV fields");
        }
        cases.push_back({
            fields[0], fields[1], parse_hex32(fields[2], line_number),
            parse_hex32(fields[3], line_number), fields[4],
        });
    }
    if (cases.empty()) {
        throw std::runtime_error("golden TSV contains no cases: " + path);
    }
    return cases;
}

static float eval_cpu(const std::string & op, float x) {
    if (op == "exp") {
        return ggml_exp_lut_mixed_fp16_f32_REEX(x);
    }
    if (op == "sigmoid") {
        return ggml_sigmoid_lut_mixed_fp16_f32_REEX(x);
    }
    if (op == "sin") {
        return ggml_sin_lut_mixed_fp16_f32_REEX(x);
    }
    if (op == "cos") {
        return ggml_cos_lut_mixed_fp16_f32_REEX(x);
    }
    if (op == "reciprocal") {
        return ggml_reciprocal_lut_mixed_fp16_f32_REEX(x);
    }
    if (op == "rsqrt") {
        return ggml_rsqrt_lut_mixed_fp16_f32_REEX(x);
    }
    if (op == "sqrt") {
        return ggml_sqrt_lut_mixed_fp16_f32_REEX(x);
    }
    if (op == "log") {
        return ggml_log_lut_mixed_fp16_f32_REEX(x);
    }
    if (op == "silu") {
        return ggml_silu_lut_mixed_fp16_f32_REEX(x);
    }
    throw std::runtime_error("unsupported op: " + op);
}

static std::vector<float> eval_all_cpu(const std::vector<test_case> & cases) {
    std::vector<float> output;
    output.reserve(cases.size());
    for (const test_case & tc : cases) {
        output.push_back(eval_cpu(tc.op, bits_f32(tc.input_bits)));
    }
    return output;
}

static bool write_and_check(
        const std::vector<test_case> & cases, const std::vector<float> & actual,
        const std::string & path, const char * backend_name, const std::string & precision) {
    if (actual.size() != cases.size()) {
        throw std::runtime_error("result count mismatch");
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open output TSV: " + path);
    }
    output << tsv_header(precision) << '\n';

    std::map<std::string, op_stats> stats;
    bool passed = true;
    bool reported_failure = false;
    for (size_t index = 0; index < cases.size(); ++index) {
        const test_case & tc = cases[index];
        const uint32_t actual_bits = f32_bits(actual[index]);
        const std::string actual_class = classify(actual[index]);
        const uint64_t ulp = ulp_distance(tc.expected_bits, tc.expected_class, actual_bits, actual_class);
        op_stats & stat = stats[tc.op];
        ++stat.total;
        stat.max_ulp = std::max(stat.max_ulp, ulp);

        const bool exact = actual_bits == tc.expected_bits ||
                           (tc.expected_class == "nan" && actual_class == "nan");
        stat.exact += exact;
        if (ulp > 1) {
            passed = false;
            ++stat.failures;
            if (!reported_failure) {
                fprintf(stderr,
                        "first %s mismatch: op=%s case=%s input=%08x expected=%08x/%s "
                        "actual=%08x/%s ulp=%llu\n",
                        backend_name, tc.op.c_str(), tc.case_id.c_str(), tc.input_bits,
                        tc.expected_bits, tc.expected_class.c_str(), actual_bits,
                        actual_class.c_str(), static_cast<unsigned long long>(ulp));
                reported_failure = true;
            }
        }

        output << tc.op << '\t' << tc.case_id << '\t'
               << std::hex << std::setw(8) << std::setfill('0') << tc.input_bits << '\t'
               << std::setw(8) << actual_bits << std::dec << '\t' << actual_class << '\n';
    }

    for (const auto & entry : stats) {
        const op_stats & stat = entry.second;
        printf("[%s] %-10s exact=%llu/%llu (%.6f%%) MaxULP=%llu failures=%llu\n",
               backend_name, entry.first.c_str(),
               static_cast<unsigned long long>(stat.exact),
               static_cast<unsigned long long>(stat.total),
               100.0 * static_cast<double>(stat.exact) / static_cast<double>(stat.total),
               static_cast<unsigned long long>(stat.max_ulp),
               static_cast<unsigned long long>(stat.failures));
    }
    return passed;
}

#ifdef GGML_CUDA
static ggml_tensor * build_cuda_graph_op(
        ggml_context * context, ggml_tensor * input, const std::string & op) {
    if (op == "exp") {
        return ggml_exp(context, input);
    }
    if (op == "sigmoid") {
        return ggml_sigmoid(context, input);
    }
    if (op == "sin") {
        return ggml_sin(context, input);
    }
    if (op == "cos") {
        return ggml_cos(context, input);
    }
    if (op == "sqrt") {
        return ggml_sqrt(context, input);
    }
    if (op == "log") {
        return ggml_log(context, input);
    }
    if (op == "silu") {
        return ggml_silu(context, input);
    }
    throw std::runtime_error("no CUDA graph op for " + op);
}

static std::vector<float> eval_cuda_graph(
        ggml_backend_t backend, const std::string & op, const std::vector<float> & input) {
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * context = ggml_init(params);
    if (context == nullptr) {
        throw std::runtime_error("cannot create CUDA graph context");
    }

    ggml_tensor * input_tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, input.size());
    ggml_tensor * output_tensor = build_cuda_graph_op(context, input_tensor, op);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output_tensor);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        throw std::runtime_error("cannot allocate CUDA graph tensors for " + op);
    }

    ggml_backend_tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        throw std::runtime_error("CUDA graph compute failed for " + op);
    }

    std::vector<float> output(input.size());
    ggml_backend_tensor_get(output_tensor, output.data(), 0, output.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return output;
}

static std::vector<float> eval_all_cuda(
        ggml_backend_t backend, const std::vector<test_case> & cases) {
    std::map<std::string, std::vector<size_t>> groups;
    for (size_t index = 0; index < cases.size(); ++index) {
        groups[cases[index].op].push_back(index);
    }

    std::vector<float> output(cases.size());
    for (const auto & entry : groups) {
        const std::string & op = entry.first;
        const std::vector<size_t> & indices = entry.second;
        std::vector<float> input;
        input.reserve(indices.size());
        for (size_t index : indices) {
            input.push_back(bits_f32(cases[index].input_bits));
        }

        std::vector<float> op_output(input.size());
        if (op == "reciprocal" || op == "rsqrt") {
            const ggml_cuda_reex_lut_test_op test_op =
                op == "reciprocal" ? GGML_CUDA_REEX_LUT_TEST_RECIPROCAL
                                   : GGML_CUDA_REEX_LUT_TEST_RSQRT;
            if (!ggml_cuda_reex_lut_test_launch(0, test_op, input.data(), op_output.data(), input.size())) {
                throw std::runtime_error("CUDA test-only launch adapter failed for " + op);
            }
        } else {
            op_output = eval_cuda_graph(backend, op, input);
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            output[indices[i]] = op_output[i];
        }
    }
    return output;
}
#endif

static void usage(const char * argv0) {
#ifdef GGML_CUDA
    fprintf(stderr, "usage: %s --golden PATH --cpu-output PATH --cuda-output PATH "
                    "[--precision mixed_fp16|fp32]\n", argv0);
#else
    fprintf(stderr, "usage: %s --golden PATH --cpu-output PATH "
                    "[--precision mixed_fp16|fp32]\n", argv0);
#endif
}

int main(int argc, char ** argv) {
#ifndef GGML_USE_REEX
    fprintf(stderr, "test-reex-lut requires GGML_USE_REEX\n");
    return 1;
#else
    std::string golden_path;
    std::string cpu_output_path;
    std::string cuda_output_path;
    std::string precision = "mixed_fp16";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--golden") == 0 && i + 1 < argc) {
            golden_path = argv[++i];
        } else if (strcmp(argv[i], "--cpu-output") == 0 && i + 1 < argc) {
            cpu_output_path = argv[++i];
        } else if (strcmp(argv[i], "--cuda-output") == 0 && i + 1 < argc) {
            cuda_output_path = argv[++i];
        } else if (strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
            precision = argv[++i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (golden_path.empty() || cpu_output_path.empty()
#ifdef GGML_CUDA
        || cuda_output_path.empty()
#endif
    ) {
        usage(argv[0]);
        return 2;
    }
    if (precision != "mixed_fp16" && precision != "fp32") {
        fprintf(stderr, "unsupported precision: %s\n", precision.c_str());
        usage(argv[0]);
        return 2;
    }

    try {
        ggml_init_trigonometric_lut_REEX();
        const std::vector<test_case> cases = load_cases(golden_path, precision);
        const bool cpu_passed = write_and_check(
            cases, eval_all_cpu(cases), cpu_output_path, "CPU", precision);
        bool cuda_passed = true;
#ifdef GGML_CUDA
        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (backend == nullptr || !ggml_backend_is_cuda(backend)) {
            throw std::runtime_error("CUDA backend unavailable; fallback is forbidden");
        }
        cuda_passed = write_and_check(
            cases, eval_all_cuda(backend, cases), cuda_output_path, "CUDA", precision);
        ggml_backend_free(backend);
#else
        if (!cuda_output_path.empty()) {
            throw std::runtime_error("this test binary was built without CUDA");
        }
#endif
        const bool passed = cpu_passed && cuda_passed;
        printf("ADA300 %s CPU/CUDA gate: %s\n", precision.c_str(), passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception & error) {
        fprintf(stderr, "test-reex-lut: %s\n", error.what());
        return 1;
    }
#endif
}
