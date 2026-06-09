/**
 * Test for REEX LUT precision (Sin/Cos + direct + normalized ops)
 *
 * Compares REEX LUT against libm and reports metrics for:
 *   - Full-chain FP16/BF16 (*_lut_fp16_f32_REEX / *_lut_bf16_f32_REEX), and
 *   - Mixed-FP16 inference path (*_mixed_fp16_f32_REEX; matches ggml unary / vec).
 * CUDA: ggml graph unary vs libm as [GPU Mixed-FP16 vs libm]; GPU vs CPU mixed parity test.
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_REEX
#include "reex/ggml-reex-lut-config.h"
#include "reex/reex_lut.h"
#include "reex/reex_lut_direct.h"
#include "reex/reex_lut_normalized.h"
#endif

#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>
#include <string>

// Same as lut_bf_fp_new test_same_as_python / run_same_experiment.py
static const int NUM_TEST_POINTS_SAME_AS_PYTHON = 1000;
static const float PI_F = 3.14159265358979323846f;

// Test configuration
struct test_config {
    float input_min;
    float input_max;
    int n_samples;
    const char * name;
};

// Precision metrics
struct precision_metrics {
    double max_abs_error;
    double mean_abs_error;
    double max_rel_error;
    double mean_rel_error;
    double cosine_similarity;
    int n_samples;
};

// Calculate precision metrics
precision_metrics calculate_metrics(const std::vector<float> & reference, 
                                    const std::vector<float> & computed) {
    precision_metrics metrics;
    metrics.n_samples = reference.size();
    
    double sum_abs_error = 0.0;
    double sum_rel_error = 0.0;
    double sum_ref_sq = 0.0;
    double sum_computed_sq = 0.0;
    double sum_dot = 0.0;
    
    metrics.max_abs_error = 0.0;
    metrics.max_rel_error = 0.0;
    
    for (size_t i = 0; i < reference.size(); ++i) {
        float ref = reference[i];
        float comp = computed[i];
        float abs_error = std::abs(ref - comp);
        float rel_error = (ref != 0.0f) ? abs_error / std::abs(ref) : 0.0f;
        
        sum_abs_error += abs_error;
        if (ref != 0.0f) {
            sum_rel_error += rel_error;
        }
        
        metrics.max_abs_error = std::max(metrics.max_abs_error, static_cast<double>(abs_error));
        if (ref != 0.0f) {
            metrics.max_rel_error = std::max(metrics.max_rel_error, static_cast<double>(rel_error));
        }
        
        sum_ref_sq += ref * ref;
        sum_computed_sq += comp * comp;
        sum_dot += ref * comp;
    }
    
    metrics.mean_abs_error = sum_abs_error / reference.size();
    metrics.mean_rel_error = sum_rel_error / reference.size();
    
    // Cosine similarity
    double norm_ref = std::sqrt(sum_ref_sq);
    double norm_computed = std::sqrt(sum_computed_sq);
    if (norm_ref > 0.0 && norm_computed > 0.0) {
        metrics.cosine_similarity = sum_dot / (norm_ref * norm_computed);
    } else {
        metrics.cosine_similarity = 1.0;
    }
    
    return metrics;
}

// Test sin function
void test_sin(const test_config & config) {
    printf("\n=== Testing SIN: %s ===\n", config.name);
    printf("Input range: [%.6f, %.6f], Samples: %d\n", 
           config.input_min, config.input_max, config.n_samples);
    
    std::vector<float> inputs(config.n_samples);
    std::vector<float> reference(config.n_samples);
    std::vector<float> computed_fp16(config.n_samples);
    std::vector<float> computed_bf16(config.n_samples);
    
    // Generate test inputs
    for (int i = 0; i < config.n_samples; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(config.n_samples - 1);
        inputs[i] = config.input_min + t * (config.input_max - config.input_min);
    }
    
    // Compute reference (standard library)
    for (int i = 0; i < config.n_samples; ++i) {
        reference[i] = sinf(inputs[i]);
    }
    
    // Compute with REEX LUT
#ifdef GGML_USE_REEX
    std::vector<float> computed_mixed(config.n_samples);
    for (int i = 0; i < config.n_samples; ++i) {
        computed_fp16[i] = ggml_sin_lut_fp16_f32_REEX(inputs[i]);
        computed_bf16[i] = ggml_sin_lut_bf16_f32_REEX(inputs[i]);
        computed_mixed[i] = ggml_sin_lut_mixed_fp16_f32_REEX(inputs[i]);
    }
    
    // Calculate metrics
    precision_metrics metrics_fp16 = calculate_metrics(reference, computed_fp16);
    precision_metrics metrics_bf16 = calculate_metrics(reference, computed_bf16);
    precision_metrics metrics_mixed = calculate_metrics(reference, computed_mixed);
    
    // Print results (full-chain = segment/compare in fp16/bf16 space; mixed = FP32 MAC, output FP16 round — inference path)
    printf("\nFull-chain FP16 LUT Results:\n");
    printf("  Max Absolute Error: %.6e\n", metrics_fp16.max_abs_error);
    printf("  Mean Absolute Error: %.6e\n", metrics_fp16.mean_abs_error);
    printf("  Max Relative Error: %.6e\n", metrics_fp16.max_rel_error);
    printf("  Mean Relative Error: %.6e\n", metrics_fp16.mean_rel_error);
    printf("  Cosine Similarity: %.9f\n", metrics_fp16.cosine_similarity);
    
    printf("\nFull-chain BF16 LUT Results:\n");
    printf("  Max Absolute Error: %.6e\n", metrics_bf16.max_abs_error);
    printf("  Mean Absolute Error: %.6e\n", metrics_bf16.mean_abs_error);
    printf("  Max Relative Error: %.6e\n", metrics_bf16.max_rel_error);
    printf("  Mean Relative Error: %.6e\n", metrics_bf16.mean_rel_error);
    printf("  Cosine Similarity: %.9f\n", metrics_bf16.cosine_similarity);

    printf("\nMixed-FP16 LUT Results (inference path):\n");
    printf("  Max Absolute Error: %.6e\n", metrics_mixed.max_abs_error);
    printf("  Mean Absolute Error: %.6e\n", metrics_mixed.mean_abs_error);
    printf("  Max Relative Error: %.6e\n", metrics_mixed.max_rel_error);
    printf("  Mean Relative Error: %.6e\n", metrics_mixed.mean_rel_error);
    printf("  Cosine Similarity: %.9f\n", metrics_mixed.cosine_similarity);
    
    // Print worst cases
    printf("\nWorst Cases (Full-chain FP16):\n");
    for (size_t i = 0; i < reference.size(); ++i) {
        float error = std::abs(reference[i] - computed_fp16[i]);
        if (error > metrics_fp16.max_abs_error * 0.9) {
            printf("  x=%.6f, ref=%.6f, computed=%.6f, error=%.6e\n",
                   inputs[i], reference[i], computed_fp16[i], error);
        }
    }
    printf("\nWorst Cases (Mixed-FP16):\n");
    for (size_t i = 0; i < reference.size(); ++i) {
        float error = std::abs(reference[i] - computed_mixed[i]);
        if (error > metrics_mixed.max_abs_error * 0.9) {
            printf("  x=%.6f, ref=%.6f, computed=%.6f, error=%.6e\n",
                   inputs[i], reference[i], computed_mixed[i], error);
        }
    }
#else
    printf("\nERROR: GGML_USE_REEX is not defined!\n");
    printf("Please compile with -DGGML_USE_REEX=ON\n");
#endif
}

// Test cos function
void test_cos(const test_config & config) {
    printf("\n=== Testing COS: %s ===\n", config.name);
    printf("Input range: [%.6f, %.6f], Samples: %d\n", 
           config.input_min, config.input_max, config.n_samples);
    
    std::vector<float> inputs(config.n_samples);
    std::vector<float> reference(config.n_samples);
    std::vector<float> computed_fp16(config.n_samples);
    std::vector<float> computed_bf16(config.n_samples);
    
    // Generate test inputs
    for (int i = 0; i < config.n_samples; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(config.n_samples - 1);
        inputs[i] = config.input_min + t * (config.input_max - config.input_min);
    }
    
    // Compute reference (standard library)
    for (int i = 0; i < config.n_samples; ++i) {
        reference[i] = cosf(inputs[i]);
    }
    
    // Compute with REEX LUT
#ifdef GGML_USE_REEX
    std::vector<float> computed_mixed(config.n_samples);
    for (int i = 0; i < config.n_samples; ++i) {
        computed_fp16[i] = ggml_cos_lut_fp16_f32_REEX(inputs[i]);
        computed_bf16[i] = ggml_cos_lut_bf16_f32_REEX(inputs[i]);
        computed_mixed[i] = ggml_cos_lut_mixed_fp16_f32_REEX(inputs[i]);
    }
    
    // Calculate metrics
    precision_metrics metrics_fp16 = calculate_metrics(reference, computed_fp16);
    precision_metrics metrics_bf16 = calculate_metrics(reference, computed_bf16);
    precision_metrics metrics_mixed = calculate_metrics(reference, computed_mixed);
    
    printf("\nFull-chain FP16 LUT Results:\n");
    printf("  Max Absolute Error: %.6e\n", metrics_fp16.max_abs_error);
    printf("  Mean Absolute Error: %.6e\n", metrics_fp16.mean_abs_error);
    printf("  Max Relative Error: %.6e\n", metrics_fp16.max_rel_error);
    printf("  Mean Relative Error: %.6e\n", metrics_fp16.mean_rel_error);
    printf("  Cosine Similarity: %.9f\n", metrics_fp16.cosine_similarity);
    
    printf("\nFull-chain BF16 LUT Results:\n");
    printf("  Max Absolute Error: %.6e\n", metrics_bf16.max_abs_error);
    printf("  Mean Absolute Error: %.6e\n", metrics_bf16.mean_abs_error);
    printf("  Max Relative Error: %.6e\n", metrics_bf16.max_rel_error);
    printf("  Mean Relative Error: %.6e\n", metrics_bf16.mean_rel_error);
    printf("  Cosine Similarity: %.9f\n", metrics_bf16.cosine_similarity);

    printf("\nMixed-FP16 LUT Results (inference path):\n");
    printf("  Max Absolute Error: %.6e\n", metrics_mixed.max_abs_error);
    printf("  Mean Absolute Error: %.6e\n", metrics_mixed.mean_abs_error);
    printf("  Max Relative Error: %.6e\n", metrics_mixed.max_rel_error);
    printf("  Mean Relative Error: %.6e\n", metrics_mixed.mean_rel_error);
    printf("  Cosine Similarity: %.9f\n", metrics_mixed.cosine_similarity);
    
    printf("\nWorst Cases (Full-chain FP16):\n");
    for (size_t i = 0; i < reference.size(); ++i) {
        float error = std::abs(reference[i] - computed_fp16[i]);
        if (error > metrics_fp16.max_abs_error * 0.9) {
            printf("  x=%.6f, ref=%.6f, computed=%.6f, error=%.6e\n",
                   inputs[i], reference[i], computed_fp16[i], error);
        }
    }
    printf("\nWorst Cases (Mixed-FP16):\n");
    for (size_t i = 0; i < reference.size(); ++i) {
        float error = std::abs(reference[i] - computed_mixed[i]);
        if (error > metrics_mixed.max_abs_error * 0.9) {
            printf("  x=%.6f, ref=%.6f, computed=%.6f, error=%.6e\n",
                   inputs[i], reference[i], computed_mixed[i], error);
        }
    }
#else
    printf("\nERROR: GGML_USE_REEX is not defined!\n");
    printf("Please compile with -DGGML_USE_REEX=ON\n");
#endif
}

#ifdef GGML_USE_REEX
// Generic unary LUT test: full-chain fp16/bf16 (lut_bf_fp_new style) + Mixed-FP16 (inference path)
using unary_f32 = float (*)(float);
static void test_unary_lut(
    const char * op_name,
    const test_config & config,
    unary_f32 ref_func,
    unary_f32 lut_full_fp16,
    unary_f32 lut_full_bf16,
    unary_f32 lut_mixed_fp16)
{
    printf("\n=== Testing %s: %s ===\n", op_name, config.name);
    printf("Input range: [%.6f, %.6f], Samples: %d\n",
           config.input_min, config.input_max, config.n_samples);

    std::vector<float> inputs(config.n_samples);
    std::vector<float> reference(config.n_samples);
    std::vector<float> computed_full_fp16(config.n_samples);
    std::vector<float> computed_full_bf16(config.n_samples);
    std::vector<float> computed_mixed(config.n_samples);

    for (int i = 0; i < config.n_samples; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(config.n_samples - 1);
        inputs[i] = config.input_min + t * (config.input_max - config.input_min);
    }
    for (int i = 0; i < config.n_samples; ++i) {
        reference[i] = ref_func(inputs[i]);
        computed_full_fp16[i] = lut_full_fp16(inputs[i]);
        computed_full_bf16[i] = lut_full_bf16(inputs[i]);
        computed_mixed[i] = lut_mixed_fp16(inputs[i]);
    }

    precision_metrics m_full_fp16 = calculate_metrics(reference, computed_full_fp16);
    precision_metrics m_full_bf16 = calculate_metrics(reference, computed_full_bf16);
    precision_metrics m_mixed = calculate_metrics(reference, computed_mixed);

    printf("  full_fp16 MAE=%g max_err=%g cos_sim=%g  (full-chain FP16)\n",
           m_full_fp16.mean_abs_error, m_full_fp16.max_abs_error, m_full_fp16.cosine_similarity);
    printf("  full_bf16 MAE=%g max_err=%g cos_sim=%g  (full-chain BF16)\n",
           m_full_bf16.mean_abs_error, m_full_bf16.max_abs_error, m_full_bf16.cosine_similarity);
    printf("  mixed_fp16 MAE=%g max_err=%g cos_sim=%g  (Mixed-FP16, inference)\n",
           m_mixed.mean_abs_error, m_mixed.max_abs_error, m_mixed.cosine_similarity);
}
#endif

#ifdef GGML_USE_REEX
// Load reference dump (1000 floats) and compare with ggml LUT; report MAE, max_err, cos_sim vs ref.
// Returns true if max_diff <= tolerance.
static bool compare_with_ref(
    const char * ref_dir, const char * op_name, float x_min, float x_max,
    float (*lut_fp16)(float), float (*lut_bf16)(float),
    float tol)
{
    std::string path_fp16 = std::string(ref_dir) + "/" + op_name + "_fp16.bin";
    std::string path_bf16 = std::string(ref_dir) + "/" + op_name + "_bf16.bin";
    std::vector<float> x(NUM_TEST_POINTS_SAME_AS_PYTHON);
    for (int i = 0; i < NUM_TEST_POINTS_SAME_AS_PYTHON; ++i)
        x[i] = x_min + (x_max - x_min) * (float)i / (NUM_TEST_POINTS_SAME_AS_PYTHON - 1);

    bool ok = true;
    for (const char * prec : {"fp16", "bf16"}) {
        std::string path = (strcmp(prec, "fp16") == 0) ? path_fp16 : path_bf16;
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) { printf("  [Ref] %s %s: skip (no file %s)\n", op_name, prec, path.c_str()); continue; }
        std::vector<float> ref(NUM_TEST_POINTS_SAME_AS_PYTHON);
        size_t nr = fread(ref.data(), sizeof(float), NUM_TEST_POINTS_SAME_AS_PYTHON, f);
        fclose(f);
        if (nr != (size_t)NUM_TEST_POINTS_SAME_AS_PYTHON) {
            printf("  [Ref] %s %s: skip (read %zu)\n", op_name, prec, nr);
            continue;
        }
        std::vector<float> y(NUM_TEST_POINTS_SAME_AS_PYTHON);
        for (int i = 0; i < NUM_TEST_POINTS_SAME_AS_PYTHON; ++i)
            y[i] = (strcmp(prec, "fp16") == 0) ? lut_fp16(x[i]) : lut_bf16(x[i]);
        precision_metrics m = calculate_metrics(ref, y);
        bool pass = m.max_abs_error <= (double)tol;
        if (!pass) ok = false;
        printf("  [Ref] %s %s: MAE=%g max_err=%g cos_sim=%g -> %s\n",
               op_name, prec, m.mean_abs_error, m.max_abs_error, m.cosine_similarity,
               pass ? "PASS" : "FAIL");
    }
    return ok;
}

// libm / analytical references for GPU vs reference metrics.
static float ref_silu_libm(float x) { return x / (1.f + expf(-x)); }
static float ref_sigmoid_libm(float x) { return 1.f / (1.f + expf(-x)); }
static float ref_sqr_libm(float x) { return x * x; }

using ggml_build_unary_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *);
using ref_float_fn        = float (*)(float);

// Run one unary op on GPU and report MAE / max_abs_err / cosine similarity vs libm reference.
static bool gpu_lut_metrics_vs_libm(
        ggml_backend_t backend_gpu, int n, const char * op_name,
        float x_min, float x_max, ggml_build_unary_fn build, ref_float_fn ref_fn) {
    ggml_init_params params = { 512 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  [GPU Mixed-FP16 vs libm] %s: failed to create ggml context\n", op_name);
        return false;
    }
    ggml_tensor * in  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * out = build(ctx, in);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_gpu);
    if (!buf) {
        printf("  [GPU Mixed-FP16 vs libm] %s: failed to allocate GPU buffer\n", op_name);
        ggml_free(ctx);
        return false;
    }
    std::vector<float> inputs(n), reference(n), gpu(n);
    for (int i = 0; i < n; ++i) {
        float t = (n > 1) ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.5f;
        inputs[i]    = x_min + t * (x_max - x_min);
        reference[i] = ref_fn(inputs[i]);
    }
    ggml_backend_tensor_set(in, inputs.data(), 0, n * sizeof(float));
    ggml_status st = ggml_backend_graph_compute(backend_gpu, gf);
    if (st != GGML_STATUS_SUCCESS) {
        printf("  [GPU Mixed-FP16 vs libm] %s: ggml_backend_graph_compute failed: %s\n", op_name, ggml_status_to_string(st));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(out, gpu.data(), 0, n * sizeof(float));
    precision_metrics m = calculate_metrics(reference, gpu);
    printf("  [GPU Mixed-FP16 vs libm] %s: MAE=%g max_err=%g cos_sim=%g (n=%d)\n",
           op_name, m.mean_abs_error, m.max_abs_error, m.cosine_similarity, n);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}
#endif

int main(int argc, char ** argv) {
    printf("REEX LUT Precision Test (Sin/Cos + Direct + Normalized)\n");
    printf("  Full-chain FP16/BF16 vs Mixed-FP16 (inference) are reported separately.\n");
    printf("=====================================\n");

    const char * ref_dir = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--ref-dir") == 0) { ref_dir = argv[i + 1]; break; }
    }
    if (ref_dir) printf("Ref comparison dir: %s\n", ref_dir);

#ifdef GGML_USE_REEX
    // Initialize GGML backend
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (backend == NULL) {
        fprintf(stderr, "Failed to initialize CPU backend\n");
        return 1;
    }
    
    // Initialize LUT (shared; also called by CPU backend init)
    ggml_init_trigonometric_lut_REEX();
    
    // Test configurations
    std::vector<test_config> test_configs = {
        {-3.14159f, 3.14159f, 1000, "Single Period [-π, π]"},
        {-31.416f, 31.416f, 2000, "Multi Period [-10π, 10π]"},
        {-3.14159f, 3.14159f, 100, "Boundary Points [-π, π]"},
        {0.0f, 6.28318f, 500, "Positive Range [0, 2π]"},
    };
    
    // Run tests
    for (const auto & config : test_configs) {
        test_sin(config);
        test_cos(config);
    }

    // ---- 1000 pts: full-chain (lut_bf_fp_new) + Mixed-FP16 (ggml inference path) ----
    printf("\n=== Unary 1000-pt vs libm: full_fp16 / full_bf16 / mixed_fp16 ===\n");

    // Direct: exponential [-20,0], sigmoid [-6,6]
    {
        test_config cfg = {-20.0f, 0.0f, NUM_TEST_POINTS_SAME_AS_PYTHON, "exponential"};
        printf("\n[Direct] exponential x in [%g, %g]\n", cfg.input_min, cfg.input_max);
        test_unary_lut("EXP", cfg, [](float x) { return expf(x); },
            ggml_exp_lut_fp16_f32_REEX, ggml_exp_lut_bf16_f32_REEX, ggml_exp_lut_mixed_fp16_f32_REEX);
    }
    {
        test_config cfg = {0.0f, 11.0f, NUM_TEST_POINTS_SAME_AS_PYTHON, "exponential_positive"};
        printf("\n[Direct] exponential (positive range) x in [%g, %g]\n", cfg.input_min, cfg.input_max);
        test_unary_lut("EXP_POS", cfg, [](float x) { return expf(x); },
            ggml_exp_lut_fp16_f32_REEX, ggml_exp_lut_bf16_f32_REEX, ggml_exp_lut_mixed_fp16_f32_REEX);
    }
    {
        test_config cfg = {-20.0f, 11.0f, NUM_TEST_POINTS_SAME_AS_PYTHON, "exponential_full"};
        printf("\n[Direct] exponential (full range) x in [%g, %g]\n", cfg.input_min, cfg.input_max);
        test_unary_lut("EXP_FULL", cfg, [](float x) { return expf(x); },
            ggml_exp_lut_fp16_f32_REEX, ggml_exp_lut_bf16_f32_REEX, ggml_exp_lut_mixed_fp16_f32_REEX);
    }
    {
        test_config cfg = {-6.0f, 6.0f, NUM_TEST_POINTS_SAME_AS_PYTHON, "sigmoid"};
        printf("\n[Direct] sigmoid x in [%g, %g]\n", cfg.input_min, cfg.input_max);
        test_unary_lut("SIGMOID", cfg, [](float x) { return 1.f / (1.f + expf(-x)); },
            ggml_sigmoid_lut_fp16_f32_REEX, ggml_sigmoid_lut_bf16_f32_REEX, ggml_sigmoid_lut_mixed_fp16_f32_REEX);
    }
    {
        test_config cfg = {-6.0f, 6.0f, NUM_TEST_POINTS_SAME_AS_PYTHON, "silu"};
        printf("\n[Direct] silu x in [%g, %g]\n", cfg.input_min, cfg.input_max);
        test_unary_lut("SILU", cfg, ref_silu_libm,
            ggml_silu_lut_fp16_f32_REEX, ggml_silu_lut_bf16_f32_REEX, ggml_silu_lut_mixed_fp16_f32_REEX);
    }

    // Periodic: sin, cos on [-pi, pi]
    {
        test_config cfg = {-PI_F, PI_F, NUM_TEST_POINTS_SAME_AS_PYTHON, "sin"};
        printf("\n[Periodic] sin (sin LUT, phase=0)\n");
        test_unary_lut("SIN", cfg, [](float x) { return sinf(x); },
            ggml_sin_lut_fp16_f32_REEX, ggml_sin_lut_bf16_f32_REEX, ggml_sin_lut_mixed_fp16_f32_REEX);
    }
    {
        test_config cfg = {-PI_F, PI_F, NUM_TEST_POINTS_SAME_AS_PYTHON, "cos"};
        printf("\n[Periodic] cos (sin LUT, phase=pi/2)\n");
        test_unary_lut("COS", cfg, [](float x) { return cosf(x); },
            ggml_cos_lut_fp16_f32_REEX, ggml_cos_lut_bf16_f32_REEX, ggml_cos_lut_mixed_fp16_f32_REEX);
    }

    // Normalized: same ranges as test_same_as_python
    {
        test_config cfg = {0.01f, 8.0f, NUM_TEST_POINTS_SAME_AS_PYTHON, "reciprocal"};
        printf("\n[Normalized] reciprocal x in [0.01, 8]\n");
        test_unary_lut("RECIPROCAL", cfg, [](float x) { return 1.f / x; },
            ggml_reciprocal_lut_fp16_f32_REEX, ggml_reciprocal_lut_bf16_f32_REEX, ggml_reciprocal_lut_mixed_fp16_f32_REEX);
    }
    {
        test_config cfg = {0.01f, 8.0f, NUM_TEST_POINTS_SAME_AS_PYTHON, "rsqrt"};
        printf("\n[Normalized] rsqrt x in [0.01, 8]\n");
        test_unary_lut("RSQRT", cfg, [](float x) { return 1.f / sqrtf(x); },
            ggml_rsqrt_lut_fp16_f32_REEX, ggml_rsqrt_lut_bf16_f32_REEX, ggml_rsqrt_lut_mixed_fp16_f32_REEX);
    }
    {
        test_config cfg = {0.0f, 8.0f, NUM_TEST_POINTS_SAME_AS_PYTHON, "sqrt"};
        printf("\n[Normalized] sqrt x in [0, 8]\n");
        test_unary_lut("SQRT", cfg, [](float x) { return sqrtf(x); },
            ggml_sqrt_lut_fp16_f32_REEX, ggml_sqrt_lut_bf16_f32_REEX, ggml_sqrt_lut_mixed_fp16_f32_REEX);
    }
    {
        test_config cfg = {0.1f, 8.0f, NUM_TEST_POINTS_SAME_AS_PYTHON, "log"};
        printf("\n[Normalized] log x in [0.1, 8]\n");
        test_unary_lut("LOG", cfg, [](float x) { return logf(x); },
            ggml_log_lut_fp16_f32_REEX, ggml_log_lut_bf16_f32_REEX, ggml_log_lut_mixed_fp16_f32_REEX);
    }
    {
        test_config cfg = {-4.0f, 4.0f, NUM_TEST_POINTS_SAME_AS_PYTHON, "power_2"};
        printf("\n[Normalized] power_2 x in [-4, 4] (use_abs)\n");
        /* power_2 in lut_bf_fp_new = x² (square), not 2^x; ref = x*x */
        test_unary_lut("SQR(power_2)", cfg, [](float x) { return x * x; },
            ggml_sqr_fp16_f32_REEX, ggml_sqr_bf16_f32_REEX, ggml_sqr_mixed_fp16_f32_REEX);
    }

    // Optional: compare with reference dumps from run_same_experiment.py --dump-ref
    const float ref_tol = 2e-4f;
    if (ref_dir) {
        printf("\n=== Compare with ref dir (tol=%.0e, full-chain FP16/BF16 only) ===\n", (double)ref_tol);
        bool all_ok = true;
        all_ok &= compare_with_ref(ref_dir, "exponential", -20.f, 0.f, ggml_exp_lut_fp16_f32_REEX, ggml_exp_lut_bf16_f32_REEX, ref_tol);
        all_ok &= compare_with_ref(ref_dir, "sigmoid", -6.f, 6.f, ggml_sigmoid_lut_fp16_f32_REEX, ggml_sigmoid_lut_bf16_f32_REEX, ref_tol);
        all_ok &= compare_with_ref(ref_dir, "sin", -PI_F, PI_F, ggml_sin_lut_fp16_f32_REEX, ggml_sin_lut_bf16_f32_REEX, ref_tol);
        all_ok &= compare_with_ref(ref_dir, "cos", -PI_F, PI_F, ggml_cos_lut_fp16_f32_REEX, ggml_cos_lut_bf16_f32_REEX, ref_tol);
        all_ok &= compare_with_ref(ref_dir, "reciprocal", 0.01f, 8.f, ggml_reciprocal_lut_fp16_f32_REEX, ggml_reciprocal_lut_bf16_f32_REEX, ref_tol);
        all_ok &= compare_with_ref(ref_dir, "rsqrt", 0.01f, 8.f, ggml_rsqrt_lut_fp16_f32_REEX, ggml_rsqrt_lut_bf16_f32_REEX, ref_tol);
        all_ok &= compare_with_ref(ref_dir, "sqrt", 0.f, 8.f, ggml_sqrt_lut_fp16_f32_REEX, ggml_sqrt_lut_bf16_f32_REEX, ref_tol);
        all_ok &= compare_with_ref(ref_dir, "log", 0.1f, 8.f, ggml_log_lut_fp16_f32_REEX, ggml_log_lut_bf16_f32_REEX, ref_tol);
        all_ok &= compare_with_ref(ref_dir, "power_2", -4.f, 4.f, ggml_sqr_fp16_f32_REEX, ggml_sqr_bf16_f32_REEX, ref_tol);
        if (!all_ok) {
            fprintf(stderr, "Ref comparison FAILED (max_err > %.0e)\n", (double)ref_tol);
            ggml_backend_free(backend);
            return 1;
        }
        printf("Ref comparison PASSED.\n");
    }

    // Test specific critical points
    printf("\n=== Testing Critical Points ===\n");
    std::vector<float> critical_points = {
        -3.14159f, -1.5708f, 0.0f, 1.5708f, 3.14159f,
        -6.28318f, 6.28318f, -9.42478f, 9.42478f
    };
    
    printf("\nSIN Critical Points (full_fp16 | mixed_fp16 | bf16 vs ref):\n");
    for (float x : critical_points) {
        float ref = sinf(x);
#ifdef GGML_USE_REEX
        float full_fp16 = ggml_sin_lut_fp16_f32_REEX(x);
        float mixed = ggml_sin_lut_mixed_fp16_f32_REEX(x);
        float bf16 = ggml_sin_lut_bf16_f32_REEX(x);
        printf("  x=%.6f: ref=%.6f | full_fp16=%.6f (e=%.2e) mixed=%.6f (e=%.2e) bf16=%.6f (e=%.2e)\n",
               x, ref, full_fp16, std::abs(ref - full_fp16), mixed, std::abs(ref - mixed),
               bf16, std::abs(ref - bf16));
#endif
    }
    
    printf("\nCOS Critical Points (full_fp16 | mixed_fp16 | bf16 vs ref):\n");
    for (float x : critical_points) {
        float ref = cosf(x);
#ifdef GGML_USE_REEX
        float full_fp16 = ggml_cos_lut_fp16_f32_REEX(x);
        float mixed = ggml_cos_lut_mixed_fp16_f32_REEX(x);
        float bf16 = ggml_cos_lut_bf16_f32_REEX(x);
        printf("  x=%.6f: ref=%.6f | full_fp16=%.6f (e=%.2e) mixed=%.6f (e=%.2e) bf16=%.6f (e=%.2e)\n",
               x, ref, full_fp16, std::abs(ref - full_fp16), mixed, std::abs(ref - mixed),
               bf16, std::abs(ref - bf16));
#endif
    }

#ifdef GGML_USE_REEX
    // CUDA/GPU backend LUT test: run unary ops on GPU and compare with CPU REEX Mixed-FP16
    {
        const int n_cuda = 512;
        ggml_backend_t backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!backend_gpu) {
            printf("\n=== CUDA/GPU backend not available, skipping GPU LUT test ===\n");
        } else {
            const int n_gpu_ref = NUM_TEST_POINTS_SAME_AS_PYTHON;
            printf("\n=== GPU Mixed-FP16 vs libm (CUDA ggml unary, GGML_LUT_NUM_SEGMENTS_REEX=%d, n=%d) ===\n",
                   GGML_LUT_NUM_SEGMENTS_REEX, n_gpu_ref);
            (void) gpu_lut_metrics_vs_libm(backend_gpu, n_gpu_ref, "sin", -PI_F, PI_F,
                [](ggml_context * c, ggml_tensor * x) { return ggml_sin(c, x); }, sinf);
            (void) gpu_lut_metrics_vs_libm(backend_gpu, n_gpu_ref, "cos", -PI_F, PI_F,
                [](ggml_context * c, ggml_tensor * x) { return ggml_cos(c, x); }, cosf);
            (void) gpu_lut_metrics_vs_libm(backend_gpu, n_gpu_ref, "exp", -20.0f, 0.0f,
                [](ggml_context * c, ggml_tensor * x) { return ggml_exp(c, x); }, expf);
            (void) gpu_lut_metrics_vs_libm(backend_gpu, n_gpu_ref, "sigmoid", -6.0f, 6.0f,
                [](ggml_context * c, ggml_tensor * x) { return ggml_sigmoid(c, x); }, ref_sigmoid_libm);
            (void) gpu_lut_metrics_vs_libm(backend_gpu, n_gpu_ref, "sqrt", 0.0f, 8.0f,
                [](ggml_context * c, ggml_tensor * x) { return ggml_sqrt(c, x); }, sqrtf);
            (void) gpu_lut_metrics_vs_libm(backend_gpu, n_gpu_ref, "log", 0.1f, 8.0f,
                [](ggml_context * c, ggml_tensor * x) { return ggml_log(c, x); }, logf);
            (void) gpu_lut_metrics_vs_libm(backend_gpu, n_gpu_ref, "sqr", -4.0f, 4.0f,
                [](ggml_context * c, ggml_tensor * x) { return ggml_sqr(c, x); }, ref_sqr_libm);
            (void) gpu_lut_metrics_vs_libm(backend_gpu, n_gpu_ref, "silu", -6.0f, 6.0f,
                [](ggml_context * c, ggml_tensor * x) { return ggml_silu(c, x); }, ref_silu_libm);

            auto run_gpu_op_test = [backend_gpu, n_cuda](const char * op_name, float x_min, float x_max,
                    ggml_tensor * (*build)(ggml_context *, ggml_tensor *),
                    float (*cpu_ref)(float)) -> bool {
                ggml_init_params params = { 256 * 1024, nullptr, true };
                ggml_context * ctx = ggml_init(params);
                if (!ctx) { printf("  %s: failed to create context\n", op_name); return false; }
                ggml_tensor * in  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_cuda);
                ggml_tensor * out = build(ctx, in);
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, out);
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_gpu);
                if (!buf) { printf("  %s: failed to allocate GPU buffer\n", op_name); ggml_free(ctx); return false; }
                std::vector<float> inputs(n_cuda);
                for (int i = 0; i < n_cuda; ++i) {
                    float t = (n_cuda > 1) ? static_cast<float>(i) / static_cast<float>(n_cuda - 1) : 0.5f;
                    inputs[i] = x_min + t * (x_max - x_min);
                }
                ggml_backend_tensor_set(in, inputs.data(), 0, n_cuda * sizeof(float));
                ggml_status st = ggml_backend_graph_compute(backend_gpu, gf);
                if (st != GGML_STATUS_SUCCESS) {
                    printf("  %s: ggml_backend_graph_compute failed\n", op_name);
                    ggml_backend_buffer_free(buf);
                    ggml_free(ctx);
                    return false;
                }
                std::vector<float> gpu_out(n_cuda);
                ggml_backend_tensor_get(out, gpu_out.data(), 0, n_cuda * sizeof(float));
                float max_err = 0.0f;
                for (int i = 0; i < n_cuda; ++i) {
                    float cpu = cpu_ref(inputs[i]);
                    float e = std::abs(gpu_out[i] - cpu);
                    if (e > max_err) max_err = e;
                }
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
                const float tol = 2e-2f;
                bool ok = (max_err <= tol);
                printf("  %s: max |err|=%.2e (tolerance %.2e) %s\n", op_name, (double)max_err, (double)tol, ok ? "PASS" : "FAIL");
                return ok;
            };

            bool all_ok = true;

            printf("\n=== Testing Sin/Cos on GPU (REEX LUT vs CPU REEX) ===\n");
            {
                const float x_min = -3.14159f;
                const float x_max =  3.14159f;
                ggml_init_params params = { 256 * 1024, nullptr, true };
                ggml_context * ctx = ggml_init(params);
                if (!ctx) {
                    printf("  Failed to create ggml context\n");
                    all_ok = false;
                } else {
                    ggml_tensor * in   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_cuda);
                    ggml_tensor * sout = ggml_sin(ctx, in);
                    ggml_tensor * cout = ggml_cos(ctx, in);
                    ggml_cgraph * gf = ggml_new_graph(ctx);
                    ggml_build_forward_expand(gf, sout);
                    ggml_build_forward_expand(gf, cout);
                    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_gpu);
                    if (!buf) {
                        printf("  Failed to allocate GPU buffer\n");
                        ggml_free(ctx);
                        all_ok = false;
                    } else {
                        std::vector<float> inputs(n_cuda);
                        for (int i = 0; i < n_cuda; ++i) {
                            float t = static_cast<float>(i) / static_cast<float>(n_cuda - 1);
                            inputs[i] = x_min + t * (x_max - x_min);
                        }
                        ggml_backend_tensor_set(in, inputs.data(), 0, n_cuda * sizeof(float));
                        ggml_status st = ggml_backend_graph_compute(backend_gpu, gf);
                        if (st != GGML_STATUS_SUCCESS) {
                            printf("  ggml_backend_graph_compute failed: %s\n", ggml_status_to_string(st));
                            all_ok = false;
                        } else {
                            std::vector<float> gpu_sin(n_cuda), gpu_cos(n_cuda);
                            ggml_backend_tensor_get(sout, gpu_sin.data(), 0, n_cuda * sizeof(float));
                            ggml_backend_tensor_get(cout, gpu_cos.data(), 0, n_cuda * sizeof(float));
                            float max_err_sin = 0.0f, max_err_cos = 0.0f;
                            for (int i = 0; i < n_cuda; ++i) {
                                float cpu_sin = ggml_sin_lut_mixed_fp16_f32_REEX(inputs[i]);
                                float cpu_cos = ggml_cos_lut_mixed_fp16_f32_REEX(inputs[i]);
                                float es = std::abs(gpu_sin[i] - cpu_sin);
                                float ec = std::abs(gpu_cos[i] - cpu_cos);
                                if (es > max_err_sin) max_err_sin = es;
                                if (ec > max_err_cos) max_err_cos = ec;
                            }
                            const float tol = 2e-2f;
                            bool ok = (max_err_sin <= tol && max_err_cos <= tol);
                            printf("  GPU vs CPU REEX: max |err| sin=%.2e, cos=%.2e (tolerance %.2e) %s\n",
                                   (double)max_err_sin, (double)max_err_cos, (double)tol, ok ? "PASS" : "FAIL");
                            if (!ok) all_ok = false;
                        }
                        ggml_backend_buffer_free(buf);
                    }
                    ggml_free(ctx);
                }
            }

            printf("\n=== Testing Sqrt/Log/Sigmoid/Exp/Sqr/Silu on GPU (REEX LUT vs CPU REEX) ===\n");
            if (!run_gpu_op_test("sqrt",   0.0f,  8.0f,  [](ggml_context * c, ggml_tensor * x) { return ggml_sqrt(c, x); },   ggml_sqrt_lut_mixed_fp16_f32_REEX))   all_ok = false;
            if (!run_gpu_op_test("log",    0.1f,  8.0f,  [](ggml_context * c, ggml_tensor * x) { return ggml_log(c, x); },    ggml_log_lut_mixed_fp16_f32_REEX))    all_ok = false;
            if (!run_gpu_op_test("sigmoid", -6.0f, 6.0f, [](ggml_context * c, ggml_tensor * x) { return ggml_sigmoid(c, x); }, ggml_sigmoid_lut_mixed_fp16_f32_REEX)) all_ok = false;
            if (!run_gpu_op_test("exp",    -20.0f, 0.0f,  [](ggml_context * c, ggml_tensor * x) { return ggml_exp(c, x); },    ggml_exp_lut_mixed_fp16_f32_REEX))    all_ok = false;
            if (!run_gpu_op_test("sqr",    -4.0f, 4.0f,  [](ggml_context * c, ggml_tensor * x) { return ggml_sqr(c, x); },    ggml_sqr_mixed_fp16_f32_REEX))        all_ok = false;
            if (!run_gpu_op_test("silu",   -6.0f, 6.0f,  [](ggml_context * c, ggml_tensor * x) { return ggml_silu(c, x); },  ggml_silu_lut_mixed_fp16_f32_REEX))   all_ok = false;

            if (!all_ok) {
                printf("  CUDA LUT test FAILED (one or more ops exceeded tolerance).\n");
                ggml_backend_free(backend_gpu);
                ggml_backend_free(backend);
                return 1;
            }
            printf("  All CUDA LUT tests PASSED.\n");
            ggml_backend_free(backend_gpu);
        }
    }
#endif

    ggml_backend_free(backend);
    printf("\n=== Test Complete ===\n");
#else
    printf("\nERROR: GGML_USE_REEX is not defined!\n");
    printf("Please compile with -DGGML_USE_REEX=ON\n");
    return 1;
#endif
    
    return 0;
}
