/*
 * test-reex-cuda-verify.cpp  —  CUDA W4×INT8(Q8_1) 精度测试
 *
 * CUDA 对 Q4_K 权重做 MUL_MAT 时，默认会把 F32 激活量化为 Q8_1，并走
 * Q4_K×Q8_1 的 MMQ/MMVQ 路径。该测试检查：
 *
 *   1. 最终验收标准：相对 float×float 金标准，cosine > 0.9999 且 mae < 1 LSB(W4+Q8)
 *   2. 增量误差参考：相对 Q4_K×f32（仅权重量化），mae < 1 LSB(Q8)
 *
 * 编译条件：GGML_CUDA=ON，且当前构建不能启用 REEX Q16 路径。
 */

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-cpu/quants.h"
#include "ggml-quants.h"
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const float COSINE_MIN   = 0.9999f;
static const float LSB_REL_W4   = 1.0f / 16.0f;
static const float LSB_REL_Q8   = 1.0f / 127.0f;

extern "C" void ggml_reex_cuda_reset_q8_stats(void);
extern "C" int  ggml_reex_cuda_get_q8_mul_mat_hits(void);
extern "C" int  ggml_reex_cuda_get_q8_mul_mat_id_hits(void);

static void fill_deterministic(float * dst, int n, float amplitude) {
    for (int i = 0; i < n; i++) {
        dst[i] = amplitude * sinf(i * 0.1f + (i % 7) * 0.3f);
    }
}

static float cosine_sim(const float * a, const float * b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0f;
    return (float)(dot / sqrt(na * nb));
}

static float mae(const float * a, const float * b, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) {
        s += fabs((double)a[i] - (double)b[i]);
    }
    return (float)(s / n);
}

static float calc_energy(const float * a, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) {
        s += fabs((double)a[i]);
    }
    return (float)(s / n);
}

static void compute_cpu_float_reference(
    const float * weight_f32, int64_t K, int64_t N,
    const float * act_f32, int64_t M,
    float * out)
{
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            double sum = 0.0;
            for (int64_t k = 0; k < K; k++) {
                sum += (double)weight_f32[n * K + k] * (double)act_f32[m * K + k];
            }
            out[m * N + n] = (float)sum;
        }
    }
}

static void compute_cpu_q4k_float_reference(
    const void * weight_q4k, int64_t K, int64_t N,
    const float * act_f32, int64_t M,
    float * out)
{
    size_t q4k_row_size = ggml_row_size(GGML_TYPE_Q4_K, K);
    std::vector<float> deq(K);

    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            const block_q4_K * row = (const block_q4_K *)((const char *)weight_q4k + n * q4k_row_size);
            dequantize_row_q4_K(row, deq.data(), K);
            double sum = 0.0;
            for (int64_t k = 0; k < K; k++) {
                sum += (double)deq[k] * (double)act_f32[m * K + k];
            }
            out[m * N + n] = (float)sum;
        }
    }
}

static bool run_cuda_mul_mat(
    ggml_backend_t backend,
    const void * weight_q4k_data, int64_t K, int64_t N,
    const float * act_f32, int64_t M,
    float * out)
{
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * r = ggml_mul_mat(ctx, w, a);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "  [ERROR] Failed to allocate backend buffer\n");
        ggml_free(ctx);
        return false;
    }

    size_t w_nbytes = ggml_nbytes(w);
    ggml_backend_tensor_set(w, weight_q4k_data, 0, w_nbytes);
    ggml_backend_tensor_set(a, act_f32, 0, K * M * sizeof(float));

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "  [ERROR] Backend graph compute failed: %d\n", (int)status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(r, out, 0, N * M * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main(void) {
    printf("=== test-reex-cuda-q8: CUDA W4×INT8(Q8_1) precision test ===\n\n");

    ggml_cpu_init();

    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) {
        fprintf(stderr, "Failed to init CUDA backend (no GPU available?)\n");
        printf("\n[SKIP] CUDA not available, skipping Q8 test.\n");
        return 0;
    }

    printf("CUDA backend: %s\n\n", ggml_backend_name(cuda_backend));

    const int64_t K = 2048;
    const int64_t N = 512;
    const int64_t M = 4;

    std::vector<float> weight_f32(K * N);
    fill_deterministic(weight_f32.data(), K * N, 1.0f);

    size_t q4k_row_size = ggml_row_size(GGML_TYPE_Q4_K, K);
    std::vector<uint8_t> weight_q4k(q4k_row_size * N);
    for (int64_t row = 0; row < N; row++) {
        quantize_row_q4_K(weight_f32.data() + row * K,
                          weight_q4k.data() + row * q4k_row_size, K);
    }

    std::vector<float> act_f32(K * M);
    fill_deterministic(act_f32.data(), K * M, 0.8f);

    std::vector<float> ref_f32xf32(N * M, 0.0f);
    compute_cpu_float_reference(weight_f32.data(), K, N, act_f32.data(), M, ref_f32xf32.data());

    std::vector<float> ref_q4k_float(N * M, 0.0f);
    compute_cpu_q4k_float_reference(weight_q4k.data(), K, N, act_f32.data(), M, ref_q4k_float.data());

    std::vector<float> out_cuda(N * M, 0.0f);

    printf("Running MUL_MAT on CUDA backend (Q4_K × F32)...\n");
    ggml_reex_cuda_reset_q8_stats();
    bool ok_cuda = run_cuda_mul_mat(
        cuda_backend, weight_q4k.data(), K, N,
        act_f32.data(), M, out_cuda.data());
    if (!ok_cuda) {
        fprintf(stderr, "CUDA MUL_MAT failed\n");
        ggml_backend_free(cuda_backend);
        return 1;
    }

    int total_elements = (int)(N * M);
    float cos_vs_f32  = cosine_sim(ref_f32xf32.data(), out_cuda.data(), total_elements);
    float mae_vs_f32  = mae(ref_f32xf32.data(), out_cuda.data(), total_elements);
    float cos_vs_q4kf = cosine_sim(ref_q4k_float.data(), out_cuda.data(), total_elements);
    float mae_vs_q4kf = mae(ref_q4k_float.data(), out_cuda.data(), total_elements);

    float energy_f32  = calc_energy(ref_f32xf32.data(), total_elements);
    float energy_q4kf = calc_energy(ref_q4k_float.data(), total_elements);

    float lsb_gold = energy_f32  * (LSB_REL_W4 + LSB_REL_Q8);
    float lsb_q8   = energy_q4kf * LSB_REL_Q8;

    float mae_lsb_vs_f32  = (lsb_gold > 1e-30f) ? mae_vs_f32 / lsb_gold : 0.0f;
    float mae_lsb_vs_q4kf = (lsb_q8   > 1e-30f) ? mae_vs_q4kf / lsb_q8   : 0.0f;
    const int q8_mul_mat_hits = ggml_reex_cuda_get_q8_mul_mat_hits();
    const int q8_mul_mat_id_hits = ggml_reex_cuda_get_q8_mul_mat_id_hits();

    printf("\n--- Results (CUDA W4×INT8) ---\n");
    printf("  Matrix: W[%lld×%lld] × A[%lld×%lld] → Out[%lld×%lld]\n",
           (long long)N, (long long)K, (long long)K, (long long)M,
           (long long)N, (long long)M);
    printf("  vs f32×f32 (final requirement):\n");
    printf("    cosine:       %.6f  (threshold: %.4f)\n", cos_vs_f32, COSINE_MIN);
    printf("    mae/LSB(W4+Q8): %.4f  (threshold: < 1.0)\n", mae_lsb_vs_f32);
    printf("  vs Q4_K×f32 (activation-only delta):\n");
    printf("    cosine:       %.6f  (threshold: %.4f)\n", cos_vs_q4kf, COSINE_MIN);
    printf("    mae/LSB(Q8):  %.4f  (threshold: < 1.0)\n", mae_lsb_vs_q4kf);
    printf("  path trace:\n");
    printf("    q8_mul_mat_hits:    %d  (threshold: > 0)\n", q8_mul_mat_hits);
    printf("    q8_mul_mat_id_hits: %d  (info)\n", q8_mul_mat_id_hits);

    bool pass_cos = cos_vs_f32 >= COSINE_MIN;
    bool pass_mae = mae_lsb_vs_f32 < 1.0f;
    bool pass_hit = q8_mul_mat_hits > 0;
    bool pass = pass_cos && pass_mae && pass_hit;

    printf("\n  final cosine check: %s\n", pass_cos ? "PASS" : "FAIL");
    printf("  final mae/LSB check: %s\n", pass_mae ? "PASS" : "FAIL");
    printf("  path hit check: %s\n", pass_hit ? "PASS" : "FAIL");
    printf("\n=== %s ===\n", pass ? "ALL PASSED" : "FAILED");

    ggml_backend_free(cuda_backend);
    return pass ? 0 : 1;
}
