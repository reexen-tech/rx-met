/*
 * test-reex-cuda-q16.cpp  —  CUDA W4×INT16 (REEX Q16) 精度测试
 *
 * 将 F32 激活量化为 block_q16_K，与 Q4_K 权重做 GEMM，
 * 比较 CUDA REEX Q16 输出与 CPU float 参考（Q4_K 权重反量化后 × F32 激活）。
 *
 * 测试矩阵：
 *   1. 单向量 (M=1)
 *   2. 小 batch (M=4)
 *   3. 中等 batch (M=8)
 *
 * 评估指标：
 *   - 最终验收：vs f32×f32，cosine > 0.9999, MAE < 1 LSB(W4+Q16)
 *   - 增量参考：vs Q4_K×f32，MAE < 1 LSB(Q16)
 *
 * 编译条件：GGML_CUDA=ON && GGML_REEX_GEMM=ON
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

static const float COSINE_MIN    = 0.9999f;
static const float LSB_REL_W4    = 1.0f / 16.0f;
static const float LSB_REL_Q16   = 1.0f / 32767.0f;

// Trace helpers are defined in the CUDA backend. Forward declare them here so
// this C++ test does not need to include CUDA-only headers.
extern "C" void ggml_reex_cuda_reset_stats(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_hits(void);
extern "C" int  ggml_reex_cuda_get_q16_batch_fallbacks(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_id_hits(void);
extern "C" int  ggml_reex_cuda_get_q16_mul_mat_id_unsupported(void);

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

static float calc_mae(const float * a, const float * b, int n) {
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

static float silu_scalar(float x) {
    return x / (1.0f + expf(-x));
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

static void compute_cpu_float_reference_mul_mat_id(
    const float * weight_f32, int64_t K, int64_t M_rows, int64_t N_mats,
    const float * act_f32, int64_t n_act_channels, int64_t n_used, int64_t n_tokens,
    const int32_t * ids_data,
    float * out)
{
    GGML_UNUSED(N_mats);
    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        for (int64_t used = 0; used < n_used; ++used) {
            const int32_t expert = ids_data[tok * n_used + used];
            const int64_t act_channel = n_act_channels > 1 ? (used % n_act_channels) : 0;
            const float * act_row = act_f32 + (tok * n_act_channels + act_channel) * K;
            const float * weight_mat = weight_f32 + (int64_t)expert * M_rows * K;
            float * out_row = out + (tok * n_used + used) * M_rows;

            for (int64_t row = 0; row < M_rows; ++row) {
                const float * w = weight_mat + row * K;
                double sum = 0.0;
                for (int64_t k = 0; k < K; ++k) {
                    sum += (double)w[k] * (double)act_row[k];
                }
                out_row[row] = (float)sum;
            }
        }
    }
}

static void compute_cpu_q4k_float_reference_mul_mat_id(
    const void * weight_q4k, int64_t K, int64_t M_rows, int64_t N_mats,
    const float * act_f32, int64_t n_act_channels, int64_t n_used, int64_t n_tokens,
    const int32_t * ids_data,
    float * out)
{
    GGML_UNUSED(N_mats);
    const size_t q4k_row_size = ggml_row_size(GGML_TYPE_Q4_K, K);
    std::vector<float> deq(K);

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        for (int64_t used = 0; used < n_used; ++used) {
            const int32_t expert = ids_data[tok * n_used + used];
            const int64_t act_channel = n_act_channels > 1 ? (used % n_act_channels) : 0;
            const float * act_row = act_f32 + (tok * n_act_channels + act_channel) * K;
            float * out_row = out + (tok * n_used + used) * M_rows;
            const char * weight_mat = (const char *)weight_q4k + (int64_t)expert * M_rows * q4k_row_size;

            for (int64_t row = 0; row < M_rows; ++row) {
                const block_q4_K * w = (const block_q4_K *)(weight_mat + row * q4k_row_size);
                dequantize_row_q4_K(w, deq.data(), K);
                double sum = 0.0;
                for (int64_t k = 0; k < K; ++k) {
                    sum += (double)deq[k] * (double)act_row[k];
                }
                out_row[row] = (float)sum;
            }
        }
    }
}

static void compute_cpu_float_reference_moe_chain(
    const float * gate_w_f32, const float * up_w_f32, int64_t K, int64_t M_rows, int64_t N_mats,
    const float * act_f32, int64_t n_act_channels, int64_t n_used, int64_t n_tokens,
    const int32_t * ids_data,
    float * out)
{
    std::vector<float> gate(M_rows * n_used * n_tokens);
    std::vector<float> up(M_rows * n_used * n_tokens);

    compute_cpu_float_reference_mul_mat_id(gate_w_f32, K, M_rows, N_mats, act_f32, n_act_channels, n_used, n_tokens, ids_data, gate.data());
    compute_cpu_float_reference_mul_mat_id(up_w_f32,   K, M_rows, N_mats, act_f32, n_act_channels, n_used, n_tokens, ids_data, up.data());

    const int total = (int)(M_rows * n_used * n_tokens);
    for (int i = 0; i < total; ++i) {
        out[i] = silu_scalar(gate[i]) * up[i];
    }
}

static void compute_cpu_q4k_float_reference_moe_chain(
    const void * gate_w_q4k, const void * up_w_q4k, int64_t K, int64_t M_rows, int64_t N_mats,
    const float * act_f32, int64_t n_act_channels, int64_t n_used, int64_t n_tokens,
    const int32_t * ids_data,
    float * out)
{
    std::vector<float> gate(M_rows * n_used * n_tokens);
    std::vector<float> up(M_rows * n_used * n_tokens);

    compute_cpu_q4k_float_reference_mul_mat_id(gate_w_q4k, K, M_rows, N_mats, act_f32, n_act_channels, n_used, n_tokens, ids_data, gate.data());
    compute_cpu_q4k_float_reference_mul_mat_id(up_w_q4k,   K, M_rows, N_mats, act_f32, n_act_channels, n_used, n_tokens, ids_data, up.data());

    const int total = (int)(M_rows * n_used * n_tokens);
    for (int i = 0; i < total; ++i) {
        out[i] = silu_scalar(gate[i]) * up[i];
    }
}

static void compute_cpu_float_reference_qwen3moe_routed_ffn(
    const float * gate_w_f32, const float * up_w_f32, const float * down_w_f32,
    int64_t K, int64_t N_ff, int64_t N_embd, int64_t N_mats,
    const float * act_f32, const float * weights_f32,
    int64_t n_used, int64_t n_tokens,
    const int32_t * ids_data,
    float * out)
{
    std::vector<float> gate(N_ff * n_used * n_tokens);
    std::vector<float> up(N_ff * n_used * n_tokens);
    std::vector<float> hidden(N_ff * n_used * n_tokens);
    std::vector<float> down(N_embd * n_used * n_tokens);

    compute_cpu_float_reference_mul_mat_id(gate_w_f32, K, N_ff, N_mats, act_f32, 1, n_used, n_tokens, ids_data, gate.data());
    compute_cpu_float_reference_mul_mat_id(up_w_f32,   K, N_ff, N_mats, act_f32, 1, n_used, n_tokens, ids_data, up.data());

    for (int i = 0; i < (int)hidden.size(); ++i) {
        hidden[i] = silu_scalar(gate[i]) * up[i];
    }

    compute_cpu_float_reference_mul_mat_id(down_w_f32, N_ff, N_embd, N_mats, hidden.data(), n_used, n_used, n_tokens, ids_data, down.data());

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        for (int64_t used = 0; used < n_used; ++used) {
            const float w = weights_f32[tok * n_used + used];
            float * out_row = out + (tok * n_used + used) * N_embd;
            const float * down_row = down.data() + (tok * n_used + used) * N_embd;
            for (int64_t i = 0; i < N_embd; ++i) {
                out_row[i] = down_row[i] * w;
            }
        }
    }
}

static void compute_cpu_q4k_float_reference_qwen3moe_routed_ffn(
    const void * gate_w_q4k, const void * up_w_q4k, const void * down_w_q4k,
    int64_t K, int64_t N_ff, int64_t N_embd, int64_t N_mats,
    const float * act_f32, const float * weights_f32,
    int64_t n_used, int64_t n_tokens,
    const int32_t * ids_data,
    float * out)
{
    std::vector<float> gate(N_ff * n_used * n_tokens);
    std::vector<float> up(N_ff * n_used * n_tokens);
    std::vector<float> hidden(N_ff * n_used * n_tokens);
    std::vector<float> down(N_embd * n_used * n_tokens);

    compute_cpu_q4k_float_reference_mul_mat_id(gate_w_q4k, K, N_ff, N_mats, act_f32, 1, n_used, n_tokens, ids_data, gate.data());
    compute_cpu_q4k_float_reference_mul_mat_id(up_w_q4k,   K, N_ff, N_mats, act_f32, 1, n_used, n_tokens, ids_data, up.data());

    for (int i = 0; i < (int)hidden.size(); ++i) {
        hidden[i] = silu_scalar(gate[i]) * up[i];
    }

    compute_cpu_q4k_float_reference_mul_mat_id(down_w_q4k, N_ff, N_embd, N_mats, hidden.data(), n_used, n_used, n_tokens, ids_data, down.data());

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        for (int64_t used = 0; used < n_used; ++used) {
            const float w = weights_f32[tok * n_used + used];
            float * out_row = out + (tok * n_used + used) * N_embd;
            const float * down_row = down.data() + (tok * n_used + used) * N_embd;
            for (int64_t i = 0; i < N_embd; ++i) {
                out_row[i] = down_row[i] * w;
            }
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

    ggml_backend_tensor_set(w, weight_q4k_data, 0, ggml_nbytes(w));
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

static bool run_cuda_mul_mat_id(
    ggml_backend_t backend,
    const void * weight_q4k_data, int64_t K, int64_t M_rows, int64_t N_mats,
    const float * act_f32, int64_t n_act_channels, int64_t n_used, int64_t n_tokens,
    const int32_t * ids_data,
    float * out_data)
{
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 6 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, K, M_rows, N_mats);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    struct ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_act_channels, n_tokens);
    struct ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(as,  weight_q4k_data, 0, ggml_nbytes(as));
    ggml_backend_tensor_set(ids, ids_data,        0, ggml_nbytes(ids));
    ggml_backend_tensor_set(b,   act_f32,         0, ggml_nbytes(b));

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(out, out_data, 0, ggml_nbytes(out));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS;
}

static bool run_cuda_moe_chain(
    ggml_backend_t backend,
    const void * gate_w_q4k, const void * up_w_q4k, int64_t K, int64_t M_rows, int64_t N_mats,
    const float * act_f32, int64_t n_act_channels, int64_t n_used, int64_t n_tokens,
    const int32_t * ids_data,
    float * out_data)
{
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * gate_w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, K, M_rows, N_mats);
    struct ggml_tensor * up_w   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, K, M_rows, N_mats);
    struct ggml_tensor * ids    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    struct ggml_tensor * cur    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_act_channels, n_tokens);

    struct ggml_tensor * gate   = ggml_mul_mat_id(ctx, gate_w, cur, ids);
    struct ggml_tensor * up     = ggml_mul_mat_id(ctx, up_w, cur, ids);
    struct ggml_tensor * out    = ggml_mul(ctx, ggml_silu(ctx, gate), up);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(gate_w, gate_w_q4k, 0, ggml_nbytes(gate_w));
    ggml_backend_tensor_set(up_w,   up_w_q4k,   0, ggml_nbytes(up_w));
    ggml_backend_tensor_set(ids,    ids_data,   0, ggml_nbytes(ids));
    ggml_backend_tensor_set(cur,    act_f32,    0, ggml_nbytes(cur));

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(out, out_data, 0, ggml_nbytes(out));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS;
}

static bool run_cuda_qwen3moe_routed_ffn(
    ggml_backend_t backend,
    const void * gate_w_q4k, const void * up_w_q4k, const void * down_w_q4k,
    int64_t K, int64_t N_ff, int64_t N_embd, int64_t N_mats,
    const float * act_f32, const float * weights_f32,
    int64_t n_used, int64_t n_tokens,
    const int32_t * ids_data,
    float * out_data)
{
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 12 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * gate_w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, K,    N_ff,   N_mats);
    struct ggml_tensor * up_w   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, K,    N_ff,   N_mats);
    struct ggml_tensor * down_w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, N_ff, N_embd, N_mats);
    struct ggml_tensor * ids    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    struct ggml_tensor * cur    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, n_tokens);
    struct ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, n_tokens);

    struct ggml_tensor * gate   = ggml_mul_mat_id(ctx, gate_w, cur, ids);
    struct ggml_tensor * up     = ggml_mul_mat_id(ctx, up_w,   cur, ids);
    struct ggml_tensor * hidden = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    struct ggml_tensor * down   = ggml_mul_mat_id(ctx, down_w, hidden, ids);
    struct ggml_tensor * out    = ggml_mul(ctx, down, weights);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(gate_w, gate_w_q4k, 0, ggml_nbytes(gate_w));
    ggml_backend_tensor_set(up_w,   up_w_q4k,   0, ggml_nbytes(up_w));
    ggml_backend_tensor_set(down_w, down_w_q4k, 0, ggml_nbytes(down_w));
    ggml_backend_tensor_set(ids,    ids_data,   0, ggml_nbytes(ids));
    ggml_backend_tensor_set(cur,    act_f32,    0, ggml_nbytes(cur));
    ggml_backend_tensor_set(weights, weights_f32, 0, ggml_nbytes(weights));

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(out, out_data, 0, ggml_nbytes(out));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS;
}

struct TestCase {
    const char * name;
    int64_t K, N, M;
    bool expect_q16_hit;
};

int main(void) {
    printf("=== test-reex-cuda-q16: CUDA W4×INT16 REEX precision test ===\n\n");

    ggml_cpu_init();

    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) {
        printf("[SKIP] CUDA not available, skipping test.\n");
        return 0;
    }
    printf("CUDA backend: %s\n\n", ggml_backend_name(cuda_backend));

    TestCase tests[] = {
        {"Single vector (M=1)", 2048, 512, 1, true},
        {"Small batch  (M=4)", 2048, 512, 4, true},
        {"Medium batch (M=8)", 2048, 512, 8, true},
        {"Fallback batch (M=9)", 2048, 256, 9, false},
        {"Large K     (K=4096, M=1)", 4096, 256, 1, true},
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);

    bool all_pass = true;

    for (int t = 0; t < ntests; t++) {
        TestCase & tc = tests[t];
        printf("--- Test %d: %s (K=%lld, N=%lld, M=%lld) ---\n",
               t+1, tc.name, (long long)tc.K, (long long)tc.N, (long long)tc.M);

        std::vector<float> weight_f32(tc.K * tc.N);
        fill_deterministic(weight_f32.data(), tc.K * tc.N, 1.0f);

        size_t q4k_row_size = ggml_row_size(GGML_TYPE_Q4_K, tc.K);
        std::vector<uint8_t> weight_q4k(q4k_row_size * tc.N);
        for (int64_t row = 0; row < tc.N; row++) {
            quantize_row_q4_K(weight_f32.data() + row * tc.K,
                              weight_q4k.data() + row * q4k_row_size, tc.K);
        }

        std::vector<float> act_f32(tc.K * tc.M);
        fill_deterministic(act_f32.data(), tc.K * tc.M, 0.8f);

        int total = (int)(tc.N * tc.M);

        std::vector<float> ref_f32xf32(total);
        compute_cpu_float_reference(weight_f32.data(), tc.K, tc.N,
                                     act_f32.data(), tc.M, ref_f32xf32.data());

        std::vector<float> ref_q4k_float(total);
        compute_cpu_q4k_float_reference(weight_q4k.data(), tc.K, tc.N,
                                         act_f32.data(), tc.M, ref_q4k_float.data());

        std::vector<float> out_cuda(total, 0.0f);
        ggml_reex_cuda_reset_stats();
        bool ok = run_cuda_mul_mat(cuda_backend, weight_q4k.data(), tc.K, tc.N,
                                    act_f32.data(), tc.M, out_cuda.data());
        if (!ok) {
            printf("  CUDA MUL_MAT failed!\n");
            all_pass = false;
            continue;
        }

        const int q16_hits = ggml_reex_cuda_get_q16_mul_mat_hits();
        const int q16_batch_fallbacks = ggml_reex_cuda_get_q16_batch_fallbacks();

        const bool hit_ok = tc.expect_q16_hit ? (q16_hits > 0) : (q16_hits == 0);
        const bool fallback_ok = tc.expect_q16_hit ? (q16_batch_fallbacks == 0) : (q16_batch_fallbacks > 0);

        float cos_vs_f32    = cosine_sim(ref_f32xf32.data(), out_cuda.data(), total);
        float cos_vs_q4kf   = cosine_sim(ref_q4k_float.data(), out_cuda.data(), total);

        float mae_vs_f32    = calc_mae(ref_f32xf32.data(), out_cuda.data(), total);
        float mae_vs_q4kf   = calc_mae(ref_q4k_float.data(), out_cuda.data(), total);

        float energy_f32    = calc_energy(ref_f32xf32.data(), total);
        float energy_q4kf   = calc_energy(ref_q4k_float.data(), total);

        float lsb_f32  = energy_f32  * (LSB_REL_W4 + LSB_REL_Q16);
        float lsb_q4kf = energy_q4kf * LSB_REL_Q16;

        float mae_lsb_vs_f32  = (lsb_f32 > 1e-30f) ? mae_vs_f32 / lsb_f32 : 0.0f;
        float mae_lsb_vs_q4kf = (lsb_q4kf > 1e-30f) ? mae_vs_q4kf / lsb_q4kf : 0.0f;

        printf("  vs f32×f32 (final requirement):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_f32, COSINE_MIN);
        printf("    mae/LSB(W4+Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_f32);
        printf("  vs Q4_K×f32 (activation-only delta):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_q4kf, COSINE_MIN);
        printf("    mae/LSB(Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_q4kf);
        printf("  Q16 trace:\n");
        printf("    q16 hits: %d  (expected %s)\n", q16_hits, tc.expect_q16_hit ? "> 0" : "0");
        printf("    batch fallbacks: %d  (expected %s)\n", q16_batch_fallbacks, tc.expect_q16_hit ? "0" : "> 0");

        bool pass_cos = cos_vs_f32 >= COSINE_MIN;
        bool pass_mae = mae_lsb_vs_f32 < 1.0f;
        bool pass_trace = hit_ok && fallback_ok;
        bool pass = pass_cos && pass_mae && pass_trace;

        printf("  Result: %s (cosine: %s, mae: %s, trace: %s)\n\n",
               pass ? "PASS" : "FAIL",
               pass_cos ? "ok" : "FAIL",
               pass_mae ? "ok" : "FAIL",
               pass_trace ? "ok" : "FAIL");

        if (!pass) all_pass = false;
    }

    // MUL_MAT_ID 小 token-batch 命中 Q16 CUDA 路径，并满足最终精度要求。
    {
        printf("--- Test %d: MUL_MAT_ID small token-batch (Q16 hit) ---\n", ntests + 1);

        const int64_t K = 256;
        const int64_t M_rows = 32;
        const int64_t N_mats = 4;
        const int64_t n_used = 2;
        const int64_t n_tokens = 4;

        std::vector<float> weight_f32(K * M_rows * N_mats);
        fill_deterministic(weight_f32.data(), weight_f32.size(), 1.0f);

        size_t q4k_row_size = ggml_row_size(GGML_TYPE_Q4_K, K);
        std::vector<uint8_t> weight_q4k(q4k_row_size * M_rows * N_mats);
        for (int64_t mat = 0; mat < N_mats; ++mat) {
            for (int64_t row = 0; row < M_rows; ++row) {
                const int64_t src_off = (mat * M_rows + row) * K;
                const int64_t dst_off = (mat * M_rows + row) * q4k_row_size;
                quantize_row_q4_K(weight_f32.data() + src_off, weight_q4k.data() + dst_off, K);
            }
        }

        std::vector<float> act_f32(K * n_used * n_tokens);
        fill_deterministic(act_f32.data(), act_f32.size(), 0.7f);

        std::vector<int32_t> ids(n_used * n_tokens);
        for (int64_t tok = 0; tok < n_tokens; ++tok) {
            ids[tok * n_used + 0] = tok % N_mats;
            ids[tok * n_used + 1] = (tok + 1) % N_mats;
        }

        const int total = (int)(M_rows * n_used * n_tokens);
        std::vector<float> ref_f32xf32(total);
        std::vector<float> ref_q4k_float(total);
        compute_cpu_float_reference_mul_mat_id(
            weight_f32.data(), K, M_rows, N_mats,
            act_f32.data(), n_used, n_used, n_tokens,
            ids.data(), ref_f32xf32.data());
        compute_cpu_q4k_float_reference_mul_mat_id(
            weight_q4k.data(), K, M_rows, N_mats,
            act_f32.data(), n_used, n_used, n_tokens,
            ids.data(), ref_q4k_float.data());

        std::vector<float> out_cuda(total, 0.0f);
        ggml_reex_cuda_reset_stats();
        const bool ok = run_cuda_mul_mat_id(
            cuda_backend, weight_q4k.data(), K, M_rows, N_mats,
            act_f32.data(), n_used, n_used, n_tokens, ids.data(), out_cuda.data());

        const int q16_hits = ggml_reex_cuda_get_q16_mul_mat_hits();
        const int q16_mul_mat_id_hits = ggml_reex_cuda_get_q16_mul_mat_id_hits();
        const int q16_batch_fallbacks = ggml_reex_cuda_get_q16_batch_fallbacks();
        const int q16_mul_mat_id_unsupported = ggml_reex_cuda_get_q16_mul_mat_id_unsupported();

        float cos_vs_f32    = cosine_sim(ref_f32xf32.data(), out_cuda.data(), total);
        float cos_vs_q4kf   = cosine_sim(ref_q4k_float.data(), out_cuda.data(), total);
        float mae_vs_f32    = calc_mae(ref_f32xf32.data(), out_cuda.data(), total);
        float mae_vs_q4kf   = calc_mae(ref_q4k_float.data(), out_cuda.data(), total);
        float energy_f32    = calc_energy(ref_f32xf32.data(), total);
        float energy_q4kf   = calc_energy(ref_q4k_float.data(), total);
        float lsb_f32       = energy_f32  * (LSB_REL_W4 + LSB_REL_Q16);
        float lsb_q4kf      = energy_q4kf * LSB_REL_Q16;
        float mae_lsb_vs_f32  = (lsb_f32 > 1e-30f) ? mae_vs_f32 / lsb_f32 : 0.0f;
        float mae_lsb_vs_q4kf = (lsb_q4kf > 1e-30f) ? mae_vs_q4kf / lsb_q4kf : 0.0f;

        const bool pass = ok
            && cos_vs_f32 >= COSINE_MIN
            && mae_lsb_vs_f32 < 1.0f
            && q16_hits == 0
            && q16_mul_mat_id_hits > 0
            && q16_batch_fallbacks == 0
            && q16_mul_mat_id_unsupported == 0;

        printf("  compute: %s\n", ok ? "ok" : "FAIL");
        printf("  vs f32×f32 (final requirement):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_f32, COSINE_MIN);
        printf("    mae/LSB(W4+Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_f32);
        printf("  vs Q4_K×f32 (activation-only delta):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_q4kf, COSINE_MIN);
        printf("    mae/LSB(Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_q4kf);
        printf("  Q16 trace:\n");
        printf("    q16 hits: %d  (expected 0)\n", q16_hits);
        printf("    q16 mul_mat_id hits: %d  (expected > 0)\n", q16_mul_mat_id_hits);
        printf("    batch fallbacks: %d  (expected 0)\n", q16_batch_fallbacks);
        printf("    mul_mat_id unsupported: %d  (expected 0)\n", q16_mul_mat_id_unsupported);
        printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");

        if (!pass) all_pass = false;
    }

    // 更大 token-batch 现在也应命中 Q16 MUL_MAT_ID，不再受 n_tokens<=8 的限制。
    {
        printf("--- Test %d: MUL_MAT_ID larger token-batch ---\n", ntests + 2);

        const int64_t K = 256;
        const int64_t M_rows = 32;
        const int64_t N_mats = 4;
        const int64_t n_used = 2;
        const int64_t n_tokens = 16;

        std::vector<float> weight_f32(K * M_rows * N_mats);
        fill_deterministic(weight_f32.data(), weight_f32.size(), 1.0f);

        size_t q4k_row_size = ggml_row_size(GGML_TYPE_Q4_K, K);
        std::vector<uint8_t> weight_q4k(q4k_row_size * M_rows * N_mats);
        for (int64_t mat = 0; mat < N_mats; ++mat) {
            for (int64_t row = 0; row < M_rows; ++row) {
                const int64_t src_off = (mat * M_rows + row) * K;
                const int64_t dst_off = (mat * M_rows + row) * q4k_row_size;
                quantize_row_q4_K(weight_f32.data() + src_off, weight_q4k.data() + dst_off, K);
            }
        }

        std::vector<float> act_f32(K * n_used * n_tokens);
        fill_deterministic(act_f32.data(), act_f32.size(), 0.65f);

        std::vector<int32_t> ids(n_used * n_tokens);
        for (int64_t tok = 0; tok < n_tokens; ++tok) {
            ids[tok * n_used + 0] = tok % N_mats;
            ids[tok * n_used + 1] = (tok + 2) % N_mats;
        }

        const int total = (int)(M_rows * n_used * n_tokens);
        std::vector<float> ref_f32xf32(total);
        std::vector<float> ref_q4k_float(total);
        compute_cpu_float_reference_mul_mat_id(
            weight_f32.data(), K, M_rows, N_mats,
            act_f32.data(), n_used, n_used, n_tokens,
            ids.data(), ref_f32xf32.data());
        compute_cpu_q4k_float_reference_mul_mat_id(
            weight_q4k.data(), K, M_rows, N_mats,
            act_f32.data(), n_used, n_used, n_tokens,
            ids.data(), ref_q4k_float.data());

        std::vector<float> out_cuda(total, 0.0f);
        ggml_reex_cuda_reset_stats();
        const bool ok = run_cuda_mul_mat_id(
            cuda_backend, weight_q4k.data(), K, M_rows, N_mats,
            act_f32.data(), n_used, n_used, n_tokens, ids.data(), out_cuda.data());

        const int q16_hits = ggml_reex_cuda_get_q16_mul_mat_hits();
        const int q16_mul_mat_id_hits = ggml_reex_cuda_get_q16_mul_mat_id_hits();
        const int q16_batch_fallbacks = ggml_reex_cuda_get_q16_batch_fallbacks();
        const int q16_mul_mat_id_unsupported = ggml_reex_cuda_get_q16_mul_mat_id_unsupported();

        const float cos_vs_f32 = cosine_sim(ref_f32xf32.data(), out_cuda.data(), total);
        const float cos_vs_q4kf = cosine_sim(ref_q4k_float.data(), out_cuda.data(), total);
        const float mae_vs_f32 = calc_mae(ref_f32xf32.data(), out_cuda.data(), total);
        const float mae_vs_q4kf = calc_mae(ref_q4k_float.data(), out_cuda.data(), total);
        const float energy_f32 = calc_energy(ref_f32xf32.data(), total);
        const float energy_q4kf = calc_energy(ref_q4k_float.data(), total);
        const float lsb_f32 = energy_f32 * (LSB_REL_W4 + LSB_REL_Q16);
        const float lsb_q4kf = energy_q4kf * LSB_REL_Q16;
        const float mae_lsb_vs_f32 = (lsb_f32 > 1e-30f) ? mae_vs_f32 / lsb_f32 : 0.0f;
        const float mae_lsb_vs_q4kf = (lsb_q4kf > 1e-30f) ? mae_vs_q4kf / lsb_q4kf : 0.0f;

        const bool pass = ok
            && q16_hits == 0
            && cos_vs_f32 >= COSINE_MIN
            && mae_lsb_vs_f32 < 1.0f
            && q16_mul_mat_id_hits > 0
            && q16_batch_fallbacks == 0
            && q16_mul_mat_id_unsupported == 0;

        printf("  compute: %s\n", ok ? "ok" : "FAIL");
        printf("  vs f32×f32 (final requirement):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_f32, COSINE_MIN);
        printf("    mae/LSB(W4+Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_f32);
        printf("  vs Q4_K×f32 (activation-only delta):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_q4kf, COSINE_MIN);
        printf("    mae/LSB(Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_q4kf);
        printf("  Q16 trace:\n");
        printf("    q16 hits: %d  (expected 0)\n", q16_hits);
        printf("    q16 mul_mat_id hits: %d  (expected > 0)\n", q16_mul_mat_id_hits);
        printf("    batch fallbacks: %d  (expected 0)\n", q16_batch_fallbacks);
        printf("    mul_mat_id unsupported: %d  (expected 0)\n", q16_mul_mat_id_unsupported);
        printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");

        if (!pass) all_pass = false;
    }

    // 更贴近实际 MoE：广播输入（ne11=1）+ 多个 experts per token，并验证更大 token-batch 仍命中 Q16。
    {
        printf("--- Test %d: MUL_MAT_ID MoE-like broadcast input ---\n", ntests + 3);

        const int64_t K = 256;
        const int64_t M_rows = 128;
        const int64_t N_mats = 16;
        const int64_t n_act_channels = 1;
        const int64_t n_used = 8;
        const int64_t n_tokens = 16;

        std::vector<float> weight_f32(K * M_rows * N_mats);
        fill_deterministic(weight_f32.data(), weight_f32.size(), 0.9f);

        const size_t q4k_row_size = ggml_row_size(GGML_TYPE_Q4_K, K);
        std::vector<uint8_t> weight_q4k(q4k_row_size * M_rows * N_mats);
        for (int64_t mat = 0; mat < N_mats; ++mat) {
            for (int64_t row = 0; row < M_rows; ++row) {
                const int64_t src_off = (mat * M_rows + row) * K;
                const int64_t dst_off = (mat * M_rows + row) * q4k_row_size;
                quantize_row_q4_K(weight_f32.data() + src_off, weight_q4k.data() + dst_off, K);
            }
        }

        std::vector<float> act_f32(K * n_act_channels * n_tokens);
        fill_deterministic(act_f32.data(), act_f32.size(), 0.6f);

        std::vector<int32_t> ids(n_used * n_tokens);
        for (int64_t tok = 0; tok < n_tokens; ++tok) {
            for (int64_t used = 0; used < n_used; ++used) {
                ids[tok * n_used + used] = (tok * 3 + used * 2) % N_mats;
            }
        }

        const int total = (int)(M_rows * n_used * n_tokens);
        std::vector<float> ref_f32xf32(total);
        std::vector<float> ref_q4k_float(total);
        compute_cpu_float_reference_mul_mat_id(
            weight_f32.data(), K, M_rows, N_mats,
            act_f32.data(), n_act_channels, n_used, n_tokens,
            ids.data(), ref_f32xf32.data());
        compute_cpu_q4k_float_reference_mul_mat_id(
            weight_q4k.data(), K, M_rows, N_mats,
            act_f32.data(), n_act_channels, n_used, n_tokens,
            ids.data(), ref_q4k_float.data());

        std::vector<float> out_cuda(total, 0.0f);
        ggml_reex_cuda_reset_stats();
        const bool ok = run_cuda_mul_mat_id(
            cuda_backend, weight_q4k.data(), K, M_rows, N_mats,
            act_f32.data(), n_act_channels, n_used, n_tokens, ids.data(), out_cuda.data());

        const int q16_hits = ggml_reex_cuda_get_q16_mul_mat_hits();
        const int q16_mul_mat_id_hits = ggml_reex_cuda_get_q16_mul_mat_id_hits();
        const int q16_batch_fallbacks = ggml_reex_cuda_get_q16_batch_fallbacks();
        const int q16_mul_mat_id_unsupported = ggml_reex_cuda_get_q16_mul_mat_id_unsupported();

        const float cos_vs_f32 = cosine_sim(ref_f32xf32.data(), out_cuda.data(), total);
        const float cos_vs_q4kf = cosine_sim(ref_q4k_float.data(), out_cuda.data(), total);
        const float mae_vs_f32 = calc_mae(ref_f32xf32.data(), out_cuda.data(), total);
        const float mae_vs_q4kf = calc_mae(ref_q4k_float.data(), out_cuda.data(), total);
        const float energy_f32 = calc_energy(ref_f32xf32.data(), total);
        const float energy_q4kf = calc_energy(ref_q4k_float.data(), total);
        const float lsb_f32 = energy_f32 * (LSB_REL_W4 + LSB_REL_Q16);
        const float lsb_q4kf = energy_q4kf * LSB_REL_Q16;
        const float mae_lsb_vs_f32 = (lsb_f32 > 1e-30f) ? mae_vs_f32 / lsb_f32 : 0.0f;
        const float mae_lsb_vs_q4kf = (lsb_q4kf > 1e-30f) ? mae_vs_q4kf / lsb_q4kf : 0.0f;

        const bool pass = ok
            && cos_vs_f32 >= COSINE_MIN
            && mae_lsb_vs_f32 < 1.0f
            && q16_hits == 0
            && q16_mul_mat_id_hits > 0
            && q16_batch_fallbacks == 0
            && q16_mul_mat_id_unsupported == 0;

        printf("  compute: %s\n", ok ? "ok" : "FAIL");
        printf("  vs f32×f32 (final requirement):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_f32, COSINE_MIN);
        printf("    mae/LSB(W4+Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_f32);
        printf("  vs Q4_K×f32 (activation-only delta):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_q4kf, COSINE_MIN);
        printf("    mae/LSB(Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_q4kf);
        printf("  Q16 trace:\n");
        printf("    q16 hits: %d  (expected 0)\n", q16_hits);
        printf("    q16 mul_mat_id hits: %d  (expected > 0)\n", q16_mul_mat_id_hits);
        printf("    batch fallbacks: %d  (expected 0)\n", q16_batch_fallbacks);
        printf("    mul_mat_id unsupported: %d  (expected 0)\n", q16_mul_mat_id_unsupported);
        printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");

        if (!pass) all_pass = false;
    }

    // 小型 MoE 图：两个 MUL_MAT_ID（gate/up）+ silu + mul，验证真实图中多节点都会命中 Q16 CUDA。
    {
        printf("--- Test %d: MoE graph chain (gate/up) ---\n", ntests + 4);

        const int64_t K = 256;
        const int64_t M_rows = 96;
        const int64_t N_mats = 8;
        const int64_t n_act_channels = 1;
        const int64_t n_used = 4;
        const int64_t n_tokens = 16;

        std::vector<float> gate_w_f32(K * M_rows * N_mats);
        std::vector<float> up_w_f32(K * M_rows * N_mats);
        fill_deterministic(gate_w_f32.data(), gate_w_f32.size(), 0.7f);
        fill_deterministic(up_w_f32.data(),   up_w_f32.size(),   0.5f);

        const size_t q4k_row_size = ggml_row_size(GGML_TYPE_Q4_K, K);
        std::vector<uint8_t> gate_w_q4k(q4k_row_size * M_rows * N_mats);
        std::vector<uint8_t> up_w_q4k(q4k_row_size * M_rows * N_mats);
        for (int64_t mat = 0; mat < N_mats; ++mat) {
            for (int64_t row = 0; row < M_rows; ++row) {
                const int64_t src_off = (mat * M_rows + row) * K;
                const int64_t dst_off = (mat * M_rows + row) * q4k_row_size;
                quantize_row_q4_K(gate_w_f32.data() + src_off, gate_w_q4k.data() + dst_off, K);
                quantize_row_q4_K(up_w_f32.data()   + src_off, up_w_q4k.data()   + dst_off, K);
            }
        }

        std::vector<float> act_f32(K * n_act_channels * n_tokens);
        fill_deterministic(act_f32.data(), act_f32.size(), 0.45f);

        std::vector<int32_t> ids(n_used * n_tokens);
        for (int64_t tok = 0; tok < n_tokens; ++tok) {
            for (int64_t used = 0; used < n_used; ++used) {
                ids[tok * n_used + used] = (tok + used * 3) % N_mats;
            }
        }

        const int total = (int)(M_rows * n_used * n_tokens);
        std::vector<float> ref_f32xf32(total);
        std::vector<float> ref_q4k_float(total);
        compute_cpu_float_reference_moe_chain(
            gate_w_f32.data(), up_w_f32.data(), K, M_rows, N_mats,
            act_f32.data(), n_act_channels, n_used, n_tokens,
            ids.data(), ref_f32xf32.data());
        compute_cpu_q4k_float_reference_moe_chain(
            gate_w_q4k.data(), up_w_q4k.data(), K, M_rows, N_mats,
            act_f32.data(), n_act_channels, n_used, n_tokens,
            ids.data(), ref_q4k_float.data());

        std::vector<float> out_cuda(total, 0.0f);
        ggml_reex_cuda_reset_stats();
        const bool ok = run_cuda_moe_chain(
            cuda_backend, gate_w_q4k.data(), up_w_q4k.data(), K, M_rows, N_mats,
            act_f32.data(), n_act_channels, n_used, n_tokens, ids.data(), out_cuda.data());

        const int q16_hits = ggml_reex_cuda_get_q16_mul_mat_hits();
        const int q16_mul_mat_id_hits = ggml_reex_cuda_get_q16_mul_mat_id_hits();
        const int q16_batch_fallbacks = ggml_reex_cuda_get_q16_batch_fallbacks();
        const int q16_mul_mat_id_unsupported = ggml_reex_cuda_get_q16_mul_mat_id_unsupported();

        const float cos_vs_f32 = cosine_sim(ref_f32xf32.data(), out_cuda.data(), total);
        const float cos_vs_q4kf = cosine_sim(ref_q4k_float.data(), out_cuda.data(), total);
        const float mae_vs_f32 = calc_mae(ref_f32xf32.data(), out_cuda.data(), total);
        const float mae_vs_q4kf = calc_mae(ref_q4k_float.data(), out_cuda.data(), total);
        const float energy_f32 = calc_energy(ref_f32xf32.data(), total);
        const float energy_q4kf = calc_energy(ref_q4k_float.data(), total);
        const float lsb_f32 = energy_f32 * (LSB_REL_W4 + LSB_REL_Q16);
        const float lsb_q4kf = energy_q4kf * LSB_REL_Q16;
        const float mae_lsb_vs_f32 = (lsb_f32 > 1e-30f) ? mae_vs_f32 / lsb_f32 : 0.0f;
        const float mae_lsb_vs_q4kf = (lsb_q4kf > 1e-30f) ? mae_vs_q4kf / lsb_q4kf : 0.0f;

        const bool pass = ok
            && cos_vs_f32 >= COSINE_MIN
            && mae_lsb_vs_f32 < 1.0f
            && q16_hits == 0
            && q16_mul_mat_id_hits >= 2
            && q16_batch_fallbacks == 0
            && q16_mul_mat_id_unsupported == 0;

        printf("  compute: %s\n", ok ? "ok" : "FAIL");
        printf("  vs f32×f32 (final requirement):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_f32, COSINE_MIN);
        printf("    mae/LSB(W4+Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_f32);
        printf("  vs Q4_K×f32 (activation-only delta):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_q4kf, COSINE_MIN);
        printf("    mae/LSB(Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_q4kf);
        printf("  Q16 trace:\n");
        printf("    q16 hits: %d  (expected 0)\n", q16_hits);
        printf("    q16 mul_mat_id hits: %d  (expected >= 2)\n", q16_mul_mat_id_hits);
        printf("    batch fallbacks: %d  (expected 0)\n", q16_batch_fallbacks);
        printf("    mul_mat_id unsupported: %d  (expected 0)\n", q16_mul_mat_id_unsupported);
        printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");

        if (!pass) all_pass = false;
    }

    // 更接近 Qwen3MoE routed FFN：gate/up/down 三次 MUL_MAT_ID + silu + mul(weights)。
    {
        printf("--- Test %d: Qwen3MoE-like routed FFN ---\n", ntests + 5);

        const int64_t K = 256;
        const int64_t N_ff = 256;
        const int64_t N_embd = 64;
        const int64_t N_mats = 8;
        const int64_t n_used = 4;
        const int64_t n_tokens = 16;

        std::vector<float> gate_w_f32(K * N_ff * N_mats);
        std::vector<float> up_w_f32(K * N_ff * N_mats);
        std::vector<float> down_w_f32(N_ff * N_embd * N_mats);
        fill_deterministic(gate_w_f32.data(), gate_w_f32.size(), 0.08f);
        fill_deterministic(up_w_f32.data(),   up_w_f32.size(),   0.09f);
        fill_deterministic(down_w_f32.data(), down_w_f32.size(), 0.07f);

        const size_t gate_row_size = ggml_row_size(GGML_TYPE_Q4_K, K);
        const size_t down_row_size = ggml_row_size(GGML_TYPE_Q4_K, N_ff);
        std::vector<uint8_t> gate_w_q4k(gate_row_size * N_ff * N_mats);
        std::vector<uint8_t> up_w_q4k(gate_row_size * N_ff * N_mats);
        std::vector<uint8_t> down_w_q4k(down_row_size * N_embd * N_mats);

        for (int64_t mat = 0; mat < N_mats; ++mat) {
            for (int64_t row = 0; row < N_ff; ++row) {
                const int64_t src_off = (mat * N_ff + row) * K;
                const int64_t dst_off = (mat * N_ff + row) * gate_row_size;
                quantize_row_q4_K(gate_w_f32.data() + src_off, gate_w_q4k.data() + dst_off, K);
                quantize_row_q4_K(up_w_f32.data()   + src_off, up_w_q4k.data()   + dst_off, K);
            }
            for (int64_t row = 0; row < N_embd; ++row) {
                const int64_t src_off = (mat * N_embd + row) * N_ff;
                const int64_t dst_off = (mat * N_embd + row) * down_row_size;
                quantize_row_q4_K(down_w_f32.data() + src_off, down_w_q4k.data() + dst_off, N_ff);
            }
        }

        std::vector<float> act_f32(K * n_tokens);
        fill_deterministic(act_f32.data(), act_f32.size(), 0.12f);

        std::vector<float> weights_f32(n_used * n_tokens);
        for (int64_t tok = 0; tok < n_tokens; ++tok) {
            for (int64_t used = 0; used < n_used; ++used) {
                weights_f32[tok * n_used + used] = 0.05f + 0.03f * (float)((tok + used) % 5);
            }
        }

        std::vector<int32_t> ids(n_used * n_tokens);
        for (int64_t tok = 0; tok < n_tokens; ++tok) {
            for (int64_t used = 0; used < n_used; ++used) {
                ids[tok * n_used + used] = (tok * 2 + used * 3) % N_mats;
            }
        }

        const int total = (int)(N_embd * n_used * n_tokens);
        std::vector<float> ref_f32xf32(total);
        std::vector<float> ref_q4k_float(total);
        compute_cpu_float_reference_qwen3moe_routed_ffn(
            gate_w_f32.data(), up_w_f32.data(), down_w_f32.data(),
            K, N_ff, N_embd, N_mats,
            act_f32.data(), weights_f32.data(),
            n_used, n_tokens, ids.data(), ref_f32xf32.data());
        compute_cpu_q4k_float_reference_qwen3moe_routed_ffn(
            gate_w_q4k.data(), up_w_q4k.data(), down_w_q4k.data(),
            K, N_ff, N_embd, N_mats,
            act_f32.data(), weights_f32.data(),
            n_used, n_tokens, ids.data(), ref_q4k_float.data());

        std::vector<float> out_cuda(total, 0.0f);
        ggml_reex_cuda_reset_stats();
        const bool ok = run_cuda_qwen3moe_routed_ffn(
            cuda_backend,
            gate_w_q4k.data(), up_w_q4k.data(), down_w_q4k.data(),
            K, N_ff, N_embd, N_mats,
            act_f32.data(), weights_f32.data(),
            n_used, n_tokens, ids.data(), out_cuda.data());

        const int q16_hits = ggml_reex_cuda_get_q16_mul_mat_hits();
        const int q16_mul_mat_id_hits = ggml_reex_cuda_get_q16_mul_mat_id_hits();
        const int q16_batch_fallbacks = ggml_reex_cuda_get_q16_batch_fallbacks();
        const int q16_mul_mat_id_unsupported = ggml_reex_cuda_get_q16_mul_mat_id_unsupported();

        const float cos_vs_f32 = cosine_sim(ref_f32xf32.data(), out_cuda.data(), total);
        const float cos_vs_q4kf = cosine_sim(ref_q4k_float.data(), out_cuda.data(), total);
        const float mae_vs_f32 = calc_mae(ref_f32xf32.data(), out_cuda.data(), total);
        const float mae_vs_q4kf = calc_mae(ref_q4k_float.data(), out_cuda.data(), total);
        const float energy_f32 = calc_energy(ref_f32xf32.data(), total);
        const float energy_q4kf = calc_energy(ref_q4k_float.data(), total);
        const float lsb_f32 = energy_f32 * (LSB_REL_W4 + LSB_REL_Q16);
        const float lsb_q4kf = energy_q4kf * LSB_REL_Q16;
        const float mae_lsb_vs_f32 = (lsb_f32 > 1e-30f) ? mae_vs_f32 / lsb_f32 : 0.0f;
        const float mae_lsb_vs_q4kf = (lsb_q4kf > 1e-30f) ? mae_vs_q4kf / lsb_q4kf : 0.0f;

        const bool pass_trace = ok
            && q16_hits == 0
            && q16_mul_mat_id_hits >= 3
            && q16_batch_fallbacks == 0
            && q16_mul_mat_id_unsupported == 0;

        printf("  compute: %s\n", ok ? "ok" : "FAIL");
        printf("  vs f32×f32 (final requirement):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_f32, COSINE_MIN);
        printf("    mae/LSB(W4+Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_f32);
        printf("  vs Q4_K×f32 (activation-only delta):\n");
        printf("    cosine:    %.6f  (min: %.4f)\n", cos_vs_q4kf, COSINE_MIN);
        printf("    mae/LSB(Q16): %.4f  (max: 1.0)\n", mae_lsb_vs_q4kf);
        printf("  Q16 trace:\n");
        printf("    q16 hits: %d  (expected 0)\n", q16_hits);
        printf("    q16 mul_mat_id hits: %d  (expected >= 3)\n", q16_mul_mat_id_hits);
        printf("    batch fallbacks: %d  (expected 0)\n", q16_batch_fallbacks);
        printf("    mul_mat_id unsupported: %d  (expected 0)\n", q16_mul_mat_id_unsupported);
        printf("  Result: %s (graph trace coverage)\n\n", pass_trace ? "PASS" : "FAIL");

        if (!pass_trace) all_pass = false;
    }

    printf("=== %s ===\n", all_pass ? "ALL PASSED" : "SOME FAILED");

    ggml_backend_free(cuda_backend);
    return all_pass ? 0 : 1;
}
