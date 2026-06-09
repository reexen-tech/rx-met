/*
 * test-reex-gemm.cpp  —  Reex GEMM 综合测试
 *
 * 评估基准：float×float（原始 float 权重 × float 激活）
 * 每组 4 个类型：f32×f32 / W4×f32 / W4×Q16 / W4×Q8
 * 评估指标：cosine > 0.9999, mae < 1 LSB
 *
 * 测试层级：
 *   1. 激活量化 roundtrip（Q16 / Q8）
 *   2. 单 vec_dot 4-way 对比
 *   3. 单 block GEMM 4-way 对比
 *   4. 多次 GEMM 串联 4-way 对比（qwen3moe 单 block 拓扑模拟）
 *
 * 编译 & 运行：
 *   cmake --build build --target test-reex-gemm
 *   ctest --test-dir build -R test-reex-gemm -V
 */

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "reex/reex_gemm_cpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* ================================================================
 *  全局评估标准
 * ================================================================ */

static const float COSINE_MIN = 0.9999f;

/*
 * 1 LSB 定义（依据 ADC/DAC 与 JEDEC 惯例）
 *
 * 标准含义（JEDEC / 数据手册）：
 *   - 1 LSB = 量化器的模拟分辨率 = Full_Scale / 2^N = 一个量化步长
 *   - 误差常以 "½ LSB"、"1 LSB" 的倍数表示
 *
 * 本测试中「输出 1 LSB」= 各量化器步长传播到 dot product 输出后的等效界：
 *
 *   • basis（传播参考量，误差界与之成正比）：
 *     - 数值误差分析中 dot product 误差界 ∝ Σ|x_i||y_i|（见 e.g. Higham, RR 等）
 *     - 单 vec_dot：basis = energy = Σ|w_i×a_i|
 *     - GEMM 矩阵：basis = mean_energy = (1/(N*M)) Σ_{j,m} Σ_i |W_ji||act_im|
 *     - Chain 等无逐行 energy 时退化为 mean|gold| 作为输出量级参考
 *
 *   • LSB_REL_factor：各格式 1 LSB 相对系数（见下）
 *
 *     Q4_K（4-bit）：dequant = d*scale_s*q4 - dmin*min_s，q4∈[0,15]
 *       → 1 LSB_weight = 一步 q4 = d*scale_s → 相对子块范围 = 1/16
 *
 *     Q16（15-bit int16，per-block scale, QK_K=256, d = amax/32767）：
 *       → 1 LSB_activation = 1/32767（与 REEX_Q16_SCALE 一致）
 *
 *     Q8_K（7-bit 有效，per-block scale, QK_K=256）：
 *       → 1 LSB_activation ≈ 1/127
 *
 * 组合路径（权重+激活均量化）时，按最坏叠加：output_1LSB = basis × (LSB_w + LSB_a)。
 */
static const float LSB_REL_Q4K = (1.f / 16.f);   /* 4-bit: 1/2^4 */
static const float LSB_REL_Q16  = (1.f / 32767.f);
static const float LSB_REL_Q8   = (1.f / 127.f);

enum quant_path { PATH_F32F32, PATH_W4F32, PATH_W4Q16, PATH_W4Q8 };

static float lsb_rel_factor(quant_path p) {
    switch (p) {
        case PATH_F32F32: return 0.f;
        case PATH_W4F32:  return LSB_REL_Q4K;
        case PATH_W4Q16:  return LSB_REL_Q4K + LSB_REL_Q16;
        case PATH_W4Q8:   return LSB_REL_Q4K + LSB_REL_Q8;
    }
    return 0.f;
}

static float compute_1lsb(float basis, quant_path p) {
    if (p == PATH_F32F32) return 1e-10f;
    return basis * lsb_rel_factor(p);
}

static const char * path_tag(quant_path p) {
    switch (p) {
        case PATH_F32F32: return "f32×f32";
        case PATH_W4F32:  return "W4×f32 ";
        case PATH_W4Q16:  return "W4×Q16 ";
        case PATH_W4Q8:   return "W4×Q8  ";
    }
    return "???";
}

static const char * basis_label(size_t n) {
    return (n == 1) ? "energy" : "mean|gold|";
}

/* ================================================================
 *  辅助函数
 * ================================================================ */

static void generate_data(float amplitude, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++)
        dst[i] = amplitude * sinf((float)i * 0.1f + (float)(i % 7) * 0.3f);
}

static void generate_data_negative(float amplitude, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++)
        dst[i] = -fabsf(amplitude * sinf((float)i * 0.1f + (float)(i % 7) * 0.3f));
}

static void generate_data_positive(float amplitude, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++)
        dst[i] = fabsf(amplitude * sinf((float)i * 0.1f + (float)(i % 7) * 0.3f) + 0.01f);
}

static double array_cosine(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return (na < 1e-30 && nb < 1e-30) ? 1.0 : 0.0;
    return dot / sqrt(na * nb);
}

static void block_metrics(const float * ref, const float * test, size_t n,
                           float * out_mae, float * out_rel, float * out_mean_abs_ref)
{
    double mae = 0, abs_ref = 0;
    for (size_t i = 0; i < n; i++) {
        mae     += fabs((double)ref[i] - test[i]);
        abs_ref += fabs((double)ref[i]);
    }
    *out_mae = (float)(mae / n);
    *out_mean_abs_ref = (float)(abs_ref / n);
    *out_rel = (*out_mean_abs_ref > 1e-12f) ? *out_mae / *out_mean_abs_ref : 0.f;
}

/*
 * 统一评估，返回 1=FAIL, 0=PASS
 *
 * n > 1 (GEMM 矩阵输出): cosine > 0.9999 AND mae < 1LSB
 * n == 1 (单 vec_dot 标量): cosine 恒为 ±1 无意义，仅检查 mae < 1LSB
 *
 * lsb_basis: 用于计算 1LSB 的基准（传播参考量）
 *   - 传 >0：使用该值（vec_dot 用 energy；GEMM 用 mean_energy，见下）
 *   - 传 <=0：对矩阵用 mean|gold|（仅 chain 等无逐行 energy 时用）
 * basis_is_energy: 为 true 时打印标签为 mean_energy/energy（与误差传播定义一致）
 */
static int evaluate(const char * prefix, quant_path path,
                    const float * gold, const float * test, size_t n,
                    float lsb_basis = -1.f, bool basis_is_energy = false)
{
    float mae, rel, mar;
    block_metrics(gold, test, n, &mae, &rel, &mar);
    float basis = (lsb_basis > 0.f) ? lsb_basis : mar;
    float lsb = compute_1lsb(basis, path);
    float lsb_ratio = (lsb > 1e-30f) ? mae / lsb : 0.f;
    int mae_ok = (mae <= lsb) ? 1 : 0;
    float factor = lsb_rel_factor(path);
    const char * basis_name = basis_is_energy ? (n == 1 ? "energy" : "mean_energy") : basis_label(n);

    if (n == 1) {
        /* 单标量: cosine(a,b) = sign(a)*sign(b) = ±1，无区分度 */
        int fail = mae_ok ? 0 : 1;
        fprintf(stderr, "  %s %s %s  gold=%.4f  test=%.4f  |err|=%.2e\n",
                fail ? "FAIL" : "  OK", prefix, path_tag(path),
                gold[0], test[0], fabsf(test[0] - gold[0]));
        if (path != PATH_F32F32)
            fprintf(stderr, "       1LSB = %s(%.2e) × %.1e = %.2e    |err|/1LSB = %.2f  %s\n",
                    basis_name, basis, factor, lsb, lsb_ratio,
                    mae_ok ? "<1LSB OK" : ">1LSB FAIL");
        return fail;
    }

    /* n > 1: cosine 有意义 */
    double cos = array_cosine(gold, test, n);
    int cos_ok = (cos >= COSINE_MIN) ? 1 : 0;
    int fail = (cos_ok && mae_ok) ? 0 : 1;

    fprintf(stderr, "  %s %s %s  cosine=%.6f  mae=%.2e\n",
            fail ? "FAIL" : "  OK", prefix, path_tag(path), cos, mae);
    if (path != PATH_F32F32)
        fprintf(stderr, "       1LSB = %s(%.2e) × %.1e = %.2e    mae/1LSB = %.2f  %s\n",
                basis_name, basis, factor, lsb, lsb_ratio,
                mae_ok ? "<1LSB OK" : ">1LSB FAIL");
    return fail;
}

/* ================================================================
 *  Test 1: Q16 激活 roundtrip
 * ================================================================ */

static int test_q16_roundtrip(long k) {
    assert(k > 0 && k % QK_K == 0);
    std::vector<float> src(k), recovered(k);
    generate_data(1.0f, k, src.data());

    size_t row_bytes = ggml_q16_activation_row_size_reex(k);
    std::vector<uint8_t> buf(row_bytes);
    ggml_quantize_row_f32_to_q16_scale_row_reex(src.data(), buf.data(), k);

    block_q16_K * bq = (block_q16_K *)buf.data();
    int nb = k / QK_K;
    for (int b = 0; b < nb; b++) {
        float d = bq[b].d;
        for (int j = 0; j < QK_K; j++)
            recovered[b * QK_K + j] = bq[b].qs[j] * d;
    }

    float mae, rel, mar;
    block_metrics(src.data(), recovered.data(), k, &mae, &rel, &mar);
    double cos = array_cosine(src.data(), recovered.data(), k);

    int fail = (cos < COSINE_MIN) ? 1 : 0;
    fprintf(stderr, "  %s q16_roundtrip k=%ld cosine=%.6f mae=%.2e\n",
            fail ? "FAIL" : "  OK", k, cos, mae);
    return fail;
}

/* ================================================================
 *  Test 2: Q8 激活 roundtrip
 * ================================================================ */

static int test_q8_roundtrip(long k) {
    assert(k > 0 && k % QK_K == 0);
    std::vector<float> src(k), recovered(k);
    generate_data(1.0f, k, src.data());

    size_t row_bytes = ggml_q8_activation_row_size_reex(k);
    std::vector<uint8_t> buf(row_bytes);
    ggml_quantize_row_f32_to_q8_scale_row_reex(src.data(), buf.data(), k);

    block_q8_K * bq = (block_q8_K *)buf.data();
    int nb = k / QK_K;
    for (int b = 0; b < nb; b++) {
        float d = bq[b].d;
        for (int j = 0; j < QK_K; j++)
            recovered[b * QK_K + j] = bq[b].qs[j] * d;
    }

    float mae, rel, mar;
    block_metrics(src.data(), recovered.data(), k, &mae, &rel, &mar);
    double cos = array_cosine(src.data(), recovered.data(), k);

    int fail = (cos < COSINE_MIN) ? 1 : 0;
    fprintf(stderr, "  %s q8_roundtrip k=%ld cosine=%.6f mae=%.2e\n",
            fail ? "FAIL" : "  OK", k, cos, mae);
    return fail;
}

/* ================================================================
 *  Test 3: 单 vec_dot 4-way 对比
 *  gold = 原始 float 权重 · float 激活（无任何量化）
 * ================================================================ */

static int test_vec_dot_group(int n, const char * label,
                               void (*gen_act)(float, size_t, float *))
{
    assert(n > 0 && n % QK_K == 0);
    int nb = n / QK_K;

    std::vector<float> wf(n), af(n);
    generate_data(1.0f, n, wf.data());
    gen_act(1.0f, n, af.data());

    /* gold: 原始 float × float（无量化） */
    double gold_d = 0;
    for (int i = 0; i < n; i++) gold_d += (double)wf[i] * af[i];
    float gold = (float)gold_d;

    /* Q4_K 量化权重 */
    std::vector<block_q4_K> wq(nb);
    quantize_row_q4_K_ref(wf.data(), wq.data(), n);

    /*
     * 输入能量 = sum(|w_i| × |a_i|)
     * 用作 1LSB 基准，避免 dot product 抵消导致 |gold| 过小而误判
     */
    float energy = 0;
    for (int i = 0; i < n; i++) energy += fabsf(wf[i]) * fabsf(af[i]);

    char prefix[128];
    int fail = 0;

    snprintf(prefix, sizeof(prefix), "vec_dot n=%d %s", n, label);
    fail += evaluate(prefix, PATH_F32F32, &gold, &gold, 1, energy, true);

    /* W4×f32 */
    {
        float out = 0;
        ggml_vec_dot_q4_K_f32_reex(n, &out, 0, wq.data(), 0, af.data(), 0, 1);
        fail += evaluate(prefix, PATH_W4F32, &gold, &out, 1, energy, true);
    }

    /* W4×Q16 */
    {
        size_t q16_bytes = ggml_q16_activation_row_size_reex(n);
        std::vector<uint8_t> q16_buf(q16_bytes);
        ggml_quantize_row_f32_to_q16_scale_row_reex(af.data(), q16_buf.data(), n);
        float out = 0;
        ggml_vec_dot_q4_K_q16_reex(n, &out, 0, wq.data(), 0, q16_buf.data(), 0, 1);
        fail += evaluate(prefix, PATH_W4Q16, &gold, &out, 1, energy, true);
    }

    /* W4×Q8 */
    {
        size_t q8_bytes = ggml_q8_activation_row_size_reex(n);
        std::vector<uint8_t> q8_buf(q8_bytes);
        ggml_quantize_row_f32_to_q8_scale_row_reex(af.data(), q8_buf.data(), n);
        float out = 0;
        ggml_vec_dot_q4_K_q8_reex(n, &out, 0, wq.data(), 0, q8_buf.data(), 0, 1);
        fail += evaluate(prefix, PATH_W4Q8, &gold, &out, 1, energy, true);
    }

    return fail;
}

/* ================================================================
 *  Single Block GEMM 公共数据
 * ================================================================ */

struct gemm_data {
    std::vector<float> wf;
    std::vector<block_q4_K> wq;
    std::vector<float> act;
    std::vector<float> gold;
    float mean_energy;  /* (1/(N*M)) * Σ_{j,m} Σ_i |W_ji||act_im|，误差传播参考量 */
    int K, M, N, nb;
};

static gemm_data make_gemm_data(int K, int M, int N) {
    assert(K > 0 && K % QK_K == 0);
    gemm_data d;
    d.K = K; d.M = M; d.N = N; d.nb = K / QK_K;

    d.wf.resize(N * K);
    for (int j = 0; j < N; j++)
        generate_data(1.0f + j * 0.1f, K, d.wf.data() + j * K);

    d.wq.resize(N * d.nb);
    for (int j = 0; j < N; j++)
        quantize_row_q4_K_ref(d.wf.data() + j * K, d.wq.data() + j * d.nb, K);

    d.act.resize(K * M);
    for (int m = 0; m < M; m++)
        generate_data(0.5f + m * 0.05f, K, d.act.data() + m * K);

    d.gold.resize(N * M, 0.f);
    double sum_energy = 0.0;
    for (int j = 0; j < N; j++) {
        for (int m = 0; m < M; m++) {
            double sum = 0;
            double energy_jm = 0;
            for (int i = 0; i < K; i++) {
                sum += (double)d.wf[j * K + i] * d.act[m * K + i];
                energy_jm += fabs(d.wf[j * K + i]) * fabs(d.act[m * K + i]);
            }
            d.gold[j * M + m] = (float)sum;
            sum_energy += energy_jm;
        }
    }
    d.mean_energy = (float)(sum_energy / (N * M));
    return d;
}

/* ================================================================
 *  Test 4: Single Block GEMM 4-way
 *  gold = float×float（原始 float 权重 × float 激活）
 * ================================================================ */

static int test_single_block_gemm_group(int K, int M, int N) {
    gemm_data d = make_gemm_data(K, M, N);
    size_t sz = (size_t)(N * M);

    char prefix[128];
    snprintf(prefix, sizeof(prefix), "single_block K=%d M=%d N=%d", K, M, N);
    int fail = 0;

    /* f32×f32 */
    fail += evaluate(prefix, PATH_F32F32, d.gold.data(), d.gold.data(), sz, d.mean_energy, true);

    /* W4×f32 */
    {
        std::vector<float> out(sz, 0.f);
        for (int j = 0; j < N; j++)
            for (int m = 0; m < M; m++)
                ggml_vec_dot_q4_K_f32_reex(K, &out[j * M + m], 0,
                    d.wq.data() + j * d.nb, 0,
                    d.act.data() + m * K, 0, 1);
        fail += evaluate(prefix, PATH_W4F32, d.gold.data(), out.data(), sz, d.mean_energy, true);
    }

    /* W4×Q16 */
    {
        size_t q16_bytes = ggml_q16_activation_row_size_reex(K);
        std::vector<uint8_t> q16_buf(M * q16_bytes);
        for (int m = 0; m < M; m++)
            ggml_quantize_row_f32_to_q16_scale_row_reex(
                d.act.data() + m * K, q16_buf.data() + m * q16_bytes, K);
        std::vector<float> out(sz, 0.f);
        for (int j = 0; j < N; j++)
            for (int m = 0; m < M; m++)
                ggml_vec_dot_q4_K_q16_reex(K, &out[j * M + m], 0,
                    d.wq.data() + j * d.nb, 0,
                    q16_buf.data() + m * q16_bytes, 0, 1);
        fail += evaluate(prefix, PATH_W4Q16, d.gold.data(), out.data(), sz, d.mean_energy, true);
    }

    /* W4×Q8 */
    {
        size_t q8_bytes = ggml_q8_activation_row_size_reex(K);
        std::vector<uint8_t> q8_buf(M * q8_bytes);
        for (int m = 0; m < M; m++)
            ggml_quantize_row_f32_to_q8_scale_row_reex(
                d.act.data() + m * K, q8_buf.data() + m * q8_bytes, K);
        std::vector<float> out(sz, 0.f);
        for (int j = 0; j < N; j++)
            for (int m = 0; m < M; m++)
                ggml_vec_dot_q4_K_q8_reex(K, &out[j * M + m], 0,
                    d.wq.data() + j * d.nb, 0,
                    q8_buf.data() + m * q8_bytes, 0, 1);
        fail += evaluate(prefix, PATH_W4Q8, d.gold.data(), out.data(), sz, d.mean_energy, true);
    }

    return fail;
}

/* ================================================================
 *  Test 5: GEMM Chain 4-way
 *
 *  模拟单 block 内多次 GEMM 串联的误差累积：
 *    GEMM_0: W0[K,K] × act[K,M] → out0[K,M]
 *    GEMM_1: W1[K,K] × out0     → out1[K,M]
 *    ...
 *  gold = 全部用原始 float 权重 × float 激活
 * ================================================================ */

static int test_gemm_chain_group(int K, int M, int num_gemm) {
    assert(K > 0 && K % QK_K == 0);
    int N = K;
    int nb = K / QK_K;
    size_t sz = (size_t)(N * M);

    /* 初始激活 */
    std::vector<float> init_act(K * M);
    for (int m = 0; m < M; m++)
        generate_data(0.5f + m * 0.05f, K, init_act.data() + m * K);

    /* 预生成所有轮次的权重 */
    struct round_weights {
        std::vector<float> wf;
        std::vector<block_q4_K> wq;
    };
    std::vector<round_weights> rounds(num_gemm);
    for (int g = 0; g < num_gemm; g++) {
        rounds[g].wf.resize(N * K);
        for (int j = 0; j < N; j++)
            generate_data(0.3f + g * 0.1f + j * 0.01f, K, rounds[g].wf.data() + j * K);
        rounds[g].wq.resize(N * nb);
        for (int j = 0; j < N; j++)
            quantize_row_q4_K_ref(rounds[g].wf.data() + j * K, rounds[g].wq.data() + j * nb, K);
    }

    /* 跑 4 条路径，每条独立串联 */
    auto run_chain = [&](quant_path mode) -> std::vector<float> {
        std::vector<float> act = init_act;
        for (int g = 0; g < num_gemm; g++) {
            std::vector<float> out(sz, 0.f);
            if (mode == PATH_F32F32) {
                for (int j = 0; j < N; j++)
                    for (int m = 0; m < M; m++) {
                        double sum = 0;
                        for (int i = 0; i < K; i++)
                            sum += (double)rounds[g].wf[j * K + i] * act[m * K + i];
                        out[j * M + m] = (float)sum;
                    }
            } else if (mode == PATH_W4F32) {
                for (int j = 0; j < N; j++)
                    for (int m = 0; m < M; m++)
                        ggml_vec_dot_q4_K_f32_reex(K, &out[j * M + m], 0,
                            rounds[g].wq.data() + j * nb, 0,
                            act.data() + m * K, 0, 1);
            } else if (mode == PATH_W4Q16) {
                size_t q16_bytes = ggml_q16_activation_row_size_reex(K);
                std::vector<uint8_t> buf(M * q16_bytes);
                for (int m = 0; m < M; m++)
                    ggml_quantize_row_f32_to_q16_scale_row_reex(
                        act.data() + m * K, buf.data() + m * q16_bytes, K);
                for (int j = 0; j < N; j++)
                    for (int m = 0; m < M; m++)
                        ggml_vec_dot_q4_K_q16_reex(K, &out[j * M + m], 0,
                            rounds[g].wq.data() + j * nb, 0,
                            buf.data() + m * q16_bytes, 0, 1);
            } else {
                size_t q8_bytes = ggml_q8_activation_row_size_reex(K);
                std::vector<uint8_t> buf(M * q8_bytes);
                for (int m = 0; m < M; m++)
                    ggml_quantize_row_f32_to_q8_scale_row_reex(
                        act.data() + m * K, buf.data() + m * q8_bytes, K);
                for (int j = 0; j < N; j++)
                    for (int m = 0; m < M; m++)
                        ggml_vec_dot_q4_K_q8_reex(K, &out[j * M + m], 0,
                            rounds[g].wq.data() + j * nb, 0,
                            buf.data() + m * q8_bytes, 0, 1);
            }
            act = out;
        }
        return act;
    };

    std::vector<float> gold = run_chain(PATH_F32F32);
    std::vector<float> w4f32 = run_chain(PATH_W4F32);
    std::vector<float> w4q16 = run_chain(PATH_W4Q16);
    std::vector<float> w4q8  = run_chain(PATH_W4Q8);

    char prefix[128];
    snprintf(prefix, sizeof(prefix), "chain K=%d M=%d G=%d", K, M, num_gemm);
    int fail = 0;

    fail += evaluate(prefix, PATH_F32F32, gold.data(), gold.data(), sz);
    fail += evaluate(prefix, PATH_W4F32,  gold.data(), w4f32.data(), sz);
    fail += evaluate(prefix, PATH_W4Q16,  gold.data(), w4q16.data(), sz);
    fail += evaluate(prefix, PATH_W4Q8,   gold.data(), w4q8.data(), sz);

    return fail;
}

/* ================================================================
 *  main
 * ================================================================ */

int main(void) {
    ggml_cpu_init();

    int total_fail = 0;

    fprintf(stderr, "reex gemm tests (gold = float×float, criteria: cosine>%.4f, mae<1LSB)\n", COSINE_MIN);

    /* --- 1. 激活量化 roundtrip --- */
    fprintf(stderr, "\n--- Activation Roundtrip ---\n");
    total_fail += test_q16_roundtrip(256);
    total_fail += test_q16_roundtrip(512);
    total_fail += test_q8_roundtrip(256);
    total_fail += test_q8_roundtrip(512);

    /* --- 2. 单 vec_dot 4-way --- */
    fprintf(stderr, "\n--- Vec Dot 4-way (gold = float×float) ---\n");
    total_fail += test_vec_dot_group(256, "neg", generate_data_negative);
    fprintf(stderr, "\n");
    total_fail += test_vec_dot_group(256, "pos", generate_data_positive);
    fprintf(stderr, "\n");
    total_fail += test_vec_dot_group(512, "neg", generate_data_negative);
    fprintf(stderr, "\n");
    total_fail += test_vec_dot_group(512, "mix", generate_data);

    /* --- 3. Single Block GEMM 4-way --- */
    fprintf(stderr, "\n--- Single Block GEMM 4-way (gold = float×float) ---\n");
    total_fail += test_single_block_gemm_group(256, 8, 16);
    fprintf(stderr, "\n");
    total_fail += test_single_block_gemm_group(256, 4, 32);

    /* --- 4. GEMM Chain 4-way（误差累积观察，不计入 pass/fail）--- */
    fprintf(stderr, "\n--- GEMM Chain 4-way (gold = float×float, 误差累积观察) ---\n");
    int chain_fail = test_gemm_chain_group(256, 8, 4);
    fprintf(stderr, "  [chain: %d items >1LSB — 多轮串联误差累积，仅供参考]\n", chain_fail);

    fprintf(stderr, "\n=== SUMMARY: %d failures (single-op) ===\n", total_fail);
    fprintf(stderr, "reex gemm tests: %s\n", total_fail ? "FAILED" : "all passed");
    return total_fail ? 1 : 0;
}
