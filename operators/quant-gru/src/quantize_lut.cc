#include "quantize_ops_helper.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

void generate_piecewise_linear_lut_to_params(GRUQuantParams &params) {
    const auto &config = params.bitwidth_config_;
    // 直接使用单一权威 scale（仿射=连续校准值，POT2=2^-shift），无需 fixed/shift 往返
    const float update_in_scale = params.update_gate_input_.scale;
    const float update_out_scale = params.update_gate_output_.scale;
    const float reset_in_scale = params.reset_gate_input_.scale;
    const float reset_out_scale = params.reset_gate_output_.scale;
    const float new_in_scale = params.new_gate_input_.scale;
    const float new_out_scale = params.new_gate_output_.scale;

    // update gate Sigmoid
    params.sigmoid_update_gate_lut_ = generate_sigmoid_lut(
        update_in_scale, params.update_gate_input_.zero_point,
        update_out_scale, params.update_gate_output_.zero_point,
        config.update_gate_input_, config.update_gate_output_);

    // reset gate Sigmoid
    params.sigmoid_reset_gate_lut_ = generate_sigmoid_lut(
        reset_in_scale, params.reset_gate_input_.zero_point,
        reset_out_scale, params.reset_gate_output_.zero_point,
        config.reset_gate_input_, config.reset_gate_output_);

    // new gate Tanh
    params.tanh_new_gate_lut_ = generate_tanh_lut(
        new_in_scale, params.new_gate_input_.zero_point,
        new_out_scale, params.new_gate_output_.zero_point,
        config.new_gate_input_, config.new_gate_output_);

#ifdef DEBUG
    printf("[DEBUG] generate_piecewise_linear_lut_to_params: LUTs stored in params\n");
#endif
}

// ==================== 分段线性量化参数生成函数 ====================

// 线性拟合函数（最小二乘法）
inline void linear_fit(const std::vector<float> &x, const std::vector<float> &y, float &b,
                       float &c) {
    int n = x.size();
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;

    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }

    float denom = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denom) < 1e-9f) {
        b = 0.0f;
        c = sum_y / n;
        return;
    }

    b = (n * sum_xy - sum_x * sum_y) / denom;
    c = (sum_y - b * sum_x) / n;
}

// 自适应分段（Sigmoid/Tanh 专用）
// 🔥 基于导数的权重分配，与 Python 参考 (bc_ds_U8.py) 保持一致
// 关键：中心区域固定在 x = 0 附近（sigmoid/tanh 的特性），不是输入范围的中心
std::vector<float> adaptive_segmentation_sigmoid(float x_min, float x_max, int num_segments) {
    // Sigmoid/Tanh 的权重配置（与 Python 参考一致）
    // centerWeight: 中心区域的权重倍数
    // centerRange: 中心区域的半宽度
    const float centerWeight = 5.0f;  // sigmoid: 5.0, tanh: 4.0
    const float centerRange = 2.0f;   // |x| < 2.0 的区域权重增加

    // 1. 在输入范围内均匀采样，计算权重
    const int numSamples = 1000;
    std::vector<float> xSamples(numSamples);
    std::vector<float> weights(numSamples - 1);

    for (int i = 0; i < numSamples; i++) {
        xSamples[i] = x_min + (x_max - x_min) * static_cast<float>(i) / (numSamples - 1);
    }

    // 2. 计算导数（斜率）和权重
    for (int i = 0; i < numSamples - 1; i++) {
        float x = xSamples[i];
        float x_next = xSamples[i + 1];

        // 计算 sigmoid 的导数 y' = y * (1 - y)，其中 y = sigmoid(x)
        float y = 1.0f / (1.0f + std::exp(-x));
        float y_next = 1.0f / (1.0f + std::exp(-x_next));
        float slope = std::abs(y_next - y) / (x_next - x + 1e-9f);

        // 距离 x = 0 的距离（与 Python 参考一致）
        float distToCenter = std::abs(x);

        // 计算权重
        if (distToCenter < centerRange) {
            // 中心区域：权重随距离线性递减
            weights[i] = centerWeight * (1.0f - distToCenter / centerRange) + 1.0f;
        } else {
            // 外侧区域：基于斜率的权重
            weights[i] = 1.0f + slope * 0.5f;
        }
    }

    // 3. 归一化权重
    float sumWeights = 0.0f;
    for (int i = 0; i < numSamples - 1; i++) {
        sumWeights += weights[i];
    }
    for (int i = 0; i < numSamples - 1; i++) {
        weights[i] /= sumWeights;
    }

    // 4. 计算累积权重
    std::vector<float> cumWeights(numSamples - 1);
    cumWeights[0] = weights[0];
    for (int i = 1; i < numSamples - 1; i++) {
        cumWeights[i] = cumWeights[i - 1] + weights[i];
    }

    // 5. 根据累积权重生成分段点
    std::vector<float> points;
    points.push_back(x_min);

    for (int i = 1; i < num_segments; i++) {
        float target = static_cast<float>(i) / num_segments;

        // 二分查找目标累积权重对应的 x 值
        auto it = std::lower_bound(cumWeights.begin(), cumWeights.end(), target);
        int idx = static_cast<int>(std::distance(cumWeights.begin(), it));
        if (idx >= numSamples - 1) idx = numSamples - 2;
        if (idx < 0) idx = 0;

        points.push_back(xSamples[idx]);
    }

    points.push_back(x_max);

    // 6. 确保点单调递增且无重复
    std::sort(points.begin(), points.end());
    auto last = std::unique(points.begin(), points.end(),
                            [](float a, float b) { return std::abs(a - b) < 1e-9f; });
    points.erase(last, points.end());

    // 如果去重后点数不够，在最大间隔处插入点
    while (static_cast<int>(points.size()) < num_segments + 1) {
        float max_gap = 0.0f;
        size_t max_gap_idx = 0;
        for (size_t i = 0; i < points.size() - 1; i++) {
            float gap = points[i + 1] - points[i];
            if (gap > max_gap) {
                max_gap = gap;
                max_gap_idx = i;
            }
        }
        float new_point = (points[max_gap_idx] + points[max_gap_idx + 1]) / 2.0f;
        points.insert(points.begin() + max_gap_idx + 1, new_point);
    }

    return points;
}

// ==================== 统一的 LUT 生成函数 ====================
//
// 【设计原则】
//   - 所有位宽配置使用统一的 SigmoidLUT 结构
//   - q_b 使用 int32_t 避免溢出（tanh 斜率 1.0 需要此精度）
//   - 根据 input_bw 自动确定输入范围
//
// =========================================================================

/**
 * @brief 统一的 Sigmoid LUT 生成函数
 * @param shift_bits_x 输入量化 shift bits
 * @param zp_x 输入 zero-point
 * @param shift_bits_y 输出量化 shift bits
 * @param zp_y 输出 zero-point
 * @param input_bw 输入位宽（决定输入范围）
 * @param output_bw 输出位宽（决定 shift_bits_b 精度）
 */
static inline std::pair<int32_t, int8_t> quantize_multiplier_signed(float value, QuantBitWidth output_bw) {
    if (std::abs(value) < 1e-12f) {
        return {0, 0};
    }
    const int32_t max_q = output_bw.qmax();
    int8_t n = static_cast<int8_t>(std::floor(std::log2(static_cast<double>(max_q) / std::abs(value))));
    int64_t q = round_to_int64(static_cast<double>(value) * std::ldexp(1.0, n));
    q = std::max<int64_t>(-max_q, std::min<int64_t>(max_q, q));
    return {static_cast<int32_t>(q), n};
}

SigmoidLUT generate_sigmoid_lut(float effective_scale_x, int32_t zp_x, float effective_scale_y,
                                 int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw) {
    // 根据输入位宽确定量化范围（任意位宽支持）
    int32_t quant_min = input_bw.qmin();
    int32_t quant_max = input_bw.qmax();

    float scale_x = effective_scale_x;
    float x_min = static_cast<float>(quant_min - zp_x) * scale_x;
    float x_max = static_cast<float>(quant_max - zp_x) * scale_x;

    // Sigmoid 有效范围限制
    constexpr float SIGMOID_EFFECTIVE_RANGE = 8.0f;
    x_min = std::max(x_min, -SIGMOID_EFFECTIVE_RANGE);
    x_max = std::min(x_max, SIGMOID_EFFECTIVE_RANGE);

#ifdef DEBUG
    printf("[DEBUG] generate_sigmoid_lut: input_bw=%d, eff_scale_x=%.8f, zp_x=%d, x_range=[%.4f, %.4f]\n",
           static_cast<int>(input_bw), scale_x, zp_x, x_min, x_max);
#endif

    SigmoidLUT lut{};
    lut.shift_bits_x = 0;
    lut.zp_x = zp_x;
    lut.shift_bits_y = 0;
    lut.zp_y = zp_y;
    lut.effective_scale_x = effective_scale_x;
    lut.effective_scale_y = effective_scale_y;

    // 生成分段点
    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    // 第一遍扫描：拟合所有分段
    struct SegmentCoeffs { float x_start, x_end, b, c; };
    std::vector<SegmentCoeffs> all_coeffs(NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        const int num_samples = 100;
        std::vector<float> x_seg(num_samples), y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = 1.0f / (1.0f + std::exp(-x_val));  // Sigmoid
        }

        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);
        all_coeffs[i] = {x_start, x_end, b_fp, c_fp};
    }

    const float scale_y = effective_scale_y;

    // 第三遍扫描：量化每段
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto &coeff = all_coeffs[i];
        const float m_real = coeff.b * (scale_x / scale_y);
        auto [q_b, n_BX_total] = quantize_multiplier_signed(m_real, output_bw);
        int32_t term_c_precomputed = round_to_int(coeff.c / scale_y + static_cast<float>(zp_y));

        // threshold 量化（任意位宽支持，存储为 int32_t）
        int32_t threshold = round_to_int(coeff.x_end / scale_x + zp_x);
        threshold = clamp_by_bitwidth(threshold, input_bw);

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

SigmoidLUT generate_sigmoid_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                                int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw) {
    return generate_sigmoid_lut(exp2_scale(shift_bits_x), zp_x, exp2_scale(shift_bits_y), zp_y, input_bw, output_bw);
}

/**
 * @brief 统一的 Tanh LUT 生成函数
 * @param shift_bits_x 输入量化 shift bits
 * @param zp_x 输入 zero-point
 * @param shift_bits_y 输出量化 shift bits
 * @param zp_y 输出 zero-point
 * @param input_bw 输入位宽（决定输入范围）
 * @param output_bw 输出位宽（决定 shift_bits_b 精度）
 */
SigmoidLUT generate_tanh_lut(float effective_scale_x, int32_t zp_x, float effective_scale_y,
                              int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw) {
    // 根据输入位宽确定量化范围（任意位宽支持）
    int32_t quant_min = input_bw.qmin();
    int32_t quant_max = input_bw.qmax();

    float scale_x = effective_scale_x;
    float x_min = static_cast<float>(quant_min - zp_x) * scale_x;
    float x_max = static_cast<float>(quant_max - zp_x) * scale_x;

    // Tanh 有效范围限制
    constexpr float TANH_EFFECTIVE_RANGE = 4.0f;
    x_min = std::max(x_min, -TANH_EFFECTIVE_RANGE);
    x_max = std::min(x_max, TANH_EFFECTIVE_RANGE);

#ifdef DEBUG
    printf("[DEBUG] generate_tanh_lut: input_bw=%d, eff_scale_x=%.8f, zp_x=%d, x_range=[%.4f, %.4f]\n",
           static_cast<int>(input_bw), scale_x, zp_x, x_min, x_max);
#endif

    SigmoidLUT lut{};
    lut.shift_bits_x = 0;
    lut.zp_x = zp_x;
    lut.shift_bits_y = 0;
    lut.zp_y = zp_y;
    lut.effective_scale_x = effective_scale_x;
    lut.effective_scale_y = effective_scale_y;

    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    struct SegmentCoeffs { float x_start, x_end, b, c; };
    std::vector<SegmentCoeffs> all_coeffs(NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        const int num_samples = 100;
        std::vector<float> x_seg(num_samples), y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = std::tanh(x_val);  // Tanh
        }

        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);
        all_coeffs[i] = {x_start, x_end, b_fp, c_fp};
    }

    const float scale_y = effective_scale_y;

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto &coeff = all_coeffs[i];
        const float m_real = coeff.b * (scale_x / scale_y);
        auto [q_b, n_BX_total] = quantize_multiplier_signed(m_real, output_bw);
        int32_t term_c_precomputed = round_to_int(coeff.c / scale_y + static_cast<float>(zp_y));

        // threshold 量化（任意位宽支持，存储为 int32_t）
        int32_t threshold = round_to_int(coeff.x_end / scale_x + zp_x);
        threshold = clamp_by_bitwidth(threshold, input_bw);

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

SigmoidLUT generate_tanh_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                             int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw) {
    return generate_tanh_lut(exp2_scale(shift_bits_x), zp_x, exp2_scale(shift_bits_y), zp_y, input_bw, output_bw);
}
