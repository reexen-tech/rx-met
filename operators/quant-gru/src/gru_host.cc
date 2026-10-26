#include "gru_host.h"
#include "gru_calibration_helpers.h"
#include "calibration_utils.h"
#include "gru_quant_cpu.h"
#include "pot_sqnr_calibrator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>


GRUQuantParams calculateGRUQuantitativeParameters(
    const GRUQuantizationRanges &quant_ranges, const OperatorQuantConfig &bitwidth_config) {
    ensureSymmetricWeightBiasConfig(bitwidth_config);
    GRUQuantParams quant_params;
    quant_params.hidden_ = quant_ranges.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;
    const bool usePOT2 = bitwidth_config.usePOT2_;
    const PotScaleMethod potMethod = bitwidth_config.pot_scale_method_;
    const float potTol = bitwidth_config.pot_scale_tolerance_;

    // 辅助 lambda：单值校准 -> 唯一权威 QuantParam（scale + zp）
    auto calibrateScalar = [usePOT2, potMethod, potTol](QuantBitWidth bw, bool symmetric,
                                     float min_val, float max_val) -> QuantParam {
        const EncodedScaleResult encoded =
            encodeFromRange(min_val, max_val, bw, symmetric, usePOT2, potMethod, potTol);
        return QuantParam{storedScaleForMode(encoded, usePOT2), encoded.zero_point};
    };

    // 输入 x 和隐藏状态 h
    quant_params.x_ = calibrateScalar(bitwidth_config.x_, bitwidth_config.x_symmetric_,
                                      quant_ranges.min_x_, quant_ranges.max_x_);
    quant_params.h_ = calibrateScalar(bitwidth_config.h_, bitwidth_config.h_symmetric_,
                                      quant_ranges.min_h_, quant_ranges.max_h_);

    // 权重 W/R 和偏置 bw/br：唯一 per-channel 权威数组，按粒度广播
    const int channel_size = quant_ranges.hidden_ * 3;
    const int hidden_size = quant_ranges.hidden_;

    auto calibrateChannel = [&](ChannelQuantParam& out,
                                OperatorQuantConfig::QuantizationGranularity gran,
                                QuantBitWidth bw, bool symmetric,
                                const std::vector<float>& mins, const std::vector<float>& maxs) {
        out.granularity = gran;
        out.hidden = hidden_size;
        out.resize(channel_size);
        if (gran == OperatorQuantConfig::PER_TENSOR) {
            float mn = *std::min_element(mins.begin(), mins.end());
            float mx = *std::max_element(maxs.begin(), maxs.end());
            const QuantParam qp = calibrateScalar(bw, symmetric, mn, mx);
            std::fill(out.channels.begin(), out.channels.end(), qp);
        } else if (gran == OperatorQuantConfig::PER_GATE) {
            for (int gate = 0; gate < 3; ++gate) {
                float mn = std::numeric_limits<float>::max();
                float mx = std::numeric_limits<float>::lowest();
                for (int c = gate * hidden_size; c < (gate + 1) * hidden_size; ++c) {
                    mn = std::min(mn, mins[c]);
                    mx = std::max(mx, maxs[c]);
                }
                const QuantParam qp = calibrateScalar(bw, symmetric, mn, mx);
                for (int c = gate * hidden_size; c < (gate + 1) * hidden_size; ++c) out.channel(c) = qp;
            }
        } else {  // PER_CHANNEL
            for (int c = 0; c < channel_size; ++c) {
                out.channel(c) = calibrateScalar(bw, symmetric, mins[c], maxs[c]);
            }
        }
    };

    calibrateChannel(quant_params.W_, bitwidth_config.W_granularity_, bitwidth_config.W_,
                     bitwidth_config.W_symmetric_, quant_ranges.min_W_, quant_ranges.max_W_);
    calibrateChannel(quant_params.R_, bitwidth_config.R_granularity_, bitwidth_config.R_,
                     bitwidth_config.R_symmetric_, quant_ranges.min_R_, quant_ranges.max_R_);
    calibrateBias(quant_params.bw_, hidden_size, bitwidth_config.bw_granularity_,
        bitwidth_config.bw_, usePOT2, [&](int c) { return std::make_pair(quant_ranges.min_bw_[c], quant_ranges.max_bw_[c]); });
    calibrateBias(quant_params.br_, hidden_size, bitwidth_config.br_granularity_,
        bitwidth_config.br_, usePOT2, [&](int c) { return std::make_pair(quant_ranges.min_br_[c], quant_ranges.max_br_[c]); });

    // Linear 输出 (GEMM+bias)
    quant_params.weight_ih_linear_ = calibrateScalar(bitwidth_config.weight_ih_linear_,
        bitwidth_config.weight_ih_linear_symmetric_, quant_ranges.min_Wx_, quant_ranges.max_Wx_);
    quant_params.weight_hh_linear_ = calibrateScalar(bitwidth_config.weight_hh_linear_,
        bitwidth_config.weight_hh_linear_symmetric_, quant_ranges.min_Rh_, quant_ranges.max_Rh_);

    // 门激活输入
    quant_params.update_gate_input_ = calibrateScalar(bitwidth_config.update_gate_input_,
        bitwidth_config.update_gate_input_symmetric_, quant_ranges.min_update_gate_input_, quant_ranges.max_update_gate_input_);
    quant_params.reset_gate_input_ = calibrateScalar(bitwidth_config.reset_gate_input_,
        bitwidth_config.reset_gate_input_symmetric_, quant_ranges.min_reset_gate_input_, quant_ranges.max_reset_gate_input_);
    quant_params.new_gate_input_ = calibrateScalar(bitwidth_config.new_gate_input_,
        bitwidth_config.new_gate_input_symmetric_, quant_ranges.min_new_gate_input_, quant_ranges.max_new_gate_input_);

    // 激活函数输出（确保最小范围）
    constexpr float MIN_ACTIVATION_RANGE = 0.5f;
    auto calibrateWithMinRange = [&](const char* name, QuantBitWidth bw, bool symmetric,
                                     float min_val, float max_val) -> QuantParam {
        ensureMinRange(min_val, max_val, MIN_ACTIVATION_RANGE, name);
        return calibrateScalar(bw, symmetric, min_val, max_val);
    };

    quant_params.update_gate_output_ = calibrateWithMinRange("scale_update_gate_output",
        bitwidth_config.update_gate_output_, bitwidth_config.update_gate_output_symmetric_,
        quant_ranges.min_update_gate_output_, quant_ranges.max_update_gate_output_);
    quant_params.reset_gate_output_ = calibrateWithMinRange("scale_reset_gate_output",
        bitwidth_config.reset_gate_output_, bitwidth_config.reset_gate_output_symmetric_,
        quant_ranges.min_reset_gate_output_, quant_ranges.max_reset_gate_output_);
    quant_params.new_gate_output_ = calibrateWithMinRange("scale_new_gate_output",
        bitwidth_config.new_gate_output_, bitwidth_config.new_gate_output_symmetric_,
        quant_ranges.min_new_gate_output_, quant_ranges.max_new_gate_output_);

    // 中间计算
    quant_params.mul_reset_hidden_ = calibrateScalar(bitwidth_config.mul_reset_hidden_,
        bitwidth_config.mul_reset_hidden_symmetric_, quant_ranges.min_mul_reset_hidden_, quant_ranges.max_mul_reset_hidden_);
    quant_params.mul_new_contribution_ = calibrateScalar(bitwidth_config.mul_new_contribution_,
        bitwidth_config.mul_new_contribution_symmetric_, quant_ranges.min_mul_new_contribution_, quant_ranges.max_mul_new_contribution_);
    quant_params.mul_old_contribution_ = calibrateScalar(bitwidth_config.mul_old_contribution_,
        bitwidth_config.mul_old_contribution_symmetric_, quant_ranges.min_mul_old_contribution_, quant_ranges.max_mul_old_contribution_);

    // 生成 LUT 并存储到参数中
    generate_piecewise_linear_lut_to_params(quant_params);
    return quant_params;
}

void quantitativeWeightCPU(const int input_size, const int hidden_size, const float *W, const float *R,
                           const float *bw, const float *br,
                           const GRUQuantParams &quant_parms, int32_t *W_quant,
                           int32_t *R_quant, int32_t *bw_quant, int32_t *br_quant) {
    const int hidden3 = hidden_size * 3;
    const auto &bw_cfg = quant_parms.bitwidth_config_;
    const bool usePOT2 = bw_cfg.usePOT2_;
    const std::vector<FixedPointScale> fW = toFixedScales(quant_parms.W_, usePOT2);
    const std::vector<FixedPointScale> fR = toFixedScales(quant_parms.R_, usePOT2);
    const std::vector<FixedPointScale> fbw = toFixedScales(quant_parms.bw_, usePOT2);
    const std::vector<FixedPointScale> fbr = toFixedScales(quant_parms.br_, usePOT2);

    // 量化权重矩阵（per-channel）
    for (int k = 0; k < input_size; k++) {
        for (int m = 0; m < hidden3; m++) {
            int idx = k * hidden3 + m;
            W_quant[idx] = quantize<false>(W[idx], fW[m], 0, bw_cfg.W_);
        }
    }

    for (int k = 0; k < hidden_size; k++) {
        for (int m = 0; m < hidden3; m++) {
            int idx = k * hidden3 + m;
            R_quant[idx] = quantize<false>(R[idx], fR[m], 0, bw_cfg.R_);
        }
    }

    // 量化偏置（per-channel）
    for (int m = 0; m < hidden3; m++) {
        bw_quant[m] = quantize<false>(bw[m], fbw[m], 0, bw_cfg.bw_);
        br_quant[m] = quantize<false>(br[m], fbr[m], 0, bw_cfg.br_);
    }
}

void quantGRUForwardCPU(bool is_training, int time_steps, int batch_size, int input_size,
                        int hidden_size, const int32_t *W, const int32_t *R, const int32_t *bw,
                        const int32_t *br, const float *x, const float *h0,
                        const GRUQuantParams &quant_parms, float *h, float *v) {
    const std::size_t x_size = time_steps * batch_size * input_size;
    const std::size_t h_total_size = (time_steps + 1) * batch_size * hidden_size;
    const auto &bw_cfg = quant_parms.bitwidth_config_;
    const bool usePOT2 = bw_cfg.usePOT2_;

    // 量化输入序列
    std::vector<int32_t> x_quant(x_size);
    for (std::size_t i = 0; i < x_size; i++) {
        x_quant[i] = quantize(x[i], toFixedScale(quant_parms.x_, usePOT2), quant_parms.x_.zero_point, bw_cfg.x_);
    }

    // 分配隐藏状态缓冲区
    std::vector<int32_t> h_quant(h_total_size);

    // 初始化 h0 为零点值
    for (int i = 0; i < batch_size * hidden_size; i++) {
        h_quant[i] = quant_parms.h_.zero_point;
    }

    // 如果提供了初始状态，量化到 h_quant[0]
    if (h0 != nullptr) {
        for (int i = 0; i < batch_size * hidden_size; i++) {
            h_quant[i] = quantize(h0[i], toFixedScale(quant_parms.h_, usePOT2), quant_parms.h_.zero_point, bw_cfg.h_);
        }
    }

    // 分配中间值缓冲区（v: update_gate, reset_gate, new_gate, weight_hh_linear_g）
    std::vector<int32_t> v_quant(time_steps * batch_size * hidden_size * 4);

    // 创建 CPU 版本的 ForwardPassQuantCPU
    cpu::ForwardPassQuantCPU forward(is_training, batch_size, input_size, hidden_size);
    forward.setRescaleParam(quant_parms);

    // 运行前向传播
    forward.Run(time_steps, W, R, bw, br, x_quant.data(), h_quant.data(),
                v != nullptr ? v_quant.data() : nullptr, 0.0f, nullptr);

    // 反量化隐藏状态输出
    for (std::size_t i = 0; i < h_total_size; i++) {
        h[i] = dequantize(h_quant[i], toFixedScale(quant_parms.h_, usePOT2), quant_parms.h_.zero_point);
    }

    // 如果需要中间值，反量化 v
    if (v != nullptr) {
        const int hidden3 = hidden_size * 3;
        for (int t = 0; t < time_steps; t++) {
            for (int b = 0; b < batch_size; b++) {
                for (int j = 0; j < hidden_size; j++) {
                    const int base_idx = (t * batch_size + b) * hidden_size * 4 + j;
                    // v[0] = update_gate, v[1] = reset_gate, v[2] = new_gate, v[3] = weight_hh_linear_g
                    v[base_idx + hidden_size * 0] = dequantize(
                        v_quant[base_idx + hidden_size * 0], toFixedScale(quant_parms.update_gate_output_, usePOT2),
                        quant_parms.update_gate_output_.zero_point);
                    v[base_idx + hidden_size * 1] = dequantize(
                        v_quant[base_idx + hidden_size * 1], toFixedScale(quant_parms.reset_gate_output_, usePOT2),
                        quant_parms.reset_gate_output_.zero_point);
                    v[base_idx + hidden_size * 2] = dequantize(
                        v_quant[base_idx + hidden_size * 2], toFixedScale(quant_parms.new_gate_output_, usePOT2),
                        quant_parms.new_gate_output_.zero_point);
                    v[base_idx + hidden_size * 3] = dequantize(
                        v_quant[base_idx + hidden_size * 3], toFixedScale(quant_parms.weight_hh_linear_, usePOT2),
                        quant_parms.weight_hh_linear_.zero_point);
                }
            }
        }
    }
}

void quantGRUForwardCPU(bool is_training, int time_steps, int batch_size, int input_size,
                        int hidden_size, const float *W, const float *R, const float *bw,
                        const float *br, const float *x, const float *h0,
                        const GRUQuantParams &quant_parms, float *h, float *v) {
    const int hidden3 = hidden_size * 3;

    // 量化权重（使用统一接口）
    std::vector<int32_t> W_quant(input_size * hidden3);
    std::vector<int32_t> R_quant(hidden_size * hidden3);
    std::vector<int32_t> bw_quant(hidden3);
    std::vector<int32_t> br_quant(hidden3);

    quantitativeWeightCPU(input_size, hidden_size, W, R, bw, br, quant_parms,
                          W_quant.data(), R_quant.data(), bw_quant.data(), br_quant.data());

    // 调用量化权重版本
    quantGRUForwardCPU(is_training, time_steps, batch_size, input_size, hidden_size,
                       W_quant.data(), R_quant.data(), bw_quant.data(), br_quant.data(), x, h0,
                       quant_parms, h, v);
}

GRUQuantParams calculateGRUQuantitativeParametersFromHistograms(
    const GRUHistogramCollectors &hist_collectors, const OperatorQuantConfig &bitwidth_config,
    bool use_percentile, float percentile_value) {
    ensureSymmetricWeightBiasConfig(bitwidth_config);
    GRUQuantParams quant_params;
    quant_params.hidden_ = hist_collectors.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;
    const bool usePOT2 = bitwidth_config.usePOT2_;
    const PotScaleMethod potMethod = bitwidth_config.pot_scale_method_;
    const float potTol = bitwidth_config.pot_scale_tolerance_;

    const int channel_size = hist_collectors.hidden_ * 3;
    const int hidden_size = hist_collectors.hidden_;

    // 辅助 lambda：校准单个直方图 -> 唯一权威 QuantParam（scale + zp）
    auto histCalibrate = [&](const HistogramCollector& hist, QuantBitWidth bw, bool sym,
                             const char* name) -> QuantParam {
        if (!hist.is_valid()) throw std::runtime_error(std::string(name) + " is invalid.");
        ContinuousScaleResult continuous = calibrateContinuousScaleFromHistogram(
            hist.histogram(), bw, sym, use_percentile, percentile_value);
        EncodedScaleResult encoded = encodeScaleResult(
            continuous, bw, sym, usePOT2, potMethod, potTol);
        return QuantParam{storedScaleForMode(encoded, usePOT2), encoded.zero_point};
    };

    // 权重/偏置：唯一 per-channel 权威数组，按粒度从对应直方图广播
    auto histCalibrateChannel = [&](ChannelQuantParam& out,
                                    OperatorQuantConfig::QuantizationGranularity gran,
                                    QuantBitWidth bw, bool sym,
                                    const HistogramCollector& tensor_hist,
                                    const std::array<HistogramCollector, 3>& gate_hist,
                                    const std::vector<HistogramCollector>& chan_hist,
                                    const char* name) {
        out.granularity = gran;
        out.hidden = hidden_size;
        out.resize(channel_size);
        if (gran == OperatorQuantConfig::PER_TENSOR) {
            const QuantParam qp = histCalibrate(tensor_hist, bw, sym, name);
            std::fill(out.channels.begin(), out.channels.end(), qp);
        } else if (gran == OperatorQuantConfig::PER_GATE) {
            for (int gate = 0; gate < 3; ++gate) {
                const QuantParam qp = histCalibrate(gate_hist[gate], bw, sym, name);
                for (int c = gate * hidden_size; c < (gate + 1) * hidden_size; ++c) out.channel(c) = qp;
            }
        } else {  // PER_CHANNEL
            for (int c = 0; c < channel_size; ++c) {
                out.channel(c) = histCalibrate(chan_hist[c], bw, sym, name);
            }
        }
    };

    histCalibrateChannel(quant_params.W_, bitwidth_config.W_granularity_, bitwidth_config.W_,
        bitwidth_config.W_symmetric_, hist_collectors.W_tensor_hist, hist_collectors.W_gate_hist, hist_collectors.W_hist, "W");
    histCalibrateChannel(quant_params.R_, bitwidth_config.R_granularity_, bitwidth_config.R_,
        bitwidth_config.R_symmetric_, hist_collectors.R_tensor_hist, hist_collectors.R_gate_hist, hist_collectors.R_hist, "R");
    calibrateBias(quant_params.bw_, hidden_size, bitwidth_config.bw_granularity_,
        bitwidth_config.bw_, usePOT2, [&](int c) { const auto& h = hist_collectors.bw_hist[c].histogram();
            return std::make_pair(h.observed_min, h.observed_max); });
    calibrateBias(quant_params.br_, hidden_size, bitwidth_config.br_granularity_,
        bitwidth_config.br_, usePOT2, [&](int c) { const auto& h = hist_collectors.br_hist[c].histogram();
            return std::make_pair(h.observed_min, h.observed_max); });

    // 标量参数（不受 granularity 影响）
    quant_params.x_ = histCalibrate(hist_collectors.x_hist, bitwidth_config.x_, bitwidth_config.x_symmetric_, "scale_x");
    quant_params.h_ = histCalibrate(hist_collectors.h_hist, bitwidth_config.h_, bitwidth_config.h_symmetric_, "scale_h");
    quant_params.weight_ih_linear_ = histCalibrate(hist_collectors.Wx_hist, bitwidth_config.weight_ih_linear_, bitwidth_config.weight_ih_linear_symmetric_, "scale_weight_ih_linear");
    quant_params.weight_hh_linear_ = histCalibrate(hist_collectors.Rh_hist, bitwidth_config.weight_hh_linear_, bitwidth_config.weight_hh_linear_symmetric_, "scale_weight_hh_linear");
    quant_params.update_gate_input_ = histCalibrate(hist_collectors.update_gate_input_hist, bitwidth_config.update_gate_input_, bitwidth_config.update_gate_input_symmetric_, "scale_update_gate_input");
    quant_params.reset_gate_input_ = histCalibrate(hist_collectors.reset_gate_input_hist, bitwidth_config.reset_gate_input_, bitwidth_config.reset_gate_input_symmetric_, "scale_reset_gate_input");
    quant_params.new_gate_input_ = histCalibrate(hist_collectors.new_gate_input_hist, bitwidth_config.new_gate_input_, bitwidth_config.new_gate_input_symmetric_, "scale_new_gate_input");
    quant_params.update_gate_output_ = histCalibrate(hist_collectors.update_gate_output_hist, bitwidth_config.update_gate_output_, bitwidth_config.update_gate_output_symmetric_, "scale_update_gate_output");
    quant_params.reset_gate_output_ = histCalibrate(hist_collectors.reset_gate_output_hist, bitwidth_config.reset_gate_output_, bitwidth_config.reset_gate_output_symmetric_, "scale_reset_gate_output");
    quant_params.new_gate_output_ = histCalibrate(hist_collectors.new_gate_output_hist, bitwidth_config.new_gate_output_, bitwidth_config.new_gate_output_symmetric_, "scale_new_gate_output");
    quant_params.mul_reset_hidden_ = histCalibrate(hist_collectors.mul_reset_hidden_hist, bitwidth_config.mul_reset_hidden_, bitwidth_config.mul_reset_hidden_symmetric_, "scale_mul_reset_hidden");
    quant_params.mul_new_contribution_ = histCalibrate(hist_collectors.mul_new_contribution_hist, bitwidth_config.mul_new_contribution_, bitwidth_config.mul_new_contribution_symmetric_, "scale_mul_new_contribution");
    quant_params.mul_old_contribution_ = histCalibrate(hist_collectors.mul_old_contribution_hist, bitwidth_config.mul_old_contribution_, bitwidth_config.mul_old_contribution_symmetric_, "scale_mul_old_contribution");

    generate_piecewise_linear_lut_to_params(quant_params);
    return quant_params;
}
