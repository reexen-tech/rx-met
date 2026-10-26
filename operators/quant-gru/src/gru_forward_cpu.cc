#include "gru_host.h"
#include "inline_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void observe(const float* data, size_t count, float* minimum, float* maximum,
             HistogramCollector* histogram) {
    if (histogram) histogram->collect(data, count);
    if (!minimum) return;
    float lo = std::numeric_limits<float>::max();
    float hi = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < count; ++i) {
        if (std::isfinite(data[i])) {
            lo = std::min(lo, data[i]);
            hi = std::max(hi, data[i]);
        }
    }
    if (lo > hi) lo = hi = 0.0f;
    *minimum = std::min(*minimum, lo);
    *maximum = std::max(*maximum, hi);
}

void observeParameter(const float* values, int rows, int hidden,
                      std::vector<float>* minimum, std::vector<float>* maximum,
                      std::vector<HistogramCollector>* channels,
                      HistogramCollector* tensor,
                      std::array<HistogramCollector, 3>* gates,
                      OperatorQuantConfig::QuantizationGranularity granularity,
                      bool bias) {
    const int columns = 3 * hidden;
    std::vector<float> channel(rows), gate(rows * hidden);
    if (tensor && granularity == OperatorQuantConfig::PER_TENSOR)
        tensor->collect(values, static_cast<size_t>(rows) * columns);
    for (int c = 0; c < columns; ++c) {
        for (int k = 0; k < rows; ++k) channel[k] = values[k * columns + c];
        HistogramCollector* collector = channels &&
            (bias || granularity == OperatorQuantConfig::PER_CHANNEL) ? &(*channels)[c] : nullptr;
        // Static per-channel weights are observed once, as in the CUDA collector.
        if (collector && collector->is_valid()) collector = nullptr;
        observe(channel.data(), rows, minimum ? &(*minimum)[c] : nullptr,
                maximum ? &(*maximum)[c] : nullptr, collector);
    }
    if (gates && granularity == OperatorQuantConfig::PER_GATE) {
        for (int g = 0; g < 3; ++g) {
            for (int k = 0; k < rows; ++k)
                std::copy_n(values + k * columns + g * hidden, hidden, gate.data() + k * hidden);
            (*gates)[g].collect(gate.data(), gate.size());
        }
    }
}
}  // namespace

void forwardWithCalibrationCPU(int steps, int batch, int input, int hidden,
    const float* W, const float* R, const float* bw, const float* br,
    const float* x, const float* h0, CalibrationMethod method,
    GRUQuantizationRanges* ranges, GRUHistogramCollectors* hist,
    const OperatorQuantConfig& config, float* h, float* v) {
    if (steps <= 0 || batch <= 0 || input <= 0 || hidden <= 0)
        throw std::invalid_argument("GRU dimensions must be positive");
    if (method == CalibrationMethod::MINMAX && !ranges)
        throw std::invalid_argument("MINMAX requires quant_ranges");
    if ((method == CalibrationMethod::SQNR || method == CalibrationMethod::PERCENTILE) && !hist)
        throw std::invalid_argument("Histogram calibration requires hist_collectors");
    if (ranges && ranges->hidden_ != hidden) ranges->reset(hidden);
    if (hist && hist->hidden_ != hidden) hist->reset(hidden);
    const int nh = batch * hidden;
    const int columns = 3 * hidden;
    const size_t count = static_cast<size_t>(steps) * nh;
    std::vector<float> wx(count * 3), rh(count * 3);
    std::vector<float> z_pre(count), r_pre(count), n_pre(count);
    std::vector<float> z_values(count), r_values(count), n_values(count);
    std::vector<float> reset_hidden(count), new_contribution(count), old_contribution(count);
    if (h0) std::copy_n(h0, nh, h);
    else std::fill_n(h, nh, 0.0f);

    // Keep GEMM and bias addition separate, matching the calibration sample points.
    for (int t = 0; t < steps; ++t) {
        for (int b = 0; b < batch; ++b) {
            const size_t linear = static_cast<size_t>(t * batch + b) * columns;
            for (int c = 0; c < columns; ++c) {
                float input_sum = 0.0f, hidden_sum = 0.0f;
                for (int k = 0; k < input; ++k)
                    input_sum += x[(t * batch + b) * input + k] * W[k * columns + c];
                for (int k = 0; k < hidden; ++k)
                    hidden_sum += h[t * nh + b * hidden + k] * R[k * columns + c];
                wx[linear + c] = input_sum + bw[c];
                rh[linear + c] = hidden_sum + br[c];
            }
            for (int c = 0; c < hidden; ++c) {
                const size_t index = static_cast<size_t>(t) * nh + b * hidden + c;
                z_pre[index] = wx[linear + c] + rh[linear + c];
                r_pre[index] = wx[linear + hidden + c] + rh[linear + hidden + c];
                const float z = sigmoid(z_pre[index]);
                const float r = sigmoid(r_pre[index]);
                const float recurrent = rh[linear + 2 * hidden + c];
                reset_hidden[index] = r * recurrent;
                n_pre[index] = wx[linear + 2 * hidden + c] + reset_hidden[index];
                const float n = std::tanh(n_pre[index]);
                z_values[index] = z;
                r_values[index] = r;
                n_values[index] = n;
                old_contribution[index] = z * h[index];
                new_contribution[index] = (1.0f - z) * n;
                h[index + nh] = old_contribution[index] + new_contribution[index];
                if (v) {
                    const size_t base = static_cast<size_t>(t * batch + b) * hidden * 4 + c;
                    v[base] = z;
                    v[base + hidden] = r;
                    v[base + 2 * hidden] = n;
                    v[base + 3 * hidden] = recurrent;
                }
            }
        }
    }
    if (method == CalibrationMethod::NONE) return;
    if (method != CalibrationMethod::MINMAX) ranges = nullptr;

#define OBSERVE(DATA, COUNT, RANGE, HIST) \
    observe(DATA, COUNT, ranges ? &ranges->min_##RANGE##_ : nullptr, \
            ranges ? &ranges->max_##RANGE##_ : nullptr, hist ? &hist->HIST : nullptr)
    OBSERVE(x, static_cast<size_t>(steps) * batch * input, x, x_hist);
    // Preserve the existing stateless calibration contract: do not sample h0.
    OBSERVE(h + nh, count, h, h_hist);
    OBSERVE(wx.data(), wx.size(), Wx, Wx_hist);
    OBSERVE(rh.data(), rh.size(), Rh, Rh_hist);
    OBSERVE(z_pre.data(), count, update_gate_input, update_gate_input_hist);
    OBSERVE(r_pre.data(), count, reset_gate_input, reset_gate_input_hist);
    OBSERVE(n_pre.data(), count, new_gate_input, new_gate_input_hist);
    OBSERVE(z_values.data(), count, update_gate_output, update_gate_output_hist);
    OBSERVE(r_values.data(), count, reset_gate_output, reset_gate_output_hist);
    OBSERVE(n_values.data(), count, new_gate_output, new_gate_output_hist);
    OBSERVE(reset_hidden.data(), count, mul_reset_hidden, mul_reset_hidden_hist);
    OBSERVE(new_contribution.data(), count, mul_new_contribution, mul_new_contribution_hist);
    OBSERVE(old_contribution.data(), count, mul_old_contribution, mul_old_contribution_hist);
#undef OBSERVE
#define PARAMETER(VALUES, ROWS, NAME, GRANULARITY, BIAS) \
    observeParameter(VALUES, ROWS, hidden, ranges ? &ranges->min_##NAME##_ : nullptr, \
        ranges ? &ranges->max_##NAME##_ : nullptr, hist ? &hist->NAME##_hist : nullptr, \
        hist ? &hist->NAME##_tensor_hist : nullptr, hist ? &hist->NAME##_gate_hist : nullptr, \
        config.GRANULARITY, BIAS)
    PARAMETER(W, input, W, W_granularity_, false);
    PARAMETER(R, hidden, R, R_granularity_, false);
    PARAMETER(bw, 1, bw, bw_granularity_, true);
    PARAMETER(br, 1, br, br_granularity_, true);
#undef PARAMETER
}
