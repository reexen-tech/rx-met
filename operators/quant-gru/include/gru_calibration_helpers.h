#pragma once

#include "pot_sqnr_calibrator.h"

namespace {

inline void ensureSymmetricWeightBiasConfig(const OperatorQuantConfig &cfg) {
    if (!cfg.W_symmetric_ || !cfg.R_symmetric_ || !cfg.bw_symmetric_ || !cfg.br_symmetric_) {
        throw std::runtime_error(
            "W/R/bw/br must remain symmetric (zp=0). "
            "Non-symmetric weight/bias quantization is not supported in this stage.");
    }
}

inline EncodedScaleResult encodeFromRange(
    float min_val,
    float max_val,
    QuantBitWidth bw,
    bool is_symmetric,
    bool use_pot2,
    PotScaleMethod method = PotScaleMethod::CoverRange,
    float tolerance = 0.02f) {
    const ContinuousScaleResult cont = calibrateContinuousScaleFromRange(min_val, max_val, bw, is_symmetric);
    // MinMax 与直方图共用 encodeScaleResult；策略由 method 决定（默认 CoverRange）
    return encodeScaleResult(cont, bw, is_symmetric, use_pot2, method, tolerance);
}

// Biases have one value per channel. Histogram bin expansion and clipping
// are not a useful training range for them; reserve headroom from observations.
// Imported encodings never pass through this calibration helper.
template <typename RangeAt>
void calibrateBias(ChannelQuantParam& out, int hidden,
                   OperatorQuantConfig::QuantizationGranularity granularity,
                   QuantBitWidth width, bool use_pot2, RangeAt range_at) {
    out.granularity = granularity;
    out.hidden = hidden;
    out.resize(3 * hidden);
    const int group_size = granularity == OperatorQuantConfig::PER_TENSOR ? 3 * hidden
                         : granularity == OperatorQuantConfig::PER_GATE ? hidden : 1;
    for (int begin = 0; begin < 3 * hidden; begin += group_size) {
        double extent = 0.0;
        for (int c = begin; c < begin + group_size; ++c) {
            const auto range = range_at(c);
            if (!std::isfinite(range.first) || !std::isfinite(range.second) ||
                range.first > range.second) {
                throw std::invalid_argument("Bias calibration requires finite observations");
            }
            extent = std::max(extent, std::max(std::abs(static_cast<double>(range.first)),
                                               std::abs(static_cast<double>(range.second))));
        }
        double scale = std::max(2.0 * extent, 0.01) / width.qmax_auto_scale();
        if (use_pot2) scale = std::exp2(std::ceil(std::log2(scale)));
        if (!std::isfinite(scale) || scale > std::numeric_limits<float>::max()) {
            throw std::overflow_error("Bias headroom scale exceeds float32 range");
        }
        std::fill(out.channels.begin() + begin, out.channels.begin() + begin + group_size,
                  QuantParam{static_cast<float>(scale), 0});
    }
}

}  // namespace
