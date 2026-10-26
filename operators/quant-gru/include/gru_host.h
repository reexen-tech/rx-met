#pragma once

#include "gru_quantization_ranges.h"
#include "histogram_collector.h"
#include "quantize_param_types.h"

enum class CalibrationMethod : int8_t {
    NONE = 0,
    MINMAX = 1,
    SQNR = 2,
    PERCENTILE = 3,
};

GRUQuantParams calculateGRUQuantitativeParameters(
    const GRUQuantizationRanges&, const OperatorQuantConfig&);
GRUQuantParams calculateGRUQuantitativeParametersFromHistograms(
    const GRUHistogramCollectors&, const OperatorQuantConfig&, bool, float);

void quantitativeWeightCPU(int input_size, int hidden_size,
    const float* W, const float* R, const float* bw, const float* br,
    const GRUQuantParams&, int32_t* W_q, int32_t* R_q, int32_t* bw_q, int32_t* br_q);
void quantGRUForwardCPU(bool is_training, int time_steps, int batch_size,
    int input_size, int hidden_size, const int32_t* W, const int32_t* R,
    const int32_t* bw, const int32_t* br, const float* x, const float* h0,
    const GRUQuantParams&, float* h, float* v);
void quantGRUForwardCPU(bool is_training, int time_steps, int batch_size,
    int input_size, int hidden_size, const float* W, const float* R,
    const float* bw, const float* br, const float* x, const float* h0,
    const GRUQuantParams&, float* h, float* v);

// Calibration uses floating-point recurrence, independently of training support.
void forwardWithCalibrationCPU(int time_steps, int batch_size, int input_size,
    int hidden_size, const float* W, const float* R, const float* bw, const float* br,
    const float* x, const float* h0, CalibrationMethod method,
    GRUQuantizationRanges* ranges, GRUHistogramCollectors* histograms,
    const OperatorQuantConfig& config, float* h, float* v);
