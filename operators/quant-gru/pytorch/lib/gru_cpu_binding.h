#pragma once

#include "gru_quant_cpu.h"

inline void check_cpu_tensor(const torch::Tensor& value, torch::ScalarType dtype,
                             const char* name) {
    TORCH_CHECK(value.defined() && value.device().is_cpu() && value.scalar_type() == dtype,
                name, " must be a CPU tensor with dtype ", dtype);
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

inline void check_cpu_gru(int t, int b, int i, int h, const torch::Tensor& W,
    const torch::Tensor& R, const torch::Tensor& bw, const torch::Tensor& br,
    const torch::Tensor& x, const torch::Tensor& h0, bool integer_input = false) {
    TORCH_CHECK(t > 0 && b > 0 && i > 0 && h > 0, "GRU dimensions must be positive");
    check_cpu_tensor(W, torch::kFloat32, "W");
    check_cpu_tensor(R, torch::kFloat32, "R");
    check_cpu_tensor(bw, torch::kFloat32, "bw");
    check_cpu_tensor(br, torch::kFloat32, "br");
    check_cpu_tensor(x, integer_input ? torch::kInt32 : torch::kFloat32, "x");
    TORCH_CHECK(W.sizes() == torch::IntArrayRef({i, 3 * h}), "Invalid W shape");
    TORCH_CHECK(R.sizes() == torch::IntArrayRef({h, 3 * h}), "Invalid R shape");
    TORCH_CHECK(bw.sizes() == torch::IntArrayRef({3 * h}) && br.sizes() == bw.sizes(),
                "Invalid bias shape");
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({t, b, i}), "Invalid x shape");
    if (h0.defined() && h0.numel()) {
        check_cpu_tensor(h0, integer_input ? torch::kInt32 : torch::kFloat32, "h0");
        TORCH_CHECK(h0.sizes() == torch::IntArrayRef({b, h}), "Invalid h0 shape");
    }
}

std::tuple<torch::Tensor, torch::Tensor> forward_fp_wrapper(bool is_training,
    int t, int b, int i, int hidden, const torch::Tensor& W, const torch::Tensor& R,
    const torch::Tensor& bw, const torch::Tensor& br, const torch::Tensor& x,
    const torch::Tensor& h0) {
    TORCH_CHECK(!is_training, "CPU GRU supports inference and calibration, not training");
    check_cpu_gru(t, b, i, hidden, W, R, bw, br, x, h0);
    auto h = torch::empty({t + 1, b, hidden}, x.options());
    auto v = torch::empty({t, b, 4 * hidden}, x.options());
    forwardWithCalibrationCPU(t, b, i, hidden, W.data_ptr<float>(), R.data_ptr<float>(),
        bw.data_ptr<float>(), br.data_ptr<float>(), x.data_ptr<float>(),
        h0.defined() && h0.numel() ? h0.data_ptr<float>() : nullptr,
        CalibrationMethod::NONE, nullptr, nullptr, OperatorQuantConfig(),
        h.data_ptr<float>(), v.data_ptr<float>());
    return {h, v};
}

std::tuple<torch::Tensor, torch::Tensor> forward_calibrate_wrapper(bool,
    int t, int b, int i, int hidden, const torch::Tensor& W, const torch::Tensor& R,
    const torch::Tensor& bw, const torch::Tensor& br, const torch::Tensor& x,
    const torch::Tensor& h0, const std::string& method, const OperatorQuantConfigPy& config,
    GRUQuantizationRangesPy* ranges = nullptr, GRUHistogramCollectorsPy* hist = nullptr) {
    check_cpu_gru(t, b, i, hidden, W, R, bw, br, x, h0);
    auto calibration = stringToCalibMethod(method);
    TORCH_CHECK(calibration != CalibrationMethod::NONE, "Calibration method is required");
    auto h = torch::empty({t + 1, b, hidden}, x.options());
    auto v = torch::empty({t, b, 4 * hidden}, x.options());
    forwardWithCalibrationCPU(t, b, i, hidden, W.data_ptr<float>(), R.data_ptr<float>(),
        bw.data_ptr<float>(), br.data_ptr<float>(), x.data_ptr<float>(),
        h0.defined() && h0.numel() ? h0.data_ptr<float>() : nullptr, calibration,
        ranges ? &ranges->cpp_ranges : nullptr, hist ? &hist->gpu_collectors : nullptr,
        config.to_cpp(), h.data_ptr<float>(), v.data_ptr<float>());
    return {h, v};
}

py::tuple forward_quant_cpu(bool is_training, int t, int b, int i, int hidden,
    const torch::Tensor& W, const torch::Tensor& R, const torch::Tensor& bw,
    const torch::Tensor& br, const torch::Tensor& x, const torch::Tensor& h0,
    const GRUQuantParamsPy& parameters, bool float_storage) {
    TORCH_CHECK(!is_training, "CPU GRU supports PTQ inference, not QAT training");
    check_cpu_gru(t, b, i, hidden, W, R, bw, br, x, h0);
    auto params = parameters.to_cpp();
    TORCH_CHECK(params.hidden_ == hidden, "Quantization hidden size mismatch");
    const bool pot = params.bitwidth_config_.usePOT2_;
    auto options = x.options().dtype(torch::kInt32);
    auto Wq = torch::empty(W.sizes(), options), Rq = torch::empty(R.sizes(), options);
    auto bwq = torch::empty(bw.sizes(), options), brq = torch::empty(br.sizes(), options);
    quantitativeWeightCPU(i, hidden, W.data_ptr<float>(), R.data_ptr<float>(),
        bw.data_ptr<float>(), br.data_ptr<float>(), params, Wq.data_ptr<int32_t>(),
        Rq.data_ptr<int32_t>(), bwq.data_ptr<int32_t>(), brq.data_ptr<int32_t>());
    auto xq = torch::empty(x.sizes(), options);
    auto hq = torch::empty({t + 1, b, hidden}, options);
    auto vq = torch::empty({t, b, 4 * hidden}, options);
    for (int64_t k = 0; k < x.numel(); ++k)
        xq.data_ptr<int32_t>()[k] = quantize(x.data_ptr<float>()[k],
            toFixedScale(params.x_, pot), params.x_.zero_point, params.bitwidth_config_.x_);
    for (int k = 0; k < b * hidden; ++k)
        hq.data_ptr<int32_t>()[k] = h0.defined() && h0.numel()
            ? quantize(h0.data_ptr<float>()[k], toFixedScale(params.h_, pot),
                       params.h_.zero_point, params.bitwidth_config_.h_)
            : params.h_.zero_point;
    cpu::ForwardPassQuantCPU forward(true, b, i, hidden);
    forward.setRescaleParam(params);
    forward.Run(t, Wq.data_ptr<int32_t>(), Rq.data_ptr<int32_t>(), bwq.data_ptr<int32_t>(),
        brq.data_ptr<int32_t>(), xq.data_ptr<int32_t>(), hq.data_ptr<int32_t>(),
        vq.data_ptr<int32_t>(), 0.0f, nullptr);
    auto h = torch::empty(hq.sizes(), x.options());
    auto v = torch::empty(vq.sizes(), x.options());
    for (int64_t k = 0; k < h.numel(); ++k)
        h.data_ptr<float>()[k] = dequantize(hq.data_ptr<int32_t>()[k],
            toFixedScale(params.h_, pot), params.h_.zero_point);
    const QuantParam grids[] = {params.update_gate_output_, params.reset_gate_output_,
                              params.new_gate_output_, params.weight_hh_linear_};
    for (int64_t k = 0; k < v.numel(); ++k) {
        const auto& grid = grids[(k / hidden) % 4];
        v.data_ptr<float>()[k] = dequantize(vq.data_ptr<int32_t>()[k],
            toFixedScale(grid, pot), grid.zero_point);
    }
    py::tuple result(18);
    result[0] = h; result[1] = v;
    const torch::Tensor codes[] = {Wq, Rq, bwq, brq, xq};
    for (int k = 0; k < 5; ++k)
        result[k + 2] = float_storage ? codes[k].to(torch::kFloat32) : codes[k];
    for (int k = 7; k < 18; ++k) result[k] = torch::empty({0}, x.options().dtype(torch::kUInt8));
    return result;
}

py::tuple forward_quant_int_storage_wrapper(bool training, int t, int b, int i, int h,
    const torch::Tensor& W, const torch::Tensor& R, const torch::Tensor& bw,
    const torch::Tensor& br, const torch::Tensor& x, const torch::Tensor& h0,
    const GRUQuantParamsPy& params) {
    return forward_quant_cpu(training, t, b, i, h, W, R, bw, br, x, h0, params, false);
}

py::tuple forward_quant_float_storage_wrapper(bool training, int t, int b, int i, int h,
    const torch::Tensor& W, const torch::Tensor& R, const torch::Tensor& bw,
    const torch::Tensor& br, const torch::Tensor& x, const torch::Tensor& h0,
    const GRUQuantParamsPy& params) {
    return forward_quant_cpu(training, t, b, i, h, W, R, bw, br, x, h0, params, true);
}

std::tuple<torch::Tensor, torch::Tensor> forward_quantized_int_io_wrapper(
    int t, int b, int i, int hidden, const torch::Tensor& W, const torch::Tensor& R,
    const torch::Tensor& bw, const torch::Tensor& br, const torch::Tensor& x,
    const torch::Tensor& h0, const GRUQuantParamsPy& parameters) {
    check_cpu_gru(t, b, i, hidden, W, R, bw, br, x, h0, true);
    auto params = parameters.to_cpp();
    TORCH_CHECK(params.hidden_ == hidden, "Quantization hidden size mismatch");
    auto Wq = torch::empty(W.sizes(), x.options()), Rq = torch::empty(R.sizes(), x.options());
    auto bwq = torch::empty(bw.sizes(), x.options()), brq = torch::empty(br.sizes(), x.options());
    quantitativeWeightCPU(i, hidden, W.data_ptr<float>(), R.data_ptr<float>(),
        bw.data_ptr<float>(), br.data_ptr<float>(), params, Wq.data_ptr<int32_t>(),
        Rq.data_ptr<int32_t>(), bwq.data_ptr<int32_t>(), brq.data_ptr<int32_t>());
    auto hq = torch::full({t + 1, b, hidden}, params.h_.zero_point, x.options());
    if (h0.defined() && h0.numel()) hq[0].copy_(h0);
    cpu::ForwardPassQuantCPU forward(false, b, i, hidden);
    forward.setRescaleParam(params);
    forward.Run(t, Wq.data_ptr<int32_t>(), Rq.data_ptr<int32_t>(), bwq.data_ptr<int32_t>(),
        brq.data_ptr<int32_t>(), x.data_ptr<int32_t>(), hq.data_ptr<int32_t>(), nullptr, 0.0f, nullptr);
    return {hq.slice(0, 1).contiguous(), hq.slice(0, t).contiguous()};
}
