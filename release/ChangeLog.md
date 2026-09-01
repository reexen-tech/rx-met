# Changelog

## @DATE_TAG@

- 独立小模型交付：在 `ada200_docker` 上安装，不再附带 Docker 镜像。
- 预编译 `rx-met`（定制 AIMET）和 `quant_gru` wheel，对齐 `torch==2.8.0+cu128`。
- 离线附带 `torchaudio` / `librosa` / `soundfile` 及其缺失依赖。
- 不包含大模型（`rx_met_llm` / llama.cpp）和 `onnxruntime-gpu`。
- examples：`quick_start.py`、`onnx_ptq_quick_start.py`；ONNX 默认路径为 `/datasets/...`。
- 附带 MRNN 所需的 `examples/data/to_band_matrix_*.pt`。
- `install.sh` 登记 pip 自带的 NVIDIA CUDA runtime，供 AIMET `libpymo` 加载 `libcudart.so.12`。
