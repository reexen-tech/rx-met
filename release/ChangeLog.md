# Changelog

## @DATE_TAG@

组件版本：`@SEMVER@`。所属 ADA200 SDK：`@SDK_TAG@`。

- 独立小模型交付：在 `ada200_docker` 上安装，不再附带 Docker 镜像。
- 预编译 `rx-met`（定制 AIMET）和 `quant_gru` wheel，对齐 `torch==2.8.0+cu128`。
- 离线附带 `torchaudio` / `librosa` / `soundfile` / `tqdm` 及其缺失依赖。
- 不包含大模型（`rx_met_llm` / llama.cpp）和 `onnxruntime-gpu`。
- examples 整目录交付。入口为 `quick_start_kws.py`（kws_streaming att_mh_rnn + QuantGRU）和 `onnx_ptq_quick_start.py`。
- `install.sh` 登记 pip 自带的 NVIDIA CUDA runtime，供 AIMET `libpymo` 加载 `libcudart.so.12`。
