# 量化算子

本目录保存 rx-met 自行维护的量化算子。量化流程使用这些算子替换框架中的原始网络层。

## 目录说明

- `quant-gru/` 实现 QuantGRU 的 C++、CUDA、Python binding、配置、测试和文档。
- `quant-lstm/` 实现 QuantLSTM 的 C++、CUDA、Python binding、配置、测试、构建脚本和文档，
  独立构建与安装见 [QuantLSTM README](quant-lstm/README.md)，
  工具包接入见 [QuantLSTM 集成](../docs/QuantLSTM_integration.md)。

模型识别和替换逻辑位于 `src/aimet_torch/`。算子的计算和量化实现保留在本目录，
通过各自的 Python 接口提供能力。

QuantGRU 和 QuantLSTM 作为同级目录维护。GRU 和 LSTM 形成稳定共享接口后，再提取公共
实现。
