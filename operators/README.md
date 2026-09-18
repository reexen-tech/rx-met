# 量化算子

本目录保存 rx-met 自行维护的量化算子。量化流程使用这些算子替换框架中的原始网络层。

## 目录说明

- `quant-gru/` 实现 QuantGRU 的 C++、CUDA、Python binding、配置、测试和文档。

模型识别和替换逻辑位于 `src/aimet_torch/`。算子的计算和量化实现保留在本目录，
通过各自的 Python 接口提供能力。

未来的 QuantLSTM 应作为同级目录加入。GRU 和 LSTM 只有在共享实现形成稳定接口后，
才提取公共代码。
