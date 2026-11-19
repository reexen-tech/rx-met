# 量化算子

本目录保存 rx-met 自行维护的量化算子。PyTorch 模型需要显式使用这些算子替换原始
网络层并加载权重；`prepare_model()` 保留已有实例，不自动替换任意 `nn.GRU` / `nn.LSTM`。

## 按任务阅读

| 任务 | 入口 |
| --- | --- |
| 使用已安装的 rx-met 接入整模型 | [PyTorch 模型量化](../docs/PyTorch_quantization.md)，无需独立编译算子 |
| 替换循环层、理解与 AIMET 的协作 | [循环算子接入](../docs/QuantLSTM_integration.md) |
| 单独使用、构建或修改算子 | [QuantGRU](quant-gru/README.md)、[QuantLSTM](quant-lstm/README.md) |

## 目录说明

- `quant-gru/` 实现 QuantGRU 的 C++、CUDA、Python binding、配置、测试和文档。
- `quant-lstm/` 实现 QuantLSTM 的 C++、CUDA、Python binding、配置、测试、构建脚本和文档，
  独立构建与安装见 [QuantLSTM README](quant-lstm/README.md)，
  工具包接入见 [循环算子接入](../docs/QuantLSTM_integration.md)。

模型准备和算子注册逻辑位于 `src/aimet_torch/`。算子的计算和量化实现保留在本目录，
通过各自的 Python 接口提供能力。

QuantGRU 和 QuantLSTM 的计算、配置与数值语义分别维护；rx-met 通过公共集成接口
管理生命周期，算子内部行为以各自文档为准。
