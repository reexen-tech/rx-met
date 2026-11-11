# 项目文档

使用现成应用包时，从“安装与模型量化”选择任务。应用包提供 AIMET、QuantGRU 和 QuantLSTM，
安装后即可使用对应算子。修改源码、构建环境或交付制品时，参阅维护者文档。

## 安装与模型量化

| 任务 | 文档 |
| --- | --- |
| 选择 CPU/CUDA 环境、安装应用、启动容器 | [用户使用指南](User_guide.md) |
| 准备数据并运行已有示例 | [KWS 与 ONNX 示例](../examples/README.md) |
| 接入自己的模型 | [PyTorch PTQ/QAT](PyTorch_quantization.md) 或 [已有 ONNX 的 PTQ](ONNX_PTQ_COMPILER.md) |
| 调整位宽、校准策略和循环层设置 | [量化配置参考](Quant_config.md) |
| 导出部署制品、保存和恢复量化模型 | [导出与恢复参考](Export_and_reload.md) |
| 查找输出、日志和配置快照 | [运行目录与日志](Runtime_outputs.md) |

## 模块与接口

| 模块 | 说明与参考 |
| --- | --- |
| `aimet_torch` | [PyTorch 模型量化](PyTorch_quantization.md)、[配置](Quant_config.md)、[导出与恢复](Export_and_reload.md) |
| `aimet_onnx` | [ONNX PTQ、输入要求、编码格式和限制](ONNX_PTQ_COMPILER.md) |
| QuantGRU / QuantLSTM 在整模型中的协作 | [循环算子接入](QuantLSTM_integration.md) |
| 独立 QuantGRU 算子 | [算子接口、配置和构建](../operators/quant-gru/README.md) |
| 独立 QuantLSTM 算子 | [使用入口](../operators/quant-lstm/README.md)、[配置、导出与执行规格](../operators/quant-lstm/docs/README.md) |
| `aimet_common`、原生库及模块关系 | [系统架构](architecture.md)、[Python 包目录](../src/README.md)、[原生运行库](AIMET_ONNX_NATIVE.md) |

## 维护者文档

按开发与交付顺序阅读：

1. [参与贡献](../CONTRIBUTING.md)：开发环境、修改范围、测试和提交要求。
2. [公共运行环境](Shared_environment.md)：环境版本、兼容复用、构建和离线归档。
3. [AIMET ONNX 原生运行库](AIMET_ONNX_NATIVE.md)：原生源码、ABI、构建与冒烟测试。
4. [应用包构建与发布](Release_packaging.md)：应用构建、安装验收和离线交付。
5. [测试说明](../tests/README.md)：自动化回归、设备条件与真实数据验收。

## 架构与历史

- [系统架构](./architecture.md) 说明模块接口、量化数据流、制品关系和发布架构。
- [架构决策](adr/)保存设计背景、取舍与影响；历史决策按其记录的背景阅读，当前操作以用户指南和接口参考为准。
- [LSTM 技术文档](../operators/quant-lstm/docs/README.md#维护者文档)包含执行规格、公式推导和性能验证条件。
- [版本记录](../CHANGELOG.md)记录公开版本和待发布变更。

用户可见行为发生变化时应更新对应指南。实现取舍发生变化时应补充或修订 ADR。
