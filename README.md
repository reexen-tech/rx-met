# rx-met

rx-met 是面向 Linux x86_64 CPU 和 NVIDIA GPU 的小模型量化工具包，提供定制 AIMET、
QuantGRU、QuantLSTM 和可复现的 CPU / 多 CUDA 运行环境。CPU 支持校准与 PTQ；浮点训练和 QAT 使用 GPU。

## 核心能力

- **PyTorch 量化**：支持 PTQ、QAT、混合精度、Power-of-2 scale 和 ONNX 导出。
- **ONNX 量化**：直接读取已有 ONNX，输出 Clean ONNX、AIMET 原始 encodings、
编译器 encodings 和运行元数据。
- **量化循环算子**：QuantGRU / QuantLSTM 可替换 `num_layers=1`、`dropout=0` 的 PyTorch GRU / LSTM，
支持单向、双向、校准和 QAT。
- **可复现环境**：发布 `cu118`、`cu126` 和 `cu130` 三个固定 runtime 镜像及独立应用包。
- **运行记录**：自动创建时间戳目录、详细文件日志、配置快照和版本记录，导出产物统一放入 `exports/`。

## 支持范围


| 镜像变体    | 系统 / Python         | CUDA | PyTorch | ONNX Runtime | 最低 NVIDIA Driver |
| ------- | ------------------- | ---- | ------- | ---------------- | ---------------- |
| `cpu` | Ubuntu 22.04 / 3.10 | 无 | 2.8.0 CPU | 1.23.2 CPU | 无 |
| `cu118` | Ubuntu 22.04 / 3.10 | 11.8 | 2.7.1   | 1.20.1           | 520.61.05        |
| `cu126` | Ubuntu 22.04 / 3.10 | 12.6 | 2.8.0   | 1.23.2           | 560.35.05        |
| `cu130` | Ubuntu 24.04 / 3.12 | 13.0 | 2.10.0  | 1.27.0           | 580.126.20       |


宿主机需要 Docker；GPU 变体另需 NVIDIA Driver 和 NVIDIA Container Toolkit。精确依赖、基础镜像
和 CUDA 架构以 `[docker/variants.json](docker/variants.json)` 为准。

## 获取与运行

选择与宿主机驱动、GPU 架构和依赖匹配的应用包。首次离线交付同时提供对应 runtime 归档，
后续产品更新只需下载应用包。解压后执行：

```bash
./install.sh --offline
```

安装器校验并准备 runtime，在临时容器中离线安装三个 wheel，最后打印应用目录和 Docker 启动命令。
运行命令需将应用目录只读挂到 `/opt/rx-met`，设置 `PYTHONPATH=/opt/rx-met/python`，
并按自己的任务指定 GPU、工作目录和数据目录。完整参数见发布包 README，模板位于
[packaging/release-bundle/README.md.in](packaging/release-bundle/README.md.in)。

按该命令进入容器后，可运行完整 KWS 示例（需要 Speech Commands v0.02）：

```bash
cp -a /opt/rx-met/examples /workspace/rx-met-examples
cd /workspace/rx-met-examples
export RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02
python3 quick_start_kws.py
```

将示例命令中的 `python3 quick_start_kws.py` 改为 `python3 quick_start_kws.py --rnn_type lstm` 即可运行 LSTM。
两种模式共用量化配置，输出根目录分别为 `output/quick_start_kws/gru/` 和 `lstm/`，
每次任务创建独立时间戳目录；
字段说明和完整用法见 [示例说明](examples/README.md)。
日志输出“量化流程完成（已通过加载验证）”且退出码为 0，表示示例运行成功。

## 文档与示例


| 内容                         | 文档                                            |
| -------------------------- | --------------------------------------------- |
| 镜像加载、PyTorch 和 ONNX 使用流程   | [用户使用指南](docs/User_guide.md)                  |
| QuantSim、混合精度和 GRU / LSTM 配置 | [量化配置](docs/Quant_config.md)                  |
| ONNX PTQ 制品和编译器 encodings  | [ONNX PTQ 编译器制品](docs/ONNX_PTQ_COMPILER.md)   |
| 系统模块、量化数据流和发布架构            | [系统架构](docs/architecture.md)                   |
| KWS 和 ONNX PTQ 可运行示例       | [示例说明](examples/README.md)                    |
| 自动日志、运行目录和产物定位 | [运行目录与日志](docs/Runtime_outputs.md) |
| QuantGRU 接口和构建             | [QuantGRU](operators/quant-gru/README.md)     |
| QuantLSTM 集成与示例           | [QuantLSTM 接入](docs/QuantLSTM_integration.md) |
| QuantLSTM 接口和构建           | [QuantLSTM](operators/quant-lstm/README.md) |
| AIMET 原生运行库来源和构建           | [AIMET ONNX 原生运行库](docs/AIMET_ONNX_NATIVE.md) |
| Docker 构建、验证和发布            | [构建与发布](docs/Release_packaging.md)            |




## 从源码构建

以下命令准备环境并构建 `cu126` 应用包与离线 runtime 归档：

```bash
./scripts/environment/build.sh --no-export cu126
./scripts/release/build.sh cu126
```

完整构建矩阵、环境镜像复用和 GPU 验收流程见
[构建与发布](docs/Release_packaging.md)。

## 仓库结构


| 目录                   | 职责                                                   |
| -------------------- | ---------------------------------------------------- |
| `src/`               | `aimet_common`、`aimet_onnx` 和 `aimet_torch` Python 包 |
| `native/`            | AIMET C++/CUDA 原生运行库和 ONNX Runtime custom op         |
| `operators/`         | QuantGRU、QuantLSTM 量化替换算子                                   |
| `examples/`          | PyTorch KWS 和 ONNX PTQ 示例                            |
| `docker/`、`scripts/` | 环境镜像、依赖锁、构建和发布命令                                     |
| `docs/`              | 用户指南、配置说明、维护者文档和架构决策                                 |




## 贡献与反馈

- 公开版本和兼容性变化记录在 [CHANGELOG.md](CHANGELOG.md)。



## 许可证

[BSD-3-Clause](LICENSE)

Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
