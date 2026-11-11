# rx-met

rx-met 是面向 Linux x86_64 CPU 和 NVIDIA GPU 的小模型量化工具包，提供定制 AIMET、
QuantGRU、QuantLSTM 和可复现的 CPU / 多 CUDA 运行环境。CPU 支持校准与 PTQ；浮点训练和 QAT 使用 GPU。

## 核心能力

- **PyTorch 量化**：支持 PTQ、QAT、混合精度、Power-of-2 scale 和 ONNX 导出。
- **ONNX 量化**：直接读取已有 ONNX，输出 Clean ONNX、AIMET 原始 encodings、
  编译器 encodings 和运行元数据。
- **量化循环算子**：QuantGRU / QuantLSTM 可替换 `num_layers=1`、`dropout=0` 的 PyTorch GRU / LSTM，
  支持单向、双向、校准及 CUDA QAT。
- **可复现环境**：提供 `cpu`、`cu118`、`cu126` 和 `cu130` 固定 runtime 镜像及独立应用包。
- **运行记录**：量化示例和 RX 导出流程自动记录日志与产物，默认在时间戳运行目录下使用 `exports/`。

## 获取与运行

宿主机需要 Linux x86_64 和 Docker；CUDA 变体还需要 NVIDIA Driver 和 NVIDIA
Container Toolkit。按[平台与版本要求](docs/User_guide.md#2-支持平台与版本要求)
选择 `cpu`、`cu118`、`cu126` 或 `cu130` 应用包及匹配的 runtime。
精确环境定义见 [docker/variants.json](docker/variants.json)。

离线安装可以复用本机已有的匹配 runtime；本机没有匹配镜像时，准备对应的 runtime 归档。
进入解压后的应用包目录执行：

```bash
./install.sh --offline
```

安装器复用或加载 runtime，在容器中离线安装项目包，并打印应用目录和 Docker 启动命令。
接着按[用户使用指南](docs/User_guide.md#3-加载并启动镜像)挂载应用、工作目录和数据并进入容器，
再按[示例说明](examples/README.md)运行 KWS 或 ONNX PTQ。
CPU 的 KWS 示例需要已有浮点检查点及 `--device cpu --ptq-only`。

## 按任务阅读

| 要完成的任务 | 阅读入口 |
| --- | --- |
| 安装应用、选择设备、启动容器 | [用户使用指南](docs/User_guide.md) |
| 运行 KWS / MobileNetV2 示例 | [示例与数据准备](examples/README.md) |
| 接入自己的 PyTorch 模型 | [PyTorch 模型量化](docs/PyTorch_quantization.md) |
| 量化已有 ONNX 模型 | [ONNX PTQ 与编译器制品](docs/ONNX_PTQ_COMPILER.md) |
| 调整位宽、校准与循环层设置 | [量化配置参考](docs/Quant_config.md) |
| 导出模型、保存权重、恢复量化状态 | [导出与恢复参考](docs/Export_and_reload.md) |
| 开发、构建环境与发布应用包 | [参与贡献](CONTRIBUTING.md)、[构建与发布](docs/Release_packaging.md) |

完整的用户、模块、架构与维护者导航见[文档索引](docs/README.md)。

## 仓库结构

| 目录 | 职责 |
| --- | --- |
| [`src/`](src/README.md) | `aimet_common`、`aimet_onnx` 和 `aimet_torch` Python 包 |
| [`native/`](native/README.md) | AIMET C++/CUDA 原生运行库和 ONNX Runtime custom op |
| [`operators/`](operators/README.md) | QuantGRU、QuantLSTM 量化替换算子 |
| [`examples/`](examples/README.md) | PyTorch KWS 和 ONNX PTQ 示例 |
| [`docker/`](docker/README.md)、[`scripts/`](scripts/README.md) | 环境镜像、依赖锁、构建和发布命令 |
| [`docs/`](docs/README.md) | 用户指南、配置说明、维护者文档和架构决策 |
| [`tests/`](tests/README.md) | 项目回归测试与真实数据验收入口 |

## 贡献与反馈

- 开发环境、测试和提交要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。
- 公开版本和兼容性变化记录在 [CHANGELOG.md](CHANGELOG.md)。

## 许可证

[BSD-3-Clause](LICENSE)

Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
