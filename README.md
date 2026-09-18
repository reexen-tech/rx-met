# rx-met

rx-met 是面向 Linux x86_64 NVIDIA GPU 的小模型量化工具包，提供定制 AIMET、
QuantGRU 和可复现的多 CUDA 运行环境。

## 核心能力

- **PyTorch 量化**：支持 PTQ、QAT、混合精度、Power-of-2 scale 和 ONNX 导出。
- **ONNX 量化**：直接读取已有 ONNX，输出 Clean ONNX、AIMET 原始 encodings、
编译器 encodings 和运行元数据。
- **量化循环算子**：QuantGRU 可替换 `num_layers=1`、`dropout=0` 的 PyTorch GRU，
支持单向、双向、校准和 QAT。
- **可复现环境**：发布 `cu118`、`cu126` 和 `cu130` 三个独立 Docker 镜像。

## 支持范围


| 镜像变体    | 系统 / Python         | CUDA | PyTorch | ONNX Runtime GPU | 最低 NVIDIA Driver |
| ------- | ------------------- | ---- | ------- | ---------------- | ---------------- |
| `cu118` | Ubuntu 22.04 / 3.10 | 11.8 | 2.7.1   | 1.20.1           | 520.61.05        |
| `cu126` | Ubuntu 22.04 / 3.10 | 12.6 | 2.8.0   | 1.23.2           | 560.35.05        |
| `cu130` | Ubuntu 24.04 / 3.12 | 13.0 | 2.10.0  | 1.27.0           | 580.126.20       |


宿主机需要 Docker、NVIDIA Driver 和 NVIDIA Container Toolkit。精确依赖、基础镜像
和 CUDA 架构以 `[docker/variants.json](docker/variants.json)` 为准。

## 获取与运行

选择与宿主机驱动和 GPU 计算能力匹配的镜像变体。每个版本包含镜像归档、
`SHA256SUMS` 和 `image-manifest.json`。

以 `1.0.0` 的 `cu126` 变体为例：

```bash
sha256sum --check --strict SHA256SUMS
zstd -dc rx-met-v1.0.0-cu126-linux-amd64.tar.zst | docker load

docker run --gpus all --rm \
  rx-met:1.0.0-cu126 \
  python3 -c "import torch, aimet_torch, aimet_onnx, quant_gru; print(torch.__version__, torch.version.cuda)"
```

命令退出码为 0，并打印与镜像变体匹配的 PyTorch 和 CUDA 版本，表示基础环境可用。

完整 KWS 示例需要 Speech Commands v0.02 数据集：

```bash
docker run --gpus all --rm \
  --ipc=host \
  -v /path/to/workspace:/workspace \
  -v /path/to/datasets:/datasets:ro \
  -w /workspace \
  -e RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02 \
  rx-met:1.0.0-cu126 \
  bash -lc 'cp -a /opt/rx-met/examples /workspace/rx-met-examples &&
    cd /workspace/rx-met-examples &&
    python3 quick_start_kws.py'
```

日志输出“量化流程完成（已通过加载验证）”，表示示例运行成功。

## 文档与示例


| 内容                         | 文档                                            |
| -------------------------- | --------------------------------------------- |
| 镜像加载、PyTorch 和 ONNX 使用流程   | [用户使用指南](docs/User_guide.md)                  |
| QuantSim、混合精度和 QuantGRU 配置 | [量化配置](docs/Quant_config.md)                  |
| ONNX PTQ 制品和编译器 encodings  | [ONNX PTQ 编译器制品](docs/ONNX_PTQ_COMPILER.md)   |
| KWS 和 ONNX PTQ 可运行示例       | [示例说明](examples/README.md)                    |
| QuantGRU 接口和构建             | [QuantGRU](operators/quant-gru/README.md)     |
| AIMET 原生运行库来源和构建           | [AIMET ONNX 原生运行库](docs/AIMET_ONNX_NATIVE.md) |
| Docker 构建、验证和发布            | [构建与发布](docs/Release_packaging.md)            |




## 从源码构建

以下命令构建 `cu126` 发布镜像并跳过归档导出：

```bash
./scripts/release/build.sh --no-export cu126
```

完整构建矩阵、环境镜像复用和 GPU 验收流程见
[构建与发布](docs/Release_packaging.md)。

## 仓库结构


| 目录                   | 职责                                                   |
| -------------------- | ---------------------------------------------------- |
| `src/`               | `aimet_common`、`aimet_onnx` 和 `aimet_torch` Python 包 |
| `native/`            | AIMET C++/CUDA 原生运行库和 ONNX Runtime custom op         |
| `operators/`         | QuantGRU 及后续量化替换算子                                   |
| `examples/`          | PyTorch KWS 和 ONNX PTQ 示例                            |
| `docker/`、`scripts/` | 环境镜像、依赖锁、构建和发布命令                                     |
| `docs/`              | 用户指南、配置说明、维护者文档和架构决策                                 |




## 贡献与反馈

- 公开版本和兼容性变化记录在 [CHANGELOG.md](CHANGELOG.md)。



## 许可证

[BSD-3-Clause](LICENSE)

Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
