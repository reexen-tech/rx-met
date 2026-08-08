# rx-met

rx-met 是面向模型量化与硬件部署的离线工具包，提供统一的 Docker
运行环境，覆盖大语言模型和 PyTorch 小模型量化流程。

## 核心能力

- 大语言模型从 Hugging Face 权重转换为 GGUF，并执行多种量化。
- 支持 REEX block-64、混合精度、PPL 评估和硬件格式导出。
- 集成 AIMET、PyTorch 和 QuantGRU，支持小模型 PTQ、QAT 与导出。
- 以 CUDA 12.8 或纯 CPU 镜像交付，客户机器无需安装 Python、PyTorch 或 CUDA Toolkit。
- Release Bundle 包含 Docker 镜像、示例配置和离线使用文档。

## 使用入口

- 客户离线使用（GPU）：[release/README.md](release/README.md)
- 客户离线使用（CPU）：[release/README.cpu.md](release/README.cpu.md)
- LLM 配置说明：[examples/config/README.md](examples/config/README.md)
- AIMET 定制说明：[aimet_README.md](aimet_README.md)
- 版本记录：[CHANGELOG.md](CHANGELOG.md)

## 构建离线 Release

### GPU 包（默认）

构建机需要 Docker、NVIDIA Driver 和可用的 NVIDIA GPU。执行：

```bash
./scripts/release_build.sh
```

默认制品位于 `.release/export/`：

```text
rx-met-<version>-release.tar.gz
rx-met-<version>-release.tar.gz.sha256
```

### CPU 包

构建机只需 Docker（无需 GPU）。执行：

```bash
./scripts/release_build_cpu.sh
```

制品：

```text
rx-met-<version>-cpu-release.tar.gz
rx-met-<version>-cpu-release.tar.gz.sha256
```

CPU 包不含 QuantGRU；`examples/quick_start.py` 小模型流程不可用，请使用 GPU 包。

脚本会依次完成多阶段镜像构建、运行时验证、镜像导出和 Release Bundle
组装。

## 主要目录

- `aimet_common/`、`aimet_onnx/`、`aimet_torch/`：AIMET 定制组件。
- `rx_met_llm/`：JSON 驱动的大模型量化流水线和 `rx-met` CLI。
- `quant-gru-pytorch/`：QuantGRU 源码与 CUDA 扩展。
- `llama.cpp/`：GGUF 转换、量化和硬件导出工具。
- `examples/`：客户示例与配置。
- `docker/`：CUDA 12.8 与 CPU 多阶段镜像定义（`Dockerfile` / `Dockerfile.cpu`）。
- `scripts/`：wheel、native、镜像验证和 Release 构建脚本（含 `release_build_cpu.sh`）。
- `native/aimet/`：rx-met 自维护的 AIMET native 源码及固定版本的 ORT 头文件。

## 开发构建

仅构建 AIMET wheel：

```bash
./scripts/build_wheel.sh
```

该命令先从仓库内源码编译 `_libpymo`、`libquant_info` 和
`libaimet_onnxrt_ops.so`，再生成 `cp310-cp310-linux_x86_64` wheel。默认
为 CPU 版本；CUDA 版本使用 `RX_MET_ENABLE_CUDA=1`，可通过
`RX_MET_CUDA_ARCHITECTURES` 指定目标架构。完整说明见
[doc/AIMET_ONNX_NATIVE.md](doc/AIMET_ONNX_NATIVE.md)。

仅运行镜像验收：

```bash
RX_MET_IMAGE=rx-met:<version> ./scripts/verify_image.sh
```

## 许可证

BSD-3-Clause

Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
