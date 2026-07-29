# rx-met

rx-met 是面向模型量化与硬件部署的离线工具包，提供统一的 Docker
运行环境，覆盖大语言模型和 PyTorch 小模型量化流程。

## 核心能力

- 大语言模型从 Hugging Face 权重转换为 GGUF，并执行多种量化。
- 支持 REEX block-64、混合精度、PPL 评估和硬件格式导出。
- 集成 AIMET、PyTorch 和 QuantGRU，支持小模型 PTQ、QAT 与导出。
- 以 CUDA 12.8 单镜像交付，客户机器无需安装 Python、PyTorch 或 CUDA Toolkit。
- Release Bundle 包含 Docker 镜像、示例配置和离线使用文档。

## 使用入口

- 客户离线使用：[release/README.md](release/README.md)
- LLM 配置说明：[examples/config/README.md](examples/config/README.md)
- AIMET 定制说明：[aimet_README.md](aimet_README.md)
- 版本记录：[CHANGELOG.md](CHANGELOG.md)

## 构建离线 Release

构建机需要 Docker、NVIDIA Driver 和可用的 NVIDIA GPU。执行：

```bash
./scripts/release_build.sh
```

脚本会依次完成多阶段镜像构建、运行时验证、镜像导出和 Release Bundle
组装。默认制品位于 `.release/export/`：

```text
rx-met-<version>-release.tar.gz
rx-met-<version>-release.tar.gz.sha256
```

## 主要目录

- `aimet_common/`、`aimet_onnx/`、`aimet_torch/`：AIMET 定制组件。
- `rx_met_llm/`：JSON 驱动的大模型量化流水线和 `rx-met` CLI。
- `quant-gru-pytorch/`：QuantGRU 源码与 CUDA 扩展。
- `llama.cpp/`：GGUF 转换、量化和硬件导出工具。
- `examples/`：客户示例与配置。
- `docker/`：CUDA 12.8 多阶段镜像定义。
- `scripts/`：wheel、native、镜像验证和 Release 构建脚本。

## 开发构建

仅构建 AIMET wheel：

```bash
./scripts/build_wheel.sh
```

仅运行镜像验收：

```bash
RX_MET_IMAGE=rx-met:<version> ./scripts/verify_image.sh
```

## 许可证

BSD-3-Clause

Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
