# rx-met

面向 ADA200 的小模型量化工具包：定制 AIMET（`aimet_torch` / `aimet_onnx` / `aimet_common`）+ QuantGRU。

大模型量化不在本仓库。交付物是预编译 wheel 软件包，装进已有的 `ada200_docker`，**不含 Docker 镜像**。

## 核心能力

- PyTorch PTQ / QAT、混合精度、Po2 scale
- QuantGRU 与 ONNX 直量化 PTQ
- ONNX + encodings 导出

## 使用入口

- 客户软件包：[release/README.md](release/README.md)
- 小模型示例：`examples/quick_start.py`
- ONNX PTQ 示例：`examples/onnx_ptq_quick_start.py`
- 量化配置：[examples/config/README.md](examples/config/README.md)
- AIMET 定制说明：[aimet_README.md](aimet_README.md)

## 主要目录

- `aimet_common/`、`aimet_onnx/`、`aimet_torch/`：AIMET 定制组件
- `quant-gru-pytorch/`：QuantGRU 源码与 CUDA 扩展
- `native/aimet/`：AIMET native 源码
- `examples/`：客户示例与配置
- `docker/`：仅用于构建 wheel 的 Builder 镜像
- `scripts/`：native / wheel / Release 构建
- `release/`：客户安装脚本与说明

## 开发构建

仅构建 AIMET wheel（需先编 native）：

```bash
./scripts/build_wheel.sh
```

## 构建离线 Release

版本号打包装时生成，不写死在仓库里：

- 对外标签：`vYYMMDD`（可用 `RX_MET_RELEASE_DATE=260831` 覆盖）
- wheel 内部版本：`YY.M.D`（例如 `260831` → `26.8.31`）

```bash
./scripts/release_build.sh
./scripts/verify_bundle.sh
```

制品默认在 `.release/export/`：

```text
ada200-rx-met-vYYMMDD-linux_x86_64.tar.gz
```

源码基线：`aimet_rx` tag `qwen35-reexen-fullstack-v0.1.0`；打包流程对齐该仓 `zcx` 分支的小模型拆分。

## 许可证

BSD-3-Clause

Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
