# rx-met

面向 ADA200 的小模型量化工具包：定制 AIMET（`aimet_torch` / `aimet_onnx` / `aimet_common`）+ QuantGRU。

大模型量化不在本仓库。交付物是预编译 wheel 软件包，装进已有的 `ada200_docker`，**不含 Docker 镜像**。

## 核心能力

- PyTorch PTQ / QAT、混合精度、Po2 scale
- QuantGRU 与 ONNX 直量化 PTQ
- ONNX + encodings 导出

## 使用入口

- 客户软件包：[release/README.md](release/README.md)
- 小模型示例：`examples/quick_start_kws.py`
- ONNX PTQ 示例：`examples/onnx_ptq_quick_start.py`
- 量化配置：[examples/config/README.md](examples/config/README.md)

## 主要目录

- `aimet_common/`、`aimet_onnx/`、`aimet_torch/`：AIMET 定制组件
- `quant-gru-pytorch/`：QuantGRU 源码与 CUDA 扩展
- `native/aimet/`：AIMET native 源码
- `examples/`：客户示例与配置
- `docker/`：团队开发 / 打包镜像（不替代官方 `ada200_docker:latest`）
- `scripts/`：native / wheel / Release 构建
- `release/`：客户安装脚本与说明

## 团队开发与打包镜像

官方运行时仍是 `ada200_docker:latest`。团队内部用派生镜像 `ada200_docker:rx-met-dev`（统一 Docker + 编译工具）：

```bash
./scripts/build_dev_image.sh
```

- `docker/Dockerfile`：在运行时底图上加编译工具，产出 `ada200_docker:rx-met-dev`
- `docker/Dockerfile.ada200`：本机没有官方镜像时，按同一契约造运行时底图（Ubuntu 22.04、Python 3.10、`torch==2.8.0+cu128`）

有 `ada200_docker:latest` 就直接用它当底图。最终只打 `ada200_docker:rx-met-dev`，**不会**覆盖官方 `ada200_docker:latest`。

## 构建离线 Release

组件版本写在仓库根目录 `VERSION`（当前 `1.0.0`），用于 wheel 和共享盘目录。
软件包文件名用 SDK 日期标签 `vYYMMDD`（`RX_MET_RELEASE_DATE`，默认当天）。

```bash
./scripts/build_dev_image.sh
./scripts/release_build.sh
./scripts/verify_bundle.sh
```

`release_build.sh` 检测到没有 `ada200_docker:rx-met-dev` 时会退出并提示先跑构建脚本。客户包仍装进官方 `ada200_docker:latest`。

制品默认在 `.release/export/`：

```text
ada200-rx-met-vYYMMDD-linux_x86_64.tar.gz
```

源码基线：`aimet_rx` tag `qwen35-reexen-fullstack-v0.1.0`；打包流程对齐该仓 `zcx` 分支的小模型拆分。

## 许可证

BSD-3-Clause

Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
