# Docker 构建配置

本目录定义 rx-met 的环境镜像、产品镜像和 CUDA 变体。维护人员通过
`scripts/environment/` 和 `scripts/release/` 中的命令使用这些配置。

## 文件说明

- `Dockerfile.environment` 构建可复用的 build-env 和 runtime-env。
- `Dockerfile` 编译当前源码并组装最终产品镜像。
- `variants.json` 记录 CUDA、Python、Torch、ONNX Runtime、基础镜像和精确依赖。
- `docker-bake.hcl` 由 `scripts/dependencies.py lock` 生成，定义 Buildx 目标。
- `requirements/build.lock` 固定 Python 构建工具。
- `requirements/cu*-py*.lock` 分别固定三个产品变体的完整运行依赖。

`variants.json` 是变体和精确版本的权威来源。`scripts/dependencies.py lock` 是
`docker-bake.hcl` 和 requirements lock 的维护入口。
