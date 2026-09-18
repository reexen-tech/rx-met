# rx-met Docker 构建与发布

rx-met 提供 Linux x86_64 GPU 镜像。构建流程将第三方运行环境与项目源码分开，
生成三个 CUDA 变体的发布镜像。

## 1. 构建矩阵

机器可读的源配置位于 `docker/variants.json`。`python3 scripts/dependencies.py lock`
是 `docker/docker-bake.hcl` 和 `docker/requirements/*.lock` 的唯一维护入口。

| 变体 | NVIDIA devel/runtime | Python | Torch 三件套 | ONNX Runtime GPU | CUDA 架构 | 严格最低驱动 |
| --- | --- | --- | --- | --- | --- | --- |
| `cu118` | 11.8.0 cuDNN 8 / Ubuntu 22.04 | 3.10 | 2.7.1 / 0.22.1 / 2.7.1 | 1.20.1 | 80、86、89、90 | 520.61.05 |
| `cu126` | 12.6.3 / Ubuntu 22.04 | 3.10 | 2.8.0 / 0.23.0 / 2.8.0 | 1.23.2 | 80、86、89、90 | 560.35.05 |
| `cu130` | 13.0.3 / Ubuntu 24.04 | 3.12 | 2.10.0 / 0.25.0 / 2.10.0 | 1.27.0 | 80、86、89、90、120 | 580.126.20 |

上述驱动值是 CUDA 发行版的严格下限。最低驱动兼容性需要在表中对应版本上完成验收。

## 2. 模块边界

构建分为三个独立生命周期：

```text
Dockerfile.environment
  NVIDIA devel -> build-env：工具链 + Torch + 全部第三方依赖
  NVIDIA runtime/base -> runtime-env：Python + Torch + 全部第三方运行依赖

Dockerfile
  build-env   -> 编译当前 rx-met/QuantGRU wheel
  runtime-env -> 安装项目 wheel、examples 和产品元数据 -> 最终镜像
```

对应脚本职责：

| 脚本 | 职责 |
| --- | --- |
| `scripts/environment/build.sh` | 构建缺失的稳定环境镜像，直接运行时默认导出全部三个变体 |
| `scripts/environment/verify.sh` | 验证环境身份、工具链、依赖版本和平台 |
| `scripts/environment/export.sh` | 在本地生成可复制的环境归档和校验文件 |
| `scripts/environment/load.sh` | 校验归档 SHA-256 后加载并验证环境 |
| `scripts/release/build.sh` | 解析环境、编译当前源码、组装和导出发布镜像 |
| `scripts/release/verify_bundle.sh` | 验证发布镜像及可选 GPU 流程 |
| `scripts/release/verify_kws_example.sh` | 使用真实 Speech Commands 数据完整运行 KWS/QAT 用例 |
| `scripts/release/verify_onnx_ptq_example.sh` | 使用指定 ONNX 模型和校准数据完整运行 PTQ 用例 |

Python 包兼容范围定义在 `pyproject.toml`。环境版本、基础镜像、CUDA/Torch/
ONNX Runtime 组合和精确依赖定义在 `docker/variants.json`。执行
`python3 scripts/dependencies.py lock` 会生成 Bake 配置、构建工具锁，以及
`cu118-py310.lock`、`cu126-py310.lock`、`cu130-py312.lock` 三份自包含
运行环境锁。环境版本默认是 `deps-v1`，与产品 `VERSION` 独立。

## 3. 环境镜像

首次准备并导出全部三个环境：

```bash
./scripts/environment/build.sh
```

无参数调用默认处理 `cu118`、`cu126`、`cu130`，构建本机缺失的镜像，验证后
生成本地可搬运归档。已有完整一对时直接复用。强制刷新环境必须显式执行：

```bash
./scripts/environment/build.sh --force --no-export cu126
```

默认 tag：

```text
rx-met-build-env:deps-v1-cu118
rx-met-runtime-env:deps-v1-cu118
rx-met-build-env:deps-v1-cu126
rx-met-runtime-env:deps-v1-cu126
rx-met-build-env:deps-v1-cu130
rx-met-runtime-env:deps-v1-cu130
```

依赖矩阵、Torch、CUDA、Python、Ubuntu 或编译工具链变化时创建新的环境版本。
AIMET、QuantGRU、examples 和产品版本变化继续复用现有环境镜像。

## 4. 本地导出和分发

环境构建脚本默认在本地生成完整、校验过、可直接复制的文件：

```bash
./scripts/environment/build.sh
```

`--no-export` 用于准备本机环境镜像并跳过归档导出。

也可以对已存在的环境镜像单独执行：

```bash
./scripts/environment/export.sh cu118 cu126 cu130
```

默认输出到 `.release/environment-images/deps-v1/`。每个 CUDA 变体只有一个
归档文件，其中包含对应的 build-env 和 runtime-env：

```text
rx-met-environment-deps-v1-cu118-linux-amd64.tar.zst
rx-met-environment-deps-v1-cu126-linux-amd64.tar.zst
rx-met-environment-deps-v1-cu130-linux-amd64.tar.zst
environment-manifest.json
README.md
SHA256SUMS
```

导出脚本先在输出目录的临时子目录生成全部所选文件，通过 zstd 和 SHA-256 校验
后再替换正式文件。独立的发布系统负责上传制品仓库或其他存储系统。

复制或移动整个目录后，应在目标位置再次校验：

```bash
cd /path/to/environment-images/deps-v1
sha256sum --check --strict SHA256SUMS
```

使用非原子写入的存储系统时，先以 `.partial` 文件名上传，完成 SHA-256 校验后再
重命名为正式文件。

加载环境：

```bash
./scripts/environment/load.sh \
  --archive-dir /path/to/environment-images/deps-v1 \
  cu118 cu126 cu130
```

也可以重复传入单文件：

```bash
./scripts/environment/load.sh \
  --archive /path/to/rx-met-environment-deps-v1-cu126-linux-amd64.tar.zst \
  cu126
```

加载器要求归档目录包含导出脚本生成的 `SHA256SUMS`。默认策略保护本机同名环境
tag，`--force` 用于显式覆盖。

## 5. 构建环境解析

`scripts/release/build.sh` 对每个所选 CUDA 变体执行以下规则：

1. 本机存在完整的 build-env 和 runtime-env：验证并直接复用。
2. 本机缺少两个环境镜像且提供环境归档：校验、加载并验证。
3. 本机缺少两个环境镜像且省略环境归档：自动调用 `scripts/environment/build.sh`。
4. 本机环境镜像缺少一个成员：立即退出，并提示使用 `--rebuild-environment` 成对重建。
5. 显式归档缺失或校验失败：立即退出，并关闭公网构建回退。

常用命令：

```bash
# 自动解析环境，构建全部产品镜像并导出
./scripts/release/build.sh

# 构建 cu126 并跳过 tar.zst 导出
./scripts/release/build.sh --no-export cu126

# 从指定目录加载缺失的环境
./scripts/release/build.sh \
  --environment-dir /path/to/environment-images/deps-v1 \
  cu118 cu126 cu130

# 强制刷新环境后发布
./scripts/release/build.sh --rebuild-environment cu126

# 关闭环境自动构建
./scripts/release/build.sh --no-auto-environment cu126
```

环境归档目录由调用方提供，并直接指向具体的环境版本目录：

```text
/path/to/environment-images/deps-v1/
```

```bash
./scripts/release/build.sh \
  --environment-dir /path/to/environment-images/deps-v1 \
  cu118 cu126 cu130

# 也可以通过环境变量指定同一目录
RX_MET_ENV_ARCHIVE_DIR=/path/to/environment-images/deps-v1 \
  ./scripts/release/build.sh cu118 cu126 cu130
```

该目录可以位于本地磁盘或已挂载的远程存储。目录必须包含当前环境版本的完整归档、
manifest 和 SHA-256 校验文件。首次构建可直接运行 `scripts/environment/build.sh`，
再将生成的归档保存到所需位置。

## 6. 版本和配置

产品版本来自仓库根目录 `VERSION`。环境版本和镜像仓库名可独立设置：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `RX_MET_ENV_VERSION` | `deps-v1` | 稳定环境版本 |
| `RX_MET_BUILD_ENV_REPOSITORY` | `rx-met-build-env` | 构建环境仓库名 |
| `RX_MET_RUNTIME_ENV_REPOSITORY` | `rx-met-runtime-env` | 运行环境仓库名 |
| `RX_MET_ENV_ARCHIVE_DIR` | 空 | 环境归档目录 |
| `RX_MET_AUTO_BUILD_ENVIRONMENT` | `1` | 缺少环境且没有归档时自动构建 |
| `RX_MET_IMAGE_REPOSITORY` | `rx-met` | 最终产品镜像仓库名 |
| `RX_MET_EXPORT_DIR` | `.release/export` | 产品归档目录 |
| `RX_MET_EXPORT_IMAGES` | `1` | `0` 时跳过导出 |
| `RX_MET_BUILDER` | 当前 builder | 显式选择 Buildx builder |
| `RX_MET_ZSTD_THREADS` | `2` | 产品归档压缩线程数 |

环境归档使用本机 Docker image store，因此环境构建、加载和产品发布均要求
Buildx 使用 `docker` driver。脚本在检测到 `docker-container`、remote 或 Kubernetes
driver 时退出。

## 7. 发布制品和验收

以产品版本 `1.0.0` 为例：

```text
.release/export/
|-- SHA256SUMS
|-- README.md
|-- ChangeLog.md
|-- image-manifest.json
|-- examples/
|   |-- README.md
|   |-- quick_start_kws.py
|   |-- onnx_ptq_quick_start.py
|   |-- prepare_onnx_ptq_data.py
|   `-- config/
|-- rx-met-v1.0.0-cu118-linux-amd64.tar.zst
|-- rx-met-v1.0.0-cu126-linux-amd64.tar.zst
`-- rx-met-v1.0.0-cu130-linux-amd64.tar.zst
```

`examples/` 从本次构建的产品镜像中提取，供用户在加载镜像前直接审阅使用方法；
其内容与镜像内 `/opt/rx-met/examples` 一致。顶层 `SHA256SUMS` 覆盖所有示例源码和
配置文件。模型、数据集、缓存和运行输出由用户单独管理。

构建脚本自动执行无 GPU 验收。发布前还必须在目标驱动环境执行：

```bash
./scripts/release/verify_bundle.sh --gpu --gpu-device 0 cu118 cu126 cu130
```

GPU 验收检查 `torch.cuda.is_available()`，运行关闭 CPU 回退的 ONNX Runtime CUDA
session，并运行 AIMET v2 和 QuantGRU CUDA forward/backward。该步骤提供快速冒烟
检查，完整发布验收还包括真实数据用例。

KWS 完整流程在目标 GPU 上执行，默认使用 GPU 0：

```bash
./scripts/release/verify_kws_example.sh \
  --dataset-dir /path/to/speech_commands_v0.02 \
  cu118 cu126 cu130
```

ONNX PTQ 完整流程需要显式传入相互匹配的模型和 NPY/NPZ 校准数据：

```bash
./scripts/release/verify_onnx_ptq_example.sh \
  --model /path/to/model.onnx \
  --dataset-dir /path/to/calib \
  cu118 cu126 cu130
```

两个脚本的 `--output-dir` 都有默认值，并会再按 CUDA 变体创建子目录。ONNX PTQ
以外部模型仓库作为只读输入，KWS 用例使用 Speech Commands 数据。模型输入名、输入
shape 和校准文件必须一致。
