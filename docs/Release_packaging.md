# rx-met Docker 构建与发布

rx-met 自行维护 Linux x86_64 GPU 镜像，不再依赖旧通用 Docker。构建
流程将稳定的第三方环境与频繁变化的项目源码分开，最终交付三个产品镜像。

## 1. 构建矩阵

机器可读的唯一矩阵位于 `docker/docker-bake.hcl`：

| 变体 | NVIDIA devel/runtime | Python | Torch 三件套 | ONNX Runtime GPU | CUDA 架构 | 严格最低驱动 |
| --- | --- | --- | --- | --- | --- | --- |
| `cu118` | 11.8.0 cuDNN 8 / Ubuntu 22.04 | 3.10 | 2.7.1 / 0.22.1 / 2.7.1 | 1.20.1 | 80、86、89、90 | 520.61.05 |
| `cu126` | 12.6.3 / Ubuntu 22.04 | 3.10 | 2.8.0 / 0.23.0 / 2.8.0 | 1.23.2 | 80、86、89、90 | 560.35.05 |
| `cu130` | 13.0.3 / Ubuntu 24.04 | 3.12 | 2.10.0 / 0.25.0 / 2.10.0 | 1.27.0 | 80、86、89、90、120 | 580.126.20 |

上述驱动值是 CUDA 发行版的严格下限。只在较新驱动上验收，不能证明镜像在
声明的最低驱动上兼容。

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
| `environment/build.sh` | 构建缺失的稳定环境镜像，直接运行时默认导出全部三个变体 |
| `environment/verify.sh` | 验证环境身份、工具链、依赖版本和平台 |
| `environment/export.sh` | 在本地生成可直接搬运的环境归档和校验文件 |
| `environment/load.sh` | 校验归档 SHA-256 后加载并验证环境 |
| `release/build.sh` | 解析环境、编译当前源码、组装和导出产品镜像 |
| `release/verify_bundle.sh` | 验证最终产品镜像及可选 GPU 流程 |
| `release/verify_kws_example.sh` | 使用真实 Speech Commands 数据完整运行 KWS/QAT 用例 |
| `release/verify_onnx_ptq_example.sh` | 使用指定 ONNX 模型和校准数据完整运行 PTQ 用例 |

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

不传参数时默认处理 `cu118`、`cu126`、`cu130`，构建本机缺失的镜像，验证后
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

只有依赖矩阵、Torch、CUDA、Python、Ubuntu 或编译工具链发生变化时，
才创建新的环境版本并重建。普通 AIMET、QuantGRU、examples 或产品版本变化不
应使环境镜像失效。

## 4. 本地导出和人工搬运

环境构建脚本默认在本地生成完整、校验过、可直接复制的文件：

```bash
./scripts/environment/build.sh
```

只准备本机环境镜像、不生成归档时使用 `--no-export`。

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
后再替换正式文件。它不负责上传到远端或共享存储。

维护人检查本地结果后，人工复制或移动整个目录，并在目标位置再次校验：

```bash
cd /path/to/environment-images/deps-v1
sha256sum --check --strict SHA256SUMS
```

上传共享存储时建议由人工使用 `.partial` 文件名，复制完成并核对 SHA-256 后再
重命名为正式文件。是否允许直接生成归档，应遵守目标存储的使用规范。

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

加载器要求归档目录存在导出脚本生成的 `SHA256SUMS`。本机已有同名环境 tag 时默认
拒绝覆盖，只有人工确认后才能使用 `--force`。

## 5. 产品发布决策

`release/build.sh` 对每个所选 CUDA 变体执行以下规则：

1. build-env 和 runtime-env 都在本机：验证并直接复用。
2. 两者都不在本机，且传入环境归档：校验、加载并验证。
3. 两者都不在本机，也没有归档：自动调用 `environment/build.sh`。
4. 只存在其中一个：立即退出，要求使用 `--rebuild-environment` 成对重建。
5. 显式提供的归档缺失或校验失败：立即退出，不回退到公网构建。

常用命令：

```bash
# 自动解析环境，构建全部产品镜像并导出
./scripts/release/build.sh

# 只构建 cu126，不导出 tar.zst
./scripts/release/build.sh --no-export cu126

# 从指定目录加载缺失的环境
./scripts/release/build.sh \
  --environment-dir /path/to/environment-images/deps-v1 \
  cu118 cu126 cu130

# 强制刷新环境后发布
./scripts/release/build.sh --rebuild-environment cu126

# 环境不存在时禁止自动联网构建
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

该目录可以位于本地磁盘或已挂载的共享存储。目录必须包含当前环境版本的完整归档、
manifest 和 SHA-256 校验文件。首次构建可直接运行 `environment/build.sh`，
生成的新归档由维护人按部署环境的存储规范放入归档目录。

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
| `RX_MET_EXPORT_IMAGES` | `1` | `0` 时构建但不导出 |
| `RX_MET_BUILDER` | 当前 builder | 显式选择 Buildx builder |
| `RX_MET_ZSTD_THREADS` | `2` | 产品归档压缩线程数 |

环境归档使用本机 Docker image store，因此环境构建、加载和产品发布均要求
Buildx 使用 `docker` driver。隔离的 `docker-container`、remote 或 Kubernetes
driver 当前不受支持，脚本会在构建前退出。

## 7. 产品制品和验收

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
其内容与镜像内 `/opt/rx-met/examples` 一致。所有示例源码和配置文件都加入顶层
`SHA256SUMS`，模型、数据集、缓存和运行输出不进入交付目录。

构建脚本自动执行无 GPU 验收。发布前还必须在目标驱动环境执行：

```bash
./scripts/release/verify_bundle.sh --gpu --gpu-device 0 cu118 cu126 cu130
```

GPU 验收检查 `torch.cuda.is_available()`，运行禁止 CPU 回退的 ONNX Runtime CUDA
session，并运行 AIMET v2 和 QuantGRU CUDA forward/backward。这是快速冒烟检查，
不替代真实数据的完整用例验证。

KWS 完整流程在目标 GPU 上执行，默认只使用 GPU 0：

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

两个脚本的 `--output-dir` 都有默认值，并会再按 CUDA 变体创建子目录。ONNX 模型
可以来自任意外部模型仓库，该仓库只作为 ONNX PTQ 的只读输入，不参与 KWS 用例。
模型输入名、输入 shape 和校准文件必须一致。

## 8. 代理和多人服务器

只有首次构建或环境版本变化需要访问 NVIDIA、Ubuntu、PyTorch 和 PyPI。源码
版本更新复用环境镜像时不需要下载第三方依赖。需要代理时只设置当前 shell：

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
export HTTP_PROXY=http://proxy.example.com:8080
export ALL_PROXY=socks5://proxy.example.com:1080
./scripts/environment/build.sh
```

脚本不会重启 Docker daemon，不会删除现有镜像，也不会清理公共 Docker/BuildKit
缓存。多人共用 daemon 时，产品验证构建可设置个人
`RX_MET_IMAGE_REPOSITORY`，避免覆盖同名产品 tag。

共享目录的复制、替换和清理由维护人按部署环境的存储规范人工执行。

完整依赖与版本选择依据见
`docs/research/cuda-torch-image-matrix.md`。
