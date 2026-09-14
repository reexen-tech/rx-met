# rx-met 脚本说明

本目录包含稳定环境构建、产品发布、镜像验收以及容器内部 wheel 构建脚本。
日常开发和发布不需要逐个运行所有脚本，通常只需要关注下面六个入口。

## 1. 最常用的脚本

### 1.1 `release_build.sh`：产品发布主入口

普通版本发布优先使用这个脚本：

```bash
cd /path/to/rx-met
./scripts/release_build.sh
```

不传 CUDA 参数时，默认依次构建 `cu118`、`cu126`、`cu130` 三个产品镜像。
脚本会：

1. 检查本机是否已有对应的 `build-env` 和 `runtime-env`。
2. 环境镜像存在时直接复用；不存在时从指定归档加载或自动构建。
3. 使用 `build-env` 编译当前源码中的 rx-met、AIMET native 和 QuantGRU。
4. 将项目 wheel 安装到 `runtime-env`，生成最终产品镜像。
5. 执行不使用 GPU 的镜像验收。
6. 导出三个产品镜像归档、说明文件、镜像清单和 `SHA256SUMS`。

默认产品 tag：

```text
rx-met:1.0.0-cu118
rx-met:1.0.0-cu126
rx-met:1.0.0-cu130
```

默认输出目录：

```text
.release/export/
|-- rx-met-v1.0.0-cu118-linux-amd64.tar.zst
|-- rx-met-v1.0.0-cu126-linux-amd64.tar.zst
|-- rx-met-v1.0.0-cu130-linux-amd64.tar.zst
|-- image-manifest.json
|-- README.md
|-- ChangeLog.md
`-- SHA256SUMS
```

只构建某个变体：

```bash
./scripts/release_build.sh cu126
```

只生成本机产品镜像，不导出 `.tar.zst`：

```bash
./scripts/release_build.sh --no-export cu126
```

### 1.2 `build_environment_images.sh`：稳定环境准备入口

只有首次准备环境，或者 CUDA、Torch、Python、Ubuntu、requirements、编译工具链
发生变化时才需要直接运行：

```bash
./scripts/build_environment_images.sh
```

不传参数时默认处理全部三个 CUDA 变体。每个变体生成两个 Docker 镜像：

```text
rx-met-build-env:deps-v1-cu118
rx-met-runtime-env:deps-v1-cu118
rx-met-build-env:deps-v1-cu126
rx-met-runtime-env:deps-v1-cu126
rx-met-build-env:deps-v1-cu130
rx-met-runtime-env:deps-v1-cu130
```

完成验证后，默认在本地生成三个可人工搬运的环境归档：

```text
.release/environment-images/deps-v1/
|-- rx-met-environment-deps-v1-cu118-linux-amd64.tar.zst
|-- rx-met-environment-deps-v1-cu126-linux-amd64.tar.zst
|-- rx-met-environment-deps-v1-cu130-linux-amd64.tar.zst
|-- environment-manifest.json
|-- README.md
`-- SHA256SUMS
```

先试构建单个环境，但不压缩归档：

```bash
./scripts/build_environment_images.sh --no-export cu126
```

已有完整的 build/runtime 环境对时会直接复用。环境只有一半时脚本会停止，必须
人工确认后使用 `--force` 成对重建。

### 1.3 `load_environment_images.sh`：在新机器加载稳定环境

环境归档被人工复制到另一台机器后，使用该脚本校验并加载：

```bash
./scripts/load_environment_images.sh \
  --archive-dir /path/to/environment-images/deps-v1 \
  cu118 cu126 cu130
```

它会检查归档在 `SHA256SUMS` 中的记录，通过后执行 `zstd` 解压和 `docker load`，
最后调用 `verify_environment_images.sh`。一个 CUDA 归档同时包含对应的 build-env
和 runtime-env。

也可以让产品发布脚本在环境缺失时直接加载：

```bash
./scripts/release_build.sh \
  --environment-dir /path/to/environment-images/deps-v1
```

### 1.4 `verify_bundle.sh`：最终产品镜像验收

`release_build.sh` 已自动执行无 GPU 验收。产品发布前还需要在目标 GPU 服务器上
执行：

```bash
./scripts/verify_bundle.sh --gpu --gpu-device 0 cu118 cu126 cu130
```

该脚本检查：

- Torch、torchvision、torchaudio 和 CUDA 版本；
- AIMET Torch、AIMET ONNX、QuantGRU 以及公共依赖能否导入；
- ONNX Runtime GPU provider 和 AIMET ONNX custom op 的最小校准与推理；
- 原生动态库是否存在未解析依赖；
- 使用 `--gpu` 时禁止 CPU 回退的 ONNX Runtime CUDA 推理；
- 使用 `--gpu` 时的 AIMET v2 CUDA 流程；
- 使用 `--gpu` 时的 QuantGRU CUDA forward、backward 和同步。

这属于镜像冒烟验收，不会读取正式 Speech Commands 数据集，也不会运行外部模型
仓库中的完整 ONNX PTQ example。完整数据集验证仍需在目标 GPU 服务器单独执行。

### 1.5 `verify_kws_example.sh`：KWS 完整用例验证

在最终产品镜像中读取真实 Speech Commands 数据，完整执行 FP 训练、PTQ、Po2、
QAT、导出和重新加载验证：

```bash
./scripts/verify_kws_example.sh \
  --dataset-dir /path/to/speech_commands_v0.02 \
  --output-dir /path/to/output/kws \
  cu126
```

`--dataset-dir` 必填。`--output-dir` 不传时默认为
`.release/example-validation/v<VERSION>/quick_start_kws/`。不传 CUDA 变体时默认按
顺序验证三个产品镜像，实际产物分别写入输出根目录的 `cu118/`、`cu126/`、
`cu130/`。脚本默认只向容器开放 GPU 0，可用 `--gpu-device` 选择其他 GPU，避免
占用同一服务器上的其他显卡。

KWS 用例从数据集自行训练模型，因此没有模型输入参数。模型权重、ONNX、encodings
和 `verify.log` 都保存在对应变体的输出目录。

### 1.6 `verify_onnx_ptq_example.sh`：ONNX PTQ 完整用例验证

在最终产品镜像中读取指定 ONNX 和真实 NPY/NPZ 校准样本，完整执行 PTQ、Po2、
产物导出和重新加载验证：

```bash
./scripts/verify_onnx_ptq_example.sh \
  --model /path/to/model.onnx \
  --dataset-dir /path/to/calib \
  --output-dir /path/to/output/onnx-ptq \
  --gpu-device 0 \
  cu126
```

`--model` 和 `--dataset-dir` 必填。模型目录整体以只读方式挂载，以兼容使用外部
权重文件的 ONNX。输出目录默认是
`.release/example-validation/v<VERSION>/onnx_ptq/`，并按 CUDA 变体隔离。

该脚本通过 `--gpus device=<ID>` 只向容器开放指定 GPU，并要求 ONNX Runtime 使用
`CUDAExecutionProvider`。模型输入名和静态 shape 必须与校准样本匹配。多输入模型
每个 NPZ 需包含所有输入，或使用 example 支持的同样本 NPY 命名。默认只向容器
开放 GPU 0，可用 `--gpu-device` 选择其他 GPU，避免占用同机其他显卡。

## 2. 目录约定

```text
scripts/
|-- README.md
|-- release_build.sh
|-- build_environment_images.sh
|-- load_environment_images.sh
|-- export_environment_images.sh
|-- verify_bundle.sh
|-- verify_kws_example.sh
|-- verify_onnx_ptq_example.sh
|-- verify_environment_images.sh
|-- build_aimet_native.sh
|-- internal/
|   |-- build_wheels_in_container.sh
|   |-- build_quant_gru_wheel.sh
|   |-- download_runtime_wheels.sh
|   |-- prepare_onnxruntime_headers.sh
|   |-- prepare_packaging.py
|   `-- verify_dependency_wheelhouse.py
`-- lib/
    `-- environment_images.sh
```

- `scripts/` 根目录是维护人员或开发人员可以直接运行的稳定接口。
- `scripts/internal/` 是 Dockerfile 或其他脚本调用的内部实现，不作为日常命令。
- `scripts/lib/` 只放通过 `source` 引入的函数库，不能作为命令单独运行。

## 3. 两条构建流程

### 3.1 稳定环境流程

```text
build_environment_images.sh
  -> docker/docker-bake.hcl
  -> docker/Dockerfile.environment
     -> internal/download_runtime_wheels.sh
        -> internal/verify_dependency_wheelhouse.py
     -> internal/prepare_onnxruntime_headers.sh
  -> verify_environment_images.sh
  -> export_environment_images.sh
```

稳定环境只包含操作系统、CUDA、Torch、公共 Python 依赖和构建工具，不包含当前
版本的 rx-met、AIMET native 或 QuantGRU wheel。因此它可以跨多个产品版本复用。

### 3.2 产品发布流程

```text
release_build.sh
  -> 复用本机环境，或调用 load_environment_images.sh/build_environment_images.sh
  -> docker/docker-bake.hcl
  -> docker/Dockerfile
     -> internal/build_wheels_in_container.sh
        -> internal/prepare_packaging.py
        -> build_aimet_native.sh
        -> 构建 rx-met wheel
        -> internal/build_quant_gru_wheel.sh
  -> verify_bundle.sh
  -> 导出产品镜像归档
```

产品版本变化时只重新编译项目 wheel 并组装产品镜像，不重新下载和安装稳定依赖。

## 4. 环境镜像脚本

### `export_environment_images.sh`

将本机已有的稳定环境镜像导出成可搬运文件。每个 CUDA 变体对应一个归档，归档
中包含 build-env 和 runtime-env 两个 tag。

```bash
./scripts/export_environment_images.sh cu126
```

脚本在输出目录的临时子目录执行 `docker save`、zstd 压缩和 SHA-256 校验，全部
成功后才替换正式文件。脚本不自动上传；向共享存储或其他机器复制、移动文件由
维护人员人工完成。

### `verify_environment_images.sh`

验证稳定环境本身。默认检查 GPU provider 已安装但不访问 GPU；在 GPU 服务器上应
增加 `--gpu`，实际创建 CUDA session：

```bash
./scripts/verify_environment_images.sh cu126
./scripts/verify_environment_images.sh --gpu --gpu-device 0 cu118 cu126 cu130
```

主要检查镜像平台、环境角色、环境版本、Python、CUDA/Torch、ONNX Runtime GPU
版本、provider 和 `pip check`。build-env
必须包含 `nvcc`、CMake 等编译工具；runtime-env 必须不包含 `nvcc`，也不应提前
安装项目自己的 AIMET/QuantGRU wheel。

### `lib/environment_images.sh`

供其他 Shell 脚本 `source` 的公共函数库，不能作为命令单独执行。它统一定义：

- 环境版本，默认 `deps-v1`；
- build-env/runtime-env 仓库名；
- 镜像 tag 和环境归档命名；
- 环境版本和镜像仓库名校验；
- Buildx 必须使用 `docker` driver 的检查。

## 5. 容器内部构建脚本

下面这些脚本通常不需要维护人员直接运行，由 Dockerfile 在 build-env 中调用。

### `internal/download_runtime_wheels.sh`

由 `Dockerfile.environment` 调用。根据以下锁文件下载完整离线 wheelhouse：

```text
docker/requirements/build.lock
docker/requirements/common.lock
docker/requirements/cu118.txt
docker/requirements/cu126.txt
docker/requirements/cu130.txt
docker/requirements/onnxruntime-cu118.txt
docker/requirements/onnxruntime-cu126.txt
docker/requirements/onnxruntime-cu130.txt
```

如果输出目录已有完整、版本匹配且平台兼容的 wheel，会跳过网络下载。

### `internal/prepare_onnxruntime_headers.sh`

由 `Dockerfile.environment` 调用。它按 CUDA 变体准备与 ONNX Runtime GPU wheel
版本一致的 C/C++ 头文件。cu126 使用仓库内已校验的 1.23.2 头文件；cu118 和
cu130 从 ONNX Runtime 官方 tag 下载源码归档，并在解压前校验固定 SHA-256。
这些头文件只保留在 build-env，供 AIMET custom-op 编译使用。
维护调试时可通过 `RX_MET_ONNXRUNTIME_ARCHIVE` 指定已下载的同版本官方源码归档；
SHA-256 校验仍不会跳过。

### `internal/verify_dependency_wheelhouse.py`

由 `download_runtime_wheels.sh` 调用。它使用 Python 的 packaging/wheel 元数据进行
结构化校验，检查：

- 所有 requirements 都使用精确的 `==` 版本；
- wheel 数量、包名和版本与锁文件完全一致；
- 不允许缺少依赖或存在 lock 之外的额外依赖；
- wheel 文件名与内部 METADATA 一致；
- wheel 与当前 Python ABI 和 Linux x86_64 平台兼容。

### `internal/build_wheels_in_container.sh`

由产品 `Dockerfile` 调用，是容器内项目 wheel 构建的调度入口。它先检查源码布局，
再把源码复制到 `/tmp/rx-met-build`，确保 `prepare_packaging.py` 只修改临时副本，
随后分别构建 rx-met 和 QuantGRU wheel。

### `internal/prepare_packaging.py`

只应针对容器内的临时源码副本执行。它负责：

- 校验产品版本为 `MAJOR.MINOR.PATCH`；
- 将 wheel 版本设置为 `<产品版本>+cu118/cu126/cu130`；
- 根据 CUDA 变体写入匹配的 Torch 三件套依赖；
- 根据 CUDA 变体写入匹配的 ONNX Runtime GPU 依赖；
- 设置 QuantGRU wheel 的 CUDA local version；
- 从小模型发布 wheel 中排除 `rx_met_llm` 和对应命令行入口。

不要直接对日常开发工作区运行该脚本，因为它会修改指定目录中的 `pyproject.toml`、
`VERSION`、`requirements.txt` 和 QuantGRU `_version.py`。

### `build_aimet_native.sh`

使用 CMake、Ninja、pybind11、Eigen 和 CUDA 编译 AIMET native 运行库，主要产物
包括：

```text
aimet_common/_libpymo.cpython-3xx-x86_64-linux-gnu.so
aimet_common/libquant_info.cpython-3xx-x86_64-linux-gnu.so
aimet_common/libaimet_onnxrt_ops.so
```

该脚本也可用于 AIMET native 的独立开发和验证。

### `internal/build_quant_gru_wheel.sh`

使用对应 CUDA Toolkit、CUDA 版 Torch 和目标 SM 架构编译 `quant-gru/`，然后生成
与目标镜像 Python ABI 匹配的 `quant_gru` wheel。没有 `nvcc` 或使用 CPU Torch 时
会明确失败。

## 6. 环境版本和产品版本

产品版本来自仓库根目录 `VERSION`，例如 `1.0.0`。环境版本默认是 `deps-v1`，二者
独立：

```text
rx-met-build-env:deps-v1-cu126  # 稳定依赖环境
rx-met:1.0.0-cu126              # 当前产品
```

普通 rx-met、AIMET、QuantGRU 或 examples 修改不需要升级环境版本。正式环境交付
后，CUDA、Torch、Python、Ubuntu、requirements lock 或工具链发生变化时，应创建
新的环境版本，不要覆盖已经交付的版本。例如：

```bash
RX_MET_ENV_VERSION=deps-v2 ./scripts/build_environment_images.sh
```

## 7. 环境归档和多人服务器约束

- 所有导出脚本只生成可搬运文件，不自动复制或移动到共享存储。
- `--environment-dir` 直接读取指定环境版本目录中的归档，不写共享存储。
- 脚本不会重启 Docker daemon，也不会清理公共 Docker/BuildKit 缓存。
- 多人共用 Docker daemon 时，可设置个人 `RX_MET_IMAGE_REPOSITORY`，避免覆盖他人
  的产品 tag。
- 环境构建、加载和产品发布要求 Buildx 使用 `docker` driver。
- 正式发布前必须校验输出目录中的 `SHA256SUMS`。

完整 Docker 构建矩阵、版本和发布规则见 `docs/Release_packaging.md`。
