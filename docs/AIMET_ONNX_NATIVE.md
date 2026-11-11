# AIMET ONNX 原生运行库

rx-met 直接从仓库源码编译 `aimet_common` 和 `aimet_onnx` 所需的原生运行库。
wheel 中的 AIMET 动态库全部来自本次源码构建。

## 源码归属与基线

`native` 下的实现由 rx-met 维护。最初的 C++/CUDA 源码来自与当前 Python
代码基线一致的 Qualcomm AIMET [`2.17.0`](https://github.com/qualcomm/aimet/tree/0e679b705818df7f408971e34693b8098f0971a9)
tag，commit 为 `0e679b705818df7f408971e34693b8098f0971a9`，导入时未修改内容。对应的
BSD-3-Clause 许可证保存在 `native/LICENSE`。

初始源码映射如下：

- `ModelOptimizations/DlQuantization` -> `native/common`；
- `ModelOptimizations/PyModelOptimizations` ->
  `native/common/bindings/python`；
- `TrainingExtensions/onnx/src` -> `native/onnx/src`。

该映射记录初始源码来源。后续实现按 rx-met 的运行职责组织，所有变更由 Git 记录。

`docker/environment/third_party/onnxruntime` 保存 ONNX Runtime 1.23.2 头文件，供本地独立
构建使用，并保留许可证、第三方声明、版本和 `SHA256SUMS`。Docker 构建会
按变体准备与运行时一致的头文件：`cpu=1.23.2`、`cu118=1.20.1`、
`cu126=1.23.2`、`cu130=1.27.0`。当前变体优先使用仓库中已校验的离线头文件；
准备其他版本时，官方源码归档须通过指定的 SHA-256 校验后才能进入编译。

## 构建产物

原生构建会向 `aimet_common` 安装三个运行文件：

- `_libpymo<EXT_SUFFIX>`；
- `libquant_info<EXT_SUFFIX>`；
- `libaimet_onnxrt_ops.so`。

源码构建支持 Linux x86_64 上的 CPython 3.10、3.11 和 3.12。构建依赖包括 C++17
编译器、CMake 3.19 或更新版本、Ninja、Eigen3、pybind11、所用 Python 的开发头文件，
以及匹配版本的 ONNX Runtime 头文件；
CUDA 构建还需要 CUDA Toolkit。发行 wheel 使用的 Python 版本由
[`docker/variants.json`](../docker/variants.json) 定义。

## 构建与冒烟测试

以下命令在源码仓库根目录执行，使用同一 Python 解释器构建和测试。冒烟测试会导入
`aimet_onnx` 并运行 QuantSim，因此除构建工具外还需要匹配的 Python 运行依赖，包括
PyTorch、NumPy、ONNX 和 ONNX Runtime。对应变体的完整依赖见
[`docker/requirements/`](../docker/requirements/)；CPU 使用 `onnxruntime`，CUDA 使用
兼容目标 CUDA/cuDNN 的 `onnxruntime-gpu`，测试时还需有可用 GPU。

命令使用默认安装目录 `src/aimet_common`，供冒烟测试从源码树加载新库。如果通过
`RX_MET_PYTHON` 选择了其他解释器，用同一解释器运行 `smoke_onnx_runtime.py`。

CPU：

```bash
RX_MET_ENABLE_CUDA=0 ./scripts/build_aimet_native.sh
python3 native/tests/smoke_onnx_runtime.py \
  --provider CPUExecutionProvider
```

CUDA：

```bash
RX_MET_ENABLE_CUDA=1 \
RX_MET_CUDA_ARCHITECTURES='80;90' \
./scripts/build_aimet_native.sh
python3 native/tests/smoke_onnx_runtime.py \
  --provider CUDAExecutionProvider
```

测试成功时输出 `AIMET ONNX native smoke passed`，并列出 provider、自定义算子域和加载的库。
该测试使用一个小型 MatMul + Relu 模型，检查原生库导入、校准及有限的推理输出。

直接运行 `build_aimet_native.sh` 默认使用仓库内已归档的 1.23.2 头文件；上述构建依赖已
安装时，无需下载 ORT 源码即可离线构建。这要求当前运行环境使用匹配版本的 ONNX Runtime。其他版本通过
`RX_MET_ONNXRUNTIME_ROOT` 指定头文件目录，并用 `RX_MET_ONNXRUNTIME_VERSION`
要求该目录的版本。例如，已有 cu118 依赖的环境可使用：

```bash
RX_MET_ENABLE_CUDA=1 \
RX_MET_ONNXRUNTIME_ROOT="$PWD/docker/environment/third_party/onnxruntime-prepared/1.20.1" \
RX_MET_ONNXRUNTIME_VERSION=1.20.1 \
./scripts/build_aimet_native.sh
```

环境镜像构建由 `scripts/internal/prepare_onnxruntime_headers.sh` 负责准备所选版本头文件。

支持的环境变量：

- `RX_MET_ENABLE_CUDA`：`0` 表示 CPU（默认），`1` 表示 CUDA；
- `RX_MET_AIMET_BUILD_DIR`：CMake 构建目录，默认 `.release/build/aimet-cpu` 或 `.release/build/aimet-cuda`；
- `RX_MET_AIMET_INSTALL_DIR`：安装目录，默认是仓库中的 `src/aimet_common`；
- `RX_MET_PYTHON`：Python 解释器，默认 `python3`；
- `RX_MET_CUDA_ARCHITECTURES`：CMake CUDA 架构列表；
- `RX_MET_ONNXRUNTIME_ROOT`：指定 ONNX Runtime 头文件根目录；
- `RX_MET_ONNXRUNTIME_VERSION`：要求头文件匹配的 ONNX Runtime 版本。

`scripts/internal/build_wheels_in_container.sh` 会先执行该原生构建，再构建 rx-met
wheel。产物按目标环境分别是 `cp310-cp310-linux_x86_64` 或
`cp312-cp312-linux_x86_64`；缺少任何原生文件都会使 wheel 构建失败。
