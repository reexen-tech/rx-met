# AIMET ONNX 原生运行库

rx-met 直接从仓库源码编译 `aimet_common` 和 `aimet_onnx` 所需的原生运行库。
构建过程不会下载预编译 AIMET 动态库，也不会把仓库外的 AIMET 二进制文件放入
wheel。

## 源码归属与基线

`native` 下的实现由 rx-met 维护。最初的 C++/CUDA 源码来自与当前 Python
代码基线一致的 Qualcomm AIMET `2.17.0` tag，commit 为
`0e679b705818df7f408971e34693b8098f0971a9`，导入时未修改内容。

初始源码映射如下：

- `ModelOptimizations/DlQuantization` -> `native/common`；
- `ModelOptimizations/PyModelOptimizations` ->
  `native/common/bindings/python`；
- `TrainingExtensions/onnx/src` -> `native/onnx/src`。

该映射只记录来源，不表示后续同步契约。原生实现可以按 rx-met 需求直接修改，并按
运行职责组织，而不继续镜像上游 AIMET 目录。所有改动均由 Git 跟踪，因此不再维护
单独的 AIMET 源码清单。

`native/third_party/onnxruntime` 保存 ONNX Runtime 1.23.2 头文件，供本地独立
构建使用，并保留许可证、第三方声明、版本和 `SHA256SUMS`。Docker 多 CUDA 构建会
按变体准备与运行时一致的头文件：`cu118=1.20.1`、`cu126=1.23.2`、
`cu130=1.27.0`。下载的官方源码归档必须通过固定 SHA-256 后才能用于编译，避免
custom-op 请求比实际运行库更新的 ORT C API。

## 构建产物

原生构建会向 `aimet_common` 安装三个运行文件：

- `_libpymo<EXT_SUFFIX>`；
- `libquant_info<EXT_SUFFIX>`；
- `libaimet_onnxrt_ops.so`。

支持的 ABI 是 Linux x86_64 上的 CPython 3.10-3.12。构建依赖包括 C++17
编译器、CMake、Ninja、Eigen3、pybind11 和匹配版本的 ONNX Runtime 头文件；
CUDA 构建还需要 CUDA Toolkit。

## 构建与冒烟测试

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

直接运行 `build_aimet_native.sh` 默认使用仓库内已归档的 1.23.2 头文件，不访问
网络。环境镜像构建由 `prepare_onnxruntime_headers.sh` 负责准备其他版本头文件。

支持的环境变量：

- `RX_MET_ENABLE_CUDA`：`0` 表示 CPU，`1` 表示 CUDA；
- `RX_MET_AIMET_BUILD_DIR`：CMake 构建目录；
- `RX_MET_AIMET_INSTALL_DIR`：安装目录，默认是仓库中的 `src/aimet_common`；
- `RX_MET_PYTHON`：Python 解释器，默认 `python3`；
- `RX_MET_CUDA_ARCHITECTURES`：CMake CUDA 架构列表；
- `RX_MET_ONNXRUNTIME_ROOT`：指定 ONNX Runtime 头文件根目录；
- `RX_MET_ONNXRUNTIME_VERSION`：要求头文件匹配的 ONNX Runtime 版本。

`scripts/internal/build_wheels_in_container.sh` 会先执行该原生构建，再构建 rx-met
wheel。产物按目标环境分别是 `cp310-cp310-linux_x86_64` 或
`cp312-cp312-linux_x86_64`；缺少任何原生文件都会使 wheel 构建失败。
