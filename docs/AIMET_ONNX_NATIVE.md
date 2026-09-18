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

`native/third_party/onnxruntime` 保存 ONNX Runtime 1.23.2 头文件，供本地独立
构建使用，并保留许可证、第三方声明、版本和 `SHA256SUMS`。Docker 多 CUDA 构建会
按变体准备与运行时一致的头文件：`cu118=1.20.1`、`cu126=1.23.2`、
`cu130=1.27.0`。下载的官方源码归档通过固定 SHA-256 校验后进入编译，确保 custom-op
使用与实际运行库匹配的 ORT C API。

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

构建命令和冒烟测试均以退出码 0 结束，并成功创建对应 Execution Provider 的 ONNX
Runtime session，即表示原生运行库通过基础验证。

直接运行 `build_aimet_native.sh` 默认使用仓库内已归档的 1.23.2 头文件，可以离线
完成构建。环境镜像构建由 `prepare_onnxruntime_headers.sh` 负责准备其他版本头文件。

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
