# quant-lstm 安装指南

quant-lstm 提供 PyTorch CPU/CUDA 模块和 CMake C++ package。CPU Python 模块支持
校准和 PTQ；CUDA 模块另支持浮点训练和 QAT。CMake package 面向独立 C++ 集成，
也支持不依赖 CUDA 的 reference-only 安装。

## 1. 环境要求

| 组件 | 要求 | 用途 |
| --- | --- | --- |
| CMake | 3.24 或更高版本 | 配置原生库和安装 package |
| C++ compiler | 支持 C++17 | 编译 C++ 核心和 reference |
| `nlohmann_json` | 3.11.2 或更高版本 | 配置与参数 JSON |
| CUDA Toolkit | 提供 `nvcc`、cuBLAS 和 CUDA Runtime | CUDA C++ 与 Python 模块 |
| Python | 具有开发头文件 | 构建 PyTorch extension |
| PyTorch | CPU 或 CUDA 版，与 extension 构建环境 ABI 兼容 | Python 接口 |

CPU-only C++ package 不要求 CUDA Toolkit 或 PyTorch。CPU Python 模块需要 CPU
PyTorch，不要求 CUDA Toolkit；通过 `RX_MET_ENABLE_CUDA=0` 选择 CPU extension。
CUDA wheel 不会自动转换为 CPU wheel。

Ubuntu 可以使用系统包提供编译工具和头文件；安装后确认 CMake 和
`nlohmann_json` 满足上表的最低版本要求。若系统源版本较低，使用提供所需版本的
开发环境。CMake 也会尝试从所选 Python 解释器中的 `nlohmann_json` 包查找头文件。

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake nlohmann-json3-dev python3-dev
```

CUDA Toolkit、驱动和 CUDA-enabled PyTorch 需要按照目标 GPU 和 PyTorch 官方安装
矩阵选择。`nvcc --version`、`python -c 'import torch; print(torch.version.cuda)'`
和驱动支持的 CUDA 版本需要兼容。

## 2. 安装 PyTorch CUDA 模块

Python package 支持普通环境安装和 wheel 安装。本文构建命令均从算子根目录执行
（rx-met 中为 `operators/quant-lstm/`），
原生核心默认从 `build/` 目录链接 `libquant_lstm.a`，可通过
`RX_MET_QUANT_LSTM_BUILD_DIR` 指定其他构建目录。

以下 Python 构建命令使用单配置 CMake 生成器。新构建目录默认启用 CUDA、
关闭测试和示例，并使用 `Release`。
普通安装无需显式指定这些开关。需要开发构建时可传入
`-DCMAKE_BUILD_TYPE=Debug`、`-DQUANT_LSTM_BUILD_TESTS=ON` 或
`-DQUANT_LSTM_BUILD_EXAMPLES=ON`；仓库验证脚本会显式开启所需目标。
如果使用多配置生成器，构建时用 `--config Release` 选择配置，并将
`RX_MET_QUANT_LSTM_BUILD_DIR` 指向实际包含 `libquant_lstm.a` 的目录。

已有构建目录保留 CMake 缓存中的设置；需要恢复当前默认值时，使用
`cmake --fresh -S . -B build` 重新配置。

CUDA 构建默认使用 `CMAKE_CUDA_ARCHITECTURES=native`，需要构建环境能检测到目标 GPU。
在无 GPU 的构建机或需要分发到其他 GPU 架构时，配置 CMake 时显式传入
`-DCMAKE_CUDA_ARCHITECTURES=<目标架构列表>`，并确认目标架构受所用 CUDA Toolkit 支持。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 按目标 CUDA 环境安装 CUDA-enabled PyTorch 后执行：
cmake -S . -B build
cmake --build build --parallel
python -m pip install ./pytorch --no-build-isolation
```

`--no-build-isolation` 使 extension 使用当前环境中已经安装且与 CUDA 匹配的
PyTorch。pip 构建并安装包含 Python 模块、`_quant_lstm` native extension 和默认
量化配置的 wheel。安装完成后可以删除或移动源码目录。

需要把 wheel 复制到另一台 ABI 兼容的机器时，先生成制品：

```bash
python -m pip wheel \
  --no-build-isolation \
  --no-deps \
  --wheel-dir dist \
  ./pytorch
```

输出文件类似：

```text
dist/quant_lstm-0.1.0-cp312-cp312-linux_x86_64.whl
```

在目标环境预先安装兼容的 CUDA-enabled PyTorch，再安装 wheel：

```bash
python -m pip install \
  dist/quant_lstm-0.1.0-cp312-cp312-linux_x86_64.whl
```

wheel 绑定构建时的 Python ABI、平台、PyTorch C++ ABI 和 CUDA 依赖，不能跨不兼容
环境复用。项目没有让 pip 自动选择 PyTorch CUDA variant，因此 PyTorch 必须由用户
根据目标环境先行安装。

### CPU Python 模块

在装有 `pip`、`setuptools`、`wheel` 和 CPU PyTorch 的 Python 环境中，
从算子根目录执行：

```bash
cmake -S . -B build_cpu -DQUANT_LSTM_ENABLE_CUDA=OFF
cmake --build build_cpu --parallel 2
RX_MET_ENABLE_CUDA=0 RX_MET_QUANT_LSTM_BUILD_DIR="$PWD/build_cpu" \
  python -m pip install ./pytorch --no-build-isolation --no-deps
```

如需只生成 wheel，保留两个环境变量，将 `pip install` 改为
`pip wheel --wheel-dir dist-cpu`。安装后，模块、输入和状态保持在 CPU，
调用 `eval()` 执行校准和 PTQ。
普通前向无需额外关闭梯度，推荐使用 `torch.no_grad()` 或 `torch.inference_mode()`
减少推理开销。CPU 不提供训练 checkpoint、反向传播或 QAT；实际跨算子的反向传播会明确报错。
构建 rx-met 发布包时使用 `cpu` 变体，三个
wheel 的版本与 CPU PyTorch / ONNX Runtime 锁文件配套。

### 验证 CUDA 安装

在可访问 CUDA GPU 的环境执行验证：

```bash
python - <<'PY'
import torch
from quant_lstm import QuantLSTM

assert torch.cuda.is_available()
module = QuantLSTM(4, 8, batch_first=True, device="cuda").eval()
inputs = torch.randn(2, 3, 4, device="cuda", dtype=torch.float32)
output, (hidden, cell) = module(inputs)
assert output.shape == (2, 3, 8)
assert hidden.shape == cell.shape == (1, 2, 8)
assert torch.isfinite(output).all()
print("QuantLSTM CUDA FP32 forward passed")
PY
```

打印 `QuantLSTM CUDA FP32 forward passed` 且进程退出码为 0 表示 Python 模块、
extension、CUDA Runtime 和 native FP32 forward 已正确加载。该检查使用随机输入，
只验证安装和接口，不用于量化校准。

卸载 package：

```bash
python -m pip uninstall quant-lstm
```

单独使用 QuantLSTM 时可以按照本节命令构建 wheel。通过 rx-met 应用包安装时，
应用包已提供配套 wheel。

### 2.1 升级已有安装

Python 模块和 `_quant_lstm` extension 必须来自同一版本。Po2 修复增加了校准
binding 返回的内存范围信息，升级时需要同时重建扩展并安装 Python 模块：

```bash
cmake -S . -B build
cmake --build build --parallel 2
python -m pip install ./pytorch --force-reinstall --no-deps --no-build-isolation
```

重新生成分发 wheel 时使用上文的 `pip wheel` 命令，并在兼容环境中安装新制品。
rx-met 中的副本由工具包发布流程构建；同步源码后也应更新配套算子 wheel 或镜像。
升级后可从算子根目录执行接口回归，验证校准 binding、Po2 转换和编码回读。
完整测试需要 CUDA 模块、可见 GPU 和 `onnx`：

```bash
(cd pytorch && python -m unittest -v tests.test_aimet_interface)
```

### 2.2 ONNX Runtime PTQ Bridge

可选的 `quant_lstm_onnx_runtime` bridge 在构建时读取 rx-met 公共目录
`docker/environment/third_party/onnxruntime/include`。独立仓库构建时，通过
`QUANT_LSTM_ONNXRUNTIME_INCLUDE` 指向包含 `onnxruntime_cxx_api.h` 的目录后重建
wheel。未提供头文件时不构建 bridge；普通 PyTorch 执行和标准 ONNX 导出不依赖它。

bridge 注册时请求构建头文件中的 `ORT_API_VERSION`，运行时必须支持该 API。
仓库默认头文件为 ORT 1.23.2（API 23）。目标运行时只支持较低的 API 时，
需要设置 `QUANT_LSTM_ONNXRUNTIME_INCLUDE` 并重建 wheel；
`RX_MET_ONNXRUNTIME_ROOT` 不会切换本算子的头文件。rx-met 的 cu118 默认构建链
未传入此覆盖项，使用 ORT 1.20.1（API 20）执行 ONNX LSTM PTQ 时会注册失败。
在 cu118 builder 中重建 LSTM wheel 时，设置
`QUANT_LSTM_ONNXRUNTIME_INCLUDE=/opt/rx-quant-env/onnxruntime/include`，
使用已准备好的匹配头文件。rx-met 的版本锁文件和各环境变体见
[共享环境说明](../../../docs/Shared_environment.md)。

## 3. 安装 CUDA C++ package

使用自定义安装前缀可以避免修改系统目录：

```bash
install_prefix=/path/to/quant-lstm-install

cmake -S . -B build-install \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}"
cmake --build build-install --parallel
cmake --install build-install
```

安装结果包括：

```text
<prefix>/include/
<prefix>/lib/libquant_lstm.a
<prefix>/lib/cmake/quant-lstm/
<prefix>/share/quant-lstm/config/
<prefix>/share/quant-lstm/schemas/
<prefix>/share/doc/quant-lstm/
```

下游项目使用导出的 target，不需要手工拼接 include 和 library 路径：

```cmake
cmake_minimum_required(VERSION 3.24)
project(quant_lstm_consumer LANGUAGES CXX)

find_package(quant-lstm CONFIG REQUIRED)
add_executable(app main.cc)
target_link_libraries(app PRIVATE quant_lstm::quant_lstm)
```

配置并构建下游项目：

```bash
cmake -S /path/to/consumer -B /path/to/consumer/build \
  -DCMAKE_PREFIX_PATH=/path/to/quant-lstm-install
cmake --build /path/to/consumer/build --parallel
```

`find_package()` 会加载 `quant-lstm-config.cmake`，查找 `nlohmann_json`，并在该
package 包含 CUDA 时查找 `CUDAToolkit`。成功生成并运行下游可执行文件是 C++
安装的完成判定。

将 `CMAKE_INSTALL_PREFIX` 设置为 `/usr/local` 可以执行系统级安装，但这不是使用
quant-lstm 的必要条件。自定义 prefix 更易于并存、升级和删除。

## 4. 安装 CPU-only reference package

CPU-only package 包含浮点和 int32 C++ reference，不包含 PyTorch 模块，也不链接
CUDA 或 cuBLAS：

```bash
install_prefix=/path/to/quant-lstm-cpu-install

cmake -S . -B build-cpu-install \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DQUANT_LSTM_ENABLE_CUDA=OFF \
  -DQUANT_LSTM_BUILD_EXAMPLES=ON
cmake --build build-cpu-install --parallel
cmake --install build-cpu-install
"${install_prefix}/bin/lstm_float_example"
"${install_prefix}/bin/lstm_int32_example"
```

两个示例均退出码为 0 表示 reference package 安装成功。完整的独立 consumer 验收
可以运行：

```bash
tools/run_cpu_only_package_check.sh
```

该脚本会配置独立构建树、运行 CTest、安装到构建树内的独立 prefix、构建外部
`find_package()` consumer，并确认 consumer 没有链接 CUDA 动态库。

## 5. 使用 Docker 构建环境

独立 quant-lstm 仓库的 `docker/` 提供 CUDA、PyTorch、CMake 和测试依赖的开发镜像。
rx-met 的 `operators/quant-lstm/` 副本不包含 `docker/` 和 `.github/`，使用 rx-met
自身的镜像与构建流程。以下命令仅从包含 `docker/Dockerfile` 的独立仓库根目录执行：

```bash
docker build -f docker/Dockerfile -t quant-lstm:cuda .
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  quant-lstm:cuda
```

容器进入 `/workspace`。挂载目录后，按照第 2 节执行 CMake 构建和普通环境安装。
宿主机需要 NVIDIA Container Toolkit，且驱动需要支持镜像内 CUDA Runtime。

## 6. 常见安装错误

| 错误 | 原因与处理 |
| --- | --- |
| `未找到 CUDA 编译器` | `QUANT_LSTM_ENABLE_CUDA=ON`，但 `nvcc` 不在 `PATH` 或 CUDA Toolkit 未安装 |
| `未找到 build/libquant_lstm.a` | Python extension 构建前未在默认 `build/` 或 `RX_MET_QUANT_LSTM_BUILD_DIR` 指定目录完成 CMake 构建 |
| `_quant_lstm 扩展未找到` | wheel 安装尚未完成、当前 Python 环境错误，或 extension 与 Python ABI 不匹配 |
| `PyTorch ... 只支持 CUDA input` | CUDA wheel 收到 CPU 输入或状态；将模块、输入和状态移动到 CUDA，或显式构建 CPU wheel 后以 `eval()` 执行 |
| `Could not find quant-lstm` | 下游没有设置安装 prefix 的 `CMAKE_PREFIX_PATH` |
| `Could not find CUDAToolkit` | 下游正在消费 CUDA package，但 CMake 无法定位 CUDA Toolkit |
