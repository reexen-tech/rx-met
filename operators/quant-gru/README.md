# QuantGRU

QuantGRU 提供 PyTorch GRU(门控循环单元) 的量化实现，基于原生 C++、CUDA 和 PyTorch，包含浮点计算、PTQ、QAT、
整数输入输出和 ONNX 导出。模型参数命名与单层 `torch.nn.GRU` 对齐。

通过 rx-met 应用包安装的用户已经具备对应 wheel。整模型量化从
[PyTorch 模型量化](../../docs/PyTorch_quantization.md)开始，循环层替换与统一校准见
[循环算子接入](../../docs/QuantLSTM_integration.md)。本文的构建和独立校准流程面向
单独使用 QuantGRU 或维护算子源码的读者。

## 支持范围

- 支持浮点和量化两种模式；CUDA 包支持训练与推理，CPU 包支持校准和 PTQ 推理。
- 模型使用 `num_layers=1`、`dropout=0`，支持单向和双向计算。
- 输入为三维序列张量，`batch_first=True` 对应 `[B, T, I]`，默认布局为 `[T, B, I]`。
- CUDA 包使用 CUDA float32 张量，CPU 包支持 float32 前向和校准。输出为序列和最终隐藏状态。
- 隐藏状态形状为 `[D, B, H]`，其中单向 `D=1`，双向 `D=2`。
- 校准方法包括 MinMax、SQNR 和 Percentile，默认使用 MinMax。
- 权重和偏置采用对称量化，支持 per-tensor、per-gate 和 per-channel 粒度。
- 位宽配置接口接受 1 至 32，32 位使用有符号 INT32 范围。配置效果通过目标模型评估。

多层模型通过串联多个 QuantGRU 实例构建。校准隐藏状态范围的统计从第一个输出时间步
开始；有状态模型需要验证初始隐藏状态在部署量化网格中的覆盖范围。

## 环境与安装

以下命令均从 rx-met 的 `operators/quant-gru` 目录执行。发布环境与镜像说明见
[rx-met Docker 文档](../../docker/README.md)。

两种 Python 模块均在 Linux 上构建，需要 C++17 编译器、CMake 3.18 及以上版本、
OpenMP，以及已安装 `pip`、`setuptools`、`wheel` 和对应 PyTorch 的 Python 环境。
rx-met 的 wheel 构建流程使用 Python 3.10 至 3.12。CUDA 构建另需 CUDA Toolkit
和 CUDA 版 PyTorch，两者的 CUDA 版本须兼容。

CPU 发布包支持浮点推理、MinMax/SQNR/Percentile 校准、PTQ、整数输入输出和 ONNX
适配，不支持浮点训练或 QAT。使用 CPU 版 PyTorch 构建：

```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release -DRX_MET_ENABLE_CUDA=OFF
cmake --build build_release --parallel 2
RX_MET_ENABLE_CUDA=0 python3 -m pip install ./pytorch --no-deps --no-build-isolation
```

CPU 构建不需要 CUDA Toolkit，生成的 wheel 包含 CPU 原生共享库。使用时将模块和输入
放在 CPU，校准和推理均要求 `eval()`，普通前向无需额外关闭梯度。推荐使用
`torch.no_grad()` 或 `torch.inference_mode()` 减少推理开销；实际跨 CPU 算子的反向传播
会明确报错，CPU 不支持训练或 QAT。量化参数导入导出接口与 CUDA 包相同。自定义 CPU 构建目录时，安装前
设置 `RX_MET_QUANT_GRU_BUILD_DIR` 为构建目录的绝对路径。
需要从 ONNX 模型校准时，参见[构建 ONNX Runtime 适配扩展](#onnx-输入校准)。

CUDA 构建使用：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
python3 -m pip install ./pytorch --no-deps --no-build-isolation
```

CMake 生成 `pytorch/lib/libgru_quant_shared.so` 和 `build/gru_example`，pip 构建
`gru_interface_binding` 扩展并安装 `quant_gru`。构建机器缺少可见 GPU 时，在 CMake
配置中传入 `-DCMAKE_CUDA_ARCHITECTURES`，并在 pip 构建时设置
`TORCH_CUDA_ARCH_LIST`，为两个构建步骤指定同一目标架构。

量化激活由统一编译宏 `USE_REAL_ACTIVATION` 控制。环境变量
`RX_MET_USE_REAL_ACTIVATION=0`（默认）不定义宏，两种 CUDA carrier 都使用 LUT；
设为 `1` 时定义宏，均执行反量化、标准 sigmoid/tanh、再量化，并跳过 LUT 生成、
参数初始化和传递。它与 `quant_storage_dtype` 独立，也不随 train/eval 切换。
切换后须用相同设置重新编译共享库和 Python 扩展，例如：

```bash
RX_MET_USE_REAL_ACTIVATION=1 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
RX_MET_USE_REAL_ACTIVATION=1 python3 -m pip install ./pytorch --no-deps --no-build-isolation
```

CMake 也接受 `-DUSE_REAL_ACTIVATION=ON/OFF`；Python 扩展和 wheel 构建使用上述环境变量。
显式 CMake 选项和已有构建缓存优先；环境变量仅在首次配置且未设置该选项时提供默认值。
关闭时不向 C++ 编译器传入该宏，不能用 `-DUSE_REAL_ACTIVATION=0` 代替取消定义。

## 最小示例

以下示例检查 CUDA 模块加载、前向接口和输出形状。使用 CPU wheel 时将 `device`
改为 `"cpu"`，下方校准和重载示例沿用同一设备。

```python
import torch
from quant_gru import QuantGRU

device = "cuda"
gru = QuantGRU(input_size=64, hidden_size=128, batch_first=True).to(device).eval()
inputs = torch.randn(2, 10, 64, device=device)
with torch.no_grad():
    output, hidden = gru(inputs)

assert output.shape == (2, 10, 128)
assert hidden.shape == (1, 2, 128)
assert torch.isfinite(output).all()
print("QuantGRU forward OK")
```

完成断言并打印 `QuantGRU forward OK` 表示前向冒烟检查通过。示例中的随机输入用于
接口检查，模型量化使用代表性校准数据。

## 量化流程

独立使用 QuantGRU 时，流程为配置、收集校准数据、生成量化参数、启用量化。
以下代码接续已创建的 `gru`，`calibration_loader` 每次返回形状为 `[B, T, 64]`
的代表性输入张量：

```python
gru.load_bitwidth_config(
    "pytorch/config/gru_quant_bitwidth_config.json"
)
gru.calibrating = True
with torch.no_grad():
    for inputs in calibration_loader:
        gru(inputs.to(device=device, dtype=torch.float32))
gru.calibrating = False
gru.finalize_calibration()
gru.use_quantization = True
```

独立模块默认使用 affine scale。如需 Po2，在校准后、参数锁定和 QAT 前调用
`gru.enable_po2()`；默认取整策略为 `CoverRange`，容差为 2%。
JSON 字段、默认值和加载时机见[配置说明](pytorch/config/README.md)。

输入和循环 bias 采用带训练余量的独立网格，普通层的 INT32 乘积 scale 对齐不会
覆盖这些网格。校准规则、数值例子和部署约束见
[Bias 编码差异](../../docs/QuantLSTM_integration.md#bias-编码差异)。

CUDA QAT 在校准后使用 `gru.train()` 和 PyTorch 训练循环。量化前向记录截断 mask，
反向传播使用浮点梯度。候选门输入或输出发生截断时，其 reset gate 上游路径也按同一个
clamp STE mask 截断；序列输出与最终隐藏状态的损失均遵循此链式法则。
输入线性输出的截断同样作用于输入、权重和 bias 梯度，并在跨 stream 消费梯度前完成。
训练和部署评估分别记录损失、输出误差和任务指标。
非零更新门 zero point 下，`q_one-q_z` 已经是 `(1-z)` 的 centered 码，隐藏状态
更新直接使用该差值；两种 CUDA carrier 与独立 CPU 参考均遵守此约定。

CUDA 下 `quant_storage_dtype` 选择独立计算路径：

| 配置 | 量化码载体与计算 |
| --- | --- |
| `"float32"`（默认） | `quantGRUForwardFP()` 调用 `ForwardPassQuantFP`，使用 FP32 量化码、cuBLAS SGEMM，以及浮点重缩放、显式舍入和截断 |
| `"int32"` | `quantGRUForwardInt()` 使用 INT32 量化码、整数矩阵乘和重缩放，乘累加使用 INT64 |

两条路径都通过 `USE_REAL_ACTIVATION` 选择真实激活或 16 段 PWL。
量化边界统一使用 `clamp(round_to_nearest_even(x / scale) + zero_point)`；零点在舍入后相加，
输入、初始状态和真实激活输出遵循同一顺序。
调用层统一使用完整的 `quantize()` / `quantize_f()`，由入口负责饱和和训练 mask，
不再分别调用未截断码值计算与 clamp；内部共享数值步骤不作为独立调用入口。
CPU 包目前只有整数计算核心，`float32` 只改变返回的量化码缓存类型。
常规量化前向接收浮点输入和 master 权重，入口量化、出口反量化；FP32 carrier 的码值
如 `3.0` 仍表示整数码，不是反量化实数。标准 STE 反向使用反量化门值和截断 mask，
LUT 模式下也不使用分段斜率作为导数。

FP32 可精确表示 8/16 位码，但累加和乘积仍受浮点有效精度限制，绝对值超过 `2^24`
的整数不能保证精确表示。两条路径的缩放实现、舍入误差和 mask 可能不同，不保证跨
carrier 逐位一致；性能和任务精度应在目标模型、设备与激活构建模式下分别评估。
浮点到 INT32 的边界量化在类型转换前完成饱和；FP32 饱和端点取整数范围内最近的可表示值，避免
`INT32_MAX` 转成 FP32 后超出量化范围。训练 mask 以未截断码值是否处于合法范围为准。

`forward_quantized()` 和 `get_io_quant_meta()` 要求量化参数已就绪，不会自动完成待更新
参数的校准。更改位宽、scale 模式或校准统计后，先调用 `finalize_calibration()`，
再读取元数据和准备整数输入；已锁定的参数需先解锁才能重新计算。

公共 AIMET 编码重载根据所有标准 scale 是否为精确 2 的幂恢复 Po2/affine 执行模式，
与 LSTM 的重载约定一致；不向编译器编码添加执行元数据。锁定前会完成尚未执行的网格
更新（例如 affine 切换 Po2），避免冻结旧 scale 与新执行模式的混合状态；若缺少用于
推导新网格的校准数据则报错，保持未锁定。验证见
rx-met 的 `tests/recurrent/test_gru_storage.py`，分别验证 carrier 的输出、缓存、mask、
梯度和单向/双向重载。

## 保存与重载

模型权重与量化参数共同确定推理结果。以下代码接续已校准的 `gru`：

```python
torch.save(gru.state_dict(), "gru_weights.pth")
gru.export_quant_params("gru_encodings.json")

restored = QuantGRU(64, 128, batch_first=True).to(device).eval()
restored.load_state_dict(
    torch.load("gru_weights.pth", map_location=device, weights_only=True)
)
restored.load_quant_params("gru_encodings.json")
restored.use_quantization = True
```

重建实例沿用原模型的形状、方向和存储模式，并用相同输入比较保存前后的输出。
模型与量化文件按同次导出结果配套使用。

## ONNX 导出

安装 `onnx` 后，使用 `export_mode=True` 和 PyTorch legacy exporter 导出。
此导出路径要求 `bias=True`；`bias=False` 模型可用于普通前向和校准，但不能通过该路径导出。
符号注册的 opset 与 `torch.onnx.export(opset_version=...)` 一致，支持 opset 13 及以上版本：

```python
from quant_gru import ensure_quant_gru_onnx_registered, get_quant_gru_custom_opsets

opset = 18
ensure_quant_gru_onnx_registered(opset=opset)
gru.cpu().eval()
try:
    gru.export_mode = True
    torch.onnx.export(
        gru,
        torch.zeros(1, 10, 64),
        "gru.onnx",
        opset_version=opset,
        dynamo=False,
        custom_opsets=get_quant_gru_custom_opsets(),
        input_names=["input"],
        output_names=["output", "hidden"],
    )
finally:
    gru.export_mode = False
    gru.to(device)
```

产物使用标准 ONNX `GRU` 节点，量化参数单独导出。下游整图工具负责组合模型图、
节点命名和部署 encodings。

## ONNX 输入校准

`pytorch/quant_gru_onnx.py` 提供标准 ONNX GRU 的输入适配，转换门顺序、方向、
W/R/B 和输入输出布局，直接调用 QuantGRU 的校准、量化前向、Po2 与编码接口。
`pytorch/lib/gru_onnx_runtime.cc` 只负责 ONNX Runtime 与 Python 接口之间的张量传递，
不实现门计算。CUDA 包的桥接经主机缓冲区调用原生 CUDA 算子，存在额外数据拷贝；
CPU 包直接使用 CPU 原生算子。

RX-MET 的 `create_onnx_ptq_sim` 自动接入此能力，读取相同的 `GRU_config`。
导出保存标准 GRU，双向参数编码按方向组合，内部编码沿用原生 schema。
当前支持范围和验证见 [ONNX PTQ 指南](../../docs/ONNX_PTQ_COMPILER.md)和
[原生 GRU 测试](../../tests/onnx/test_native_gru.py)。

`quant_gru_onnx_runtime` 扩展在注册时请求构建头文件的 `ORT_API_VERSION`，运行时
必须提供该 API。构建时优先使用 `QUANT_GRU_ONNXRUNTIME_INCLUDE` 指定的头文件目录，
其次使用 `$RX_MET_ONNXRUNTIME_ROOT/include`，最后使用 rx-met 仓库内的
`docker/environment/third_party/onnxruntime/include`。rx-met 发布构建通过
`RX_MET_ONNXRUNTIME_ROOT` 使用变体对应的头文件。独立构建时可显式指定与目标
运行时兼容的头文件。缺少 `onnxruntime_cxx_api.h` 时仍可构建 PyTorch wheel，
但不会包含 ONNX 输入校准所需的适配扩展。

## 接口与验证

| 接口 | 用途 |
| --- | --- |
| `forward(input, hx=None)` | 浮点张量输入输出，按量化开关执行浮点或量化计算 |
| `forward_quantized(input, hx=None)` | 已量化的 INT32 输入输出 |
| `get_io_quant_meta()` | 返回输入、输出和隐藏状态的有效 scale、零点和位宽 |
| `set_all_bitwidth()`、`load_bitwidth_config()` | 设置位宽或加载 JSON 配置 |
| `finalize_calibration()`、`reset_calibration()` | 生成量化参数或清除校准状态 |
| `set_quant_params_locked()` | 控制量化参数锁定状态 |
| `get_quant_config()`、`adjust_quant_config()` | 查询或调整指定算子的配置 |

[示例脚本](pytorch/example/example_usage.py) 覆盖校准、QAT、双向计算、ONNX 导出和
重载；[测试脚本](tests/test_quant_gru.py) 检查输出、梯度和误差阈值。这两个脚本需要
CUDA 包和可见 GPU（`--list` 只列出示例）。命令从 QuantGRU 目录执行：

```bash
python3 pytorch/example/example_usage.py --list
python3 pytorch/example/example_usage.py -e basic
python3 tests/test_quant_gru.py
```

示例脚本会捕获并打印异常，运行失败时返回非零退出码。
测试数据规模、位宽和通过阈值以测试脚本为准。

已安装 CPU 包的开发者使用以下测试检查浮点前向、PTQ、整数输入输出和参数重载：

```bash
python3 -m pytest -q tests/test_cpu_ptq.py
```

该测试需要 `pytest`。ONNX Runtime 适配测试另需 `onnx`、`onnxruntime` 和构建时生成的
`quant_gru_onnx_runtime` 扩展，入口为 `tests/test_cpu_onnx.py`。

## 文档与源码

| 路径 | 职责 |
| --- | --- |
| [pytorch/config/](pytorch/config/README.md) | JSON 字段、默认值和配置接口 |
| [docs/GRU量化计算流程.md](docs/GRU量化计算流程.md) | 参数结构、逐算子公式、重缩放与 16 段 PWL |
| [quant-gru-cpu-only/](quant-gru-cpu-only/README.md) | 独立的 C++ CPU 定点参考实现 |
| `include/`、`src/` | 主 C++/CUDA 计算、校准和 LUT 实现 |
| `pytorch/` | Python 模块、扩展 binding 与使用示例 |
| [根目录 CHANGELOG.md](../../CHANGELOG.md) | 集成版本与迁移记录 |

项目使用的 GRU 实现源自 [Haste](https://github.com/lmnt-com/haste)，并与
[PyTorch GRU 接口](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) 集成。
