# QuantGRU

QuantGRU 提供 PyTorch GRU(门控循环单元) 的高性能量化实现，基于原生 C++、CUDA 和 PyTorch，包含浮点计算、PTQ、QAT、
整数输入输出和 ONNX 导出。模型参数命名与单层 `torch.nn.GRU` 对齐。

## 支持范围

- 浮点和量化两种模式：可在训练和推理时自由切换
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

QuantGRU 可从独立仓库或 rx-met 的 `operators/quant-gru` 目录构建。以下命令均从
包含本 README 的 QuantGRU 目录执行。独立仓库的镜像说明位于 `docker/README.md`，
rx-met 的发布镜像说明位于其仓库根目录的 `docker/README.md`。

源码构建需要 Linux、C++17 编译器、CMake 3.18 及以上版本、OpenMP、CUDA Toolkit
和 CUDA 版 PyTorch。项目 wheel 构建流程使用 Python 3.10 至 3.12。编译工具链与
PyTorch CUDA 版本保持兼容。

CPU 发布包支持浮点推理、MinMax/SQNR/Percentile 校准、PTQ、整数输入输出和 ONNX
适配，不支持浮点训练或 QAT。使用 CPU 版 PyTorch 构建：

```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release -DRX_MET_ENABLE_CUDA=OFF
cmake --build build_release --parallel 2
RX_MET_ENABLE_CUDA=0 python3 -m pip install ./pytorch --no-deps --no-build-isolation
```

CPU 构建不需要 CUDA Toolkit，生成的 wheel 包含 CPU 原生共享库。使用时将模块和输入
放在 CPU，校准和推理均要求 `eval()` 且在 `torch.no_grad()` 或 `torch.inference_mode()`
上下文中调用；量化参数导入导出接口与 CUDA 包相同。自定义 CPU 构建目录时，安装前
设置 `RX_MET_QUANT_GRU_BUILD_DIR` 为构建目录的绝对路径。
ONNX Runtime 适配还需要构建时提供 C++ headers，可通过
`QUANT_GRU_ONNXRUNTIME_INCLUDE` 指定其目录。

CUDA 构建使用：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
python3 -m pip install ./pytorch --no-deps --no-build-isolation
```

CMake 生成 `pytorch/lib/libgru_quant_shared.so` 和 `build/gru_example`，pip 构建
`gru_interface_binding` 扩展并安装 `quant_gru`。构建机器缺少可见 GPU 时，通过
`CMAKE_CUDA_ARCHITECTURES` 和 `TORCH_CUDA_ARCH_LIST` 指定目标架构。

## 最小示例

以下示例检查 CUDA 模块加载、前向接口和输出形状：

```python
import torch
from quant_gru import QuantGRU

gru = QuantGRU(input_size=64, hidden_size=128, batch_first=True).cuda().eval()
inputs = torch.randn(2, 10, 64, device="cuda")
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
        gru(inputs.to(device="cuda", dtype=torch.float32))
gru.calibrating = False
gru.finalize_calibration()
gru.use_quantization = True
```

独立模块默认 `use_pot2_scale=False`，使用 affine scale。配置中的
`use_pot2_scale=true` 启用 Po2；Po2 默认取整策略为 `CoverRange`，容差为 2%。
JSON 字段、默认值和加载时机见[配置说明](pytorch/config/README.md)。

QAT 在校准后使用 `gru.train()` 和 PyTorch 训练循环。量化前向记录截断 mask，
反向传播使用浮点梯度。候选门输入或输出发生截断时，其 reset gate 上游路径也按同一个
clamp STE mask 截断；序列输出与最终隐藏状态的损失均遵循此链式法则。
输入线性输出的截断同样作用于输入、权重和 bias 梯度，并在跨 stream 消费梯度前完成。
训练和部署评估分别记录损失、输出误差和任务指标。
非零更新门 zero point 下，`q_one-q_z` 已经是 `(1-z)` 的 centered 码，隐藏状态
更新直接使用该差值；主整数核、浮点参考与独立 CPU 参考均遵守此约定。

`quant_storage_dtype="float32"` 与 `"int32"` 共用整数/LUT 前向，包括量化、重缩放、门值和
截断 mask；该选项只改变提供给反向传播的量化码缓存类型。FP32 carrier 通过临时 INT32
缓存再转换为 FP32，增加分配和转换成本。8/16 位码可精确表示；超过 2^24 的整数码不能
保证 FP32 缓存精度。标准 STE 反向继续使用反量化门值，未改为 LUT 的分段导数。
底层 `ForwardPassQuantFP` 是保留的真实激活参考，不再是公共 QAT 前向。

公共 AIMET 编码重载根据所有标准 scale 是否为精确 2 的幂恢复 Po2/affine 执行模式，
与 LSTM 的重载约定一致；不向编译器编码添加执行元数据。锁定前会完成尚未执行的网格
更新（例如 affine 切换 Po2），避免冻结旧 scale 与新执行模式的混合状态；若缺少用于
推导新网格的校准数据则报错，保持未锁定。验证见
`tests/test_gru_carrier_contract.py` 的输出/缓存/mask/梯度和单向/双向重载对照。

## 保存与重载

模型权重与量化参数共同确定推理结果。以下代码接续已校准的 `gru`：

```python
torch.save(gru.state_dict(), "gru_weights.pth")
gru.export_quant_params("gru_encodings.json")

restored = QuantGRU(64, 128, batch_first=True).cuda().eval()
restored.load_state_dict(
    torch.load("gru_weights.pth", map_location="cuda", weights_only=True)
)
restored.load_quant_params("gru_encodings.json")
restored.use_quantization = True
```

重建实例沿用原模型的形状、方向和存储模式，并用相同输入比较保存前后的输出。
模型与量化文件按同次导出结果配套使用。

## ONNX 导出

导出使用 `export_mode=True` 和 PyTorch legacy exporter。符号注册的 opset 与
`torch.onnx.export(opset_version=...)` 一致，支持 opset 13 及以上版本：

```python
from quant_gru import ensure_quant_gru_onnx_registered, get_quant_gru_custom_opsets

opset = 18
ensure_quant_gru_onnx_registered(opset=opset)
gru.cpu().eval()
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
gru.export_mode = False
gru.cuda()
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
当前支持范围和验证见 rx-met 仓库中的 `docs/ONNX_PTQ_COMPILER.md` 和
`tests/test_onnx_gru_native.py`。

在 rx-met 中构建 wheel 时，仓库内的 ONNX Runtime 1.20.1 头文件用于构建
`quant_gru_onnx_runtime` 扩展，支持本项目使用的 ONNX Runtime 1.20.1 及以上版本。
独立构建需通过 `QUANT_GRU_ONNXRUNTIME_INCLUDE` 指定头文件目录；没有头文件时仍可
构建仅用于 Torch 的 wheel，ONNX 输入校准会提示缺少运行时扩展。

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
重载；[测试脚本](tests/test_quant_gru.py) 检查输出、梯度和误差阈值。命令从
QuantGRU 目录执行：

```bash
python3 pytorch/example/example_usage.py --list
python3 pytorch/example/example_usage.py -e basic
python3 tests/test_quant_gru.py
```

示例脚本会捕获并打印异常，运行失败时返回非零退出码。
测试数据规模、位宽和通过阈值以测试脚本为准。

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

### 固定 bias 网格的训练余量

新校准的输入/循环 bias 在各自粒度下，由观测到的最大绝对值建立 scale
`max(2 * max_abs_bias, 0.01) / qmax`；Po2 模式再严格向上取整为 2 的幂。
此策略将观测范围至少扩大一倍，并为零 bias 提供至少 ±0.01 的可训练区间。
8 位下最小 affine 步长约 7.87e-5、Po2 步长约 1.22e-4，可表示 1e-4 量级的更新。
位宽、对称零点和部署算术保持原有约定。MinMax、Percentile、SQNR 均对 bias 使用此规则；bias 不使用直方图裁剪或常数 bin 扩展范围。

已加载的编码与 QAT 期间的固定编码不会被扩张。

余量以较粗的 bias 分辨率为代价，不能保证任意学习率/训练时长都不饱和；应同时检查
偏置饱和、梯度与任务指标。可加载经过独立验证的显式编码以采用其他范围策略。
