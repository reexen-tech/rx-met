# QuantGRU

QuantGRU 提供 PyTorch GRU(门控循环单元) 的高性能量化实现，基于 CUDA 和 PyTorch，包含 CUDA 浮点计算、PTQ、QAT、
整数输入输出和 ONNX 导出。模型参数命名与单层 `torch.nn.GRU` 对齐。

## 支持范围

- 浮点和量化两种模式：可在训练和推理时自由切换
- 模型使用 `num_layers=1`、`dropout=0`，支持单向和双向计算。
- 输入为三维序列张量，`batch_first=True` 对应 `[B, T, I]`，默认布局为 `[T, B, I]`。
- 常规前向和校准使用 CUDA float32 张量。输出为序列和最终隐藏状态。
- 隐藏状态形状为 `[D, B, H]`，其中单向 `D=1`，双向 `D=2`。
- 校准方法包括 MinMax、SQNR 和 Percentile，默认使用 MinMax。
- 权重和偏置采用对称量化，支持 per-tensor、per-gate 和 per-channel 粒度。
- 位宽配置接口接受 1 至 32，32 位使用有符号 INT32 范围。配置效果通过目标模型评估。

多层模型通过串联多个 QuantGRU 实例构建。校准隐藏状态范围的统计从第一个输出时间步
开始；有状态模型需要验证初始隐藏状态在部署量化网格中的覆盖范围。

## 环境与安装

rx-met 发布镜像包含 QuantGRU，环境说明见
[Docker 构建配置](../../docker/README.md)。

源码构建需要 Linux、C++17 编译器、CMake 3.18 及以上版本、OpenMP、CUDA Toolkit
和 CUDA 版 PyTorch。项目 wheel 构建流程使用 Python 3.10 至 3.12。编译工具链与
PyTorch CUDA 版本保持兼容。

先进入包含本 README 的 QuantGRU 目录，再执行：

```bash
cmake -S operators/quant-gru -B build -DCMAKE_BUILD_TYPE=Release
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
反向传播使用浮点梯度。训练和部署评估分别记录损失、输出误差和任务指标。

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

示例脚本会捕获并打印异常，结果检查同时读取日志中的完成信息和错误信息。
测试数据规模、位宽和通过阈值以测试脚本为准。

## 文档与源码

| 路径 | 职责 |
| --- | --- |
| [pytorch/config/](pytorch/config/README.md) | JSON 字段、默认值和配置接口 |
| [docs/GRU量化计算流程.md](docs/GRU量化计算流程.md) | 参数结构、逐算子公式、重缩放与 16 段 PWL |
| [quant-gru-cpu-only/](quant-gru-cpu-only/README.md) | 独立的 C++ CPU 定点参考实现 |
| `include/`、`src/` | 主 C++/CUDA 计算、校准和 LUT 实现 |
| `pytorch/` | Python 模块、扩展 binding 与使用示例 |
| [CHANGELOG.md](CHANGELOG.md) | 组件版本与迁移记录 |

项目使用的 GRU 实现源自 [Haste](https://github.com/lmnt-com/haste)，并与
[PyTorch GRU 接口](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) 集成。
