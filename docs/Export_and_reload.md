# 导出与恢复参考

本文档说明量化结果的制品关系、导出接口、图处理约束和 PyTorch 恢复顺序。
首次接入流程见 [PyTorch 模型量化](PyTorch_quantization.md)；已有 ONNX 的 PTQ、
编译器字段和原生循环层 ONNX 限制见 [ONNX PTQ 编译器制品](ONNX_PTQ_COMPILER.md)。
配置字段与匹配规则见 [量化配置](Quant_config.md)。

## 制品与用途

| 制品 | 内容与用途 |
| --- | --- |
| 主权重检查点 | 保存模型权重及业务所需的网络类型、前处理状态等信息，供 PyTorch 重建模型 |
| `<prefix>.onnx` | 不含 Q/DQ 的部署图，保存计算结构和模型参数 |
| `<prefix>.json` | 与同次 ONNX 配套的公共编译器编码，当前为 schema 4 |
| AIMET 原始 `.encodings` / `.aimet.encodings` | 原始量化模拟编码，用于对应框架的调试、重载和追溯 |
| 阶段编码 JSON | `save_quantizer_encodings()` 保存的全部或部分量化网格，供分阶段校准和训练恢复 |

ONNX 与最终编译器 JSON 必须配套使用。它们不代替 PyTorch 主权重检查点；编码文件
不保存模型权重。调用方应使用导出接口返回的路径，不通过扫描前缀或替换文件后缀猜测
最终编码。运行目录和配置快照见 [运行目录与日志](Runtime_outputs.md)。

## 导出入口

### PyTorch QuantSim

```python
from aimet_torch.rx_export import export_onnx_json

onnx_path, encodings_path = export_onnx_json(
    sim,
    export_dir="output/export",
    filename_prefix="model",
    dummy_input_shape=(1, 49, 20),
    opset=18,
)
```

以上为接口片段，`sim` 须已完成校准。`dummy_input_shape` 必须匹配实际 `forward`
输入；省略时的兼容默认值是 `(1, 16000, 1)`，不是所有模型通用的 shape。
`opset` 默认是 18，导出使用 PyTorch legacy exporter（`dynamo=False`）。返回值是
`(onnx_path, encodings_path)` 两个字符串，分别指向最终 ONNX 和 `.json`。

多输入模型可按 `forward` 的位置参数顺序传入多个 shape、Tensor 或 NumPy 数组。
数组输入保留实际值、dtype 和 shape，并转到 CPU；shape 输入生成浮点零张量。
单输入模型继续支持一个 shape tuple。例如双输入 Add：

```python
onnx_path, encodings_path = export_onnx_json(
    sim, export_dir, "two_input_add",
    dummy_input_shape=(left_array, right_array),
)
```

这里的 `left_array`、`right_array` 是模型所需的实际示例输入，`export_dir` 是输出根目录。
单输入导出时请求 ONNX 输入名 `input`；多输入时由导出器生成输入名，后处理从生成的图
读取实际输入名和静态 shape。`filename_prefix` 只能是单个文件名，不能包含目录。

### ONNX QuantSim

```python
from aimet_onnx.rx_ptq import export_onnx_json

artifacts = export_onnx_json(
    sim,
    original_model,
    output_dir="output/export",
    filename_prefix="model",
)
```

此片段接收 ONNX QuantSim 和对应的原模型。返回字典包含：

| 键 | 值 |
| --- | --- |
| `onnx` | Clean ONNX 的 `Path` |
| `encodings` | 最终编译器 `.json` 的 `Path` |
| `aimet_encodings` | AIMET 原始 `.aimet.encodings` 的 `Path` |
| `coverage` | 编译器编码覆盖率记录 |
| `onnx_shapes` | ONNX shape 补全报告 |

旧 ONNX 名称 `export_onnx_compiler_artifacts()` 保留原参数和字典返回值。两框架
共用编译器字段规范化与校验，框架相关的图处理分别维护。ONNX PTQ 的完整输出、
metadata 和重载验证见 [ONNX PTQ 编译器制品](ONNX_PTQ_COMPILER.md)。

## 导出前提与校验

PyTorch 导出前应完成模型准备、配置、校准及所需的 Po2/QAT，并将 `sim.model`
切换到 CPU 和 eval 模式。QuantGRU 当前要求 `bias=True` 才能导出，关闭量化也不豁免；
QuantLSTM 支持 `bias=False`。无 bias GRU 的接入选择见[循环层约束](QuantLSTM_integration.md#使用与配置)。
导出器在图后处理和循环算子编码合并后校验最终 ONNX
及 encodings，通过后才返回路径。校验以 `sim.model` 中启用的量化器为基准，检查：

- 输入、输出和参数编码是否完整，包括输入编号、每个通道及原生 GRU/LSTM 内部编码。
- 编码值是否与模型一致，scale 是否为正数且有限。
- ONNX 结构和 shape 是否有效、节点名是否重复，以及原生循环层节点是否匹配。
- 每个浮点计算节点是否有 activation encoding key，参数输入是否有 initializer 对应的
  parameter encoding key；编码 key 是否能在最终图或简化前图中找到来源。
- 按最终 ONNX 输入检查计算覆盖率：Add/Mul/MatMul 等双浮点输入必须覆盖两路，单输入
  激活检查一路，Conv/Gemm 的激活和权重分别检查。量化器开关不能绕过此项检查。

缺少编码、通道缺失、编码值不一致或启用的张量量化器未校准时抛出 `RuntimeError`。
错误会给出模块及字段，例如 `encodings.activation_encodings.module_add.output.0`。
标量常量、shape/index 等非浮点输入不要求编码；简化消除的节点可以保留有原始图来源
的编码。关闭量化的计算输入仍留在最终浮点计算节点上时，会因缺少编码而报错。

这项校验不替代精度验证。失败时已写出的文件只用于排查，不应作为通过验证的部署制品。
现有静态制品也可用 `scripts/verify_compiler_artifacts.py` 只读检查，参数见
[工具脚本](../scripts/README.md)。

## 图准备与后处理

### Linear 与量化边界

带偏置的高维 Linear 在导出副本中转换为 Reshape → Gemm → Reshape，保留融合计算
和原有输入、输出、参数编码；无偏置 Linear 保持单个 MatMul。Gemm 使用原模块名，
权重与偏置名称保持对应，不新增中间量化位置，也不修改训练侧 QuantSim。
乘一、加零和单输入 Concat 的量化边界保留在最终 ONNX。

### GELU

`prepare_model()` 自动把 `nn.GELU` 和 `F.gelu` 替换为显式计算模块，保留
`approximate="none"` 的 Erf 公式和 `approximate="tanh"` 的近似公式。
Div/Erf/Add/Mul/Pow/Tanh 按所选公式参与校准与 QAT，标量常量仍豁免。
准备必须在创建 QuantSim 前完成；旧浮点权重可复用，原模块边界编码不能覆盖新增
中间量化器，需要重新校准并评估 QAT 精度。

### BatchNorm 与 FX 参数

KWS 使用 `QuantizableBatchNorm2d`，准备时展开为 Sub/Div/Mul/Add，统计量通过算子
输入参与校准。标准 BatchNorm 缺少统计量编码时，仍不能通过全部浮点输入覆盖检查。
导出器按 FX 参数的使用位置，把 reshape/常量折叠后的 initializer 恢复为参数编码名称，
不通过参数数值猜测归属。

FX 展开模块的参数只由原模块 `param_quantizers` 量化一次，导出通过 `param_encodings`
覆盖参数边，不再写一份不同的 activation encoding。普通主权重模型提取会移除辅助
FX 量化调用；提取 QDQ 权重模型时则把实际执行的参数网格应用到副本。
旧版同时保存无效父参数编码和子输入编码的阶段文件不保证兼容，应从固定 FP 权重重新校准。
配置和梯度关系见 [FX 参数量化](Quant_config.md#fx-展开模块的参数量化)。

## 权重和编码恢复

首先确认检查点保存的是主权重还是量化模型状态。恢复时必须使用相同的网络类型、模型
准备方式、两份配置、默认位宽和量化方案，不能用 `strict=False` 隐藏模型结构差异。

### 主权重检查点

KWS 在保存前调用 `sim.get_original_model(sim.model, qdq_weights=False)`，检查点中的
`model` 是提取后的主权重，另保存 `rnn_type`、`model_variant` 和外置 MFCC 状态
`speech_features`。它不依赖 QuantSim 的内部 Python 状态。使用 KWS 时由示例的
`load_checkpoint()` 校验网络类型、前处理和兼容迁移，并严格加载权重。

自有模型可以采用相同的分离方式。以下是接续量化流程的 API 片段，`build_model()`
须创建保存时对应的模型，`sim_options` 须包含相同的 dummy input、配置和量化方案：

```python
import torch
from aimet_torch import model_preparer
from aimet_torch.v2 import quantsim
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
from aimet_torch.staged_quantization_utils import load_quantizer_encodings

torch.save(
    sim.get_original_model(sim.model, qdq_weights=False).state_dict(),
    "output/master_weights.pth",
)

fresh_model = model_preparer.prepare_model(build_model().eval())
state = torch.load("output/master_weights.pth", map_location="cpu", weights_only=True)
fresh_model.load_state_dict(state, strict=True)
fresh_sim = quantsim.QuantizationSimModel(fresh_model, **sim_options)
fresh_sim.set_percentile_value(99.99)  # 使用 Percentile 时沿用保存侧的设置
apply_mixed_precision_bitwidth(fresh_sim.model, config_file=bitwidth_config)
load_quantizer_encodings(
    fresh_sim.model,
    load_path=encodings_path,
    skip_if_not_found=False,
    allow_overwrite=False,
)
fresh_sim.model.eval()
```

模型有自定义量化开关或输入量化策略时，还需在恢复侧重新应用，例如 KWS 的 attention
量化器设置。`get_original_model()` 提取后的参数名须与准备后的目标模型一致；如果模型
适配改变了命名或权重布局，需要显式迁移并校验，不可跳过不匹配项。

### QuantSim state_dict

若保存的是 `sim.model.state_dict()`，目标应先重建相同 QuantSim 和量化配置，再加载
该状态。普通量化器参数及其形状依赖相同的图结构和实现版本；
这类状态不能直接当作主权重加载到普通模型。原生循环层还需要其完整量化参数，不能
假定任意筛选后的 PyTorch state dict 就足以恢复编码。

旧流程若刻意只把量化模型中的主权重部分加载到新 QuantSim，可以使用非严格加载，
但必须检查返回的 `missing_keys` 只包含预期的量化器参数、`unexpected_keys` 为空，
随后从配套文件完整加载编码。该方式不是处理任意缺失键的通用修复。

### 恢复结果验证

比较保存前和恢复后的同批输入输出及任务指标，记录容差和设备。KWS 按
`atol=1e-5`、`rtol=1e-4` 比较输出，并对比测试精度；这些阈值不是所有模型的精度保证。
名称错位通常来自模型准备、配置或默认位宽差异。固定编码后再创建训练 optimizer，
需要继续 QAT 时按 [QAT 冻结要求](Quant_config.md#qat-和重新加载)设置量化器和 BatchNorm。

## 分阶段编码保存与恢复

`save_quantizer_encodings(sim.model, save_path, layer_types=None)` 保存全部或选定类型
的量化参数，不保存模型权重。`load_quantizer_encodings()` 支持 `layer_types` 和
优先级更高的 `exclude_layer_types` 筛选。`skip_if_not_found` 默认 `True`；设为 `False`
可让所选原生循环层缺失编码和部分普通量化器加载失败成为错误。普通层的编译器 JSON
按文件内的条目加载，缺失条目或无法匹配的模块和输入槽仍可能被跳过。完整恢复时，
检查加载统计及对应的量化器，并比较恢复前后的输出。
`allow_overwrite=False` 禁止后续校准覆盖已加载的量化网格。

阶段文件加载以保存的 `qmin/qmax/symmetric/scale/offset` 为准，包括 Po2 后与初始
配置不同的 bias 位宽。普通 affine 量化器通过 scale/offset 参数化精确恢复，不从
实数 min/max 反推网格；目标量化器形状必须匹配，非法 scale、offset 或不一致范围
会抛出 `ValueError`。加载后再冻结量化器参数或创建训练 optimizer。

关闭量化的原生循环层可以不写编码；文件中有效的原生循环层编码被显式加载后，会启用
该层量化。原生阶段参数、状态和编码回读规则见 [循环算子接入](QuantLSTM_integration.md)。
每个阶段应配套保存配置、模型权重和评估结果，并记录可追溯的校准数据版本与前处理。
精度结论还应注明模型版本、验证集和复现命令。
