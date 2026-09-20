# AIMET 量化配置

本文档说明 rx-met 当前使用的 QuantSim 和混合精度配置。完整可运行流程位于
`examples/quick_start_kws.py`，ONNX PTQ 流程位于 `examples/onnx_ptq_quick_start.py`。

## 配置文件

仓库提供两份配置：

| 文件 | 使用位置 | 作用 |
| --- | --- | --- |
| `examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json` | `QuantizationSimModel(..., config_file=...)` | 控制量化器是否启用、对称性和逐通道量化 |
| `examples/config/quick_start_full_quant.json` | `apply_mixed_precision_bitwidth(..., config_file=...)` | 控制模块位宽、禁用规则和 QuantGRU 内部量化 |

基础配置负责创建 QuantSim，混合精度配置负责设置 `sim.model`。配置路径相对于运行
命令的当前目录解析。

## QuantSim 基础配置

基础配置使用 AIMET QuantSim 格式。当前示例配置包含以下顶层字段：

| 字段 | 作用 |
| --- | --- |
| `defaults.ops` | 设置算子输出量化和默认对称性 |
| `defaults.params` | 设置权重量化和默认对称性 |
| `defaults.strict_symmetric` | 控制严格对称量化范围 |
| `defaults.per_channel_quantization` | 控制参数是否默认逐通道量化 |
| `params` | 覆盖特定参数类型，例如 `bias` |
| `op_type` | 按 ONNX 算子类型覆盖默认值 |
| `supergroups` | 定义可消除中间量化器的算子组合 |
| `model_input`、`model_output` | 控制模型边界量化 |

`op_type` 遵循 ONNX 算子命名，例如 `Conv`、`Gemm`、`Add`。当前仓库的基础配置
将 `op_type` 留空，由全局默认值控制。

```python
from aimet_torch import model_preparer
from aimet_torch.v2 import quantsim

prepared_model = model_preparer.prepare_model(model)
sim = quantsim.QuantizationSimModel(
    prepared_model,
    dummy_input=dummy_input,
    quant_scheme="percentile",
    config_file="examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json",
    default_output_bw=8,
    default_param_bw=8,
)
sim.set_percentile_value(99.99)
```

`set_percentile_value()` 必须在 `compute_encodings` 前调用。重建 QuantSim 时，模型
预处理参数、基础配置、默认位宽和量化方案必须与首次创建时一致。

## 混合精度配置

混合精度配置由 `apply_mixed_precision_bitwidth()` 读取：

```python
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth

apply_mixed_precision_bitwidth(
    sim.model,
    config_file="examples/config/quick_start_full_quant.json",
    verbose=True,
)
```

顶层字段如下：

| 字段 | 作用 |
| --- | --- |
| `default_bitwidth` | 声明默认 `weight` 和 `output` 位宽，应与创建 QuantSim 时的默认位宽保持一致 |
| `layer_type_config` | 按量化模块类名设置规则，例如 `QuantizedConv2d` |
| `layer_name_config` | 按完整模块名或 `*` 通配符设置规则 |
| `GRU_config` | 配置 QuantGRU 的量化开关和内部算子 |
| `default_config` | 控制未匹配模块，支持 `disable_quantization` |

普通量化模块支持以下配置项：

| 配置项 | 说明 |
| --- | --- |
| `weight_bitwidth`、`bias_bitwidth` | 权重和偏置位宽 |
| `input_bitwidth`、`input_bitwidths` | 所有输入或按输入下标设置位宽 |
| `output_bitwidth` | 输出位宽 |
| `param_bitwidth` | 除 `weight`、`bias` 外的参数位宽 |
| `weight_symmetric`、`bias_symmetric` | 权重和偏置的对称性 |
| `input_symmetric`、`output_symmetric` | 输入和输出的对称性 |
| `param_symmetric` | 其他参数的对称性 |
| `per_channel_quantization` | 是否使用逐通道参数量化 |
| `disable_quantization` | 是否禁用匹配模块的量化器 |

当前实现按以下顺序选择规则：

1. `layer_name_config` 完整名称。
2. `layer_type_config` 模块类型。
3. `layer_name_config` 通配符。
4. 默认配置。

通配符在类型规则之后生效。配置器跳过当前模型中缺少的已注册 `Quantized*` 类型，
并对未注册的类型名抛出拼写错误。

## QuantGRU 配置

QuantGRU 使用独立的 `GRU_config` 配置路径。`apply_mixed_precision_bitwidth()` 会把
同一 JSON 文件交给每个 QuantGRU 实例，并读取该节点：

```json
{
  "GRU_config": {
    "default_config": {
      "disable_quantization": false,
      "use_pot2_scale": true
    },
    "operator_config": {
      "input": {
        "bitwidth": 8,
        "is_symmetric": true,
        "is_unsigned": false
      },
      "weight_ih": {
        "bitwidth": 8,
        "is_symmetric": true,
        "is_unsigned": false,
        "quantization_granularity": "PER_CHANNEL"
      }
    }
  }
}
```

`operator_config` 可以配置输入、输出、权重、偏置、线性计算结果以及三个门的输入和
输出。完整键集合以 `examples/config/quick_start_full_quant.json` 为准。

`quantization_granularity` 支持 `PER_TENSOR`、`PER_GATE` 和 `PER_CHANNEL`。
`is_unsigned` 适用于值域非负的量化点，例如 sigmoid 门输出。已校准或已加载量化参数
的 QuantGRU 会跳过位宽配置加载，因此配置应用位于校准之前。

一次 `compute_encodings` 前向过程同时完成 QuantGRU 和其他模块的校准。

## 校准和 Power-of-2

校准必须使用能代表实际输入分布的数据：

```python
import torch
import aimet_torch.v2 as aimet

sim.model.eval()
with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
    for index, (inputs, _) in enumerate(calib_loader):
        if index >= 100:
            break
        sim.model(inputs.to(device))
```

校准后可以把 scale 调整为 2 的幂，并按部署要求对齐 bias scale：

```python
from aimet_torch.utils_rx import apply_power_of_2_workflow

apply_power_of_2_workflow(
    sim.model,
    align_bias_scale=True,
    verbose=True,
)
```

`apply_power_of_2_workflow()` 默认使用 `cover_range`。该策略在原范围与最近的
Power-of-2 差异超过默认 2% 容差时扩大 scale，以覆盖校准范围。

## QAT 和重新加载

进入 QAT 前应冻结量化参数，并在训练模式下保持 BatchNorm 统计量冻结：

```python
from aimet_torch.utils_rx import freeze_quantizer_parameters
from aimet_torch.utils_rx import set_train_mode_freeze_bn

freeze_quantizer_parameters(sim.model, freeze_bn_affine=True)
set_train_mode_freeze_bn(sim.model)
```

重新加载时应先按原配置重建 QuantSim，再加载模型状态和 encodings：

```python
from aimet_torch.staged_quantization_utils import load_quantizer_encodings

fresh_sim.model.load_state_dict(state, strict=False)
load_quantizer_encodings(
    fresh_sim.model,
    load_path="output/model.encodings",
    verbose=True,
)
```

应比较保存前和重新加载后的输出或评估指标。量化器名称差异通常来自模型预处理、
配置文件或默认位宽差异。

## 分阶段量化

仓库提供 `save_quantizer_encodings()` 和 `load_quantizer_encodings()`，可以按模块类型
保存或恢复部分量化参数。每个模型根据自身结构和精度评估结果维护阶段划分与配置文件。

采用分阶段量化时，应为每个模型单独维护配置、校准数据、模型权重和评估结果。新阶段
加载已有 encodings 后，可以通过 `allow_overwrite=False` 防止后续校准覆盖已确认的
量化参数。任何精度结论都应同时记录模型版本、数据集、配置和复现命令。

## 检查清单

- 配置文件路径指向仓库中实际存在的文件。
- 基础配置在创建 QuantSim 时加载，混合精度配置在校准前应用。
- 百分位参数在 `compute_encodings` 前设置。
- 校准数据与部署输入的名称、形状、类型和分布一致。
- QuantGRU 配置在校准前加载，并使用共享校准过程。
- QAT 前冻结量化参数和 BatchNorm。
- 重建模型使用与训练侧相同的模型预处理和量化配置。
- 性能或精度结论包含模型版本、数据集、配置和复现命令。
