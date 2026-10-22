# AIMET 量化配置

本文档说明 rx-met 当前使用的 QuantSim 和混合精度配置。完整可运行流程位于
`examples/quick_start_kws.py`，ONNX PTQ 流程位于 `examples/quick_start_onnx_ptq.py`。

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
| `default_config` | 支持未匹配模块的 `disable_quantization`，以及整数符号回退值 `signed` |

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
| `weight_signed`、`bias_signed`、`param_signed` | 权重、偏置及其他参数的整数符号，`true` 为 INT，`false` 为 UINT |
| `input_signed`、`input_signeds`、`output_signed` | 全部输入、按输入编号或输出的整数符号；列表中 `null` 不施加局部覆盖 |
| `per_channel_quantization` | 是否使用逐通道参数量化 |
| `disable_quantization` | 是否禁用匹配模块的量化器 |

Torch 和 ONNX 的整数量化默认使用 INT 和对称量化，JSON1 遗漏激活或参数的
`is_symmetric` 时统一按 True 处理；显式 False 继续使用非对称量化。
符号与对称性独立，`*_symmetric` 只调整
校准策略，修改位宽保留当前符号。UINT 对称网格使用零点 0、范围 `[0, 2^bitwidth-1]`，
校准及推理将负值截为零；INT 对称网格使用零点 0 和完整有符号范围。
显式层设置优先于 `default_config.signed`。例如：

```json
{
  "default_config": {"signed": true},
  "layer_type_config": {
    "QuantizedLinear": {
      "input_signed": false,
      "input_symmetric": true,
      "weight_signed": true,
      "weight_symmetric": false,
      "output_signed": true
    }
  }
}
```

该示例表示 UINT 对称输入、INT 非对称权重和 INT 输出。当前下游编译器仅支持对称
量化，部署配置仍须满足其限制。GRU/LSTM 的内部配置继续由各算子包维护。

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

GRU 和 LSTM 的所有内部量化点默认均为对称量化。新建实例的 JSON 省略整个量化点
或该点的 `is_symmetric` 时，使用 `true`；激活量化点显式设置 `false` 时仍按非对称
配置执行。对称性与 signedness 独立，sigmoid 门输出保持默认无符号量化。

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
    load_path="output/model.json",
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

阶段文件加载以保存的 `qmin/qmax/symmetric/scale/offset` 为准，包括 Po2 对齐后
与初始配置不同的 bias 位宽。普通 affine 量化器使用 scale/offset 参数化精确恢复，
不再从实数 min/max 反推网格；加载后再冻结量化器参数或创建训练 optimizer。
目标量化器形状必须匹配，非法 scale、offset 或不一致的范围会抛出 `ValueError`。

## 检查清单

- 配置文件路径指向仓库中实际存在的文件。
- 基础配置在创建 QuantSim 时加载，混合精度配置在校准前应用。
- 百分位参数在 `compute_encodings` 前设置。
- 校准数据与部署输入的名称、形状、类型和分布一致。
- QuantGRU 配置在校准前加载，并使用共享校准过程。
- QAT 前冻结量化参数和 BatchNorm。
- 重建模型使用与训练侧相同的模型预处理和量化配置。
- 性能或精度结论包含模型版本、数据集、配置和复现命令。

## QuantLSTM 配置

`LSTM_config` 独立控制 LSTM 量化开关和原生 `quant_config`，在校准前应用。
KWS 的两种模式共用 `examples/config/quick_start_full_quant.json`，分别读取
`GRU_config` / `LSTM_config`。LSTM 示例设置 cell state 为 16 位、权重为 8 位
per-channel；未列出的量化点保留默认值。用户可直接修改配置中的对应节。
字段对照、最小 JSON 和默认值见 [示例配置说明](../examples/config/README.md)，
Po2 和参数锁见 [LSTM 接入说明](QuantLSTM_integration.md)。

### Conv/Linear bias 对齐的范围检查

`apply_power_of_2_workflow(..., align_bias_scale=True, bias_bitwidth=32)` 在 PyTorch 中
实际将已校准 Conv/Linear 的 bias 量化器设为指定的有符号位宽（默认 32），并使用精确的
`Sb = Sx * Sw`。对齐前逐通道检查当前 bias 的整数码是否可表示。显式请求 16 位但范围
不足、32 位仍溢出、scale 小于量化器下限或通道布局不兼容时抛出 `ValueError`，不再静默
截断或吞掉错误。普通未校准/未量化层仍报告为 skipped，需在使用部署结果前检查统计。

支持标量输入 scale，以及逐张量或与输出通道一一对应的权重/bias scale。目标后端需要支持
所选 bias 位宽；不能仅修改元数据而让部署继续按 bias16 执行。对齐只检查当前参数，QAT
期间仍应监测固定编码下的参数饱和；重复调用不会改变已经对齐的网格。

### FX 展开模块的参数量化

自定义 BN 等模块展开后，`weight/bias` 仍由原模块的 `param_quantizers` 管理。
QuantSim 在 FX `get_attr` 后、reshape/view 前执行这些量化器；直接消费该参数
或其形状变换结果的计算节点不再重复启用该路 input quantizer。激活、运行均值
和标准差等 buffer 输入仍按普通算子配置量化。参数边只有一个量化归属，导出通过
`param_encodings` 覆盖该边，而不是再写一份不同的 activation encoding。

按模块名称修改参数位宽后需重新校准。参数量化不会 detach master 权重，
`param_quantizers[name] = None` 会旁路该参数 QDQ。普通浮点检查点会移除新增的
FX 量化调用；带 QDQ 权重的导出则使用实际执行的参数网格。旧版为同一参数
同时保存父参数编码和子输入编码的阶段文件不保证兼容，应从固定 FP 权重重新校准。
