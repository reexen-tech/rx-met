# AIMET 量化配置

本文说明 PyTorch QuantSim、普通层混合精度和原生循环层的配置字段、匹配规则及生效时机。
ONNX 量化器使用相同的配置思路，但匹配对象是 ONNX 图；见 [ONNX PTQ](ONNX_PTQ_COMPILER.md)。
首次接入模型见 [PyTorch 模型量化](PyTorch_quantization.md)，运行现有模型见
[示例说明](../examples/README.md)。制品格式、权重和编码恢复见
[导出与恢复参考](Export_and_reload.md)。

## 配置文件

仓库提供两份配置：

| 文件 | 使用位置 | 作用 |
| --- | --- | --- |
| `examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json` | `QuantizationSimModel(..., config_file=...)` | 控制量化器是否启用、对称性和逐通道量化 |
| `examples/config/quick_start_full_quant.json` | `apply_mixed_precision_bitwidth(..., config_file=...)` | 控制普通层位宽、禁用规则，以及独立的 `GRU_config` / `LSTM_config` |

基础配置负责创建 QuantSim，混合精度配置负责设置 `sim.model`。配置路径相对于运行
命令的当前目录解析。

## QuantSim 基础配置

基础配置使用 AIMET QuantSim 格式。当前示例配置包含以下顶层字段：

| 字段 | 作用 |
| --- | --- |
| `defaults.ops` | 设置算子输出量化和默认对称性 |
| `defaults.params` | 设置权重量化和默认对称性 |
| `defaults.strict_symmetric` | 控制严格对称量化范围；PyTorch v2 的对称量化要求为 `"False"` |
| `defaults.per_channel_quantization` | 控制参数是否默认逐通道量化 |
| `params` | 覆盖特定参数类型，例如 `bias` |
| `op_type` | 按 ONNX 算子类型覆盖默认值 |
| `supergroups` | 定义可消除中间量化器的算子组合 |
| `model_input`、`model_output` | 控制模型边界量化 |

`op_type` 遵循 ONNX 算子命名，例如 `Conv`、`Gemm`、`Add`。当前仓库的基础配置
将 `op_type` 留空，由全局默认值控制。

`strict_symmetric` 与 `is_symmetric` 不同：本仓库的 PyTorch v2 INT 对称量化使用完整有符号
整数范围，启用严格对称会在创建量化器时报错。两条路径共用配置时保持
`defaults.strict_symmetric="False"`；ONNX 可单独启用严格对称，使用窄范围（例如
INT8 的 `[-127,127]`）。

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

使用 `quant_scheme="percentile"` 时，可在 `compute_encodings` 前调用
`set_percentile_value()` 调整校准百分位。重建 QuantSim 时，模型预处理参数、基础配置、
默认位宽和量化方案必须与首次创建时一致。

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
| `default_bitwidth` | 为未匹配普通层设置默认位宽：`weight` 用于权重，`output` 同时用于输入和输出；省略的字段保留现有位宽 |
| `layer_type_config` | 按量化模块类名设置规则，例如 `QuantizedConv2d` |
| `layer_name_config` | 按完整模块名或 `*` 通配符设置规则 |
| `GRU_config` | 配置 QuantGRU 的量化开关和内部算子 |
| `LSTM_config` | 配置 QuantLSTM 的量化开关和原生 `quant_config` |
| `default_config` | 支持未匹配模块的 `disable_quantization`，以及整数符号回退值 `signed` |

### 普通层字段

普通量化模块支持以下配置项。GRU/LSTM 内部量化点由各自配置节控制。

| 配置项 | 说明 |
| --- | --- |
| `weight_bitwidth`、`bias_bitwidth` | 权重和偏置位宽 |
| `input_bitwidth`、`input_bitwidths` | 所有输入或按输入下标设置位宽 |
| `output_bitwidth` | PyTorch 模块的第 0 路输出位宽 |
| `param_bitwidth` | 除 `weight`、`bias` 外的参数位宽 |
| `weight_symmetric`、`bias_symmetric` | 权重和偏置的对称性 |
| `input_symmetric`、`output_symmetric` | 输入和输出的对称性 |
| `param_symmetric` | 其他参数的对称性 |
| `weight_signed`、`bias_signed`、`param_signed` | 权重、偏置及其他参数的整数符号，`true` 为 INT，`false` 为 UINT |
| `input_signed`、`input_signeds`、`output_signed` | 全部输入、按输入编号或输出的整数符号；列表中 `null` 不施加局部覆盖 |
| `per_channel_quantization` | 校准前调整已有 `weight` 量化器的粒度；逐通道支持 Linear、Conv1d/2d/3d 和 `groups=1` 的 ConvTranspose1d/2d/3d |
| `disable_quantization` | 是否禁用匹配模块的量化器 |

### 整数符号与对称性

Torch 和 ONNX 的整数量化默认使用 INT 和对称量化，JSON1 遗漏激活或参数的
`is_symmetric` 时统一按 True 处理；显式 False 继续使用非对称量化。
符号与对称性独立，`*_symmetric` 只调整
校准策略，修改位宽保留当前符号。UINT 对称网格使用零点 0、范围 `[0, 2^bitwidth-1]`，
校准及推理将负值截为零；默认非 strict 的 INT 对称网格使用零点 0 和完整有符号范围。
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

### 匹配与覆盖规则

当前实现按以下顺序选择规则：

1. `layer_name_config` 完整名称。
2. `layer_type_config` 模块类型。
3. `layer_name_config` 通配符。
4. 默认配置。

多条通配符同时匹配时，采用文件中首条匹配规则。配置器跳过当前模型中缺少的已注册
`Quantized*` 类型，对既未注册也不在当前模型中的类型名抛出 `ValueError`。

匹配时选择一整条规则，不逐字段合并其他类型、通配符或默认位宽规则；匹配规则中
省略的位宽保留量化器当前值。`default_config.signed` 单独作为符号回退值。
普通层位宽字段接受 2 到 32 的整数，布尔值和未知的层规则字段会报错。
`input_bitwidths` 列表存在时替代该规则的 `input_bitwidth`，其中 `null` 或缺失下标
保留对应输入位宽。

位宽和量化器是否存在是两个设置。显式输入位宽规则可为第 0 路非标量输入补建量化器；
第 1 路及后续输入必须已有量化器，才能调整其位宽。对多输入计算，先检查对应输入槽，
再通过基础配置启用需要量化的输入，或在创建 QuantSim 后显式设置量化器。若两端共享
同一个量化器，对它指定冲突的整数符号会报错。

`output_bitwidth` 在模块已有输出槽且第 0 路为 `None` 时会尝试补建量化器，默认位宽
规则也可能触发此行为；它不会为没有输出槽的模块新增槽位。输入补建和输出补建后都应
检查实际量化器，再校准；仅有配置加载统计不能证明补建成功。

### 按单个 PyTorch 算子设置位宽

按名称配置时，以本次 `sim.model` 中的完整模块名为准。ONNX/Netron 可辅助查看
计算关系，但 ONNX 节点名、张量名和 FX 节点的 `name` 不一定等于模块名，不能直接
当作 `layer_name_config` 的键。下面按“找名称、确认计算链、配置、验证”完成一次调整。

**1. 在创建 QuantSim 后、应用混合精度配置前查找模块名。**

```python
for name, module in sim.model.named_modules():
    if hasattr(module, "input_quantizers"):
        print(name, type(module).__name__)
```

自定义归一化模块经 `prepare_model()` 展开后，实际量化对象可能是内部 Add、Sqrt、
Divide 等算子，需要使用展开后的名称。重新准备模型或修改图结构后，应重新查询，
不要沿用另一张图中的自动编号。

对于 FX `GraphModule`，还可以用 `call_module` 节点的 `target` 和输入来源确认
哪个算子执行了目标计算；`target` 才是模块名：

```python
from torch.fx import GraphModule

if isinstance(sim.model, GraphModule):
    for node in sim.model.graph.nodes:
        if node.op != "call_module":
            continue
        module = sim.model.get_submodule(str(node.target))
        if type(module).__name__ in (
            "QuantizedAdd", "QuantizedSqrt", "QuantizedDivide"
        ):
            print(node.target, type(module).__name__, node.args)
```

例如查找 `(x - mean) / sqrt(var + eps)` 时，沿输入关系确认加 `eps` 的 Add、
后继 Sqrt 和使用其结果的 Divide。普通非 FX 模型仍可使用 `named_modules()`，
不能假定它具有 `.graph`。

**2. 将精确名称规则合入项目的混合精度 JSON。**

假设本次确认的模块名为 `module_add_2` 和 `module_sqrt`，可以分别调整 Add 输出
和 Sqrt 输入。下面的名称和 32 位均为示例，需替换为本次模型名称与硬件支持的
整数量化位宽；32 位不表示切换为 FP32。

```json
{
  "layer_name_config": {
    "module_add_2": {
      "output_bitwidth": 32
    },
    "module_sqrt": {
      "input_bitwidth": 32
    }
  }
}
```

保留项目 JSON 的其他配置，将以上两项合入已有的 `layer_name_config`。
`input_bitwidth` 设置全部输入；只改某一路时使用 `input_bitwidths`，例如
Divide 的 `[16, 32]` 分别设置分子、分母，`[null, 32]` 保留分子当前位宽。
`output_bitwidth` 设置当前模块的第 0 路输出。若后继模块使用独立的输入量化器，
它的位宽不会随之改变。上述规则不禁用量化器。

**3. 应用配置并检查目标量化器。**

名称规则未匹配到实际模块时，不会自动按名称报错。先用 `get_submodule()` 校验
名称存在，再应用配置并断言实际位宽；仅看到配置文件读取成功并不足以确认生效。
下例 `bitwidth_config` 是已合入上述规则的项目 JSON 路径：

```python
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth

add = sim.model.get_submodule("module_add_2")
sqrt = sim.model.get_submodule("module_sqrt")
apply_mixed_precision_bitwidth(
    sim.model, config_file=str(bitwidth_config), verbose=True,
)
assert add.output_quantizers[0] is not None
assert add.output_quantizers[0].bitwidth == 32
assert sqrt.input_quantizers[0] is not None
assert sqrt.input_quantizers[0].bitwidth == 32
```

请在 `compute_encodings` 前执行；使用已有校准结果的模型修改位宽后，应重新校准。
创建和恢复模型时使用相同的图准备步骤与配置。

**4. 校准及 Po2 转换后检查最终网格和实际数值。**

按[PyTorch 接入流程](PyTorch_quantization.md#3-校准并评估-ptq)重新校准；目标需要
Po2 时，再完成转换。此时量化器已初始化，可以查看实际 scale：

```python
for name, side in (
    ("module_add_2", "output_quantizers"),
    ("module_sqrt", "input_quantizers"),
):
    quantizer = getattr(sim.model.get_submodule(name), side)[0]
    assert quantizer is not None and quantizer.is_initialized()
    print(name, side, quantizer.bitwidth, quantizer.get_scale().detach().cpu())
```

针对零分母，使用同批输入检查本应为正的值是否在量化前后变成 0，并继续检查
后继 Sqrt 输出和 Divide 分母输入。方差或分子本身为 0 可能正常；提高上游位宽后，
下游较粗的输入网格也可能再次将正数舍入为 0。最终位宽、Po2 scale、分母非零
和任务精度都需验证，不能仅凭最终输出有限判断问题已解决。

若本次另有保存 `*_torch.encodings`，还可用其 `activation_encodings` 下的模块名
核对输入、输出编码。该文件不是所有导出入口都保证生成的产物；普通 ONNX 张量编码
也不能直接当作 PyTorch 模块名。实际制品以[导出接口约定](Export_and_reload.md)为准。
原生 ONNX PTQ 的名称定位与配置应按[ONNX PTQ 流程](ONNX_PTQ_COMPILER.md)处理。

### FX 展开模块的参数量化

自定义 BN 等模块展开后，`weight/bias` 仍由原模块的 `param_quantizers` 管理。
QuantSim 在 FX `get_attr` 后、reshape/view 前执行这些量化器；直接消费该参数
或其形状变换结果的计算节点不再重复启用该路 input quantizer。激活、运行均值
和标准差等 buffer 输入仍按普通算子配置量化。参数边只有一个量化归属。

按模块名称修改参数位宽后需重新校准。参数量化不会 detach master 权重，
`param_quantizers[name] = None` 会旁路该参数 QDQ。编码归属和旧检查点兼容条件见
[图准备与后处理](Export_and_reload.md#图准备与后处理)。

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

`operator_config` 可以配置输入、输出、权重、偏置、线性计算结果、门的输入输出和
逐元素乘法结果。完整键集合与缺省字段规则见
[GRU 配置参考](../operators/quant-gru/pytorch/config/README.md)；
`examples/config/quick_start_full_quant.json` 提供整模型使用的配置实例。

GRU 和 LSTM 的所有内部量化点默认均为对称量化。新建实例的 JSON 省略整个量化点
或该点的 `is_symmetric` 时，使用 `true`；激活量化点显式设置 `false` 时仍按非对称
配置执行。对称性与 signedness 独立，sigmoid 门输出保持默认无符号量化。

已有实例的配置合并规则不同：GRU 省略整个量化点时沿用当前值，但出现条目时，
该条目省略的字段采用加载器默认值；LSTM 将 sparse override 合并到当前配置。
修改已有配置时应按各算子的字段规则核对，不能假定省略字段总会保留原值。

`quantization_granularity` 支持 `PER_TENSOR`、`PER_GATE` 和 `PER_CHANNEL`。
`is_unsigned` 适用于值域非负的量化点，例如 sigmoid 门输出。已校准或已加载量化参数
的 QuantGRU 会跳过位宽配置加载，因此配置应用位于校准之前。

一次 `compute_encodings` 前向过程同时完成 QuantGRU 和其他模块的校准。

## QuantLSTM 配置

`LSTM_config` 独立控制 LSTM 量化开关和原生 `quant_config`，在校准前应用。
KWS 的两种模式共用 `examples/config/quick_start_full_quant.json`，分别读取
`GRU_config` / `LSTM_config`。LSTM 示例设置 cell state 为 16 位、权重为 8 位
per-channel；未列出的量化点保留默认值。用户可直接修改配置中的对应节。

字段对照、最小 JSON 和示例显式值见 [示例配置说明](../examples/config/README.md)，
原生字段、默认值和配置变更后的校准状态见
[LSTM 配置参考](../operators/quant-lstm/docs/configuration.md)。
已校准后改变配置需先显式 `reset_calibration()`；相同配置可重复应用。
两种循环层通过同一次校准前向收集统计，Po2 和参数锁的框架调用关系见
[循环算子接入](QuantLSTM_integration.md)。

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

`apply_power_of_2_workflow()` 默认使用 `cover_range`。对普通量化器，若实数网格跨度
`abs(rmax - rmin)` 与最近的 2 的幂的相对差小于默认 2%，scale 取最接近的 2 的幂；
否则 scale 向上取整到 2 的幂。原生循环算子使用各自的 Po2 转换规则。

PyTorch 工作流分别处理普通 AIMET 量化器与原生 QuantGRU/QuantLSTM。模型只有原生
循环层时，普通量化器列表可以为空，仍会执行原生 Po2，并同步、验证实际生效的编码。
校准缺失、非法 scale 或算子参数锁阻止转换时会抛出异常；FX 图中明确为非 Tensor
常量的输入槽不要求校准。可选 bias 对齐完成后再验证普通量化器，最后生成比较报告。

返回值保留原有统计字段，并增加 `verification`：`generic` 记录普通量化器数量，
`native` 记录原生模块数量和编码验证结果。各分支的 `status` 为 `verified` 或
`not_applicable`；没有启用的量化目标时，总状态为 `not_applicable`。
`all_power_of_2` 保留原有布尔兼容行为，无目标时仍为 `True`，不表示实际验证过模型。

### Conv/Linear bias 对齐的范围检查

`apply_power_of_2_workflow(..., align_bias_scale=True, bias_bitwidth=32)` 在 PyTorch 中
实际将已校准 Conv/Linear 的 bias 量化器设为指定的有符号位宽（默认 32），并使用精确的
`Sb = Sx * Sw`。对齐前逐通道检查当前 bias 的整数码是否可表示。显式请求 16 位但范围
不足、32 位仍溢出、scale 小于量化器下限或通道布局不兼容时抛出 `ValueError`，不再静默
截断或吞掉错误。普通未校准/未量化层仍报告为 skipped，需在使用部署结果前检查统计。

支持标量输入 scale，以及逐张量或与输出通道一一对应的权重/bias scale。目标后端需要支持
所选 bias 位宽；不能仅修改元数据而让部署继续按 bias16 执行。对齐只检查当前参数，QAT
期间仍应监测固定编码下的参数饱和；重复调用不会改变已经对齐的网格。

该对齐不覆盖原生 QuantGRU/QuantLSTM 内部 bias。循环算子当前保留独立配置的
bias 编码，再重缩放到乘积 scale 的累加域；它与 INT32、`Sb = Sx * Sw` 的直接
累加方案存在设计差异，详见 [Bias 编码差异](QuantLSTM_integration.md#bias-编码差异)。

## QAT 和重新加载

进入 QAT 前应冻结量化参数，并在训练模式下保持 BatchNorm 统计量冻结：

```python
from aimet_torch.utils_rx import freeze_quantizer_parameters
from aimet_torch.utils_rx import set_train_mode_freeze_bn

freeze_quantizer_parameters(sim.model, freeze_bn_affine=True)
set_train_mode_freeze_bn(sim.model)
```

冻结和创建训练 optimizer 应在校准、Po2 或编码恢复完成后执行。CPU 只支持校准
和 PTQ；浮点训练与 QAT 使用 CUDA。

恢复顺序取决于权重的保存形式。KWS 保存的是主权重，先严格加载到对应模型，再创建
QuantSim、应用相同配置并加载编码；完整 QuantSim `state_dict` 则要求匹配的量化模型。
保存方式、代码片段和输出对照见
[权重和编码恢复](Export_and_reload.md#权重和编码恢复)。

## 分阶段量化

每个模型根据自身结构与精度评估维护阶段划分、配置、校准数据、权重和评估结果。
新阶段可通过 `load_quantizer_encodings(..., allow_overwrite=False)` 锁定已确认的编码，
避免后续校准覆盖。保存、筛选和精确网格恢复规则见
[分阶段编码保存与恢复](Export_and_reload.md#分阶段编码保存与恢复)。

## 检查清单

- 配置文件路径指向仓库中实际存在的文件。
- 基础配置在创建 QuantSim 时加载，混合精度配置在校准前应用。
- 百分位参数在 `compute_encodings` 前设置。
- 校准数据与部署输入的名称、形状、类型和分布一致。
- QuantGRU / QuantLSTM 配置在校准前加载，并使用共享校准过程。
- QAT 前冻结量化参数和 BatchNorm。
- 重建模型使用与训练侧相同的模型预处理和量化配置。
- 性能或精度结论包含模型版本、数据集、配置和复现命令。
