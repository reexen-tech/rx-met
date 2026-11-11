# QuantLSTM 配置与校准

本文是 `QuantLSTM` 用户配置、校准状态和量化参数交换格式的参考。量化公式与
Round/Clamp 位置以[量化执行规格](quantized-execution-spec.md)为准。

## 1. 配置来源与优先级

运行时配置具有两种 JSON 表示：

| 表示 | Schema | 用途 |
| --- | --- | --- |
| sparse override | `schemas/lstm_quant_override.schema.json` | 构造模块或修改配置时提供用户覆盖项 |
| resolved config | `schemas/lstm_quant_resolved.schema.json` | C++ resolver 生成的完整执行配置 |

默认值来自 `config/defaults/lstm_quant_default_v1.json`，构建 Python wheel 时会
打包这份文件。使用已安装的 wheel 时，通过 `quant_config` 为实例覆盖默认值。
算子根目录的 `schemas/` 存放格式校验规范。解析顺序为：

```text
default config <- user override -> canonical resolved config
```

Python 不补充默认值。`QuantLSTM` 构造函数把 override 原文交给 C++ resolver，
`get_quant_config()` 返回 resolver 产生的完整 resolved config。

三类值需要分别理解：

- 新建实例没有提供 `quant_config` 时，使用只包含 `schema_version=1` 的空 override，
  因而得到默认文件中的全部值。
- override 省略某个字段时，该字段使用默认文件中的值。
- 示例显式传入的字段只影响该实例，不改变仓库默认文件。

## 2. 顶层字段

最小有效 override 为：

```json
{
  "schema_version": 1
}
```

完整顶层结构为：

| 字段 | 类型 | 必需 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `schema_version` | integer | 是 | 无 | 当前只接受 `1` |
| `scale_mode` | string | 否 | `affine` | 可选 `affine` 或 `pot2` |
| `operators` | object | 否 | 18 个默认量化点 | key 是真实量化点名称 |

未知字段、重复 JSON key、`null`、错误类型和非法枚举会直接失败。resolver 不忽略
无法识别的配置。

## 3. Operator 字段

各 operator 的可配置字段如下：

| 字段 | 类型 | 取值 | 默认值 |
| --- | --- | --- | --- |
| `bitwidth` | integer | signed `1–32`、unsigned `1–31`；signed symmetric 从 `2` 起 | `8` |
| `is_unsigned`（仅非参数量化点） | boolean | `true` 或 `false` | 见下表 |
| `is_symmetric`（仅非参数量化点） | boolean | `true` 或 `false` | `true` |
| `granularity`（仅权重和偏置） | string | `per_tensor`、`per_gate`、`per_channel` | `per_channel` |

只有 `weight_ih`、`weight_hh`、`bias_ih` 和 `bias_hh` 可以修改 `granularity`。
这四组参数只展示和接受 `bitwidth/granularity`，内部固定为 signed symmetric；
`is_unsigned/is_symmetric` 不再作为配置字段，导入量化参数时非零 zero point 仍会失败。其余量化点在内部固定为 `per_tensor`，默认 JSON 和
`get_quant_config()` 返回的 resolved config 均不展示、也不接受该字段。
旧 resolved 配置需要删除非参数 `granularity` 和权重、偏置的
`is_unsigned/is_symmetric` 字段后再加载。
校准参数交换文档中的类型和粒度元数据仍保留，用于解释整数编码及 scale 分组。

例如两组权重可设为 `4`，而输入、状态和内部激活保持 `8`，无需更换计算后端。
权重和偏置固定 signed symmetric，所以支持 `2–32` 位；INT1 对称校准的正端点为
零，UINT32 超出现有整数范围元数据，两者均在配置阶段拒绝。失败的配置不改变原模块。
原 JSON 字段、结构和版本不变，`dtype` 仍使用 `INTn` / `UINTn`。

Signed 量化在 CPU/CUDA 和 QAT 中均使用完整范围：INT8 `[-128,127]`、
INT16 `[-32768,32767]`。除 bias 的训练余量规则外，signed `is_symmetric=true` 校准使用
`scale=max_abs/(2^(bitwidth-1)-1)`、`zero_point=0`。导出的 `real_min/real_max`
由整数上下界、最终 scale 和 zero point 计算；最小负值合法，不应提前截断。

默认 signed/unsigned 设置为：

| 量化点 | `is_unsigned` | `is_symmetric` |
| --- | ---: | ---: |
| `input_gate_output` | `true` | `true` |
| `forget_gate_output` | `true` | `true` |
| `output_gate_output` | `true` | `true` |
| 其他 15 个量化点 | `false` | `true` |

18 个真实量化点的完整列表见
[量化执行规格的真实量化点](quantized-execution-spec.md#3-真实量化点)。

将全部量化点设为 16 位，保留各自的 signed/unsigned 设置：

```python
import torch
from quant_lstm import QuantLSTM

module = QuantLSTM(16, 32, device="cuda")
module.set_all_bitwidth(16)
assert module.get_quant_config("input")["bitwidth"] == 16
```

修改单个量化点：

```python
module.adjust_quant_config(
    "cell_state",
    bitwidth=16,
    is_unsigned=False,
    is_symmetric=True,
)
```

`set_all_bitwidth()` 和 `adjust_quant_config()` 会立即重新运行 resolver，并使已有
校准参数失效。修改后必须重新校准或加载与新配置一致的参数文档。

### 3.1 量化点说明

默认 JSON 为每个量化点提供一条可选的字符串 `comment`，解释其对应张量，例如：

```json
{
  "forget_gate_input": {
    "bitwidth": 8,
    "is_unsigned": false,
    "is_symmetric": true,
    "comment": "遗忘门 f 经过 sigmoid 之前的输入，两条线性分支对应切片之和。"
  }
}
```

默认文件和 sparse override 均支持该注释。它不参与量化计算，C++ resolver 生成
canonical resolved config 时会移除，因此 `get_quant_config()` 不返回注释。
默认文件作为可编辑的配置来源不要求 canonical 字节格式；运行时 canonical config
的严格字节校验保持不变。`comment` 不是量化选项，operator override 仍须至少包含
一个实际配置字段。`schema_version` 是格式标识，当前固定为 `1`。

## 4. 校准

### 4.1 方法

构造参数 `calibration_method` 默认是 `"minmax"`，接受以下值：

| 方法 | 行为 |
| --- | --- |
| `minmax` | 使用全部观测值的最小值和最大值 |
| `percentile` | 使用 2,048-bin histogram 的中心百分位范围，`percentile_value` 默认 `99.99` |
| `sqnr` | 在 histogram 候选范围中选择量化噪声最小的范围 |

非 bias 量化点的候选范围通过统一 MinMax、minimum-scale fallback、可选 POT2
CoverRange 和执行参数派生链生成最终 standard scale/zp。`bias_ih/bias_hh` 不进行
直方图裁剪，三种方法均从原始观测范围生成带训练余量的独立网格：
`scale=max(2*max_abs,0.01)/qmax`、`zero_point=0`；POT2 将该 scale 向上取整到
2 的整数次幂。加载参数不会再次加入余量。公式与适用阶段见
[Bias 校准规格](quantized-execution-spec.md#55-bias-校准与训练余量)。
校准方法不改变外部参数格式。

`module.percentile_value` 接受 `(0, 100]` 的有限数值。修改它或
`calibration_method` 时，只有值发生变化才重置校准参数及参数锁。

正式校准需要使用来自目标训练或推理分布的代表性数据。随机 tensor 只适用于接口
冒烟测试，不能作为模型精度校准数据。

### 4.2 状态与流程

校准会话状态为：

```text
empty --collect--> dirty --finalize--> locked
  ^                                  |
  +------------- reset -------------+
```

以下流程接续已创建的 `module`。先加载模型的浮点权重，再准备同设备的
`calibration_batches`；每项是 `(inputs, state)`，其中 `state` 可以是
`(h_0, c_0)` 或 `None`（零初始状态）。CPU 模块先调用 `module.eval()`。

```python
module.reset_calibration()
module.calibrating = True

with torch.no_grad():
    for inputs, state in calibration_batches:
        module(inputs, state)

module.calibrating = False
report = module.finalize_calibration()
assert report["batch_count"] > 0
assert module.is_calibrated()

module.use_quantization = True
output, (hidden, cell) = module(inference_input, inference_state)
```

所有 module 参数、calibration batch 和状态必须是同设备 FP32 tensor。一次 CUDA
浮点 forward 同时生成输出和完整 checkpoint；range、参数分组统计、直方图和三个
cell contribution 诊断在设备端计算。只有每组 min/max 和可选 histogram bins 返回
C++ finalization，原始 tensor 不复制到 CPU，也不会执行第二次 forward。

`finalize_calibration()` 锁定会话并生成 standard scale/zp、内部执行编码和
`NumericSafetyReport`。locked 状态拒绝继续 collect；`reset_calibration()` 是开始
新会话的入口。双向模块分别维护 forward/reverse 状态，且两个方向必须共享完整
input 网格。

CPU wheel 复用 C++ 浮点 reference 的输出与 trace 进行校准，各量化点先在 batch 内
收集完整直方图，再合并为会话统计。CPU 调用要求 `eval()`，无需额外关闭梯度；推荐使用
`torch.no_grad()` / `torch.inference_mode()` 减少推理开销。CPU 不支持训练和 QAT，
不会生成训练 checkpoint，实际跨算子的反向传播会明确报错，也不会作为 CUDA wheel 的隐式 fallback。

校准会话的 `locked` 状态与 `set_quant_params_locked(True)` 的参数锁分开管理。
`calibration_state()` 查询收集会话；单向返回状态字符串，双向返回包含 `forward`
和 `reverse` 状态的字典。从参数文件加载后各会话为 `empty`，但
`is_calibrated()` 为 `True`，可以直接启用量化，无需再次收集数据。
`calibration_context()` 默认开始一轮新校准，正常退出时为实际收集到数据的模块
生成参数，异常时恢复 `calibrating` 标志。参数已锁定或已经处于校准模式时，
上下文不会启动新会话。`reset_calibration()` 会同时清除参数、校准范围缓存和参数锁。

### 4.3 校准后转换为 Po2

已校准模块可以在锁定参数前调用 `module.enable_pot2()`。转换同时更新 scale、
非对称 zero point 和由整数范围派生的 `real_min/real_max`；已校准的对称参数保持
zero point 为零。校准范围的保留、文件重载后的处理和方法参数见
[集成接口中的 Po2 转换](aimet_integration.md#已校准参数的-po2-转换)。

## 5. 参数保存与加载

校准完成后可以保存算子独立的原生参数文档：

```python
module.export_quant_params("/path/to/quant_params.json")
```

新实例应保持相同的形状、布局、方向和 bias 配置，并同时恢复主权重与量化参数。
以下代码接续前面的 `QuantLSTM(16, 32, device="cuda")`，先复制其权重，再加载参数文件：

```python
restored = QuantLSTM(16, 32, device="cuda").eval()
restored.load_state_dict(module.state_dict())
restored.load_quant_params("/path/to/quant_params.json")
restored.use_quantization = True
```

量化参数文件不包含模型主权重。跨进程恢复时，从同次保存的 `state_dict` 加载权重；
完整保存和重载示例见[算子 README](../README.md#保存与重载)。

算子独立的原生检查点使用 schema v1。下列字段不写入 rx-met 提供给编译器的
encodings；rx-met 的导出和回读格式见 [通用循环算子集成接口](aimet_integration.md)。

- 单向 schema：`schemas/lstm_pytorch_quant_params.schema.json`。
- 双向 schema：`schemas/lstm_pytorch_bidirectional_quant_params.schema.json`。
- 根字段为 `schema_version`、`model_info`、`execution_metadata`、`operators`，双向
  文档另含 `operators_reverse`。

`model_info` 记录 `input_size`、`hidden_size`、`bias`、`batch_first`、
`bidirectional` 和 `use_pot2_scale`。`execution_metadata` 记录：

```json
{
  "carrier": "cuda_fp32_qcarrier",
  "activation_mode": "real_sigmoid_tanh",
  "cublas_math_mode": "pedantic",
  "standard_scale_mode": "affine"
}
```

CPU wheel 也沿用 `carrier: "cuda_fp32_qcarrier"` 这一格式标识以兼容原生 schema v1；
它不表示该次执行使用了 CUDA。实际设备由所安装的 CPU/CUDA extension 及输入 tensor
决定，CPU binding 不执行 cuBLAS，`cublas_math_mode` 字段仅随参数文档保留。

每个 operator 使用：

```json
{
  "dtype": "INT8",
  "symmetric": true,
  "scale": 0.01,
  "zero_point": 0,
  "enc_type": "PER_TENSOR",
  "real_min": -1.28,
  "real_max": 1.27
}
```

单组参数使用 JSON number/integer，多组参数使用数组。四组 weight/bias 的
`scale/zero_point/real_min/real_max` 始终是完整 `4H` 数组，`enc_type` 记录
`PER_TENSOR`、`PER_GATE` 或 `PER_CHANNEL` 来源。`bias=False` 时 bias operator
必须缺失。

参数文档以 standard scale/zp 定义量化网格，不保存原始校准统计、raw ratio、
M+shift、POT2 shift 或 Q31 编码。公开 JSON 的 scale 使用数值，Python adapter
将其舍入到 FP32 并转换为内部 canonical bundle 的规范字符串；无需在公开文件中
使用该字符串表示。C++ 随后审计 shape、
数值范围、scale、zero point 和执行安全性，并重新派生执行编码。

## 6. 加载时机与失效规则

| 操作 | 校验时机 | 对已有校准参数的影响 |
| --- | --- | --- |
| 构造 `QuantLSTM` | 构造期间解析 override | 创建空校准状态 |
| `load_quant_config()` | 调用期间重新运行 resolver | 相同配置保留参数；已有参数时拒绝不同配置，须先 `reset_calibration()` |
| `adjust_quant_config()` | 调用期间重新运行 resolver | 失效 |
| `set_all_bitwidth()` | 调用期间重新运行 resolver | 失效 |
| 修改 `calibration_method` / `percentile_value` | property 赋值期间验证 | 值变化时失效 |
| `reset_calibration()` | 立即执行 | 清除参数、参数锁、校准范围缓存和 safety report |
| `finalize_calibration()` | finalization 期间派生并审计 | 生成参数及校准范围缓存，锁定当前收集会话 |
| `enable_pot2()` | 转换后经原生加载器审计 | 更新已有 scale/zp 和实数范围；受参数锁保护 |
| `set_quant_params_locked(True)` | 检查已有参数 | 阻止共享校准覆盖及 affine→Po2 转换 |
| `load_quant_params()` | 加载期间完整审计 | 用导入参数替换会话并清除旧校准范围缓存 |

修改模型 weight 不会自动刷新量化参数。本项目的 QAT 流程沿用首次 PTQ 校准所得的
scale、zero point 和位宽，不按 epoch 重新校准。浮点主权重继续更新，每次前向按
固定参数执行量化；静态参数缓存的调用方仍需在 master 参数变化后更新 generation key。

## 7. 错误与告警

以下情况直接报错：

- 量化 forward 前没有完成校准或加载参数；
- 非 FP32 tensor、设备、shape 或 bias 不匹配，或 CPU 模块处于训练模式；
- 在 CPU 算子上执行反向传播；
- 未知 operator/字段、非法位宽、非法固定值或 compact 参数向量；
- 双向文档缺失 reverse 参数或两个方向的 input 网格不同；
- scale 非有限、非正，或转换后不能表示为有限正 FP32；内部 bundle 另需符合 canonical FP32 字符串格式；
- 内部 shift 越界、所选执行路径无法通过数值安全审计或可能产生 Inf/NaN；int32 reference 另需通过宽整数溢出证明；
- `require_exact_accumulation=True` 时，FP32 整数运算的数值审计上界达到或超过 `2^24`。

普通模式下，达到该阈值但仍有限的配置允许继续执行。接口会发出
`RuntimeWarning`，并在 safety report 中记录 `precision_risk`。该告警不会切换到
CPU、int32 或浮点 LSTM fallback；调用方需要通过精度门禁判断该配置是否可接受。
