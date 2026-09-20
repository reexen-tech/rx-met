# QuantGRU 配置说明

本目录保存 [gru_quant_bitwidth_config.json](gru_quant_bitwidth_config.json)，用于
`QuantGRU.load_bitwidth_config()`。完整使用流程见[模块 README](../../README.md)，
算子计算与 scale 推导见[量化计算流程](../../docs/GRU量化计算流程.md)。

## 配置层次

`GRU_config.default_config` 控制量化开关和 scale 模式，
`GRU_config.operator_config` 控制各算子的位宽、对称性和粒度。

以下片段展示配置结构，完整配置以同目录 JSON 为准：

```json
{
  "GRU_config": {
    "default_config": {
      "disable_quantization": false,
      "use_pot2_scale": false
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

配置使用标准 JSON。`description` 和 `comment` 字段保存说明文字。

## 全局配置

| 字段 | 配置值与行为 |
| --- | --- |
| `disable_quantization` | `false` 设置 `use_quantization=True`，`true` 设置 `use_quantization=False` |
| `use_pot2_scale` | `false` 使用 affine scale，`true` 使用 Power-of-2 scale |

新建 QuantGRU 的 `use_quantization` 和 `use_pot2_scale` 均为 `False`。
配置文件省略全局字段时，对应属性沿用实例当前值。

同目录完整 JSON 设置 `disable_quantization=false`、`use_pot2_scale=false`。
需要 Po2 scale 的配置设置 `use_pot2_scale=true`。Po2 模式采用默认
`CoverRange` 策略和 2% 容差。

## 算子配置字段

| 字段 | 含义 | 已列出算子省略属性时的取值 |
| --- | --- | --- |
| `bitwidth` | 整数量化位宽，范围为 1 至 32 | `8` |
| `is_symmetric` | `true` 使用对称标定；`false` 按范围计算零点，零点可以为 0 | `true` |
| `is_unsigned` | `true` 使用无符号范围，`false` 使用有符号范围 | 两个 sigmoid 门输出取 `true`，其余算子取 `false` |
| `quantization_granularity` | 权重、偏置的 scale 粒度 | `PER_CHANNEL` |

加载规则按配置层次执行：

- 省略整个算子条目时，实例保留该算子的当前配置。
- 算子条目存在时，加载器按上表补齐省略的属性。
- 新建实例的初始配置来自 C++ `OperatorQuantConfig`，具体值见下一节。
- 同目录完整 JSON 显式设置各算子的 `is_symmetric=true`，两个 sigmoid 门输出设置
  `is_unsigned=true`。

权重和偏置要求 `is_symmetric=true`。32 位量化使用 INT32 整数范围。校准和执行的
有效范围还受存储类型、累加宽度和模型数值分布约束。

## 算子名称与初始配置

以下表格描述新建实例在加载 JSON 之前的配置。所有条目的初始位宽均为 8。

| JSON 算子名称 | 用途 | `is_symmetric` | `is_unsigned` |
| --- | --- | --- | --- |
| `input` | 当前输入序列 | `false` | `false` |
| `output` | 每个时间步的隐藏状态 | `false` | `false` |
| `weight_ih`、`weight_hh` | 输入权重、循环权重 | `true` | `false` |
| `bias_ih`、`bias_hh` | 输入偏置、循环偏置 | `true` | `false` |
| `weight_ih_linear`、`weight_hh_linear` | 两路矩阵乘与偏置相加结果 | `false` | `false` |
| `update_gate_input`、`reset_gate_input` | sigmoid 输入 | `false` | `false` |
| `update_gate_output`、`reset_gate_output` | sigmoid 输出 | `false` | `true` |
| `new_gate_input` | tanh 输入 | `false` | `false` |
| `new_gate_output` | tanh 输出 | `true` | `false` |
| `mul_reset_hidden` | 重置门与隐藏线性结果的乘积 | `false` | `false` |
| `mul_old_contribution` | 更新门与旧隐藏状态的乘积 | `false` | `false` |
| `mul_new_contribution` | 候选状态对新隐藏状态的贡献 | `false` | `false` |

三个 `mul_*` 条目仍用于配置、校准与参数表示。执行内核将这些乘积直接对齐到目标
量化空间，计算关系见[量化计算流程](../../docs/GRU量化计算流程.md)。

`weight_ih`、`weight_hh`、`bias_ih` 和 `bias_hh` 支持以下粒度：

| 粒度 | scale 共享范围 |
| --- | --- |
| `PER_TENSOR` | 整个参数张量 |
| `PER_GATE` | 每个门，合计三个 scale |
| `PER_CHANNEL` | 每个门的每个输出通道，合计 `3 * hidden_size` 个 scale |

## Scale 与整数范围

实数与量化整数的关系为：

```text
q = clamp(round(x / scale + zero_point), qmin, qmax)
x_hat = scale * (q - zero_point)
```

affine 模式保存连续 scale，定点执行时将缩放比编码为乘子和移位。Po2 模式的 scale
为 `2^-n`，按默认 `CoverRange` 规则生成。

| 位宽 | 有符号范围 | 无符号范围 |
| --- | --- | --- |
| 1 至 31 | `[-2^(b-1), 2^(b-1)-1]` | `[0, 2^b-1]` |
| 32 | `[-2147483648, 2147483647]` | 按 INT32 范围处理 |

## 加载和调整

配置加载位于校准之前，已校准或已加载量化参数的实例会跳过
`load_bitwidth_config()`。重新标定使用以下顺序：

```python
import torch

gru.set_quant_params_locked(False)
gru.reset_calibration()
gru.load_bitwidth_config(
    "pytorch/config/gru_quant_bitwidth_config.json"
)
gru.calibrating = True
with torch.no_grad():
    for inputs in calibration_loader:
        gru(inputs.to(device="cuda", dtype=torch.float32))
gru.calibrating = False
gru.finalize_calibration()
```

以上片段从 QuantGRU 目录解析路径。`gru` 为已创建的 CUDA QuantGRU 实例，
`calibration_loader` 返回与该实例输入布局一致的代表性张量。

`set_all_bitwidth(bitwidth, is_symmetric=True)` 设置统一位宽和激活对称性，权重、
偏置保持对称。已有标定参数可通过 `adjust_quant_config()` 显式设置
`bitwidth`、`is_symmetric`、`scale` 和 `zero_point`。修改位宽后，调用方负责
提供配套 scale 或重新标定，并验证输出变化。

`get_quant_config()` 返回当前位宽和对称性。标定完成后，结果还包含可用的
`scale` 和 `zero_point`。权重、偏置 scale 以逐通道数组表示：

```python
config = gru.get_quant_config("update_gate_output")
print(config["bitwidth"], config["is_symmetric"])
print(config.get("scale"), config.get("zero_point"))
```

## 校准与执行属性

这些属性在 Python 实例上设置：

| 属性 | 初始值 | 用途 |
| --- | --- | --- |
| `calibration_method` | `"minmax"` | MinMax 范围统计；`"sqnr"` 使用直方图误差搜索；`"percentile"` 使用百分位裁剪 |
| `percentile_value` | `100.0` | Percentile 校准百分位 |
| `quant_storage_dtype` | `"float32"` | 量化值以 float32 存储；`"int32"` 进入整数存储路径 |
| `calibrating` | `False` | 收集校准数据 |
| `export_mode` | `False` | ONNX 图导出 |

## 错误与验证

| 条件 | 结果与处理 |
| --- | --- |
| JSON 含未知算子名 | 加载器抛出 `ValueError`，并列出支持的名称 |
| 位宽超出 1 至 32 | 加载器抛出 `ValueError` |
| 权重或偏置采用非对称配置 | 校准接口报告配置错误；参数使用 `is_symmetric=true` |
| 量化启用时缺少校准数据和量化参数 | 前向报告校准状态错误；先完成校准或加载导出参数 |
| 配置或校准数据变化 | 参数更新后重新评估误差、范围覆盖和保存重载一致性 |

性能与精度结果记录模型、校准数据、位宽、scale 模式和执行后端。
