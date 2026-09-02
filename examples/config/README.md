# examples/config 配置说明（小模型）

本目录 JSON 给 AIMET QuantSim / 混合精度用，字段和 ONNX 直量化共用。
路径一律填容器内路径（例如 `/datasets/...`），不要填宿主机绝对路径。

| 文件 | 用途 | 入口 |
| --- | --- | --- |
| `mrnn_quantsim_config_custom_mixed_precision_v2.json` | QuantSim 基础配置 | `QuantizationSimModel(..., config_file=...)` |
| `quick_start_full_quant.json` | 混合精度位宽 | `apply_mixed_precision_bitwidth(..., config_file=...)` |

完整流程见 `examples/quick_start_kws.py` 和 `examples/onnx_ptq_quick_start.py`。

## 1. 基础配置（QuantSim）

控制是否量化、per-channel、对称性。计算类算子（Conv / Gemm / Add / Relu / BN）默认打开输入 + 输出量化。
要关掉某个计算类，用下一节的 `disable_quantization`。`QuantGRU` 走 `GRU_config`。

## 2. 阶段配置（混合精度位宽）

匹配优先级（高 → 低）：

1. `layer_name_config` 精确名
2. `layer_name_config` 通配符（`*`）
3. `layer_type_config`（`type(module).__name__`，如 `QuantizedConv2d`）
4. `default_bitwidth` / `default_config`

`layer_type_config` 的 key 与 `sim.model` 中 `type(module).__name__` 对应。

## 3. `GRU_config`

放在阶段配置里，控制 QuantGRU 内部算子：`bitwidth`、`is_symmetric`、`is_unsigned`、`quantization_granularity`（`PER_TENSOR` / `PER_GATE` / `PER_CHANNEL`）。
