# examples/config 配置说明

本目录 JSON 分两类，字段互不通用：

| 文件                                                    | 用途                     | 入口                                                     |
| ------------------------------------------------------- | ------------------------ | -------------------------------------------------------- |
| `llm_quant.json`                                      | 大模型量化（rx-met）     | `rx-met <config.json>`                                 |
| `mrnn_quantsim_config_custom_mixed_precision_v2.json` | 小模型 QuantSim 基础配置 | `QuantizationSimModel(..., config_file=...)`           |
| `quick_start_full_quant.json`                         | 小模型混合精度位宽       | `apply_mixed_precision_bitwidth(..., config_file=...)` |

未写出的字段使用默认值。路径一律填容器/运行环境内路径，不要填宿主机绝对路径（除非你在宿主机直接跑）。

---

## 1. 大模型（rx-met）

### 1.1 最小配置

```json
{
  "model": "/models/Qwen3-30B-A3B",
  "quant": "Q4_K_64"
}
```

`model` 填 HuggingFace 模型目录（含 `config.json`）。流水线自动 HF → FP16 GGUF → 量化，无需单独转换。

### 1.2 常用可选字段

```json
{
  "model": "/models/Qwen3-30B-A3B",
  "quant": "Q4_0_64",
  "output": "/workspace/runs/my_experiment",
  "calib": {
    "dataset": "/datasets/wiki.train.raw",
    "chunks": 100
  },
  "eval": {
    "dataset": "/datasets/evaluation.txt",
    "context": 512,
    "chunks": 8
  },
  "mixed_precision": [
    { "pattern": ".*\\.attn_v\\.weight$", "type": "Q5_K_64" },
    { "pattern": ".*\\.ffn_down\\..*", "type": "Q5_K_64" }
  ],
  "hw_export": true
}
```

| 字段                | 必填 | 说明                                                     |
| ------------------- | ---- | -------------------------------------------------------- |
| `model`           | 是   | HF 模型目录，或已有 FP16 GGUF 文件路径                   |
| `quant`           | 是   | 主量化类型，如`Q4_K_64`、`Q4_0_64`、`Q4_0`         |
| `output`          | 否   | 输出目录；默认`./runs/<model_stem>-<quant>/`           |
| `calib.dataset`   | 否   | 校准文本；有则跑 imatrix，无则跳过                       |
| `calib.chunks`    | 否   | imatrix 采样块数；默认`100`                            |
| `eval.dataset`    | 否   | PPL 数据集；有则评估，无则跳过                           |
| `eval.context`    | 否   | PPL 上下文长度；默认`512`                              |
| `eval.chunks`     | 否   | PPL 采样块数；默认`8`                                  |
| `mixed_precision` | 否   | `[{pattern, type}, ...]`，按 tensor 名正则覆盖量化类型 |
| `hw_export`       | 否   | `true` 导出硬件 tiled GGUF；也可为对象（见高级）       |

### 1.3 默认值（无需配置）

| 项                | 默认                                |
| ----------------- | ----------------------------------- |
| 二进制路径        | `$RX_MET_HOME/bin`                |
| HF → GGUF        | 自动，中间 FP16 GGUF 写入`output` |
| 量化输出名        | `<model_stem>-<quant>.gguf`       |
| GPU 层数          | 全部 offload                        |
| `flash_attn`    | `true`                            |
| `n_threads`     | CPU 核数                            |
| manifest / report | 始终生成                            |

### 1.4 高级字段

按需添加，Example 一般不展示：

```json
{
  "calib": { "reuse_imatrix": "/path/to/imatrix.gguf" },
  "eval": { "cache_type_k": "q4_0", "cache_type_v": "q4_0" },
  "hw_export": { "patterns": ["blk\\..*\\.(attn_q|attn_k|attn_v|attn_output|ffn_.*)\\.weight"] },
  "advanced": {
    "output_tensor_type": "Q6_K",
    "token_embedding_type": "Q6_K",
    "prune_layers": [0, 1],
    "override_kv": []
  }
}
```

| 字段                                     | 说明                                       |
| ---------------------------------------- | ------------------------------------------ |
| `calib.reuse_imatrix`                  | 复用已有 imatrix，跳过重新校准             |
| `eval.cache_type_k` / `cache_type_v` | KV cache 量化类型（小写，如`q4_0`）      |
| `hw_export.patterns`                   | 硬件导出 tensor 名匹配；默认 attn/ffn GEMM |
| `advanced.output_tensor_type`          | 输出 tensor 单独量化类型                   |
| `advanced.token_embedding_type`        | embedding 层量化类型                       |
| `advanced.prune_layers`                | 剪枝层列表                                 |
| `advanced.override_kv`                 | 覆盖模型 KV 元数据                         |

### 1.5 运行

```bash
rx-met examples/config/llm_quant.json
rx-met --dry-run examples/config/llm_quant.json
```

### 1.6 支持的量化类型（`quant`）

推荐使用 REEX block-64 类型（`*_64` / `*_64S`），全部 block=64：

| 类型 | 位宽 | 对称性 | 说明 |
|------|------|--------|------|
| `Q4_0_64` | 4 | 对称 | Legacy，每 64 一个 fp16 scale |
| `Q5_0_64` | 5 | 对称 | Legacy |
| `Q8_0_64` | 8 | 对称 | Legacy |
| `Q8_1_64` | 8 | 对称 | Legacy，另存 fp16 和 `s` |
| `Q4_1_64` | 4 | 非对称 | Legacy，带 fp16 `m` |
| `Q5_1_64` | 5 | 非对称 | Legacy，带 fp16 `m` |
| `Q2_K_64` | 2 | 非对称 | K-quant，带 `dmin` |
| `Q3_K_64` | 3 | 对称 | K-quant |
| `Q4_K_64` | 4 | 非对称 | K-quant，常用 |
| `Q5_K_64` | 5 | 非对称 | K-quant |
| `Q6_K_64` | 6 | 对称 | K-quant，常用于 output/embedding |
| `Q2_K_64S` | 2 | 对称 | K-quant 对称变体 |
| `Q4_K_64S` | 4 | 对称 | K-quant 对称变体 |
| `Q5_K_64S` | 5 | 对称 | K-quant 对称变体 |

也支持标准 ggml 类型，例如：`Q4_0`、`Q4_1`、`Q5_0`、`Q5_1`、`Q8_0`、`Q2_K`–`Q6_K`（含 `_S`/`_M`/`_L`）、`IQ*`、`F16`、`BF16`、`F32`。

---

## 2. 小模型（AIMET QuantSim）

小模型用**两份 JSON**，职责不同：

|      | 基础配置（JSON 1）                                      | 阶段配置（JSON 2）                    |
| ---- | ------------------------------------------------------- | ------------------------------------- |
| 文件 | `mrnn_quantsim_config_custom_mixed_precision_v2.json` | `quick_start_full_quant.json`       |
| 时机 | 创建`QuantizationSimModel`                            | `apply_mixed_precision_bitwidth`    |
| 控制 | 是否量化、per-channel、对称性                           | 位宽、对称性、禁用量化                |
| 命名 | ONNX 算子名（`Conv`、`Gemm`、`Add`）              | AIMET 量化类名（`QuantizedConv2d`） |

```python
sim = quantsim.QuantizationSimModel(
    prepared_model,
    dummy_input=dummy_input,
    quant_scheme="percentile",
    config_file="config/mrnn_quantsim_config_custom_mixed_precision_v2.json",
    default_output_bw=8,
    default_param_bw=8,
)
apply_mixed_precision_bitwidth(
    sim.model,
    config_file="config/quick_start_full_quant.json",
    verbose=True,
)
```

完整流程见 `examples/quick_start.py`。

### 2.1 基础配置（QuantSim）

```json
{
  "defaults": {
    "ops": { "is_output_quantized": "True", "is_symmetric": "True" },
    "params": { "is_quantized": "True", "is_symmetric": "True" },
    "strict_symmetric": "False",
    "per_channel_quantization": "True"
  },
  "params": {
    "bias": { "is_quantized": "True" }
  },
  "op_type": {
    "Conv": { "is_input_quantized": "True", "is_output_quantized": "True" },
    "Gemm": { "is_input_quantized": "True", "is_output_quantized": "True", "per_channel_quantization": "True" },
    "Add":  { "is_input_quantized": "True", "is_output_quantized": "True" }
  },
  "supergroups": [],
  "model_input": {},
  "model_output": {}
}
```

| 字段                                  | 说明                                          |
| ------------------------------------- | --------------------------------------------- |
| `defaults.ops.is_output_quantized`  | 默认是否量化算子输出                          |
| `defaults.params.is_quantized`      | 默认是否量化权重                              |
| `defaults.params.is_symmetric`      | 权重是否对称量化                              |
| `defaults.per_channel_quantization` | 权重是否 per-channel                          |
| `defaults.strict_symmetric`         | `True` 时对称范围为 `[-127,127]`（8bit）  |
| `params.bias`                       | 覆盖 bias 是否量化                            |
| `op_type.<Op>`                      | 按 ONNX 算子名覆盖输入/输出量化与 per-channel |

注意：`is_input_quantized` **不能**放在 `defaults`，必须写在各 `op_type` 下。

常用 ONNX 名：`Conv`、`ConvTranspose`、`Gemm`、`MatMul`、`Add`、`Mul`、`Sub`、`Div`、`Pad`、`Sigmoid`、`Tanh`、`ReduceMean`。

### 2.2 阶段配置（混合精度位宽）

```json
{
  "default_bitwidth": { "weight": 8, "output": 8 },
  "default_config": { "disable_quantization": false },
  "layer_type_config": {
    "QuantizedConv2d": {
      "weight_bitwidth": 8,
      "bias_bitwidth": 16,
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "input_symmetric": true,
      "output_symmetric": true,
      "disable_quantization": false
    },
    "QuantizedFloorDivide": { "disable_quantization": true }
  },
  "layer_name_config": {
    "*.cln.module_add_*": { "disable_quantization": true }
  },
  "GRU_config": { }
}
```

匹配优先级（高 → 低）：

1. `layer_name_config` 精确名
2. `layer_name_config` 通配符（`*`）
3. `layer_type_config`（`type(module).__name__`，如 `QuantizedConv2d`）
4. `default_bitwidth` / `default_config`

| 字段                                       | 说明                            |
| ------------------------------------------ | ------------------------------- |
| `weight_bitwidth` / `bias_bitwidth`    | 权重 / bias 位宽                |
| `input_bitwidth` / `output_bitwidth`   | 激活位宽                        |
| `input_symmetric` / `output_symmetric` | 激活是否对称                    |
| `per_channel_quantization`               | 是否 per-channel                |
| `disable_quantization`                   | `true` 时关闭该层量化         |
| `default_config.disable_quantization`    | `true` 时未匹配层全部禁用量化 |

`layer_type_config` 的 key 必须与 `sim.model` 中模块类型名一致，否则报错。

### 2.3 `GRU_config`（QuantGRU）

放在阶段配置里，控制 QuantGRU 内部算子：

```json
"GRU_config": {
  "default_config": {
    "disable_quantization": false,
    "use_pot2_scale": true
  },
  "operator_config": {
    "weight_ih": {
      "bitwidth": 8,
      "is_symmetric": true,
      "is_unsigned": false,
      "quantization_granularity": "PER_CHANNEL"
    },
    "update_gate_output": {
      "bitwidth": 8,
      "is_symmetric": true,
      "is_unsigned": true
    }
  }
}
```

| 字段                                           | 说明                                                         |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `default_config.disable_quantization`        | 关闭整个 QuantGRU 量化                                       |
| `default_config.use_pot2_scale`              | 使用 power-of-two scale                                      |
| `operator_config.*.bitwidth`                 | 位宽（1–32）                                                |
| `operator_config.*.is_symmetric`             | 对称量化（zp=0）                                             |
| `operator_config.*.is_unsigned`              | `true` 用 UINT（适合 sigmoid 输出 `[0,1]`）              |
| `operator_config.*.quantization_granularity` | 仅权重/bias：`PER_TENSOR` / `PER_GATE` / `PER_CHANNEL` |

常见算子：`input`、`output`、`weight_ih`、`weight_hh`、`bias_ih`、`bias_hh`、`weight_ih_linear`、`weight_hh_linear`、`update_gate_*`、`reset_gate_*`、`new_gate_*`。

门控对应：`update_gate_*` / `reset_gate_*` 为 sigmoid 前后；`new_gate_*` 为 tanh 前后。
