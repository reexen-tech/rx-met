# 大模型量化配置说明（rx-met）

Example 目录下的 JSON **只展示基本用法**。未写出的字段使用默认值（见下文「默认值」）。

---

## 最小配置

```json
{
  "model": "/data/models/Qwen3-30B-A3B",
  "quant": "Q4_K_64"
}
```

`model` 填 HuggingFace 下载后的模型目录（含 `config.json`、权重文件等）。rx-met 在内部将其转为 FP16 GGUF 再量化，用户无需单独跑 `convert_hf_to_gguf`。

---

## 常用可选字段

```json
{
  "model": "/data/models/Qwen3-30B-A3B",
  "quant": "Q4_K_64",

  "output": "./runs/my_experiment",

  "calib": {
    "dataset": "/data/wiki.train.raw",
    "chunks": 100
  },

  "eval": {
    "dataset": "/data/wiki.test.raw",
    "context": 512,
    "chunks": 8
  },

  "mixed_precision": [
    { "pattern": ".*\\.attn_v\\.weight$", "type": "Q5_K_64" },
    { "pattern": ".*\\.ffn_down\\..*",     "type": "Q5_K_64" }
  ],

  "hw_export": true
}
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `model` | 是 | HuggingFace 模型权重目录（下载产物）；流水线自动 HF → GGUF → 量化 |
| `quant` | 是 | 主量化类型（`Q4_K_64`、`Q4_K_M`、`Q4_0` 等） |
| `output` | 否 | 实验输出目录；默认 `runs/<model_stem>-<quant>/` |
| `calib.dataset` | 否 | 校准文本；有则跑 imatrix，无则跳过 |
| `calib.chunks` | 否 | imatrix 采样块数；默认 `100` |
| `eval.dataset` | 否 | PPL 数据集；有则跑评估，无则跳过 |
| `eval.context` | 否 | PPL 上下文长度；默认 `512` |
| `eval.chunks` | 否 | PPL 采样块数；默认 `8` |
| `mixed_precision` | 否 | 按 tensor 名正则覆盖量化类型 |
| `hw_export` | 否 | `true` 时导出硬件 tiled GGUF；默认 `false` |

---

## 默认值（用户无需配置）

| 项 | 默认 |
|----|------|
| 二进制路径 | `$RX_MET_HOME/bin`（安装后自动解析，用户 JSON 不填） |
| HF → GGUF | 自动，中间 FP16 GGUF 写入 `output` 目录 |
| 量化输出文件名 | `<model_stem>-<quant>.gguf` |
| GPU 层数 | 全部 offload |
| `flash_attn` | `true` |
| `n_threads` | CPU 核数 |
| manifest / report | 始终生成 |

---

## 高级配置

以下字段一般不需要在 Example 中展示，按需添加：

| 字段 | 说明 |
|------|------|
| `model`（GGUF 文件路径） | 已有 FP16 GGUF 时可直传文件路径，跳过 HF 转换（高级） |
| `quant.output_tensor_type` | 输出 tensor 单独量化类型 |
| `quant.token_embedding_type` | embedding 层量化类型 |
| `eval.cache_type_k` / `cache_type_v` | KV cache 量化类型（小写，如 `q4_0`） |
| `calib.reuse_imatrix` | 复用已有 imatrix 文件路径 |
| `hw_export.patterns` | 硬件导出 tensor 名匹配；默认 attn/ffn GEMM |
| `advanced.prune_layers` | 剪枝层列表 |
| `advanced.override_kv` | 覆盖模型 KV 元数据 |

---

## 不应出现在用户 JSON 中的项

由安装包 / 运行时自动处理，**不要**在配置里写：

- `binaries.*`、`ld_library_path`
- `report.*`
- `calibration.enabled` / `quantization.use_imatrix`（由 `calib.dataset` 是否存在决定）
- `evaluation.enabled`（由 `eval.dataset` 是否存在决定）
- `eval.psum_bits` / `REEX_Q64_PSUM_BITS`（内部硬件实验用，不对客户暴露）

上述内容会出现在运行产物 `resolved_config.json` / `manifest.json` 中，供审计用。

---

## 运行

```bash
rx-met examples/config/qwen3_reex_q4_k_64.json
```

REEX block-64 量化类型列表见 [`llama.cpp/REEX_Q64_USAGE.md`](../../llama.cpp/REEX_Q64_USAGE.md)。
