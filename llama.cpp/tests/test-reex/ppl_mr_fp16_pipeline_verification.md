# FP16 Pipeline MR 分支 PPL 回归验证报告

**分支**: `feature/fp16-pipeline-to-main`  
**目标**: `main`  
**模型**: Qwen3-30B-A3B-Instruct-2507-Q4_K.gguf  
**数据集**: wikitext-103-raw-v1 (validation.txt)  
**配置**: 100 chunks, n_ctx=512, batch_size=512, flash_attn=on, ngl=99  

---

## 1. 测试结果（MR 分支）

| KV Cache 配置 | PPL | 标准差 |
|---------------|-----|--------|
| **KV F16**    | **7.0388** | ± 0.11764 |
| **KV K8V4** (K=q8_0, V=q4_0) | **7.0576** | ± 0.11785 |
| **KV Q8Q8**   | **7.0343** | ± 0.11745 |

---

## 2. 与历史基线对比（无精度回归）

| 配置   | 历史基线 (feature/pwnl-mixed-fp16) | MR 分支 (feature/fp16-pipeline-to-main) | 差异 | 判定 |
|--------|-----------------------------------|----------------------------------------|------|------|
| KV F16 | 7.0445 ± 0.118                   | 7.0388 ± 0.118                         | −0.0057 | ✅ 在误差范围内 |
| KV K8V4| 7.0542 ± 0.118                   | 7.0576 ± 0.118                         | +0.0034 | ✅ 在误差范围内 |
| KV Q8Q8| 7.0375 ± 0.118                   | 7.0343 ± 0.118                         | −0.0032 | ✅ 在误差范围内 |

统计误差约 ±0.118，上述差异均小于 0.01，**判定为无精度回归**。

---

## 3. 其他验证

- **Backend-ops**: RMS_NORM F16 19/19、SOFT_MAX F16 216/216、MUL_MAT_ID 621/621 通过。
- **K/V 与 set_rows**: 自研路径允许 K/V 在注意力侧为 F16；在 `llama_kv_cache::cpy_k` / `cpy_v` 内插入 `ggml_cast(F16→F32)` 后调用上游 `ggml_set_rows`（不要求改写 ggml 的 SET_ROWS）。`llama-perplexity` 无崩溃，PPL 正常。

---

## 4. 结论

FP16 pipeline MR 分支在 **Qwen3-30B-A3B Q4_K + wikitext-103-raw-v1 (100 chunks)** 下与历史基线一致，**建议合入 main**。
