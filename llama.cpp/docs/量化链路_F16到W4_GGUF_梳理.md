# 链路二：量化（F16 → W4 GGUF）详细梳理

本文档梳理 llama.cpp 中 **F16（或 F32）源模型 → 4-bit 等量化 GGUF** 的完整流程，以 F16 → Q4_K_M（W4 的一种）为主线，覆盖命令行入口、参数、加载、逐张量类型选择、反量化、量化、写 GGUF 及底层 ggml 量化 API。

---

## 1. 总览：谁在做什么

| 层级 | 文件 / API | 职责 |
|------|------------|------|
| 命令行 | `tools/quantize/quantize.cpp` | 解析参数、imatrix、tensor_type、prune、调用 `llama_model_quantize` |
| 库入口 | `include/llama.h` → `llama_model_quantize` | 对外 C API，转发到 `llama_model_quantize_impl` |
| 量化编排 | `src/llama-quant.cpp` | 加载 GGUF、建 model 元数据、逐 tensor 决定类型、F16/F32→F32、调用 ggml 量化、写 GGUF |
| 底层量化 | `ggml/src/ggml.c`（`ggml_quantize_chunk`）、`ggml-quants.h` | 按 `ggml_type` 调用具体 `quantize_xxx`，输出量化块 |
| 格式实现 | `ggml/src/ggml-cpu/quants.c`、`ggml-quants.c` 等 | 各 `quantize_row_*` / `quantize_*`（如 Q4_K 的 block 编码） |

**数据流（F16 → W4）**：  
GGUF(F16) → 读入 F16 张量 → 反量化为 F32 → 按张量名/层/imatrix 决定 `ggml_type`（如 Q4_K）→ `ggml_quantize_chunk`(F32→Q4_K) → 写回 GGUF 元数据 + 量化数据。

---

## 2. 命令行入口：tools/quantize/quantize.cpp

### 2.1 用法

```text
llama-quantize [options] model-f32.gguf [model-quant.gguf] type [nthreads]
# 或
llama-quantize [options] model-f32.gguf type   # 输出自动为 ggml-model-<type>.gguf
```

- **type**：如 `Q4_K_M`、`Q4_0`、`F16`、`COPY` 等，对应 `llama_ftype`。
- **nthreads**：量化线程数，传 0 表示用 CPU 核心数。

### 2.2 主要选项与 params 对应关系

| 选项 | 作用 | 对应 params 字段 |
|------|------|------------------|
| `--leave-output-tensor` | 不量化 output.weight | `quantize_output_tensor = false` |
| `--output-tensor-type` | output.weight 的 ggml_type | `output_tensor_type` |
| `--token-embedding-type` | token_embd.weight 的 ggml_type | `token_embedding_type` |
| `--tensor-type TENSOR=TYPE` | 指定某张量量化类型 | `tensor_types`（正则匹配名） |
| `--tensor-type-file` | 从文件读 tensor→type 列表 | 同上 |
| `--imatrix file` | 重要性矩阵（用于 K-quant/低比特） | `imatrix` + KV 写入输出 GGUF |
| `--include-weights` / `--exclude-weights` | imatrix 只用于/排除哪些张量 | 过滤 imatrix_data |
| `--allow-requantize` | 允许输入已是量化类型 | `allow_requantize = true` |
| `--pure` | 所有可量化张量用同一 default_type，不做混合 | `pure = true` |
| `--prune-layers L0,L1,...` | 剪掉指定层 | `prune_layers` |
| `--keep-split` | 输出保持与输入相同的分片数 | `keep_split = true` |
| `--override-kv KEY=TYPE:VALUE` | 写回 GGUF 时覆盖元数据 | `kv_overrides` |
| `COPY` 作为 type | 只拷贝不量化 | `only_copy = true` |

### 2.3 main 流程摘要

1. 解析上述选项，填充 `llama_model_quantize_params params`（默认来自 `llama_model_quantize_default_params()`）。
2. 若提供 `--imatrix`：  
   - 调用 `load_imatrix`（或旧格式 `load_legacy_imatrix`）得到 `imatrix_data`（tensor 名 → 每行/每列重要性）。  
   - `prepare_imatrix` 按 include/exclude 过滤。  
   - 设置 `params.imatrix = &imatrix_data`，并通过 `kv_overrides` 把 imatrix 文件名、dataset、entries_count、chunks_count 写入输出 GGUF。
3. `params.ftype` 由 `try_parse_ftype(argv, params.ftype, ftype_str)` 解析（支持名字如 `Q4_K_M` 或数字）。
4. 若 type 为 `COPY`，设 `params.only_copy = true`。
5. `llama_backend_init()`。
6. 确定输入路径 `fname_inp`、输出路径 `fname_out`（若未显式给出则用 `ggml-model-<ftype>.gguf`）。
7. 调用 **`llama_model_quantize(fname_inp, fname_out, &params)`**；非 0 表示失败。

---

## 3. 库 API：llama_model_quantize 与 params

- **声明**：`include/llama.h`  
  `uint32_t llama_model_quantize(const char * fname_inp, const char * fname_out, const llama_model_quantize_params * params);`

- **默认参数**：`llama_model_quantize_default_params()`（在 `src/llama-quant.cpp`）  
  - `ftype = LLAMA_FTYPE_MOSTLY_Q5_1`，`nthread = 0`，`quantize_output_tensor = true`，`allow_requantize = false`，`pure = false` 等。

- **实现**：`llama_model_quantize_impl(fname_inp, fname_out, params)`；异常时返回 1 并打日志。

---

## 4. 核心实现：llama_model_quantize_impl（src/llama-quant.cpp）

### 4.1 前期：ftype → default_type、线程、加载

- **llama_ftype → ggml_type**：  
  根据 `params->ftype` 设 `default_type`，例如：
  - `LLAMA_FTYPE_MOSTLY_Q4_K_M` → `GGML_TYPE_Q4_K`
  - `LLAMA_FTYPE_MOSTLY_Q4_0` → `GGML_TYPE_Q4_0`
  - F16/BF32/F32/COPY 对应不量化或仅类型转换。
- **nthread**：`params->nthread <= 0` 时用 `std::thread::hardware_concurrency()`。
- **use_mmap**：Linux/Windows 默认 true，macOS  false；用于 loader 读入。
- **llama_model_loader**：  
  `llama_model_loader ml(fname_inp, splits, use_mmap, false, true, false, kv_overrides, nullptr);`  
  `ml.init_mappings(false);`  
  只读元数据和权重索引，不建完整推理模型。
- **llama_model**：  
  `model.load_arch(ml); model.load_hparams(ml); model.load_stats(ml);`  
  仅用于架构、超参、统计信息，供后面“按层/按张量名”选量化类型使用。
- **quantize_state_impl qs(model, params)**：  
  保存 model 与 params，并维护 `n_attention_wv`、`n_ffn_down` 等计数和 `has_imatrix`、`has_output` 等状态，供 `llama_tensor_get_type` 使用。
- 若 **only_copy**：把 `ftype` 改为输入文件的 `ml.ftype`，后面不再改类型只拷贝。

### 4.2 输出 GGUF 与 KV

- **gguf_context_ptr ctx_out**：`gguf_init_empty()`。
- 从输入 **复制 KV**：`gguf_set_kv(ctx_out.get(), ml.meta.get())`。
- 写 **量化相关元数据**：  
  - `general.quantization_version` = `GGML_QNT_VERSION`  
  - `general.file_type` = 目标 `ftype`
- 若有多分片则移除 split 相关 key；若有 **kv_overrides** 则按类型写回 bool/int/float/str。
- **prune**：若提供 `prune_layers`，用 `remap_layer` 重写 tensor 名（blk.%d.）并可能丢弃部分张量；prune 后更新 `general.block_count`。

### 4.3 权重列表与顺序

- 遍历 `ml.weights_map`，对每个 tensor 做 `remap_layer`（prune 时可能得到空名并跳过）。
- 得到 **tensors**（`vector<const llama_tensor_weight *>`）；若 `keep_split` 则按 `(idx, offs)` 排序，便于按分片写多个 GGUF。
- 统计各层/各角色张量数量（如 attn_v、output），用于 `llama_tensor_get_type` 的“前几层/后几层用更高比特”等逻辑。
- 为每个分片准备 `ctx_outs[i]`，先 **只添加 tensor 元数据**（`gguf_add_tensor`），不写数据；若多分片则设置 split_no、split_count、split_tensors_count。

### 4.4 逐 tensor 循环：读数据 → 是否量化 → 类型 → F32 → 量化 → 写文件

对 **tensors** 中每一项：

1. **切换输出文件**：若 `keep_split` 且当前 tensor 的 `idx` 变了，先关闭当前 ofstream，把当前分片的 meta 写回文件头，再打开下一分片文件并预留 meta 空间。
2. **读入当前 tensor 数据**：  
   - 若非 mmap：`read_data` 作缓冲区，`tensor->data = read_data.data()`。  
   - `ml.load_data_for(tensor)`：从 mmap 或文件中把该 tensor 的二进制数据载入 `tensor->data`。
3. **是否量化（quantize）**：  
   - 仅对“名字以 `.weight` 结尾”的 2D/3D 张量考虑量化；  
   - 排除：norm、output（除非 `quantize_output_tensor`）、ffn_gate_inp、altup、laurel、per_layer_model_proj、pos_embd、token_types、Mamba/RWKV 等小权重、attn_rel_b、position_embd 等。  
   - 若 `only_copy` 则 quantize 置 false。
4. **目标类型 new_type**：  
   - 若 quantize：`new_type = default_type`；  
   - 若 **非 pure** 且 default 为量化类型：  
     - 先查 **tensor_types**（--tensor-type/--tensor-type-file）是否匹配该 tensor 名，匹配则用指定 ggml_type；  
     - 否则 **llama_tensor_get_type(qs, default_type, tensor, ftype)**：根据张量名（output、token_embd、attn_v、attn_k、attn_q、ffn_down、ffn_gate、ffn_up、attn_output 等）、层号、n_gqa、n_expert、是否 imatrix 等，在 default 基础上做 **混合精度**（例如部分层用 Q5_K/Q6_K，attn_v 用更高比特等）。  
   - 若张量维度与 `new_type` 的 block 不兼容（如 `nx % qk_k != 0`），则做 **fallback**（例如 Q4_K → Q5_0，再不行 → F16），并递增 `n_fallback`。  
   - `output_tensor_type` / `token_embedding_type` 若已设置，则覆盖对应张量。  
   - 若最终 `new_type == tensor->type`，则不再量化，直接拷贝。
5. **得到 F32 数据**：  
   - 若已是 F32：`f32_data = (float *)tensor->data`；  
   - 若为已量化且未允许 requantize：抛错；  
   - 否则：**llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread)**，得到 `f32_data`。  
   - 即：**F16/BF16/任意已支持量化类型 → 统一反量化为 F32**。
6. **imatrix 校验**：  
   - 部分低比特类型（如 IQ2_XXS、IQ2_XS、IQ1_S、Q2_K_S 等）**强制要求** imatrix；若未提供则抛错。
7. **分配量化输出缓冲**：`work` 至少 `nelements*4`（上界），`new_data = work.data()`。
8. **按 expert（ne[2]）分块量化**：  
   - 对 3D 张量（MoE 等）按 `i03` 分开；每块有独立 `f32_data_03`、`new_data_03`、可选 `imatrix_03`。  
   - **llama_tensor_quantize_impl(new_type, f32_data_03, new_data_03, chunk_size, nrows, n_per_row, imatrix_03, workers, nthread_use)**：  
     - 内部按行块划分，多线程时每个线程调用 **ggml_quantize_chunk(new_type, f32_data, new_data, start, nrow, n_per_row, imatrix)**；  
     - 单线程则直接一次 `ggml_quantize_chunk(..., 0, nrows, n_per_row, imatrix)`。  
   - 累加 `new_size`。
9. **写回 GGUF**：  
   - `gguf_set_tensor_type(ctx_outs[cur_split], name, new_type)`；  
   - `gguf_set_tensor_data(ctx_outs[cur_split], name, new_data)`；  
   - 当前分片文件：`fout.write(new_data, new_size)`，再按对齐 padding（zeros）。
10. 累计 `total_size_org` / `total_size_new`；循环结束后关闭文件，把对应分片的 meta 写回文件头。

### 4.5 反量化：llama_tensor_dequantize_impl

- **输入**：`tensor`（F16/BF16 或已量化类型）、`output`（float 缓冲）、workers、nelements、nthread。
- **逻辑**：  
  - F16：`ggml_fp16_to_fp32_row`；  
  - BF16：`ggml_bf16_to_fp32_row`；  
  - 已量化：用 `ggml_get_type_traits(tensor->type)->to_float`。  
- 多线程时按 block 切分，每个线程处理一段连续元素，最后 join。  
- 结果写入 `f32_conv_buf`，供后续 `ggml_quantize_chunk` 使用。

### 4.6 单张量量化：llama_tensor_quantize_impl

- **参数**：`new_type`，`f32_data`，`new_data`，`chunk_size`，`nrows`，`n_per_row`，`imatrix`，workers，nthread。
- **单线程**：`ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix)`，再 `ggml_validate_row_data`。
- **多线程**：把行按 `chunk_size/n_per_row` 分成多块，每块调用 `ggml_quantize_chunk(..., first_row*n_per_row, this_nrow, n_per_row, imatrix)`，写进 `new_data` 对应偏移；每块校验，任一失败则抛错。
- **返回值**：量化后字节数（所有块之和）。

---

## 5. 底层：ggml_quantize_chunk（ggml/src/ggml.c）

### 5.1 签名与约定

```c
size_t ggml_quantize_chunk(
    enum ggml_type   type,    // 目标量化类型
    const float *    src,     // F32 源数据
    void *           dst,     // 输出缓冲
    int64_t          start,   // 起始元素下标（必须按 block 对齐）
    int64_t          nrows,   // 行数
    int64_t          n_per_row, // 每行元素数（即 ne[0]）
    const float *    imatrix  // 可选，重要性矩阵
);
```

- **start**：必须是 `type` 的 block 大小的整数倍，且与 `n_per_row` 对齐（按行起始）。
- 若 `ggml_quantize_requires_imatrix(type)` 为 true（部分 IQ/Q2_K 等），则 **imatrix 必非 NULL**。
- 内部会调用 **ggml_quantize_init(type)**（初始化量化表等，线程安全）。

### 5.2 按 type 分发

- **Q4_0/Q4_1/Q5_0/Q5_1/Q8_0**：`quantize_q4_0`、`quantize_q4_1` 等，声明在 `ggml-quants.h`，实现多在 `ggml-quants.c` 或 ggml-cpu。
- **Q2_K/Q3_K/Q4_K/Q5_K/Q6_K**：`quantize_q2_K`、`quantize_q4_K` 等，K-quant 按块编码 scale/min 等。
- **IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/IQ4_NL/IQ4_XS**：`quantize_iq2_xxs` 等，部分需 imatrix。
- **TQ1_0/TQ2_0**：三值量化。
- **MXFP4**：MoE 用。
- **F16/BF16/F32**：不做量化，只做类型转换或 memcpy（`ggml_fp32_to_fp16_row`、`ggml_fp32_to_bf16_row_ref`、memcpy）。

每个分支调用形如 `quantize_xxx(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix)`，返回写入字节数；**result 必须等于 nrows * row_size**（row_size = `ggml_row_size(type, n_per_row)`）。

---

## 6. F16 → Q4_K_M 路径小结（W4 示例）

1. 用户执行：`llama-quantize model-f16.gguf Q4_K_M`（或指定输出路径和 nthreads）。
2. **quantize.cpp**：解析 `Q4_K_M` → `params.ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M`，可选 imatrix、tensor_type、prune 等，调用 `llama_model_quantize(inp, out, &params)`。
3. **llama_model_quantize_impl**：  
   - `default_type = GGML_TYPE_Q4_K`；  
   - loader 打开 GGUF，model 只加载 arch/hparams/stats；  
   - 建输出 ctx_out，复制 KV 并设 `general.file_type`、quantization_version；  
   - 对每个权重 tensor：  
     - **load_data_for** 读入 F16 数据；  
     - 判定需要量化且非 output 等例外；  
     - **llama_tensor_get_type** 在 Q4_K 基础上对 attn_v、ffn_down、output 等做混合（如部分 Q5_K/Q6_K）；  
     - **llama_tensor_dequantize_impl**：F16 → F32；  
     - **llama_tensor_quantize_impl** → **ggml_quantize_chunk(GGML_TYPE_Q4_K 或混合类型, f32, dst, ...)**；  
     - 写 tensor 类型与数据到 ctx_out 和当前分片文件。
4. **ggml_quantize_chunk**：对 Q4_K 调用 `quantize_q4_K(src, dst, nrows, n_per_row, imatrix)`，按 block 编码为 Q4_K 格式，返回字节数。
5. 所有 tensor 处理完后，把各分片 meta 写回文件头，关闭文件。

---

## 7. 相关文件速查

| 用途 | 文件 |
|------|------|
| 命令行、imatrix 加载、params 组装 | `tools/quantize/quantize.cpp` |
| 量化 API、params 默认值 | `include/llama.h` |
| 量化编排、反量化、类型选择、写 GGUF | `src/llama-quant.cpp` |
| 张量类型选择（混合精度） | `src/llama-quant.cpp` 中 `llama_tensor_get_type` |
| quantize_chunk 分发 | `ggml/src/ggml.c`（ggml_quantize_chunk） |
| 各格式 block 与 quantize/dequant 声明 | `ggml/src/ggml-quants.h` |
| 各 quantize_xxx 实现 | `ggml/src/ggml-quants.c`、`ggml-cpu/quants.c` 等 |
| 模型加载（只读权重） | `src/llama-model-loader.cpp` |

以上即 **F16 → W4（如 Q4_K_M）GGUF 量化** 的完整链路梳理；其他 W4 类型（Q4_0、Q4_1、Q4_K_S 等）仅 ftype/default_type 和部分张量的混合策略不同，流程一致。
