# llama_model_loader 详细梳理

本文档梳理 `src/llama-model-loader.cpp` 与 `src/llama-model-loader.h` 中的 **llama_model_loader** 类：职责、数据结构、构造与加载流程、API 及与上层调用关系。

---

## 1. 概述与职责

**llama_model_loader** 是 llama.cpp 的 **GGUF 模型加载器**，负责：

1. **打开并解析 GGUF 文件**（支持单文件与多分片 split）
2. **维护元数据**（gguf context、键值对、override）
3. **维护权重索引**（每个 tensor 对应哪个文件、文件内偏移、ggml_tensor 元信息）
4. **按名称/维度创建 ggml 张量**（在 ggml_context 中）
5. **建立 mmap 或文件句柄**，为后续读数据做准备
6. **把权重数据加载到内存或 backend buffer**（mmap 直接映射 / 读文件 + 可选异步 GPU 上传）

调用链上，**llama.cpp** 在加载模型时构造 `llama_model_loader`，然后 **llama_model** 通过 `load_arch`、`load_hparams`、`load_vocab`、`load_tensors` 等使用 loader 的 API 完成整模型加载。

---

## 2. 头文件中的核心数据结构

### 2.1 llama_tensor_weight（单个权重的描述）

```cpp
struct llama_tensor_weight {
    uint16_t  idx;   // 源文件索引（0 = 主文件，1,2,... = 分片）
    size_t   offs;   // 该 tensor 数据在文件中的字节偏移
    ggml_tensor * tensor;  // 来自 GGUF 的 tensor 元数据（名字、形状、类型等）
};
```

- 构造时根据 `gguf_ctx` 查 tensor 在文件中的偏移，并校验 `offs + nbytes` 不越界。
- 所有权重最终存在 `weights_map` 中，按名字索引。

### 2.2 weight_name_comparer（权重排序）

- 用于 `weights_map` 的 key 比较：先按 `blk.%d.` 解析出的层号排序，同层再按名字字典序。
- 便于按层顺序遍历权重。

### 2.3 张量创建时的 flags

| 常量 | 含义 |
|------|------|
| `TENSOR_NOT_REQUIRED` | 该 tensor 可选，不存在时不抛错、返回 NULL |
| `TENSOR_DUPLICATED` | 该 tensor 被多份使用（如 MoE 共享），统计 size_data 时计多份 |
| `TENSOR_SKIP` | 跳过该 tensor（不创建、不加载） |

### 2.4 主要成员变量（llama-model-loader.h）

| 成员 | 类型 | 含义 |
|------|------|------|
| `n_kv` | int | GGUF 元数据键值对数量 |
| `n_tensors` | int | 权重 tensor 总数 |
| `n_created` | int | 已通过 create_tensor 创建的 tensor 数（用于 done_getting_tensors 校验） |
| `n_elements` | uint64_t | 所有权重元素总数 |
| `n_bytes` | size_t | 所有权重字节数 |
| `use_mmap` | bool | 是否用 mmap 映射文件 |
| `use_direct_io` | bool | 是否用 Direct I/O |
| `check_tensors` | bool | 加载后是否校验 tensor 数据合法性 |
| `no_alloc` | bool | 是否不在 loader 内分配 tensor 内存（由外部/backend 分配） |
| `files` | llama_files | 打开的文件列表（主文件 + 分片） |
| `ftype` | llama_ftype | 推断或元数据中的权重量化类型 |
| `fver` | llama_fver | GGUF 版本（V1/V2/V3） |
| `mappings` | llama_mmaps | mmap 映射列表（每个文件一个） |
| `weights_map` | map<name, llama_tensor_weight> | 权重名 → 权重描述 |
| `kv_overrides` | unordered_map<string, kv_override> | 元数据键覆盖（如 --override-kv） |
| `tensor_buft_overrides` | const llama_model_tensor_buft_override * | 张量 buffer 类型覆盖 |
| `meta` | gguf_context_ptr | 主文件的 GGUF 元数据 context |
| `contexts` | vector<ggml_context_ptr> | 各文件对应的 ggml context（仅元数据，no_alloc） |
| `arch_name` | string | 架构名（如 "llama"） |
| `llm_kv` | LLM_KV | 当前架构的 KV 键枚举封装 |
| `size_done` / `size_data` | size_t | 已加载字节数 / 总数据字节数（进度） |
| `mmaps_used` | vector<pair<first,last>> | 每个 mmap 实际用到的 [first, last) 范围（用于最后 unmap 未用区） |

---

## 3. 构造函数流程（llama_model_loader::llama_model_loader）

**位置**：`llama-model-loader.cpp` 约 506–769 行。

1. **KV / tensor 覆盖**
   - 若传入 `param_overrides_p`，填充 `kv_overrides`。
   - 保存 `tensor_buft_overrides` 指针。

2. **主 GGUF 加载**
   - `gguf_init_from_file(fname, { no_alloc: true, ctx })` 只读元数据和 tensor 索引，不读权重数据。
   - `meta` 指向该 context，`contexts[0]` 为 ggml context。
   - 打开主文件：`files.emplace_back(new llama_file(fname, "rb", use_direct_io))`。

3. **mmap 与 direct I/O 互斥**
   - 若同时开启 mmap 和 direct_io，则根据平台能力二选一并打日志。

4. **主文件权重索引**
   - 遍历 `ggml_get_first_tensor(ctx)` 到 `ggml_get_next_tensor`，对每个 tensor：
     - 检查名字不重复，累加 `n_elements`、`n_bytes`。
     - 在 `weights_map` 中插入 `llama_tensor_weight(主文件, idx=0, meta, cur)`，其中 `offs` 由 `gguf_get_data_offset + gguf_get_tensor_offset` 得到。

5. **分片（split）处理**
   - 读元数据 `general.split_count`（`get_key(LLM_KV_SPLIT_COUNT, n_split)`）。
   - 若 `n_split > 1`：
     - 要求主文件为 split 0（否则报错）。
     - 若调用方未提供 `splits` 列表，则用 `llama_get_list_splits(fname, idx, n_split)` 按命名规则生成（如 `xxx-00002-of-00004.gguf` → 4 个路径）。
     - 对 idx=1..n_split-1 逐个：`gguf_init_from_file` 分片、校验 split_no、打开文件、遍历该 ctx 的 tensor 加入 `weights_map`（idx 为当前分片下标）。
     - 读 `general.split_tensors_count` 做一致性检查。

6. **统计与版本**
   - `n_kv = gguf_get_n_kv(meta)`，`n_tensors = weights_map.size()`，`fver = gguf_get_version(meta)`。

7. **推断 ftype**
   - 按各 ggml_type 出现次数找最多的类型，映射到 `llama_ftype`（如 GGML_TYPE_Q4_K → LLAMA_FTYPE_MOSTLY_Q4_K_M）。
   - 先打上 `LLAMA_FTYPE_GUESSED`；若元数据里有 `general.file_type` 则用元数据值覆盖。

8. **日志**
   - 在 trace 时打印每个 tensor 的 split、名字、类型、形状、MiB。
   - 打印所有 KV 及类型统计。

9. **保存选项**
   - 若平台不支持 mmap 则强制 `use_mmap = false`。
   - 写入 `this->use_mmap/direct_io/check_tensors/no_alloc`。

---

## 4. GGUFMeta 命名空间（元数据读取与 override）

**位置**：约 95–264 行。

- **GKV_Base_Type / GKV_Base**：按 GGUF 类型（bool、u8/u16/u32/u64、i8/i16/i32/i64、f32/f64、string、array）绑定 `gguf_get_val_*`，用于类型安全读取。
- **GKV<T>**：
  - `get_kv(ctx, k)`：按 key 索引取对应类型，类型不符抛错。
  - `validate_override` / `try_override`：若调用方传了 `llama_model_kv_override`，则校验类型并覆盖 bool/int/float/string。
  - `set(ctx, key, target, ovrd)`：先尝试 override，否则从 ctx 读入 target。

这样 loader 的 `get_key` / `get_arr` 在内部通过 `GGUFMeta::GKV<T>::set(..., kv_overrides)` 实现“元数据 + 用户覆盖”。

---

## 5. 元数据读取 API（get_key / get_arr / get_key_or_arr）

- **get_arr_n(key, result, required)**  
  取数组类型键的**长度**，写入整型 `result`；找不到且 required 则抛错。

- **get_arr(key, result, required)**  
  `result` 为 `vector<T>` 或 `array<T,N_MAX>`，从 GGUF 数组键读出；支持 T=string/int32/uint32/float 等，类型与 GGUF 一致。

- **get_key(key, result, required)**  
  标量键，通过 `GKV<T>::set(meta, key, result, kv_overrides 中该 key 的 override)`；支持 bool、整型、浮点、string、以及枚举（如 llama_pooling_type 通过 uint32_t 中转）。

- **get_key_or_arr(key, result, n, required)**  
  - 若键为数组：要求长度等于 n，并读入 `result`。
  - 若键为标量：读一个值并复制 n 份到 `result`。  
  用于“可能是标量也可能是长度为 n 的数组”的字段（如某些 per-layer 配置）。

- **get_key(kid, ...) / get_arr(kid, ...)**  
  通过 `llm_kv(kid)` 转成字符串 key 再调上述接口，便于按架构枚举（LLM_KV_*）访问。

---

## 6. 架构与权重查询

- **get_arch_name()**：返回 `arch_name`（如 "llama"）。
- **get_arch()**：返回 `llm_kv.arch`（enum llm_arch）。
- **get_weight(name)**：在 `weights_map` 中按名查找，返回 `const llama_tensor_weight*`，无则 nullptr。
- **require_weight(name)**：同上，无则抛错。
- **get_tensor_meta(name)**：返回该权重在 GGUF 中的 `ggml_tensor*`（仅元数据）。
- **require_tensor_meta(name)**：无则抛错。

---

## 7. 张量创建与校验

- **check_tensor_dims(name, ne, required)**  
  用 `get_tensor_meta` 取 GGUF 中的 tensor，与期望维度 `ne` 逐维比较；不符抛错。若 required=false 且 tensor 不存在则返回 NULL。

- **create_tensor(ctx, name, ne, flags)**  
  - 调用 `check_tensor_dims(name, ne, !(flags & TENSOR_NOT_REQUIRED))`；若返回 NULL 则本函数返回 NULL。
  - 用 `ggml_dup_tensor(ctx, cur)` 在给定 ctx 中创建与 GGUF 中类型/形状一致的 tensor，并设名字。
  - 若 `flags & TENSOR_DUPLICATED`，只把 `ggml_nbytes(cur)` 累加到 `size_data`；否则 `n_created++`。
  - 返回新建的 `ggml_tensor*`（数据指针由 ggml_context 或后续 backend 分配，loader 不在这里读入数据）。

- **create_tensor_as_view(ctx, base, name, ne, offset, required)**  
  校验 name 的维度与 `ne` 一致，且类型与 `base` 相同；然后在 ctx 中创建对 `base` 的 view（`ggml_view_4d`），偏移为 `offset`，用于共享同一块存储的多个逻辑 tensor（如 MoE 专家）。`n_created++`。

- **done_getting_tensors()**  
  检查 `n_created == n_tensors`，否则抛错（防止漏建或多建 tensor）。

---

## 8. 映射与数据加载

### 8.1 init_mappings(prefetch, mlock_mmaps)

- 仅在 `use_mmap == true` 时执行。
- 对 `files` 中每个文件创建一个 `llama_mmap`（prefetch 控制预取，is_numa 可能影响 NUMA 策略）。
- 每个 mmap 在 `mmaps_used` 中占一项，初始为 `(mapping->size(), 0)`，后续加载时缩小“实际使用区间”。
- 若传入 `mlock_mmaps`，则对每个 mmap 做 mlock，防止被 swap 出去。

同时会遍历 `weights_map` 累加 `size_data`，供后续进度和 `load_all_data` 使用。

### 8.2 get_mapping_range(first, last, addr, idx, ctx)

- 给定文件索引 `idx` 和 ggml context，统计该 ctx 中**来自该分片**的 tensor 在文件中的最小偏移和最大结束偏移。
- 写回 `*first`、`*last`、`*addr = mapping->addr()`，供调用方做局部 unmap 或统计。

### 8.3 load_data_for(cur)（兼容旧路径，非 backend）

- 根据 `cur->name` 取 `require_weight`，得到文件索引和偏移。
- **use_mmap**：`cur->data` 指向 `mapping->addr() + offs`，或从该处 memcpy 到已分配的 `cur->data`。
- **否则**：`file->seek(offs)` 后 `read_raw(cur->data, nbytes)`。
- 若 `check_tensors` 为 true，调用 `ggml_validate_row_data` 校验；失败抛错。

### 8.4 load_all_data(ctx, bufs, lmlocks, progress_callback, user_data)

**位置**：约 921–1246 行。这是**主加载入口**，支持 mmap、非 mmap、以及可选的异步 GPU 上传。

- **前置**：要求已调用 `init_mappings()`（即 `size_data != 0`）。

- **非 mmap 且非 check_tensors 时**：尝试启用**异步 GPU 上传**：
  - 从 `bufs` 取 backend buffer，检查 backend 是否支持 async、host_buffer、events。
  - 若支持，分配若干块 pinned 的 host buffer（如 4×64MB）和 event，并建一个临时 upload backend。
  - 后续对每个 tensor：按对齐按块从文件读到 host buffer，再 `ggml_backend_tensor_set_async` 到 GPU，用 event 做同步，实现读与上传流水。

- **按 ctx 中 tensor 顺序遍历**：
  - 用 `get_weight(ggml_get_name(cur))` 找到对应 `llama_tensor_weight`；若为 nullptr（如 split experts 中未选中的专家）则跳过。
  - 调用 `progress_callback(size_done/size_data, user_data)`；若返回 false 视为取消并返回 false。
  - **use_mmap**：
    - 数据指针 `data = mapping->addr() + weight->offs`。
    - 若 `check_tensors`，则把校验任务丢到 `std::async`，结果存入 `validation_result`。
    - 若 `bufs` 中有该 idx 的 buffer（`buf_mmap`）且 `cur->data == nullptr`，则 `ggml_backend_tensor_alloc(buf_mmap, cur, data)` 把 tensor 绑定到 mmap 地址；若有 lmlocks 则扩展 mlock 范围，并更新 `mmaps_used[idx]`。
    - 否则 `ggml_backend_tensor_set(cur, data, 0, n_size)`。
  - **非 mmap**：
    - 若 tensor 已在 host buffer：`file->seek` + `read_raw` 到 `cur->data`，可选 async 校验。
    - 否则若启用了 upload_backend：按对齐分块读入 pinned buffer，再 `tensor_set_async` 到 GPU。
    - 否则：临时 buffer 读入后 `ggml_backend_tensor_set`，并可选校验。
  - `size_done += n_size`。

- **收尾**：
  - 同步并释放所有 event、host_buffers、upload_backend。
  - 收集所有 `validation_result`，若有校验失败则抛错。
  - 若 `size_done >= size_data`（本次或累计已加载完）：
    - 若 use_mmap，对每个 mapping 根据 `mmaps_used` unmap 未使用的前后片段（节省地址空间/资源）。
    - 最后再调用一次 `progress_callback(1.0f, ...)` 并返回其返回值。

---

## 9. 辅助接口

- **ftype_name()**：返回 `llama_model_ftype_name(ftype)`，用于日志或 UI。
- **print_info()**：打印文件格式版本、ftype、文件大小（MiB/GiB）及 BPW（bits per weight）。

---

## 10. 与上层调用关系

- **llama.cpp（示例/库入口）**  
  - 在加载模型时构造：
    - `llama_model_loader ml(fname, splits, use_mmap, use_direct_io, check_tensors, no_alloc, kv_overrides, tensor_buft_overrides);`
  - 然后通过 `llama_model::load_*` 使用 ml。

- **llama_model（src/llama-model.cpp）**  
  - `load_stats(ml)`、`load_arch(ml)`、`load_hparams(ml)`、`load_vocab(ml)`：用 `ml.get_key(kid, ...)` / `get_arr` 等读元数据和词表。
  - `load_tensors(ml)`：
    - 根据架构在 ggml_context 中 `create_tensor` / `create_tensor_as_view`（带 TENSOR_DUPLICATED 等 flags）。
    - 调用 `ml.done_getting_tensors()`。
    - 调用 `ml.init_mappings(...)`。
    - 调用 `ml.load_all_data(ctx, bufs, lmlocks, progress_callback, user_data)` 把权重灌进 backend buffer。
  - 若 `load_tensors` 返回 false（用户取消），上层会做清理并返回失败。

- **llama-quant（tools/quantize）**  
  - 构造 loader 用于读入 GGUF 和 tensor 列表，再做量化写回；不经过 llama_model 的完整 load_tensors 流程。

- **llama_vocab**  
  - `load(llama_model_loader & ml, LLM_KV)` 从 ml 的 meta 和 get_key/get_arr 读词表与特殊 token。

---

## 11. 小结

| 阶段 | 动作 |
|------|------|
| 构造 | 打开主 GGUF + 可选分片，建 weights_map、meta、contexts，推断 ftype，处理 kv/tensor 覆盖 |
| 元数据 | get_key / get_arr / get_key_or_arr，支持按 key 或 llm_kv 枚举，支持 override |
| 张量 | get_weight、create_tensor、create_tensor_as_view、check_tensor_dims、done_getting_tensors |
| 映射 | init_mappings（mmap + 可选 mlock），get_mapping_range |
| 数据 | load_data_for（旧 API），load_all_data（mmap / 读文件 + 可选异步 GPU 上传 + 校验） |
| 辅助 | get_arch、ftype_name、print_info |

整体上，**llama_model_loader** 把“GGUF 文件 + 分片 + 覆盖”抽象成统一的“元数据 + 权重索引 + 数据加载”接口，供 **llama_model** 和量化工具复用，并支持 mmap、Direct I/O、异步 GPU 上传和 tensor 校验等可选行为。
