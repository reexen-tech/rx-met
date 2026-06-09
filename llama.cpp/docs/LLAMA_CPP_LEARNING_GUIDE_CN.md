# llama.cpp 从零到量化与推理开发 — 学习路线图

面向目标：**基于 llama.cpp 完成 GPTQ / SmoothQuant 量化算法设计、量化算子开发与评估、端到端 C++/CUDA 推理框架搭建及效果评估**。

---

## 一、llama.cpp 整体架构速览

```
llama.cpp/
├── include/llama.h          # 对外 C API（加载、推理、采样等）
├── src/                     # llama 层：模型加载、图构建、量化入口
│   ├── llama-model*.cpp     # 模型结构、权重加载
│   ├── llama-graph*.cpp     # 计算图构建（forward 定义）
│   ├── llama-quant.cpp      # 量化工具逻辑，调用 ggml 量化
│   └── models/              # 各架构实现（LLaMA、Qwen、DeepSeek 等）
├── ggml/                    # 核心计算引擎
│   ├── include/ggml.h       # 张量、算子、量化类型、ggml_quantize_chunk 等
│   ├── src/
│   │   ├── ggml.c           # 算子实现、quantize 分发、type_traits
│   │   ├── ggml-quants.h    # 各量化格式 block 定义与 dequant 声明
│   │   ├── ggml-cpu/        # CPU 量化/反量化/vec_dot
│   │   └── ggml-cuda/       # CUDA：mmq/mmvq/mmvf、dequantize、quantize
│   └── ...
├── tools/
│   ├── quantize/            # llama-quantize 命令行工具
│   ├── imatrix/             # importance matrix（用于部分量化类型）
│   └── perplexity/          # 困惑度评估
└── common/                  # 公共头文件与实现
```

**核心关系**：  
- **llama** 负责「用什么模型、怎么建图」；**ggml** 负责「张量存什么类型、怎么算」。  
- 量化 = 权重从 F32/F16 变为某种 `ggml_type`（如 Q4_K）；推理时通过 **MUL_MAT** 等算子用量化权重参与计算。  
- 新增一种量化格式 = 在 ggml 里加类型 + 量化/反量化/vec_dot（及 CUDA 核），再在 llama 里挂到 ftype/quantize 工具。

---

## 二、推荐学习阶段（从 0 到能改量化与推理）

### 阶段 1：能跑起来 + 理解数据流（1–2 天）

1. **编译与运行**
   - 阅读 [docs/build.md](build.md)，完成 CPU 与 CUDA 编译。
   - 使用 [tools/quantize/README.md](../tools/quantize/README.md) 将 F16/BF16 GGUF 量化为 Q4_K_M 等，用 CLI 跑推理。
   - 命令示例：`llama-quantize model-f16.gguf model-q4.gguf Q4_K_M`，`llama-cli -m model-q4.gguf`。

2. **理解「模型 → 图 → 执行」**
   - `include/llama.h`：`llama_load_from_file`、`llama_decode`、`llama_sampling` 等。
   - `src/llama.cpp`：入口；`llama-impl.cpp`、`llama-context.cpp` 里与 load/decode 相关的调用。
   - `src/llama-graph.cpp`：如何根据架构（如 LLaMA）建计算图；找到 **MUL_MAT**、RMSNorm、Attention 等对应关系。

3. **理解权重与量化类型**
   - `include/llama.h` 中 `llama_ftype` 与 `ggml/include/ggml.h` 中 `ggml_type` 的对应（如 Q4_K_M → GGML_TYPE_Q4_K）。
   - 权重从 GGUF 按 `ggml_type` 加载到 `ggml_tensor`；推理时这些张量直接参与 MUL_MAT 等。

**阶段 1 检验**：能说清「一个 LLaMA layer 里有哪些张量、在图中对应哪些 op、权重是什么类型」。

---

### 阶段 2：GGML 张量与量化链路（2–3 天）

1. **ggml 张量与类型**
   - [ggml/include/ggml.h](../ggml/include/ggml.h)：`ggml_tensor`、`ggml_type` 枚举（F32/F16/Q4_0/Q4_K/…）、`ggml_type_size`、`ggml_row_size`、`ggml_blck_size`。
   - `ggml_get_type_traits(type)`：`to_float` / `from_float_ref`（反量化/量化）、`blck_size` 等，理解「块」概念。

2. **量化入口与格式**
   - [ggml/include/ggml.h](../ggml/include/ggml.h) 中：
     - `ggml_quantize_init` / `ggml_quantize_free`
     - `ggml_quantize_requires_imatrix(type)`
     - `ggml_quantize_chunk(type, src, dst, start, nrows, n_per_row, imatrix)`
   - [ggml/src/ggml.c](../ggml/src/ggml.c) 中 `ggml_quantize_chunk` 的 **switch(type)**：每种量化格式对应一个 `quantize_xxx`（如 `quantize_q4_0`、`quantize_q4_K`）。
   - [ggml/src/ggml-quants.h](../ggml/src/ggml-quants.h)：各格式的 **block 结构体**（如 `block_q4_0`、`block_q4_K`）和 **dequantize_row_xxx** 声明。

3. **CPU 量化与 vec_dot**
   - `ggml/src/ggml-cpu/`：  
     - 各 `quantize_xxx` 实现在 ggml.c 或 ggml-cpu 相关文件中；  
     - `quants.h`、`arch/` 下与 `vec_dot`、dequant 相关的实现（理解「按块量化 + 按块计算」）。

4. **llama 侧如何调用量化**
   - [src/llama-quant.cpp](../src/llama-quant.cpp)：  
     - `llama_model_quantize_impl`：按层/按张量决定 `new_type`，逐块调用 `ggml_quantize_chunk`。  
     - 理解 `llama_ftype` → `ggml_type` 的映射、imatrix 的用法（部分 K-quant 需要）。

**阶段 2 检验**：能画出「F32 权重 → ggml_quantize_chunk → 某 GGML_TYPE → 写入 GGUF」的流程，并知道推理时谁在读这些块。

---

### 阶段 3：推理时量化权重的使用（2–3 天）

1. **MUL_MAT 与后端**
   - [docs/ops.md](ops.md)：MUL_MAT 在各后端（CPU、CUDA 等）的支持情况。
   - 搜索 `GGML_OP_MUL_MAT`：在 ggml.c / ggml-cpu / ggml-cuda 中如何被派发。

2. **CPU：vec_dot 与 dequant**
   - 理解「不先整块 dequant 到 F32，而是用 vec_dot(quant_row, y)」的路径（性能关键）。
   - `ggml-cpu/quants.h`、`arch/` 中与 Q4_0、Q4_K 等对应的实现。

3. **CUDA：MMQ / MMVQ / 反量化**
   - [ggml/src/ggml-cuda/](../ggml/src/ggml-cuda/)：  
     - `mmq.cu/mmq.cuh`：批量 matmul 量化权重（MMQ = mul mat quantized）。  
     - `mmvq.cu`：向量×量化矩阵。  
     - `dequantize.cuh`：各格式的 `dequantize_xxx` 内联函数（按 block 解释 qs、scale 等）。  
   - `template-instances/`：为不同 `ggml_type` 实例化的 MMQ/MMVQ 核。
   - 理解「block 布局 → shared memory 装块 → 与激活做乘加」的流程，便于之后加新格式。

**阶段 3 检验**：能说清「Q4_K 权重的 MUL_MAT 在 CUDA 上走哪条路径、数据如何从 block 变成参与计算的数值」。

---

### 阶段 4：从现有量化到 GPTQ / SmoothQuant（按需持续）

1. **GPTQ 在 llama.cpp 中的位置**
   - 当前仓库主要是 **训练后量化（PTQ）**：对 F32/F16 权重按块做 min-max 或 K-quant 等，**没有**内置 GPTQ（基于 Hessian 的逐层校准）。
   - 要做 GPTQ：  
     - **方案 A**：在 **外部**用 Python（如 AutoGPTQ）做 GPTQ，导出为 **与现有 ggml 格式兼容** 的权重（例如导出成 Q4_K 的 block 布局），再由 llama.cpp 按现有路径推理。  
     - **方案 B**：在 **ggml 内** 新增一种 `ggml_type`（如 `GGML_TYPE_Q4_GPTQ`），在 `ggml_quantize_chunk` 里不实现（或只做占位），**量化阶段** 用 Python/其他工具生成 GGUF，推理时只为该类型实现 **dequant + vec_dot + CUDA MMQ/MMVQ**。

2. **SmoothQuant 在 llama.cpp 中的位置**
   - SmoothQuant 是 **激活与权重的平滑缩放**，减少激活动态范围，便于 INT8 等量化。
   - 在 llama.cpp 中可有两种用法：  
     - **离线**：在转换/量化脚本里对权重和（或）激活做平滑（乘除 scale），再量化为现有或新增的 `ggml_type`，推理端无需改算子，只当「另一种量化结果」。  
     - **在线**：在计算图中插入 **scale 节点**（乘/除 per-tensor 或 per-channel 的 scale），并配合 INT8 量化算子；这需要扩展 ggml 的 op 与类型（例如 INT8 张量 + scale 存储）。

3. **扩展新量化格式的通用步骤（便于接 GPTQ/SmoothQuant）**
   - 在 `ggml.h` 中增加 `GGML_TYPE_xxx`，在 `ggml_type_traits` 表中注册。  
   - 在 `ggml-quants.h` 中定义 `block_xxx` 和 `dequantize_row_xxx`。  
   - 在 `ggml.c` 的 `ggml_quantize_chunk` 中增加分支；若该格式由外部工具生成，可只写 `memcpy` 或简单校验。  
   - CPU：在 `ggml-cpu` 中实现 `vec_dot_xxx` 及需要的 dequant 路径。  
   - CUDA：在 `dequantize.cuh` 中加 `dequantize_xxx`，在 `mmq.cuh`/`mmvq` 中为新区块布局加 load/vec_dot 逻辑，并在 `ggml-cuda.cu` 中注册该类型的 MUL_MAT。  
   - 若需「量化工具」支持：在 `llama_ftype` 和 `llama-quant.cpp` 里增加对应选项。

---

## 三、量化算子开发与评估

1. **算子开发**
   - **量化侧**：实现 `quantize_xxx`（或对接外部 GPTQ/SmoothQuant 输出），保证 `ggml_validate_row_data` 通过。  
   - **推理侧**：实现 `dequantize_row_xxx`（CPU）、`dequantize_xxx`（CUDA）、`vec_dot_xxx`（CPU）以及 CUDA 的 MMQ/MMVQ 模板实例。  
   - 参考现有最简单格式（如 Q4_0）的 block 布局与 `ggml_row_size` 计算，保证与 `ggml_type_traits` 一致。

2. **评估流程**
   - **正确性**：同一模型 F16 vs 新量化，相同输入下对比 logits 或 perplexity（[tools/perplexity](../tools/perplexity)）。  
   - **速度**：用 [tools/llama-bench](../tools/llama-bench) 或自定义小脚本测 prompt 与 decode 的吞吐/延迟。  
   - **显存**：观察 CUDA 分配与 `ggml_backend_buffer` 大小，确认量化后显存符合预期。

3. **与现有评估脚本对接**
   - 若你有 [Model_Evaluator](https://github.com/...) 等脚本，可把 llama.cpp 作为推理后端：通过 `llama.h` 的 C API 加载量化模型、跑 batch 或逐条样本、输出 logits/困惑度，再与你现有的指标（如 MMLU、perplexity）对接。

---

## 四、端到端 C++/CUDA 推理框架与效果评估

1. **复用 vs 自建**
   - **复用**：直接使用 `include/llama.h` + 现有 ggml CUDA 后端；你的工作集中在「新量化格式 + 评估脚本」。  
   - **自建**：若需要完全掌控图调度、内存与流水线，可把 ggml 当作「算子库」：只调用 ggml 的 tensor/op 与 CUDA backend，自己写 `main`、batch 循环和评估逻辑。

2. **效果评估要点**
   - **任务**：perplexity、MMLU/常识等准确率、生成质量（若做生成）。  
   - **对比基线**：同一模型 F16/BF16 vs 各量化（Q4_K_M、你的 GPTQ、SmoothQuant 等）。  
   - **指标**：PPL、准确率、吞吐 (t/s)、首 token 延迟、显存占用；必要时报告校准数据量/校准集。

---

## 五、关键文件索引（便于跳转）

| 目标                     | 主要文件/目录 |
|--------------------------|----------------|
| 对外 API                 | `include/llama.h` |
| 模型加载与图构建         | `src/llama-model*.cpp`, `src/llama-graph.cpp` |
| 量化类型与工具逻辑       | `src/llama-quant.cpp`, `include/llama.h` (llama_ftype) |
| 张量类型与量化 API       | `ggml/include/ggml.h` (ggml_type, ggml_quantize_chunk) |
| 块结构与反量化声明       | `ggml/src/ggml-quants.h` |
| 量化实现分发             | `ggml/src/ggml.c` (ggml_quantize_chunk switch) |
| CPU 量化/vec_dot         | `ggml/src/ggml-cpu/` (quants.h, arch/) |
| CUDA 量化矩阵运算        | `ggml/src/ggml-cuda/mmq.cuh`, `mmvq.cu`, `dequantize.cuh` |
| CUDA 量化核实例化        | `ggml/src/ggml-cuda/template-instances/` |
| 量化命令行工具           | `tools/quantize/quantize.cpp` |
| 困惑度评估               | `tools/perplexity/` |
| 算子与后端支持表         | `docs/ops.md` |
| 构建与 CUDA 选项         | `docs/build.md` |

---

## 六、建议学习顺序小结

1. **第 1 周**：阶段 1 + 阶段 2 — 跑通量化与推理，理解 GGML 类型与 `ggml_quantize_chunk` 链路。  
2. **第 2 周**：阶段 3 — 理解 MUL_MAT 在 CPU/CUDA 上如何消费量化权重（vec_dot、MMQ、dequantize.cuh）。  
3. **第 3 周起**：阶段 4 — 选定 GPTQ 或 SmoothQuant 的落地方式（外挂校准 vs 新 ggml_type），实现一种新格式的推理路径并做正确性/速度/显存评估。  

按此路线，你可以从零把 llama.cpp 的「模型→图→量化→推理」串起来，并在明确扩展点的前提下做 GPTQ/SmoothQuant 与 C++/CUDA 推理与评估。
