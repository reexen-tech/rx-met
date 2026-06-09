# llama.cpp 代码架构与目录/模块说明

本文档梳理 llama.cpp 仓库的整体代码架构，以及各目录、模块的职责与功能。

---

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│  应用层：examples / tools (CLI、server、perplexity、quantize、mtmd 等)    │
├─────────────────────────────────────────────────────────────────────────┤
│  对外 API：include/llama.h（加载、解码、采样、会话等 C 接口）              │
├─────────────────────────────────────────────────────────────────────────┤
│  llama 层：src/（模型定义、图构建、内存/KV、量化入口、词表、采样）         │
│            common/（对话模板、采样工具、日志、参数解析等）                 │
├─────────────────────────────────────────────────────────────────────────┤
│  计算引擎：ggml/（张量、算子、多后端、量化类型、GGUF 读写）                │
└─────────────────────────────────────────────────────────────────────────┘
```

- **llama** 负责「用什么模型、怎么建计算图、怎么调度」；**ggml** 负责「张量存什么、怎么算、在哪个设备算」。
- 量化、推理、评估都依赖 **ggml** 的 `ggml_type`、算子实现与后端；**llama** 只做图描述与调用。

---

## 二、根目录与顶层文件

| 路径 / 文件 | 功能 |
|-------------|------|
| **CMakeLists.txt** | 主工程 CMake：选项（BUILD_TESTS/TOOLS/EXAMPLES、后端开关等）、子目录添加、安装规则 |
| **Makefile** | 传统 Make 入口（内部多走 CMake） |
| **include/** | 对外 C 头文件：`llama.h`（主 API）、`ggml.h` 等通过子项目或安装暴露 |
| **convert_hf_to_gguf.py** | 将 HuggingFace 格式模型转为 GGUF（入口脚本） |
| **convert_lora_to_gguf.py** | LoRA 权重转 GGUF |
| **gguf-py/** | Python 包：GGUF 读写、模型转换（与 convert 脚本配合） |
| **AGENTS.md / CLAUDE.md** | 给 AI 与贡献者的项目说明与规范 |
| **requirements/**, **poetry.lock** | Python 依赖（转换、脚本等） |

---

## 三、核心目录与模块

### 1. `include/` — 对外 API

| 文件 | 功能 |
|------|------|
| **llama.h** | 主 C API：模型/上下文加载与释放、decode、采样、KV、会话保存、batch、grammar、backlog 等；`llama_ftype` 与量化类型映射 |
| **ggml.h** | 通常由 ggml 子目录或安装提供：张量、算子、`ggml_type`、`ggml_quantize_chunk` 等 |

使用者（examples、tools、外部项目）只依赖 `llama.h`（及 ggml 头文件），不直接依赖 src 内部实现。

---

### 2. `src/` — llama 库实现（模型与推理流程）

**入口与总控：**

| 文件 | 功能 |
|------|------|
| **llama.cpp** | 库入口、版本与构建信息 |
| **llama-impl.cpp / .h** | 内部实现汇总、与 common 的衔接 |
| **llama-cparams.cpp / .h** | 上下文参数（context params） |
| **llama-arch.cpp / .h** | 架构注册与分发（按 arch 选择对应 model 实现） |

**模型与权重：**

| 文件 | 功能 |
|------|------|
| **llama-model.cpp / .h** | 模型结构抽象、层/张量元数据 |
| **llama-model-loader.cpp / .h** | 从 GGUF 加载模型（读元数据与权重） |
| **llama-model-saver.cpp / .h** | 保存模型为 GGUF（含量化后写入） |
| **llama-mmap.cpp / .h** | 模型文件 mmap 与访问 |
| **llama-io.cpp / .h** | 通用 IO 封装 |
| **llama-hparams.cpp / .h** | 超参（层数、维度、头数等） |
| **llama-quant.cpp / .h** | 量化入口：调用 ggml 的 `ggml_quantize_chunk`，与 ftype 映射 |

**计算图与执行：**

| 文件 | 功能 |
|------|------|
| **llama-graph.cpp / .h** | **核心**：按架构构建前向计算图（MUL_MAT、RMSNorm、Attention、FFN 等），决定哪些张量参与计算、以何种 ggml_type |
| **llama-batch.cpp / .h** | batch 构建与提交（供 decode 使用） |

**上下文与内存：**

| 文件 | 功能 |
|------|------|
| **llama-context.cpp / .h** | 上下文创建、decode 调度、与 backend 的衔接 |
| **llama-kv-cache.cpp / .h** | KV cache 结构与更新 |
| **llama-kv-cache-iswa.cpp / .h** | KV 的 ISWA 相关逻辑 |
| **llama-kv-cells.h** | KV 单元定义 |
| **llama-memory.cpp / .h** | 内存分配与 buffer 管理 |
| **llama-memory-hybrid.cpp / .h** | CPU+GPU 混合内存 |
| **llama-memory-hybrid-iswa.cpp / .h** | 混合内存 ISWA 相关 |
| **llama-memory-recurrent.cpp / .h** | 循环/状态复用的内存策略 |

**词表与采样：**

| 文件 | 功能 |
|------|------|
| **llama-vocab.cpp / .h** | 词表加载与 token 转换 |
| **llama-sampling.cpp / .h** | 采样（temperature、top_p、top_k 等）与 logits 处理 |
| **llama-grammar.cpp / .h** | 语法约束（grammar） |
| **llama-adapter.cpp / .h** | 适配器相关（如 LoRA 等） |

**其他：**

| 文件 | 功能 |
|------|------|
| **unicode.cpp / .h**, **unicode-data.cpp / .h** | Unicode 与分词边界等 |

**`src/models/`** — 各架构前向图与张量名：

- 每个文件对应一种或一类模型架构（如 **llama.cpp**、**qwen2.cpp**、**gemma.cpp**、**deepseek.cpp**、**mamba.cpp** 等）。
- 实现「该架构下如何建图」：层数、张量命名、Attention/FFN 形状、RoPE 等，与 **llama-graph.cpp** 的架构分发配合。
- 新增模型支持时在此增加对应 `.cpp` 并在 arch 中注册。

---

### 3. `common/` — 公共工具与对话/采样

被 **src** 与 **tools** 共用，不依赖 ggml/llama 核心。

| 文件 / 子目录 | 功能 |
|---------------|------|
| **common.cpp / .h** | 通用工具、构建信息、平台相关 |
| **log.cpp / .h** | 日志接口（LOG_INF、LOG_ERR 等） |
| **arg.cpp / .h** | 命令行参数解析 |
| **chat.cpp / .h** | 对话格式与模板（chat template） |
| **chat-parser.cpp / .h**, **chat-peg-parser.cpp / .h**, **chat-parser-xml-toolcall.cpp / .h** | 对话解析、XML tool call 解析 |
| **sampling.cpp / .h** | 采样工具（与 llama-sampling 配合或独立使用） |
| **console.cpp / .h** | 控制台交互 |
| **preset.cpp / .h** | 预设配置 |
| **download.cpp / .h**, **http.h** | 下载与 HTTP |
| **json-partial.cpp / .h**, **json-schema-to-grammar.cpp / .h** | 部分 JSON 解析、JSON Schema 转 grammar |
| **regex-partial.cpp / .h** | 正则部分匹配 |
| **peg-parser.cpp / .h** | PEG 解析器 |
| **speculative.cpp / .h** | 推测解码相关 |
| **ngram-cache.cpp / .h**, **ngram-map.cpp / .h**, **ngram-mod.cpp / .h** | n-gram 缓存与修改（推测等） |
| **unicode.cpp / .h** | Unicode 工具 |
| **debug.cpp / .h** | 调试辅助 |
| **llguidance.cpp** | LLGuidance 集成（可选） |
| **jinja/** | Jinja 模板引擎（C++ 实现），用于生成代码或配置 |

---

### 4. `ggml/` — 计算引擎（张量、算子、后端、量化）

#### 4.1 `ggml/include/` — 头文件

| 文件 | 功能 |
|------|------|
| **ggml.h** | 核心：张量（`ggml_tensor`）、算子枚举（`GGML_OP_*`）、`ggml_type`（F32/F16/Q4_0/Q4_K 等）、类型 trait、`ggml_quantize_chunk`、图构建 API |
| **gguf.h** | GGUF 文件格式：读写接口、元数据与张量列表 |
| **ggml-backend.h** | 后端抽象：buffer、device、graph 调度 |
| **ggml-alloc.h** | 内部分配器 |
| **ggml-cpu.h** | CPU 后端与类型 trait |
| **ggml-cuda.h** | CUDA 后端 |
| **ggml-metal.h** | Metal（Apple） |
| **ggml-opencl.h** | OpenCL |
| **ggml-vulkan.h** | Vulkan |
| **ggml-sycl.h** | SYCL（Intel/AMD 等） |
| **ggml-cann.h** | 华为 CANN |
| **ggml-zendnn.h**, **ggml-zdnn.h** | ZenDNN、zDNN |
| **ggml-hexagon.h** | Qualcomm Hexagon |
| **ggml-webgpu.h** | WebGPU |
| **ggml-blas.h** | BLAS 封装 |
| **ggml-opt.h** | 优化器（训练用） |
| **ggml-rpc.h**, **ggml-virtgpu.h** | RPC 与虚拟 GPU |
| **ggml-cpp.h** | C++ 封装 |

#### 4.2 `ggml/src/` — 实现结构

| 路径 / 文件 | 功能 |
|-------------|------|
| **ggml.c** | 图构建、算子实现、**量化**（`ggml_quantize_chunk` 的 switch、各 `quantize_xxx` 调用）、类型 trait、线程池等 |
| **ggml.cpp** | C++ 扩展与封装 |
| **gguf.cpp** | GGUF 读写实现 |
| **ggml-quants.h** | 各量化格式的 **block 结构体** 与 **dequantize_row_xxx** 声明 |
| **ggml-cpu/** | CPU 后端：算子实现、量化/反量化、vec_dot、各架构 SIMD（含 AMX 等） |
| **ggml-cuda/** | CUDA 后端：MMQ/MMVQ、dequantize、quantize、Flash Attention、各量化核的模板实例 |
| **ggml-metal/** | Metal 实现 |
| **ggml-vulkan/** | Vulkan 实现（含 .comp shader） |
| **ggml-opencl/** | OpenCL 实现（.cl kernel） |
| **ggml-sycl/** | SYCL 实现 |
| **ggml-cann/** | CANN 实现 |
| **ggml-hexagon/** | Hexagon HTP |
| **ggml-webgpu/** | WebGPU（.wgsl） |
| **ggml-zendnn/**, **ggml-zdnn/** | ZenDNN、zDNN |
| **ggml-blas/** | BLAS 封装实现 |
| **ggml-rpc/**, **ggml-virtgpu/** | RPC 与虚拟 GPU |

量化与推理的关系：

- **量化**：`ggml_quantize_chunk(type, src, dst, ...)` 将 F32 按块写入某种 `ggml_type`；llama 侧通过 **llama-quant.cpp** 和 **tools/quantize** 调用。
- **推理**：加载的权重已是某种 `ggml_type`；**MUL_MAT** 等算子在 CPU/CUDA 等后端中按类型走 vec_dot、MMQ 等，无需先整块反量化再乘。

---

### 5. `tools/` — 可执行工具

| 工具目录 | 主文件 | 功能 |
|----------|--------|------|
| **cli/** | cli.cpp | **llama-cli**：交互式/非交互式对话与补全，使用 libllama |
| **server/** | server.cpp, server-*.cpp/.h | **llama-server**：HTTP API 服务（OpenAI 兼容）、队列、WebUI（webui/）、多模型 |
| **completion/** | completion.cpp | **llama-completion**：单次补全（无对话） |
| **quantize/** | quantize.cpp | **llama-quantize**：GGUF 模型量化（F32/F16 → Q4_K 等），调用 llama 的量化 API |
| **imatrix/** | imatrix.cpp | **llama-imatrix**：计算 importance matrix，供量化时按重要性分配精度 |
| **perplexity/** | perplexity.cpp | **llama-perplexity**：困惑度与基准评测（PPL、HellaSwag、WinoGrande、multiple_choice） |
| **llama-bench/** | llama-bench.cpp | **llama-bench**：推理性能基准 |
| **batched-bench/** | batched-bench.cpp | 批处理基准 |
| **tokenize/** | tokenize.cpp | **llama-tokenize**：文本 → token id |
| **export-lora/** | export-lora.cpp | LoRA 导出 |
| **gguf-split/** | gguf-split.cpp | GGUF 分片/合并 |
| **fit-params/** | fit-params.cpp | 参数拟合（用于量化等） |
| **rpc/** | rpc-server.cpp | RPC 服务 |
| **mtmd/** | mtmd.cpp, mtmd-cli.cpp, clip*.h, models/*.cpp | **多模态**：LLaVA、Qwen2-VL、MiniCPM-V 等视觉/音频模型的前向与 CLI |
| **tts/** | tts.cpp | TTS（OutEtts 等） |
| **cvector-generator/** | — | 与 cvector 相关生成 |

---

### 6. `examples/` — 示例程序

每个子目录通常包含一个主 `.cpp` 和 README，演示某种用法：

| 示例 | 功能 |
|------|------|
| **simple** | 最简加载与生成 |
| **simple-chat** | 最简对话 |
| **batched** | 批处理推理 |
| **embedding** | 嵌入向量 |
| **save-load-state** | 保存/加载解码状态 |
| **speculative** | 推测解码 |
| **lookahead** | Lookahead 解码 |
| **retrieval** | 检索增强 |
| **training** | 训练示例 |
| **model-conversion** | 模型转换流程示例 |
| **gguf**, **gguf-hash** | GGUF 读写示例 |
| **eval-callback** | 评估回调 |
| 其他 | parallel、passkey、diffusion、idle、lookup 等 |

---

### 7. `grammars/` — 语法约束

- 存放 **.gbnf** 文件（BNF 风格语法），用于约束生成（如 JSON、列表、算术表达式等）。
- 与 **llama-grammar** 配合，由 server/CLI 等通过 API 指定。

---

### 8. `models/` — 模型定义模板

- 多为 **.jinja** 模板，用于生成或描述模型结构（与代码生成或文档相关），不是运行时直接加载的权重。

---

### 9. `tests/` — 测试

- C++ 单测、后端算子测试（如 **test-backend-ops**）、模型加载与简单推理测试等。
- **docs/ops/** 中的各后端 CSV 可由测试生成，用于 **docs/ops.md**。

---

### 10. `docs/` — 文档

| 路径 | 内容 |
|------|------|
| **build.md** | 构建说明（CPU、CUDA、Metal 等） |
| **install.md** | 安装方式 |
| **ops.md** | 算子与后端支持表（由 scripts/create_ops_docs.py 等生成） |
| **backend/** | 各后端说明（CUDA、SYCL、OpenCL、CANN、ZenDNN 等） |
| **development/** | 开发指南（如 HOWTO-add-model、debugging、performance） |
| **multimodal.md**, **multimodal/** | 多模态模型使用说明 |
| **function-calling.md** | 函数调用 |
| **speculative.md** | 推测解码 |
| **docker.md** | Docker 使用 |

---

### 11. `scripts/` — 脚本

- **get-*.sh**：下载数据集（WikiText、HellaSwag、WinoGrande 等）。
- **sync-ggml*.sh**：与上游 ggml 同步（若拆库）。
- **create_ops_docs.py**：生成 ops 文档。
- **compare-*.py**：结果对比、bench 对比。
- **apple/**、**snapdragon/**：平台相关验证与运行脚本。

---

### 12. `cmake/` — CMake 模块

- 各平台/编译器配置（如 **arm64-apple-clang**、**x64-windows-llvm**）、**build-info**、**git-vars**、**download-models** 等。

---

### 13. `vendor/` — 第三方依赖

- 少量头文件或第三方库（如 httplib 等），随构建引入。

---

### 14. `benches/`、`ci/`、`.github/`、`.devops/`

- 基准配置、CI 流水线、GitHub Actions、Docker/Nix 等运维与自动化。

---

## 四、数据流与依赖关系（简化）

1. **模型加载**：GGUF 文件 → **llama-model-loader**（读 gguf）→ **llama-model**（结构）→ 权重张量在 **ggml** 中按 `ggml_type` 存在 backend buffer。
2. **前向计算**：**llama-graph** 按架构建图 → **llama-context** 调度 **ggml** 图 → **ggml** 在各 backend（CPU/CUDA/…）上执行 MUL_MAT、RMSNorm 等。
3. **量化**：**tools/quantize** 或 API → **llama-quant** → **ggml_quantize_chunk** → 各 `quantize_xxx`（在 ggml.c 或 ggml-cpu 等）→ 写回 GGUF 或内存。
4. **推理**：**llama_decode** 提交 batch → **llama-context** 跑图 → **ggml** 执行；**llama_sampling** 在 logits 上采样得到 token。

---

## 五、与「量化 / 推理开发」相关的重点路径

| 目标 | 主要目录/文件 |
|------|----------------|
| 新增量化格式 | **ggml/include/ggml.h**（类型枚举）、**ggml/src/ggml-quants.h**（block + dequant）、**ggml/src/ggml.c**（quantize_chunk 分支）、**ggml-cpu**、**ggml-cuda**（dequant + MMQ/MMVQ） |
| 量化工具与 ftype | **src/llama-quant.cpp**、**include/llama.h**（llama_ftype）、**tools/quantize/quantize.cpp** |
| 推理图与权重类型 | **src/llama-graph.cpp**、**src/models/*.cpp** |
| 后端算子实现 | **ggml/src/ggml-cuda/**（mmq、mmvq、dequantize）、**ggml/src/ggml-cpu/**、**docs/ops.md** |
| 评估与评测 | **tools/perplexity/**、**tools/llama-bench/** |

以上结构可与 [LLAMA_CPP_LEARNING_GUIDE_CN.md](LLAMA_CPP_LEARNING_GUIDE_CN.md) 中的学习路线配合使用，便于定位到具体目录与文件进行阅读和修改。
