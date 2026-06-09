# llama.cpp 完整目录架构与各目录/文件功能说明

本文档列出仓库的完整目录树，并逐目录、逐类文件说明其功能。

---

## 一、完整目录树

```
llama.cpp/
├── CMakeLists.txt              # 主工程 CMake 配置
├── Makefile                    # Make 入口（内部调用 CMake）
├── AUTHORS                     # 贡献者列表
├── CLAUDE.md                   # 给 AI 的协作说明
├── AGENTS.md                   # 给 AI Agent 的规范
├── LICENSE                     # 许可证
├── pyrightconfig.json          # Python 类型检查配置
├── poetry.lock                 # Python 依赖锁定
│
├── include/                    # 对外 C API 头文件
│   └── llama.h                 # 主库 API（模型/上下文/解码/采样等）
│
├── convert_hf_to_gguf.py       # HuggingFace → GGUF 转换入口
├── convert_lora_to_gguf.py     # LoRA → GGUF 转换
│
├── requirements/               # 各场景 Python 依赖
│   └── *.txt                   # 如 convert、server、docs 等
│
├── common/                     # 公共 C++ 库（被 src、tools 共用）
│   ├── CMakeLists.txt
│   ├── common.cpp, common.h    # 通用工具、构建信息
│   ├── log.cpp, log.h          # 日志（LOG_INF/LOG_ERR）
│   ├── arg.cpp, arg.h          # 命令行参数解析
│   ├── chat.cpp, chat.h        # 对话模板（chat template）
│   ├── chat-parser*.cpp/.h     # 对话解析、XML tool call
│   ├── sampling.cpp, sampling.h
│   ├── console.cpp, console.h
│   ├── preset.cpp, preset.h
│   ├── download.cpp, download.h, http.h
│   ├── json-partial.*, json-schema-to-grammar.*
│   ├── regex-partial.*, peg-parser.*
│   ├── speculative.*, ngram-*.cpp/.h
│   ├── unicode.*, debug.*, base64.hpp
│   ├── llguidance.cpp          # LLGuidance 集成（可选）
│   └── jinja/                  # Jinja 模板引擎（C++）
│       └── lexer, parser, runtime, value, string, caps ...
│
├── src/                        # llama 库核心实现
│   ├── CMakeLists.txt
│   ├── llama.cpp               # 库入口
│   ├── llama-impl.cpp/.h       # 内部实现汇总
│   ├── llama-arch.cpp/.h       # 架构注册与分发
│   ├── llama-cparams.cpp/.h    # 上下文参数
│   ├── llama-model.cpp/.h      # 模型结构
│   ├── llama-model-loader.cpp/.h   # 从 GGUF 加载
│   ├── llama-model-saver.cpp/.h    # 保存为 GGUF
│   ├── llama-mmap.cpp/.h       # 模型 mmap
│   ├── llama-io.cpp/.h         # IO 封装
│   ├── llama-hparams.cpp/.h    # 超参
│   ├── llama-quant.cpp/.h      # 量化入口（调 ggml）
│   ├── llama-graph.cpp/.h      # 前向计算图构建
│   ├── llama-batch.cpp/.h      # batch 构建与提交
│   ├── llama-context.cpp/.h    # 上下文与 decode 调度
│   ├── llama-kv-cache*.cpp/.h  # KV cache
│   ├── llama-memory*.cpp/.h    # 内存管理（含 hybrid、recurrent）
│   ├── llama-vocab.cpp/.h      # 词表
│   ├── llama-sampling.cpp/.h   # 采样
│   ├── llama-grammar.cpp/.h    # 语法约束
│   ├── llama-adapter.cpp/.h    # 适配器（如 LoRA）
│   ├── unicode*.cpp/.h         # Unicode 工具
│   └── models/                 # 各架构前向图实现
│       ├── models.h            # 架构声明与注册
│       ├── llama.cpp           # LLaMA
│       ├── qwen2.cpp, qwen3.cpp, qwen2moe, qwen3moe, qwen2vl, qwen3vl ...
│       ├── gemma.cpp, gemma2-iswa, gemma3, gemma3n-iswa ...
│       ├── deepseek.cpp, deepseek2.cpp
│       ├── mistral3, falcon, mpt, bloom, gpt2, gptneox ...
│       ├── mamba.cpp, graph-context-mamba, rwkv6, rwkv7 ...
│       ├── phi2, phi3, plamo, chatglm, glm4, internlm2 ...
│       ├── llava, minicpmv, cogvlm, siglip, whisper-enc ...（多模态/视觉）
│       └── 其他架构（约百个 .cpp）
│
├── ggml/                       # 计算引擎（张量、算子、后端、量化）
│   ├── CMakeLists.txt
│   ├── include/                # 头文件
│   │   ├── ggml.h              # 张量、算子、ggml_type、量化 API
│   │   ├── gguf.h              # GGUF 格式
│   │   ├── ggml-backend.h      # 后端抽象
│   │   ├── ggml-alloc.h
│   │   ├── ggml-cpu.h, ggml-cuda.h, ggml-metal.h ...
│   │   └── ggml-opt.h, ggml-blas.h, ggml-cpp.h ...
│   └── src/
│       ├── ggml.c              # 图/算子/量化分发
│       ├── ggml.cpp            # C++ 扩展
│       ├── gguf.cpp            # GGUF 读写
│       ├── ggml-quants.h       # 量化 block 与 dequant 声明
│       ├── ggml-backend-impl.h
│       ├── ggml-cpu/           # CPU 后端（含 arch/x86, arm, riscv 等）
│       ├── ggml-cuda/          # CUDA（.cu/.cuh，含 mmq/mmvq/dequantize）
│       ├── ggml-metal/         # Metal（Apple）
│       ├── ggml-opencl/        # OpenCL（kernels/*.cl）
│       ├── ggml-vulkan/        # Vulkan（.comp/.glsl）
│       ├── ggml-sycl/          # SYCL
│       ├── ggml-cann/          # 华为 CANN
│       ├── ggml-hexagon/       # Qualcomm Hexagon
│       ├── ggml-webgpu/        # WebGPU（.wgsl）
│       ├── ggml-zendnn/, ggml-zdnn/
│       ├── ggml-blas/          # BLAS
│       ├── ggml-rpc/, ggml-virtgpu/  # RPC / 虚拟 GPU
│       └── ggml-hip/, ggml-musa/     # HIP(AMD), MUSA(摩尔线程)
│
├── tools/                      # 可执行工具
│   ├── CMakeLists.txt
│   ├── cli/                    # llama-cli（交互/补全）
│   │   └── cli.cpp
│   ├── server/                 # llama-server（HTTP API + WebUI）
│   │   ├── server.cpp, server-common.*, server-context.*
│   │   ├── server-http.*, server-models.*, server-queue.*
│   │   ├── public/, public_legacy/, public_simplechat/
│   │   ├── webui/              # Svelte/TS 前端
│   │   ├── themes/, bench/, tests/
│   │   └── chat*.sh, chat.mjs
│   ├── completion/             # llama-completion（单次补全）
│   │   └── completion.cpp
│   ├── quantize/               # llama-quantize（模型量化）
│   │   └── quantize.cpp
│   ├── imatrix/                # llama-imatrix（重要性矩阵）
│   │   └── imatrix.cpp
│   ├── perplexity/             # llama-perplexity（PPL/基准评测）
│   │   └── perplexity.cpp
│   ├── llama-bench/            # llama-bench（性能基准）
│   ├── batched-bench/          # 批处理基准
│   ├── tokenize/               # llama-tokenize（文本→token）
│   ├── export-lora/            # LoRA 导出
│   ├── gguf-split/             # GGUF 分片/合并
│   ├── fit-params/             # 参数拟合
│   ├── rpc/                    # RPC 服务
│   ├── mtmd/                   # 多模态（LLaVA、Qwen2-VL 等）
│   │   ├── mtmd.cpp, mtmd-cli.cpp, mtmd-helper.*
│   │   ├── clip.*, clip-graph.h, clip-impl.h, clip-model.h
│   │   ├── models/             # 各多模态架构（llava, qwen2vl, siglip ...）
│   │   ├── mtmd-audio.*        # 音频
│   │   └── legacy-models/      # 转换脚本（llava_surgery 等）
│   ├── tts/                    # TTS（OutEtts）
│   │   └── tts.cpp, tts-outetts.py
│   └── cvector-generator/      # cvector 生成
│
├── examples/                   # 示例程序
│   ├── CMakeLists.txt
│   ├── simple/                 # 最简加载与生成
│   ├── simple-chat/            # 最简对话
│   ├── batched/                # 批处理
│   ├── embedding/              # 嵌入
│   ├── save-load-state/        # 状态保存/加载
│   ├── speculative/, speculative-simple/  # 推测解码
│   ├── lookahead/              # Lookahead 解码
│   ├── retrieval/              # 检索
│   ├── training/               # 训练（finetune）
│   ├── model-conversion/       # 模型转换示例（脚本）
│   ├── gguf/, gguf-hash/       # GGUF 读写示例
│   ├── eval-callback/, parallel/, passkey/
│   ├── diffusion/, idle/, lookup/
│   ├── convert_legacy_llama.py, convert-llama2c-to-ggml/
│   ├── json_schema_to_grammar.py, regex_to_grammar.py 等
│   ├── llama.android/          # Android 示例
│   ├── llama.swiftui/          # SwiftUI 示例
│   └── sycl/                   # SYCL 示例
│
├── grammars/                   # 语法约束（.gbnf）
│   ├── README.md
│   ├── json.gbnf, json_arr.gbnf
│   ├── arithmetic.gbnf, list.gbnf
│   ├── c.gbnf, chess.gbnf, english.gbnf, japanese.gbnf
│   └── ...
│
├── models/                     # 模型相关模板/配置（多为 .jinja）
│   └── *.jinja, *.inp, *.out   # 用于生成或文档
│
├── gguf-py/                    # Python GGUF 库
│   ├── gguf/                   # 包实现
│   │   ├── *.py                # constants, reader, writer, tensor_mapping ...
│   │   └── ...
│   └── 测试与脚本
│
├── docs/                       # 文档
│   ├── build.md, install.md, docker.md
│   ├── ops.md                  # 算子与后端支持表
│   ├── ops/*.csv               # 各后端 CSV（由测试生成）
│   ├── backend/                # 各后端说明（CUDA, SYCL, OpenCL, CANN ...）
│   ├── development/            # 开发指南（HOWTO-add-model 等）
│   ├── multimodal.md, multimodal/  # 多模态说明
│   ├── function-calling.md, speculative.md, preset.md
│   ├── android.md, build-riscv64-spacemit.md, build-s390x.md
│   ├── LLAMA_CPP_ARCHITECTURE_CN.md
│   ├── LLAMA_CPP_LEARNING_GUIDE_CN.md
│   └── LLAMA_CPP_目录架构与文件说明_CN.md  # 本文档
│
├── scripts/                    # 脚本
│   ├── get-wikitext-2.sh, get-wikitext-103.sh
│   ├── get-hellaswag.sh, get-winogrande.sh, get-pg.sh
│   ├── sync-ggml.sh, sync-ggml.last, sync_vendor.py
│   ├── create_ops_docs.py      # 生成 ops 文档
│   ├── compare-llama-bench.py, compare-logprobs.py
│   ├── build-info.sh, gen-authors.sh, gen-unicode-data.py
│   ├── check-requirements.sh, debug-test.sh
│   ├── apple/, snapdragon/     # 平台相关
│   └── ...
│
├── tests/                      # 测试
│   ├── *.cpp, *.h              # 单测、后端算子测试（test-backend-ops 等）
│   └── *.sh
│
├── cmake/                      # CMake 模块
│   ├── *.cmake                 # 平台/编译器配置、build-info、git-vars 等
│   └── *.in                    # 模板
│
├── benches/                    # 基准配置
│   └── *.json, *.html, *.md
│
├── ci/                         # CI 脚本与说明
│   └── README.md, run.sh
│
├── .github/                    # GitHub Actions 等
│   └── *.yml, *.md
│
├── .devops/                    # Docker/Nix 等运维
│   └── *.Dockerfile, *.nix, *.spec
│
├── vendor/                     # 第三方依赖头文件等
├── media/                      # 图片、图标等资源
├── licenses/                   # 许可证文本
└── pocs/                       # 概念验证小示例
```

说明：`ggml/src` 下各后端的 `.cu`、`.comp`、`.cl` 等大量内核文件未逐条列出，仅以目录与代表文件概括。

---

## 二、根目录文件说明

| 文件 | 功能 |
|------|------|
| **CMakeLists.txt** | 主工程 CMake：定义选项（LLAMA_BUILD_TESTS/TOOLS/EXAMPLES/SERVER、GGML_CUDA/METAL/OPENCL 等）、添加 common、src、ggml、tools、examples、tests 等子目录，安装规则。 |
| **Makefile** | 传统 Make 入口，通常内部调用 cmake 构建。 |
| **convert_hf_to_gguf.py** | 将 HuggingFace 格式（safetensors/config）转为 GGUF，依赖 gguf-py。 |
| **convert_lora_to_gguf.py** | 将 LoRA 权重转换为 GGUF。 |
| **AUTHORS** | 贡献者名单。 |
| **CLAUDE.md / AGENTS.md** | 给 AI 助手与贡献者的使用与协作说明。 |

---

## 三、include/ — 对外 API

| 文件 | 功能 |
|------|------|
| **llama.h** | 主 C API：`llama_load_from_file`、`llama_new_context`、`llama_decode`、`llama_sampling_*`、`llama_kv_*`、会话保存/加载、batch、grammar、backlog 等；`llama_ftype` 与量化类型；错误码与回调。 |
| **ggml.h** | 由 ggml 子项目提供（或安装）：`ggml_tensor`、`ggml_type`、算子枚举、`ggml_quantize_chunk`、类型 trait 等。 |

---

## 四、common/ — 公共库

被 **src** 与 **tools** 共用，不依赖 ggml 内部实现。

| 文件/目录 | 功能 |
|-----------|------|
| **common.cpp/.h** | 通用工具、版本/构建信息、平台宏。 |
| **log.cpp/.h** | 日志接口（LOG_INF、LOG_ERR、LOG_TEE 等）。 |
| **arg.cpp/.h** | 命令行参数解析。 |
| **chat.cpp/.h** | 对话模板（chat template）解析与应用。 |
| **chat-parser*.cpp/.h** | 对话结构解析、PEG 解析、XML tool call。 |
| **sampling.cpp/.h** | 采样辅助（temperature、top_p、top_k 等）。 |
| **console.cpp/.h** | 控制台交互。 |
| **preset.cpp/.h** | 预设配置加载。 |
| **download.cpp/.h**, **http.h** | 下载与 HTTP 客户端。 |
| **json-partial.***, **json-schema-to-grammar.*** | 部分 JSON 解析、JSON Schema → grammar。 |
| **regex-partial.***, **peg-parser.*** | 正则部分匹配、PEG 解析器。 |
| **speculative.*** | 推测解码相关工具。 |
| **ngram-cache.***, **ngram-map.***, **ngram-mod.*** | n-gram 缓存与修改。 |
| **unicode.*** | Unicode 工具。 |
| **debug.*** | 调试辅助。 |
| **llguidance.cpp** | LLGuidance 集成（可选）。 |
| **jinja/** | C++ 实现的 Jinja 模板引擎（lexer、parser、runtime、value、string 等）。 |

---

## 五、src/ — llama 库实现

### 5.1 根级 .cpp/.h

| 文件 | 功能 |
|------|------|
| **llama.cpp** | 库入口、版本与构建信息导出。 |
| **llama-impl.cpp/.h** | 内部实现汇总、与 common 的衔接。 |
| **llama-arch.cpp/.h** | 架构枚举与注册，按 arch 选择对应 models/*.cpp。 |
| **llama-cparams.cpp/.h** | 上下文参数（n_ctx、n_batch 等）。 |
| **llama-model.cpp/.h** | 模型结构抽象、层与张量元数据。 |
| **llama-model-loader.cpp/.h** | 从 GGUF 加载模型（元数据与权重）。 |
| **llama-model-saver.cpp/.h** | 保存模型为 GGUF（含量化后写入）。 |
| **llama-mmap.cpp/.h** | 模型文件 mmap 与访问。 |
| **llama-io.cpp/.h** | 通用 IO 封装。 |
| **llama-hparams.cpp/.h** | 超参（层数、维度、头数、vocab 等）。 |
| **llama-quant.cpp/.h** | 量化入口：调用 ggml 的 `ggml_quantize_chunk`，ftype 与 ggml_type 映射。 |
| **llama-graph.cpp/.h** | **核心**：按架构构建前向计算图（MUL_MAT、RMSNorm、Attention、FFN 等）。 |
| **llama-batch.cpp/.h** | batch 构建与提交（供 decode 使用）。 |
| **llama-context.cpp/.h** | 上下文创建、decode 调度、与 backend 的衔接。 |
| **llama-kv-cache.cpp/.h** | KV cache 结构与更新。 |
| **llama-kv-cache-iswa.cpp/.h**, **llama-kv-cells.h** | KV 的 ISWA 与单元定义。 |
| **llama-memory.cpp/.h** | 内存分配与 buffer 管理。 |
| **llama-memory-hybrid*.cpp/.h**, **llama-memory-recurrent*.cpp/.h** | 混合内存、循环/状态复用。 |
| **llama-vocab.cpp/.h** | 词表加载与 token 转换。 |
| **llama-sampling.cpp/.h** | 采样（temperature、top_p、top_k、grammar）。 |
| **llama-grammar.cpp/.h** | 语法约束（grammar）解析与应用。 |
| **llama-adapter.cpp/.h** | 适配器（如 LoRA）加载与应用。 |
| **unicode*.cpp/.h** | Unicode 与分词边界。 |

### 5.2 src/models/ — 各架构前向图

| 类型 | 文件示例 | 功能 |
|------|-----------|------|
| 总控 | **models.h** | 架构枚举与注册、声明各 arch 的 graph 构建入口。 |
| LLaMA 系 | **llama.cpp** | LLaMA 前向图。 |
| Qwen 系 | **qwen.cpp, qwen2.cpp, qwen3.cpp, qwen2moe, qwen3moe, qwen2vl, qwen3vl** 等 | 各代 Qwen 文本与多模态。 |
| Gemma 系 | **gemma.cpp, gemma2-iswa, gemma3, gemma3n-iswa** | Gemma 文本与嵌入。 |
| 其他文本 | **deepseek.cpp, deepseek2, mistral3, falcon, mpt, bloom, gpt2, gptneox, phi2, phi3, chatglm, glm4, internlm2, mamba, rwkv6, rwkv7** 等 | 对应架构的前向图与张量命名。 |
| 多模态/视觉 | **llava.cpp, qwen2vl, qwen3vl, minicpmv, cogvlm, siglip, whisper-enc** 等 | 视觉编码器 + 语言模型的前向图。 |

每个 `.cpp` 实现该架构下「如何建图」：层数、张量名、Attention/FFN 形状、RoPE 等，供 **llama-graph.cpp** 按 arch 调用。

---

## 六、ggml/ — 计算引擎

### 6.1 ggml/include/

| 文件 | 功能 |
|------|------|
| **ggml.h** | 张量（`ggml_tensor`）、算子枚举（`GGML_OP_*`）、`ggml_type`（F32/F16/Q4_0/Q4_K 等）、类型 trait、`ggml_quantize_chunk`、图构建 API、线程池等。 |
| **gguf.h** | GGUF 文件格式：读写接口、元数据与张量列表。 |
| **ggml-backend.h** | 后端抽象：buffer、device、graph 调度。 |
| **ggml-alloc.h** | 内部分配器。 |
| **ggml-cpu.h** | CPU 后端与类型 trait。 |
| **ggml-cuda.h**, **ggml-metal.h**, **ggml-opencl.h**, **ggml-vulkan.h**, **ggml-sycl.h** | 各后端 API。 |
| **ggml-cann.h**, **ggml-zendnn.h**, **ggml-zdnn.h**, **ggml-hexagon.h**, **ggml-webgpu.h** | 其他后端。 |
| **ggml-blas.h** | BLAS 封装。 |
| **ggml-opt.h** | 优化器（训练）。 |
| **ggml-cpp.h** | C++ 封装。 |
| **ggml-rpc.h**, **ggml-virtgpu.h** | RPC 与虚拟 GPU。 |

### 6.2 ggml/src/

| 路径/文件 | 功能 |
|-----------|------|
| **ggml.c** | 图构建、算子实现、**量化**（`ggml_quantize_chunk` 的 switch、各 `quantize_xxx`）、类型 trait、线程池。 |
| **ggml.cpp** | C++ 扩展与封装。 |
| **gguf.cpp** | GGUF 读写实现。 |
| **ggml-quants.h** | 各量化格式的 block 结构体与 `dequantize_row_xxx` 声明。 |
| **ggml-cpu/** | CPU 后端：算子、量化/反量化、vec_dot、各架构 SIMD（x86/arm/riscv/loongarch/powerpc/s390/wasm 等）、repack、AMX、HBM。 |
| **ggml-cuda/** | CUDA 后端：MMQ/MMVQ、dequantize、quantize、Flash Attention、各量化核的 .cu/.cuh 及 template-instances。 |
| **ggml-metal/** | Metal 实现（.m/.metal）。 |
| **ggml-opencl/** | OpenCL 实现（kernels/*.cl）。 |
| **ggml-vulkan/** | Vulkan 实现（.comp/.glsl）。 |
| **ggml-sycl/** | SYCL 实现。 |
| **ggml-cann/** | 华为 CANN。 |
| **ggml-hexagon/** | Qualcomm Hexagon HTP。 |
| **ggml-webgpu/** | WebGPU（.wgsl）。 |
| **ggml-zendnn/**, **ggml-zdnn/** | ZenDNN、zDNN。 |
| **ggml-blas/** | BLAS 封装实现。 |
| **ggml-rpc/**, **ggml-virtgpu/** | RPC 与虚拟 GPU。 |
| **ggml-hip/**, **ggml-musa/** | HIP(AMD)、MUSA(摩尔线程)。 |

---

## 七、tools/ — 可执行工具

| 目录 | 主文件 | 生成可执行文件 | 功能 |
|------|--------|----------------|------|
| **cli/** | cli.cpp | llama-cli | 交互式/非交互式对话与补全。 |
| **server/** | server.cpp, server-*.cpp | llama-server | HTTP API（OpenAI 兼容）、队列、WebUI、多模型。 |
| **completion/** | completion.cpp | llama-completion | 单次补全（无对话）。 |
| **quantize/** | quantize.cpp | llama-quantize | GGUF 模型量化（F32/F16→Q4_K 等）。 |
| **imatrix/** | imatrix.cpp | llama-imatrix | 计算 importance matrix，供量化使用。 |
| **perplexity/** | perplexity.cpp | llama-perplexity | 困惑度与基准（PPL、HellaSwag、WinoGrande、multiple_choice）。 |
| **llama-bench/** | llama-bench.cpp | llama-bench | 推理性能基准。 |
| **batched-bench/** | batched-bench.cpp | — | 批处理基准。 |
| **tokenize/** | tokenize.cpp | llama-tokenize | 文本→token id。 |
| **export-lora/** | export-lora.cpp | llama-export-lora | LoRA 导出。 |
| **gguf-split/** | gguf-split.cpp | gguf-split | GGUF 分片/合并。 |
| **fit-params/** | fit-params.cpp | fit-params | 参数拟合。 |
| **rpc/** | rpc-server.cpp | — | RPC 服务。 |
| **mtmd/** | mtmd.cpp, mtmd-cli.cpp | mtmd-cli 等 | 多模态：LLaVA、Qwen2-VL、MiniCPM-V 等视觉/音频。 |
| **tts/** | tts.cpp | — | TTS（OutEtts）。 |
| **cvector-generator/** | cvector-generator.cpp | — | cvector 生成。 |

**tools/server/** 子目录：**public/**、**public_legacy/**、**public_simplechat/** 为静态前端；**webui/** 为 Svelte/TS WebUI；**themes/** 为主题；**bench/**、**tests/** 为基准与测试。

---

## 八、examples/ — 示例

| 目录/文件 | 功能 |
|-----------|------|
| **simple/** | 最简：加载模型、生成。 |
| **simple-chat/** | 最简对话。 |
| **batched/** | 批处理推理。 |
| **embedding/** | 嵌入向量。 |
| **save-load-state/** | 保存/加载解码状态。 |
| **speculative/**, **speculative-simple/** | 推测解码。 |
| **lookahead/** | Lookahead 解码。 |
| **retrieval/** | 检索。 |
| **training/** | 训练（finetune.cpp）。 |
| **model-conversion/** | 模型转换示例（Python/Shell）。 |
| **gguf/**, **gguf-hash/** | GGUF 读写、哈希示例。 |
| **eval-callback/**, **parallel/**, **passkey/** | 评估回调、并行、passkey。 |
| **diffusion/**, **idle/**, **lookup/** | 扩散、idle、lookup。 |
| **convert_legacy_llama.py**, **convert-llama2c-to-ggml/** | 旧格式/Llama2C 转换。 |
| **json_schema_to_grammar.py** 等 | JSON Schema/正则转 grammar。 |
| **llama.android/** | Android 示例（Kotlin）。 |
| **llama.swiftui/** | SwiftUI 示例。 |
| **sycl/** | SYCL 构建与运行示例。 |

---

## 九、grammars/、models/、gguf-py/

| 路径 | 功能 |
|------|------|
| **grammars/** | .gbnf 语法文件（JSON、算术、列表、C、棋类等），供 grammar 约束生成。 |
| **models/** | 多为 .jinja 模板，用于模型结构生成或文档。 |
| **gguf-py/** | Python 包：GGUF 读写、constants、tensor_mapping 等，供 convert 脚本使用。 |

---

## 十、docs/、scripts/、tests/、cmake/

| 路径 | 功能 |
|------|------|
| **docs/** | build.md、install.md、ops.md、各 backend 说明、development 指南、multimodal、中文架构与学习指南等。 |
| **scripts/** | 数据集下载（get-wikitext-*.sh、get-hellaswag.sh 等）、sync-ggml、create_ops_docs.py、compare-*、apple/snapdragon 平台脚本。 |
| **tests/** | C++ 单测、test-backend-ops 等，生成 docs/ops/*.csv。 |
| **cmake/** | 平台/编译器配置、build-info、git-vars、download-models、license 等 CMake 模块。 |

---

## 十一、其他根级目录

| 路径 | 功能 |
|------|------|
| **benches/** | 基准运行配置与结果。 |
| **ci/** | CI 说明与脚本。 |
| **.github/** | GitHub Actions 工作流。 |
| **.devops/** | Docker、Nix、spec 等运维配置。 |
| **vendor/** | 第三方头文件或依赖。 |
| **media/** | 图片、图标等资源。 |
| **licenses/** | 许可证文本。 |
| **pocs/** | 概念验证小示例。 |

---

以上为 llama.cpp 的完整目录树与各目录、关键文件的功能说明。与「量化 / 推理 / 新模型」相关的重点路径可参考 [LLAMA_CPP_ARCHITECTURE_CN.md](LLAMA_CPP_ARCHITECTURE_CN.md) 与 [LLAMA_CPP_LEARNING_GUIDE_CN.md](LLAMA_CPP_LEARNING_GUIDE_CN.md)。
