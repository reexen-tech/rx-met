# common_params_parser_init 参数配置说明

本文档整理 `common/arg.cpp` 中 `common_params_parser_init` 所注册的公共参数，供查阅与开发参考。完整定义与默认值以源码为准。

---

## 1. 函数与上下文

### 1.1 声明与用途

- **声明**：`common/arg.h`
  ```c
  common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);
  ```
- **定义**：`common/arg.cpp` 约第 981 行起。
- **作用**：根据当前示例类型 `ex` 初始化公共参数解析上下文，填充 `common_params_context.options`，供 CLI 解析、测试和 preset 使用。

### 1.2 返回值与过滤规则

- **返回**：`common_params_context`，包含：
  - `ex`：当前示例类型
  - `params`：公共参数结构引用
  - `options`：经示例过滤后的参数列表
  - `print_usage`：可选的用法打印回调

- **选项过滤规则**（`add_opt` 内的 lambda）：
  - 继承 **COMMON**：所有示例都包含 `LLAMA_EXAMPLE_COMMON` 的选项。
  - 示例专属：若选项的 `examples` 包含某 `LLAMA_EXAMPLE_*`，则仅在该示例下出现。
  - 排除：若选项的 `excludes` 包含当前示例，则该选项不出现。

### 1.3 示例类型（llama_example）

| 枚举值 | 说明 |
|--------|------|
| LLAMA_EXAMPLE_COMMON | 公共选项，所有示例继承 |
| LLAMA_EXAMPLE_BATCHED | batched 示例 |
| LLAMA_EXAMPLE_DEBUG | llama-debug |
| LLAMA_EXAMPLE_SPECULATIVE | 推测解码 |
| LLAMA_EXAMPLE_COMPLETION | llama-cli 补全 |
| LLAMA_EXAMPLE_CLI | 通用 CLI |
| LLAMA_EXAMPLE_EMBEDDING | 嵌入 |
| LLAMA_EXAMPLE_PERPLEXITY | 困惑度/评估 |
| LLAMA_EXAMPLE_RETRIEVAL | 检索 |
| LLAMA_EXAMPLE_PASSKEY | passkey 测试 |
| LLAMA_EXAMPLE_IMATRIX | imatrix |
| LLAMA_EXAMPLE_BENCH | batched-bench |
| LLAMA_EXAMPLE_SERVER | server |
| LLAMA_EXAMPLE_CVECTOR_GENERATOR | control vector 生成 |
| LLAMA_EXAMPLE_EXPORT_LORA | export-lora |
| LLAMA_EXAMPLE_MTMD | 多模态 (mtmd) |
| LLAMA_EXAMPLE_LOOKUP | lookup 解码 |
| LLAMA_EXAMPLE_PARALLEL | parallel 示例 |
| LLAMA_EXAMPLE_TTS | TTS |
| LLAMA_EXAMPLE_DIFFUSION | 扩散模型 |
| LLAMA_EXAMPLE_FINETUNE | 微调 |
| LLAMA_EXAMPLE_FIT_PARAMS | 参数拟合 |

---

## 2. 参数分类一览

以下按功能分组列出选项名（长格式为主）及简要说明。带 `set_env(...)` 的可用对应环境变量覆盖；`set_examples(...)` 表示仅部分示例可见。

### 2.1 帮助与信息

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `-h`, `--help`, `--usage` | 打印用法并退出 | - |
| `--version` | 版本与构建信息 | - |
| `--license` | 源码与依赖许可 | - |
| `-cl`, `--cache-list` | 列出缓存中的模型 | - |
| `--completion-bash` | 输出可 source 的 bash 补全脚本 | - |

### 2.2 输出与显示

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `--verbose-prompt` | 生成前打印详细 prompt | - |
| `--display-prompt` / `--no-display-prompt` | 是否在生成时打印 prompt | COMPLETION, CLI |
| `-co`, `--color` | 彩色输出：on/off/auto | COMPLETION, CLI, SPECULATIVE, LOOKUP |
| `--perf` / `--no-perf` | 是否启用内部性能计时 | LLAMA_ARG_PERF |
| `--show-timings` / `--no-show-timings` | 是否在每次响应后显示计时 | CLI, LLAMA_ARG_SHOW_TIMINGS |
| `-ptc`, `--print-token-count` | 每 N 个 token 打印一次计数 | COMPLETION |

### 2.3 CPU 与线程

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `-t`, `--threads` | 生成阶段 CPU 线程数 | LLAMA_ARG_THREADS |
| `-tb`, `--threads-batch` | batch/prompt 处理线程数 | - |
| `-C`, `--cpu-mask` | CPU 亲和掩码（十六进制） | - |
| `-Cr`, `--cpu-range` | CPU 亲和范围 lo-hi | - |
| `--cpu-strict` | 是否严格 CPU 放置 | - |
| `--prio` | 进程/线程优先级：low(-1), normal(0), medium(1), high(2), realtime(3) | - |
| `--poll` | 等待工作的轮询级别 0–100 | - |
| `-Cb`, `--cpu-mask-batch` | batch 用 CPU 掩码 | - |
| `-Crb`, `--cpu-range-batch` | batch 用 CPU 范围 | - |
| `--cpu-strict-batch`, `--prio-batch`, `--poll-batch` | batch 用 CPU 严格/优先级/轮询 | - |

### 2.4 Lookup 缓存（推测/解码）

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `-lcs`, `--lookup-cache-static` | 静态 lookup 缓存路径（不随生成更新） | LOOKUP, SERVER |
| `-lcd`, `--lookup-cache-dynamic` | 动态 lookup 缓存路径（随生成更新） | LOOKUP, SERVER |

### 2.5 上下文与批次

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `-c`, `--ctx-size` | 上下文长度（0=从模型读取） | LLAMA_ARG_CTX_SIZE |
| `-n`, `--predict`, `--n-predict` | 预测 token 数（-1=无限制，COMPLETION 下 -2=填满上下文） | LLAMA_ARG_N_PREDICT |
| `-b`, `--batch-size` | 逻辑最大 batch 大小 | LLAMA_ARG_BATCH |
| `-ub`, `--ubatch-size` | 物理最大 batch 大小 | LLAMA_ARG_UBATCH |
| `--keep` | 从初始 prompt 保留的 token 数（-1=全部） | - |
| `--swa-full` | 使用完整大小 SWA 缓存 | LLAMA_ARG_SWA_FULL |
| `--ctx-checkpoints`, `--swa-checkpoints` | 每 slot 最大上下文检查点数量 | SERVER, CLI, LLAMA_ARG_CTX_CHECKPOINTS |
| `-cram`, `--cache-ram` | 最大缓存 MiB（-1=无限制，0=禁用） | SERVER, CLI, LLAMA_ARG_CACHE_RAM |
| `-kvu`, `--kv-unified` / `-no-kvu`, `--no-kv-unified` | 是否使用统一 KV 缓冲 | SERVER, PERPLEXITY, BATCHED, BENCH, LLAMA_ARG_KV_UNIFIED |
| `--context-shift` / `--no-context-shift` | 无限生成长文本时是否使用 context shift | COMPLETION, CLI, SERVER, IMATRIX, PERPLEXITY, LLAMA_ARG_CONTEXT_SHIFT |
| `--chunks` | 最大处理 chunk 数（-1=全部） | IMATRIX, PERPLEXITY, RETRIEVAL |

### 2.6 Flash Attention 与模型能力

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `-fa`, `--flash-attn` | Flash Attention：on/off/auto | LLAMA_ARG_FLASH_ATTN |

### 2.7 Prompt 与输入

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `-p`, `--prompt` | 生成起始 prompt（排除 SERVER） | - |
| `-sys`, `--system-prompt` | 系统 prompt（依 chat template） | COMPLETION, CLI, DIFFUSION, MTMD |
| `-f`, `--file` | 从文件读取 prompt（排除 SERVER） | - |
| `-sysf`, `--system-prompt-file` | 从文件读取系统 prompt | COMPLETION, CLI, DIFFUSION |
| `--in-file` | 输入文件（可逗号分隔多个） | IMATRIX |
| `-bf`, `--binary-file` | 二进制 prompt 文件（排除 SERVER） | - |
| `-e`, `--escape` / `--no-escape` | 是否解析转义序列 | - |
| `--prompt-cache` | prompt 状态缓存文件路径 | COMPLETION |
| `--prompt-cache-all` | 将用户输入与生成也写入缓存 | COMPLETION |
| `--prompt-cache-ro` | 只读使用 prompt 缓存且不更新 | COMPLETION |
| `-r`, `--reverse-prompt` | 遇到该字符串时停止并交回控制 | COMPLETION, CLI, SERVER |
| `-sp`, `--special` | 输出特殊 token | COMPLETION, CLI, SERVER |
| `-cnv`, `--conversation` / `-no-cnv`, `--no-conversation` | 对话模式（不打印特殊 token 等） | COMPLETION, CLI |
| `-st`, `--single-turn` | 单轮对话后退出 | COMPLETION, CLI |
| `-i`, `--interactive` | 交互模式 | COMPLETION |
| `-if`, `--interactive-first` | 交互模式并立即等待输入 | COMPLETION |
| `-mli`, `--multiline-input` | 多行输入（无需每行结尾 `\`） | COMPLETION, CLI |
| `--in-prefix-bos` | 在用户输入前加 BOS（在 `--in-prefix` 前） | COMPLETION |
| `--in-prefix` | 用户输入前缀 | COMPLETION |
| `--in-suffix` | 用户输入后缀 | COMPLETION |
| `--warmup` / `--no-warmup` | 是否执行空跑预热 | 多示例 |
| `--spm-infill` | 使用 SPM 中缀模式（Suffix/Prefix/Middle） | SERVER |

### 2.8 采样（Sampling，多为 set_sparam）

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `--samplers` | 采样器顺序，分号分隔 | - |
| `-s`, `--seed` | RNG 种子 | - |
| `--sampler-seq`, `--sampling-seq` | 采样器简写序列 | - |
| `--ignore-eos` | 忽略 EOS 继续生成 | - |
| `--temp` | 温度 | - |
| `--top-k` | top-k（0=禁用） | LLAMA_ARG_TOP_K |
| `--top-p` | top-p（1.0=禁用） | - |
| `--min-p` | min-p（0.0=禁用） | - |
| `--top-nsigma` | top-n-sigma（-1=禁用） | - |
| `--xtc-probability`, `--xtc-threshold` | XTC 概率与阈值 | - |
| `--typical` | locally typical p（1.0=禁用） | - |
| `--repeat-last-n` | 惩罚考虑的最近 n 个 token | - |
| `--repeat-penalty` | 重复序列惩罚 | - |
| `--presence-penalty`, `--frequency-penalty` | 存在/频率惩罚 | - |
| `--dry-*` | DRY 采样（multiplier/base/allowed-length/penalty-last-n/sequence-breaker） | - |
| `--adaptive-target`, `--adaptive-decay` | adaptive-p 目标与衰减 | - |
| `--dynatemp-range`, `--dynatemp-exp` | 动态温度范围与指数 | - |
| `--mirostat`, `--mirostat-lr`, `--mirostat-ent` | Mirostat 开关、学习率、目标熵 | - |
| `-l`, `--logit-bias` | token 偏置（TOKEN_ID(+/-)BIAS） | - |
| `--grammar`, `--grammar-file` | BNF 语法约束 | - |
| `-j`, `--json-schema`, `-jf`, `--json-schema-file` | JSON Schema 约束 | - |
| `-bs`, `--backend-sampling` | 启用后端采样（实验） | LLAMA_ARG_BACKEND_SAMPLING |

### 2.9 嵌入 / RoPE / 注意力

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `--pooling` | 嵌入池化：none/mean/cls/last/rank | EMBEDDING, RETRIEVAL, SERVER, DEBUG, LLAMA_ARG_POOLING |
| `--attention` | 注意力类型：causal/non-causal | EMBEDDING |
| `--rope-scaling` | RoPE 缩放：none/linear/yarn | LLAMA_ARG_ROPE_SCALING_TYPE |
| `--rope-scale` | RoPE 上下文缩放因子 | LLAMA_ARG_ROPE_SCALE |
| `--rope-freq-base`, `--rope-freq-scale` | RoPE 基频与缩放 | LLAMA_ARG_* |
| `--yarn-*` | YaRN 相关（orig-ctx, ext-factor, attn-factor, beta-slow, beta-fast） | LLAMA_ARG_YARN_* |
| `-gan`, `--grp-attn-n`, `-gaw`, `--grp-attn-w` | 分组注意力因子与宽度 | COMPLETION, PASSKEY/COMPLETION, LLAMA_ARG_GRP_ATTN_* |

### 2.10 KV 与缓存类型

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `-kvo`, `--kv-offload` / `-nkvo`, `--no-kv-offload` | 是否启用 KV 卸载 | LLAMA_ARG_KV_OFFLOAD |
| `--repack` / `-nr`, `--no-repack` | 是否启用权重重打包 | LLAMA_ARG_REPACK |
| `--no-host` | 绕过 host 缓冲以使用 extra buffers | LLAMA_ARG_NO_HOST |
| `-ctk`, `--cache-type-k`, `-ctv`, `--cache-type-v` | K/V 缓存数据类型 | LLAMA_ARG_CACHE_TYPE_K/V |

### 2.11 评估（困惑度等）

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `--hellaswag`, `--hellaswag-tasks` | HellaSwag 任务数与数量 | PERPLEXITY |
| `--winogrande`, `--winogrande-tasks` | Winogrande 同上 | PERPLEXITY |
| `--multiple-choice`, `--multiple-choice-tasks` | 多选题评估 | PERPLEXITY |
| `--kl-divergence` | 计算 KL 散度 | PERPLEXITY |
| `--save-all-logits`, `--kl-divergence-base` | 作为 KL 基准的 logits 文件 | PERPLEXITY |
| `--ppl-stride`, `--ppl-output-type` | 困惑度步长与输出类型 | PERPLEXITY |
| `-dt`, `--defrag-thold` | （已废弃）KV 碎片整理阈值 | LLAMA_ARG_DEFRAG_THOLD |

### 2.12 并行与 Server 槽位

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `-np`, `--parallel` | 并行序列数 / server 槽位数（SERVER 下 -1=自动） | LLAMA_ARG_N_PARALLEL |
| `-ns`, `--sequences` | 解码序列数 | PARALLEL |
| `-cb`, `--cont-batching` / `-nocb`, `--no-cont-batching` | 是否启用连续/动态 batching | SERVER, LLAMA_ARG_CONT_BATCHING |

### 2.13 多模态（mmproj / 图像 / 音频）

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `-mm`, `--mmproj` | 多模态 projector 文件路径 | 多示例, LLAMA_ARG_MMPROJ |
| `-mmu`, `--mmproj-url` | projector 下载 URL | LLAMA_ARG_MMPROJ_URL |
| `--mmproj-auto` / `--no-mmproj` | 是否自动使用 projector（如 -hf 时） | LLAMA_ARG_MMPROJ_AUTO |
| `--mmproj-offload` / `--no-mmproj-offload` | projector 是否 GPU 卸载 | LLAMA_ARG_MMPROJ_OFFLOAD |
| `--image`, `--audio` | 图像/音频文件路径（可逗号分隔） | MTMD, CLI |
| `--image-min-tokens`, `--image-max-tokens` | 每图最少/最多 token 数（动态分辨率） | LLAMA_ARG_IMAGE_* |

### 2.14 设备与加载

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `--rpc` | RPC 服务器列表 host:port（逗号分隔） | LLAMA_ARG_RPC（若支持 RPC） |
| `--mlock` | 强制模型驻留 RAM | LLAMA_ARG_MLOCK |
| `--mmap` / `--no-mmap` | 是否 mmap 模型 | LLAMA_ARG_MMAP |
| `-dio`, `--direct-io` / `-ndio`, `--no-direct-io` | 是否使用 DirectIO | LLAMA_ARG_DIO |
| `--numa` | NUMA：distribute/isolate/numactl | LLAMA_ARG_NUMA |
| `-dev`, `--device` | 用于卸载的设备列表（逗号分隔） | LLAMA_ARG_DEVICE |
| `--list-devices` | 列出可用设备并退出 | - |
| `-ot`, `--override-tensor` | 张量 buffer 类型覆盖 | LLAMA_ARG_OVERRIDE_TENSOR |
| `-otd`, `--override-tensor-draft` | draft 模型张量覆盖 | SPECULATIVE, SERVER, CLI |
| `-cmoe`, `--cpu-moe` | MoE 权重保留在 CPU | LLAMA_ARG_CPU_MOE |
| `-ncmoe`, `--n-cpu-moe` | 前 N 层 MoE 在 CPU | LLAMA_ARG_N_CPU_MOE |
| `-cmoed`, `--cpu-moe-draft`, `-ncmoed`, `--n-cpu-moe-draft` | draft 模型 MoE 同上 | SPECULATIVE, SERVER, CLI |
| `-ngl`, `--gpu-layers`, `--n-gpu-layers` | 放入 VRAM 的层数（数字/auto/all） | LLAMA_ARG_N_GPU_LAYERS |
| `-sm`, `--split-mode` | 多 GPU 拆分：none/layer/row | LLAMA_ARG_SPLIT_MODE |
| `-ts`, `--tensor-split` | 各 GPU 张量比例（逗号分隔） | LLAMA_ARG_TENSOR_SPLIT |
| `-mg`, `--main-gpu` | 主 GPU 索引 | LLAMA_ARG_MAIN_GPU |
| `-fit`, `--fit` | 是否根据设备内存自动调整未设参数 | LLAMA_ARG_FIT |
| `-fitt`, `--fit-target` | 每设备目标余量 MiB（逗号或单值） | LLAMA_ARG_FIT_TARGET |
| `-fitc`, `--fit-ctx` | --fit 可设置的最小 ctx | LLAMA_ARG_FIT_CTX |
| `--check-tensors` | 检查模型张量非法值 | - |
| `--override-kv` | 覆盖模型元数据 KEY=TYPE:VALUE,... | - |
| `--op-offload` / `--no-op-offload` | 是否将 host 张量运算卸载到设备 | - |

### 2.15 LoRA / Control Vector / 模型与来源

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `--lora` | LoRA 适配器路径（可逗号分隔多个） | COMMON, EXPORT_LORA |
| `--lora-scaled` | LoRA 路径与缩放 FNAME:SCALE,... | COMMON, EXPORT_LORA |
| `--control-vector` | control vector 路径（可多个） | - |
| `--control-vector-scaled` | control vector 路径与缩放 | - |
| `--control-vector-layer-range` | control vector 作用层范围 START END | - |
| `-a`, `--alias` | 模型别名（REST API 用） | SERVER, LLAMA_ARG_ALIAS |
| `-m`, `--model` | 模型路径（或 export-lora 的 base 模型） | COMMON, EXPORT_LORA, LLAMA_ARG_MODEL |
| `-mu`, `--model-url` | 模型下载 URL | LLAMA_ARG_MODEL_URL |
| `-dr`, `--docker-repo` | Docker 仓库 [repo/]model[:quant] | LLAMA_ARG_DOCKER_REPO |
| `-hf`, `-hfr`, `--hf-repo` | Hugging Face 仓库 user/model[:quant] | LLAMA_ARG_HF_REPO |
| `-hfd`, `--hf-repo-draft` | draft 模型 HF 仓库 | LLAMA_ARG_HFD_REPO |
| `-hff`, `--hf-file` | HF 模型文件名（覆盖 quant） | LLAMA_ARG_HF_FILE |
| `-hfv`, `--hf-repo-v`, `-hffv`, `--hf-file-v` | 声码器 HF 仓库/文件 | LLAMA_ARG_HF_REPO_V / HF_FILE_V |
| `-hft`, `--hf-token` | HF 访问 token | HF_TOKEN |
| `--context-file` | 从文件加载 context（可多个） | - |
| `--lora-init-without-apply` | 仅加载 LoRA 不应用（SERVER 下可后续 POST 应用） | SERVER |

### 2.16 Server（host / port / API / WebUI / 缓存 / 槽位）

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `--host` | 监听地址或 .sock 路径 | SERVER, LLAMA_ARG_HOST |
| `--port` | 监听端口 | SERVER, LLAMA_ARG_PORT |
| `--path` | 静态文件路径 | SERVER, LLAMA_ARG_STATIC_PATH |
| `--api-prefix` | API 路径前缀（无尾部斜杠） | SERVER, LLAMA_ARG_API_PREFIX |
| `--webui-config`, `--webui-config-file` | WebUI 默认配置 JSON/文件 | SERVER, LLAMA_ARG_WEBUI_CONFIG* |
| `--webui` / `--no-webui` | 是否启用 Web UI | SERVER, LLAMA_ARG_WEBUI |
| `--embedding`, `--embeddings` | 仅支持嵌入（专用嵌入模型） | SERVER, DEBUG, LLAMA_ARG_EMBEDDINGS |
| `--rerank`, `--reranking` | 启用 rerank 端点 | SERVER, LLAMA_ARG_RERANKING |
| `--api-key`, `--api-key-file` | API 认证 key/文件 | SERVER, LLAMA_API_KEY |
| `--ssl-key-file`, `--ssl-cert-file` | SSL 私钥与证书 | SERVER, LLAMA_ARG_SSL_* |
| `--chat-template-kwargs` | chat 模板额外 JSON 参数 | SERVER, CLI, LLAMA_CHAT_TEMPLATE_KWARGS |
| `-to`, `--timeout` | 读写超时（秒） | SERVER, LLAMA_ARG_TIMEOUT |
| `--threads-http` | 处理 HTTP 的线程数 | SERVER, LLAMA_ARG_THREADS_HTTP |
| `--cache-prompt` / `--no-cache-prompt` | 是否启用 prompt 缓存 | SERVER, LLAMA_ARG_CACHE_PROMPT |
| `--cache-reuse` | 最小 chunk 以尝试 KV 复用 | SERVER, LLAMA_ARG_CACHE_REUSE |
| `--metrics` | Prometheus 指标端点 | SERVER, LLAMA_ARG_ENDPOINT_METRICS |
| `--props` | 允许 POST /props 改全局属性 | SERVER, LLAMA_ARG_ENDPOINT_PROPS |
| `--slots` / `--no-slots` | 槽位监控端点 | SERVER, LLAMA_ARG_ENDPOINT_SLOTS |
| `--slot-save-path` | 槽位 KV 缓存保存目录 | SERVER |
| `--media-path` | 本地媒体目录（file:// 相对路径） | SERVER |
| `--models-dir`, `--models-preset`, `--models-max`, `--models-autoload` | 路由模型目录/预设/最大数/自动加载 | SERVER, LLAMA_ARG_MODELS_* |
| `--jinja` / `--no-jinja` | 是否使用 Jinja 模板 | SERVER, COMPLETION, CLI, MTMD, LLAMA_ARG_JINJA |
| `--reasoning-format` | 思考标签解析与返回格式：none/deepseek/deepseek-legacy | SERVER, COMPLETION, CLI, LLAMA_ARG_THINK |
| `--reasoning-budget` | 思考预算（-1=无限制，0=禁用） | SERVER, COMPLETION, CLI, LLAMA_ARG_THINK_BUDGET |
| `--chat-template`, `--chat-template-file` | 自定义 Jinja chat 模板/文件 | COMPLETION, CLI, SERVER, MTMD, LLAMA_ARG_CHAT_TEMPLATE* |
| `--prefill-assistant` / `--no-prefill-assistant` | 最后一条为 assistant 时是否预填 | SERVER, LLAMA_ARG_PREFILL_ASSISTANT |
| `-sps`, `--slot-prompt-similarity` | 请求与槽位 prompt 相似度阈值 | SERVER |
| `--sleep-idle-seconds` | 空闲多少秒后 server 休眠（-1=不休眠） | SERVER |

### 2.17 其它工具与杂项

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `--simple-io` | 简化 IO（子进程/受限终端） | COMPLETION, CLI |
| `--positive-file`, `--negative-file` | control vector 正/负 prompt 文件 | CVECTOR_GENERATOR |
| `--pca-batch`, `--pca-iter`, `--method` | PCA 批大小/迭代/降维方法（pca/mean） | CVECTOR_GENERATOR |
| `--output-format` (batched-bench) | batched-bench 输出格式 md/jsonl | BENCH |
| `--log-disable`, `--log-file`, `--log-colors` | 日志禁用/文件/颜色 | LLAMA_LOG_* |
| `-v`, `--verbose`, `--log-verbose` | 最高详细度 | - |
| `--offline` | 离线模式（仅缓存、无网络） | LLAMA_OFFLINE |
| `-lv`, `--verbosity`, `--log-verbosity` | 详细度阈值 0–4 | LLAMA_LOG_VERBOSITY |
| `--log-prefix`, `--log-timestamps` | 日志前缀/时间戳 | LLAMA_LOG_PREFIX, LLAMA_LOG_TIMESTAMPS |

### 2.18 推测解码（draft 线程 / CPU）

| 选项 | 说明 | 环境变量/示例 |
|------|------|----------------|
| `-td`, `--threads-draft` | draft 生成线程数 | SPECULATIVE, SERVER |
| `-tbd`, `--threads-batch-draft` | draft batch 线程数 | SPECULATIVE, SERVER |
| `-Cd`, `--cpu-mask-draft`, `-Crd`, `--cpu-range-draft` | draft CPU 掩码/范围 | SPECULATIVE |
| `--cpu-strict-draft` 等 | draft 严格 CPU/优先级/轮询 | SPECULATIVE, SERVER |

（其余推测解码选项如 draft 模型路径、n_max/n_min、ngram 等在 arg.cpp 中继续以 add_opt 注册。）

### 2.19 嵌入 / 检索 / IMATRIX / BENCH / 其它示例

- **Embedding**：`--embd-normalize`, `--embd-output-format`, `--embd-separator`, `--cls-separator` 等（见 EMBEDDING, DEBUG）。
- **Retrieval**：`--chunk-size`, `--chunk-separator` 等（见 RETRIEVAL）。
- **IMATRIX**：`-o`, `--output`, `-ofreq`, `--output-frequency`, `--output-format`, `--save-frequency`, `--process-output`, `--ppl`/`--no-ppl`, `--chunk`, `--show-statistics`, `--parse-special` 等（见 IMATRIX）。
- **BENCH / PARALLEL**：`-pps`, `-tgs`, `-npp`, `-ntg`, `-npl` 等（见 BENCH, PARALLEL）。
- **PASSKEY**：`--junk`, `--pos`（见 PASSKEY）。

---

## 3. 使用与扩展说明

- **CLI**：各示例（如 llama-cli、server）在 main 中调用 `common_params_parse()`，内部会使用 `common_params_parser_init()` 得到的 `options` 解析命令行并填充 `common_params`。
- **Preset**：`common/preset.cpp` 等通过 `common_params_parser_init(default_params, ex)` 获取与示例匹配的选项，用于从 INI/JSON 等加载预设；部分选项通过 `common_params_add_preset_options()` 加入且仅作 preset 用（非 CLI）。
- **环境变量**：表中标注的 `LLAMA_ARG_*`、`HF_TOKEN`、`LLAMA_OFFLINE` 等可在不传 CLI 时覆盖对应参数（若实现中调用了 `get_value_from_env`）。
- **新增参数**：在 `common/arg.cpp` 的 `common_params_parser_init` 内对 `ctx_arg.options` 使用 `add_opt(common_arg(...))` 增加；通过 `set_examples`/`set_excludes`/`set_env`/`set_sparam` 控制可见性与环境变量。

---

## 4. 参考源码位置

- 参数解析上下文与初始化：`common/arg.h`（common_params_context、common_params_parser_init）、`common/arg.cpp`（约 981–3810 行及后续 add_opt）。
- 公共参数结构：`common/common.h`（common_params、common_params_sampling、common_params_speculative 等）。
- 解析入口：`common_params_parse()`、`common_params_to_map()`（arg.cpp）；preset 加载与 to_args：`common/preset.cpp`。

以上内容根据当前代码整理，具体行为与默认值以 `common/arg.cpp` 与 `common/common.h` 为准。
