# llm_quant

llama.cpp 侧的**权重量化预处理**算法库。各方法及其专属 CLI、测试按目录归档。

```
llama.cpp/
  export_awq_hf.py            # CLI：一步导出 AWQ-scaled FP16 HF
  llm_quant/
    awq/                      # AWQ search（从 llm-awq copy）
    gptq/                     # GPTQ for Q4_0_64
    smoothquant/              # SmoothQuant Step 1（从 mit-han-lab/smoothquant copy）
      scripts/                # 导出、实验、评测与交叉验证 CLI
      test/                   # SmoothQuant 专属测试
    requirements.txt
    README.md
```

## AWQ → AWQ-scaled FP16 HF

在 **llama.cpp 根目录**：

```bash
cd /path/to/llama.cpp
pip install -r llm_quant/requirements.txt

export HF_ENDPOINT=https://hf-mirror.com   # 如需
export HF_HOME=/path/to/hf_cache

python export_awq_hf.py \
  --model_path /path/to/Qwen2.5-7B \
  --output_dir /path/to/qwen2.5-7b-awq-g64-hf \
  --w_bit 4 --q_group_size 64
```

可选：`--save_awq_cache /path/awq.pt`。

## 接到 llama.cpp 量化

```bash
python convert_hf_to_gguf.py /path/to/qwen2.5-7b-awq-g64-hf \
  --outtype f16 --outfile model-awq-f16.gguf

./build_cuda_q64_noreex/bin/llama-quantize --pure \
  --token-embedding-type f16 --output-tensor-type f16 --leave-output-tensor \
  model-awq-f16.gguf model-Q4_1_64.gguf Q4_1_64
```

<!-- === REEX_SMOOTHQUANT BEGIN: SmoothQuant Step-1 usage, limits and code conventions === -->

## SmoothQuant → 平滑 FP16 HF → llama.cpp 量化

SmoothQuant 分两步：**第一步**校准激活、把激活的 outlier 通过等价变换"搬"进权重；**第二步**做
量化。本仓库只实现第一步，第二步直接复用 `llama-quantize`。`Q8_0_64` 是当前 W8A8 主对照，
`Q4_K_64`、`Q4_0_64` 和 `Q4_1_64` 用于验证平滑在更低权重位宽下的效果。

### Step 1：导出平滑后的 HF

```bash
cd /path/to/llama.cpp
pip install -r llm_quant/requirements.txt

python llm_quant/smoothquant/scripts/export_smoothquant_hf.py \
  --model_path /mnt/data8t/share/models/Qwen/Qwen2.5-7B-Instruct \
  --output_dir /path/to/qwen2.5-7b-smoothquant-hf \
  --alpha 0.85 \
  --n_samples 512 --seqlen 512 \
  --calib_data /path/to/calib.jsonl \
  --act_scales_cache /path/to/act_scales.pt
```

- 输出仍是**普通浮点 HF checkpoint**（safetensors + tokenizer），只是 RMSNorm 与
  q/k/v/gate/up 的权重数值被改写；`convert_hf_to_gguf.py` 无需任何改动。
- `--dtype` 默认跟随 checkpoint 的 `config.torch_dtype`，可显式覆盖。
- `--alpha` 默认按 `config.model_type` 查表（见 `llm_quant/smoothquant/defaults.py`，Qwen2/Llama = 0.85）。
- `--calib_data` 传本地 `.jsonl`（字段 `text`）；写 `pileval` 时从 `SMOOTHQUANT_PILEVAL_PATH`
  或 AWQ 已有的 `AWQ_PILEVAL_PATH` 读取。
- `--act_scales_cache` 保存校准结果，之后加 `--reuse_act_scales` 可跳过校准直接扫不同 alpha；
  缓存里带 model/calib/n_samples/seqlen 指纹，不匹配会直接报错（**alpha 不参与指纹**）。
- 显存：默认 `device_map="auto"`，7B 建议 ≥24GB；OOM 时降 `--n_samples 128` 或 `--device cpu`。

### Step 2：转 GGUF 并量化（手动，沿用现有工具）

```bash
python convert_hf_to_gguf.py /path/to/qwen2.5-7b-smoothquant-hf \
  --outtype f16 --outfile model-smooth-f16.gguf

# embed/output 保持 f16，其余类型选择使用 llama-quantize 默认策略
./build_cuda_q64/bin/llama-quantize \
  --token-embedding-type f16 --output-tensor-type f16 --leave-output-tensor \
  model-smooth-f16.gguf model-smooth-Q4_K_64.gguf Q4_K_64

# 可选：不保留 embed/output，完全使用 quantizer 默认策略
./build_cuda_q64/bin/llama-quantize \
  model-smooth-f16.gguf model-smooth-Q4_K_64-full.gguf Q4_K_64

./build_cuda_q64/bin/llama-perplexity -m model-smooth-Q4_K_64.gguf -ngl 99 \
  -f wiki.test.raw -c 512 --chunks 8
```

当前目标类型均不要求 imatrix。做多组 PPL 对比时，各组必须用同一套
embed/output 策略，否则差异来自 flag 而不是算法。

### 一键实验 + 报告

```bash
python llm_quant/smoothquant/scripts/run_smoothquant_experiment.py \
  --model_path /mnt/data8t/share/models/Qwen/Qwen2.5-7B-Instruct \
  --output_dir runs/qwen2.5-7b-sq-q4-k-64 \
  --calib_data /path/to/calib.jsonl \
  --ppl_dataset /path/to/wiki.test.raw \
  --alpha 0.85 \
  --quant-type Q4_K_64
```

产出 `runs/<exp>/experiment_report.md`（每个 stage 的完整命令、exit code、耗时、PPL 对比表）、
`manifest.json`（输入/输出 sha256 与环境信息）、`resolved_config.json` 和 `logs/*.log`。
`--dry-run` 只打印命令。宿主机可用
`llm_quant/smoothquant/scripts/run_smoothquant_experiment_docker.sh` 在 CUDA 容器内执行。

### Phase 1 实测（Qwen2.5-7B-Instruct）

> 整理后的 PPL、KL 散度结果以及当前结论与后续建议见
> [`SMOOTHQUANT_QWEN2_5_Q64_EVALUATION.md`](SMOOTHQUANT_QWEN2_5_Q64_EVALUATION.md)。

容器 `quant-gru-cuda128`（RTX 5090，torch 2.10+cu128），校准 512 条 × 512 token。
PPL 跑完整 WikiText、MBPP 和 MATH500 prompt-only 输入，KLD 在 WikiText 前 256 个 chunk
逐 token 配对评测。实测覆盖 `Q8_0_64`、`Q4_K_64`、`Q4_0_64`、`Q4_1_64` 的
direct quant 参考组及 SmoothQuant `alpha=0.85/0.80` 两组，FP16 是唯一 baseline。

当前结果表明 SmoothQuant 对量化格式敏感：Q8_0_64、Q4_K_64 和 Q4_1_64 的分布误差改善，
Q4_0_64 则明确退化。因此不应设置跨格式的统一开关或统一 alpha；完整数值、Qwen2.5
量化范围及按格式建议以主报告为准。

一键复现 KLD：
`./llm_quant/smoothquant/scripts/run_smoothquant_kld.sh <out_dir> 256 --quant-type <TYPE>`。
复用已有 FP16 基准时增加 `--reuse-base`（基准文件约 19 GB）。

与 MIT 原实现交叉验证结果：`act_scales`（197 个 key）与平滑后权重（62 个张量）
max diff 均为 **0.0**，即逐位一致。

若手头没有 pile-val，可用本地语料自建校准集
（见 `llm_quant/smoothquant/scripts/make_smoothquant_calib_jsonl.py`）：

```bash
python llm_quant/smoothquant/scripts/make_smoothquant_calib_jsonl.py \
  --source /mnt/data8t/share/datasets/model_evaluation/language_modeling/wikitext/wikitext-103-raw-v1/train-00000-of-00002.parquet \
  --output /path/to/calib.jsonl --n_docs 512
```

扫 alpha 时用 `--skip_baseline --reuse_act_scales`（把已有 `act_scales.pt` 放进新
`--output_dir` 即可），只跑平滑那一路，省掉重复的基线与校准。

### 与 MIT 原实现交叉验证

```bash
python llm_quant/smoothquant/scripts/compare_smoothquant_ref.py \
  --model_path /mnt/data8t/share/models/Qwen/Qwen2.5-7B-Instruct \
  --calib_data /path/to/calib.jsonl --alpha 0.85 --n_samples 32 \
  --ref_repo /mnt/data8t/zcx/smoothquant
```

同一 checkpoint、同一校准文件分别跑我们的实现和 `mit-han-lab/smoothquant`，比对 `act_scales`
（阈值 1e-5）与平滑后权重（阈值 1e-4）。上游 `smooth_lm` 只认 `LlamaDecoderLayer`，脚本会把该
名字重绑到当前架构的 decoder layer 上，从而在不改上游代码的前提下跑通 Qwen2。

### `smoothquant/` 来源与已知限制

| 文件 | 来源 | 改动 |
|------|------|------|
| `calibration.py` | mit-han-lab/smoothquant（MIT） | 只保留 `get_act_scales`；加本地 jsonl 路径解析、进度打印、hook 异常安全移除 |
| `smooth.py` | 同上 | `smooth_ln_fcs` 合并 LayerNorm/RMSNorm 两个版本；`smooth_lm` 改为走 `adapters.py` |
| `adapters.py` | 新增 | 按 decoder layer 类名分派并做结构校验，替代上游的 `isinstance` 硬编码 |
| `defaults.py` | 新增 | 按 `model_type` 给推荐 alpha |
| `runner.py` | 新增 | 校准 + 平滑编排，act_scales 缓存与指纹校验 |

已知限制（对比论文原版）：

1. **权重 scale 粒度**：实测 Q64 类型按 64 个输入元素分块，不是论文的 per-output-channel。
2. **激活量化粒度**：llama.cpp 运行时按每 64 元素在线量化，不是论文的 per-token。
3. **收益依赖最终格式**：Q8_0_64、Q4_K_64、Q4_1_64 改善，Q4_0_64 实测退化，不能跨格式外推。
4. **MoE 未实现**：`MixtralDecoderLayer` / `Qwen2MoeDecoderLayer` / `Qwen3MoeDecoderLayer` 等会
   显式抛 `NotImplementedError`，留待下一阶段（需与 `llama-quantize` 的张量清单对齐）。

### 代码注释规范

本目录及相关 CLI 的所有新增代码都用成对注释包裹，便于与上游 llama.cpp 内容区分：

```python
# === REEX_SMOOTHQUANT BEGIN: <一句话说明本段功能> ===
...
# === REEX_SMOOTHQUANT END ===
```

全新文件整体包一对；修改已有文件（如 `llm_quant/__init__.py`）只包裹新增行。C/C++ 用 `//`。

<!-- === REEX_SMOOTHQUANT END === -->

## `awq/` 来源

| 文件 | 来源 | 改动 |
|------|------|------|
| `auto_scale/auto_clip/pre_quant/quantizer` | llm-awq copy | 仅改 import |
| `utils/*` | llm-awq copy | 包路径 `awq.utils` |
| `scaled_act.py` | `qmodule.ScaledActivation` | 去掉 CUDA WQLinear |
| `real_quantize_*` | — | `NotImplementedError` |
