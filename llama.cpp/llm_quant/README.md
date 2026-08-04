# llm_quant

llama.cpp 侧的**权重量化预处理**算法库。各方法分目录；CLI 在仓库根目录。

```
llama.cpp/
  export_awq_hf.py     # CLI：一步导出 AWQ-scaled FP16 HF
  export_gptq_hf.py    # CLI：Qwen2.5/Qwen3.5 MoE W4A8/G64 GPTQ
  llm_quant/
    awq/               # AWQ search（从 llm-awq copy）
    gptq/              # Q4_0_64 / Q8_0_64 GPTQ + layer sequential
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

## GPTQ Q4_0_64 / Q8_0_64

支持 W4A8（`Q4_0_64 × Q8_0_64`）和 W8A8
（`Q8_0_64 × Q8_0_64`），两者均使用 G64、PileVal 128×512、
damping 0.01 和 lazy block 128。暂不支持 act-order、static-groups、
true-sequential 或 G32。`--bits` 默认为 4；W4/W8 共用 GPTQ 数学核，
但 Q8 按 C `roundf` 使用 ties-away-from-zero。

```bash
python export_gptq_hf.py \
  --model_path /path/to/Qwen2.5-0.5B \
  --output_dir /path/to/qwen2.5-0.5b-gptq-q8_0_64-hf \
  --bits 8

python convert_hf_to_gguf.py \
  /path/to/qwen2.5-0.5b-gptq-q8_0_64-hf \
  --outtype f16 \
  --outfile model-gptq-Q8_0_64.gguf

python validate_gptq_gguf.py \
  --sidecar /path/to/qwen2.5-0.5b-gptq-q8_0_64-hf/gptq_q8_0_64.pt \
  --gguf model-gptq-Q8_0_64.gguf
```

导出目录同时包含：

- fake-quant FP16 HF 权重，便于检查；
- `gptq_q4_0_64.pt` 或 `gptq_q8_0_64.pt`，保存 GPTQ 决定的精确
  codes/scales/packed blocks；
- `gptq_export_meta.json`，保存固定配置、逐层统计和 fixed-point 结果。

`convert_hf_to_gguf.py` 检测到唯一的 sidecar 后会直接写对应 Q64 blocks；
同目录同时存在 Q4 和 Q8 manifest 会报错。不能再对结果运行
`llama-quantize Q4_0_64/Q8_0_64`，否则会进行第二次 RTN 并可能改变 GPTQ 码字。
sidecar 加载、匹配、Qwen3.5 重排和 packed 写入集中在
`llm_quant/gptq/gguf_adapter.py`；官方 converter 只保留带 `REEX` 标记的
load、ftype、emit、F16 保留和 finish 接入点。

Qwen3.5-35B-A3B 支持完整文本塔（包括 routed experts），视觉塔保持 BF16。
所有模型统一使用按层分片的 packed-only sidecar v3；旧 v1/v2 不再兼容。
目标层由 `llm_quant/targets.py` 显式定义，零样本或 Cholesky 失败会直接报错：

```bash
CUDA_VISIBLE_DEVICES=5 python export_gptq_hf.py \
  --model_path /mnt/data2/models/Qwen3.5-35B-A3B \
  --output_dir /mnt/data2/results/qwen3.5-35b-a3b-gptq \
  --bits 8 \
  --expert-hessian-weighting route_squared \
  --resume
```

`--expert-hessian-weighting none` 用于生成 unweighted/full 对照。v3 resume
会校验 source/config/targets，已完成层只恢复 packed 权重并前向传播，不会重新量化。

完整操作和资源说明见
`docs/2026-07-29_Qwen3.5-35B-A3B_GPTQ适配与验证.md`。
评测必须用 `build_cuda_q64_noreex`（Q64 无 LUT）。在该构建下 WikiText
`-c 512 --chunks 8`：F16 6.4599、RTN 6.6723、GPTQ weighted 6.7311
（约比 RTN 高 0.9%）。此前在带 LUT 的 `build_cuda` 上测得的 ≈10.85 已作废。

完整的 F16/RTN/GPTQ PPL、固定 prompt 生成和 CPU/CUDA 对比：

```bash
scripts/run_gptq_q4_0_64_e2e.sh \
  /path/to/Qwen2.5-0.5B \
  /path/to/wiki.test.raw \
  results/2026-07-28_GPTQ_Q4_0_64
```

## `awq/` 来源

| 文件 | 来源 | 改动 |
|------|------|------|
| `auto_scale/auto_clip/pre_quant/quantizer` | llm-awq copy | 仅改 import |
| `utils/*` | llm-awq copy | 包路径 `awq.utils` |
| `scaled_act.py` | `qmodule.ScaledActivation` | 去掉 CUDA WQLinear |
| `real_quantize_*` | — | `NotImplementedError` |
