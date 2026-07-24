# llm_quant

llama.cpp 侧的**权重量化预处理**算法库。各方法分目录；CLI 在仓库根目录。

```
llama.cpp/
  export_awq_hf.py     # CLI：一步导出 AWQ-scaled FP16 HF
  llm_quant/
    awq/               # AWQ search（从 llm-awq copy）
    requirements.txt
    README.md
    # gptq/ ...        # 预留
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

## `awq/` 来源

| 文件 | 来源 | 改动 |
|------|------|------|
| `auto_scale/auto_clip/pre_quant/quantizer` | llm-awq copy | 仅改 import |
| `utils/*` | llm-awq copy | 包路径 `awq.utils` |
| `scaled_act.py` | `qmodule.ScaledActivation` | 去掉 CUDA WQLinear |
| `real_quantize_*` | — | `NotImplementedError` |
