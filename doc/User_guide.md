# AIMET RX 用户使用指南

> 本文档遵循《Reexen 工具链软件文档管理规范》第 11.1 节「工具链快速上手」基线模板编写，
> 面向首次接触 AIMET RX 的客户与内部用户，目标是让你在最短时间内跑通第一个量化流程。

| 项目 | 内容 |
|------|------|
| 文档名称 | AIMET RX 用户使用指南（快速上手） |
| 文档版本 | 2026.06-r1 |
| 适用软件版本 | 以根目录 `VERSION` 为准 |
| 文档责任人 | （待填写） |
| 最近更新 | 2026-06 |

相关文档：

- 量化配置详解：[`doc/Quant_config.md`](./Quant_config.md)
- 大模型 block-64 量化与位宽截断说明：[`llama.cpp/REEX_Q64_USAGE.md`](../llama.cpp/REEX_Q64_USAGE.md)
- 安装包总览：[`README.md`](../README.md)

---

## 1. AIMET RX 是什么

AIMET RX 是 Reexen 基于高通 AIMET（AI Model Efficiency Toolkit）定制的**模型量化工具链可重分发包**，用于把浮点模型压缩为面向 NPU / 定点硬件部署的低比特量化模型。它覆盖两类完全不同的工作流：

| 工作流 | 适用对象 | 底层引擎 | 入口 | 对应章节 |
|--------|----------|----------|------|----------|
| **普通模型量化** | CNN / RNN / KWS 等中小模型（PyTorch） | `aimet_torch` / `aimet_onnx` | `examples/quick_start.py` | [第 5 章](#5-使用方法一普通模型) |
| **大模型量化** | LLM / MoE 等大语言模型（GGUF） | 内置 REEX 版 `llama.cpp` + `aimet_llama` | `examples/llm_quick_start.py` | [第 6 章](#6-使用方法二大模型) |

选择原则：

- 模型能在单卡显存内用 PyTorch 训练 / 推理，需要 **QAT、混合精度、Power-of-2 量化、ONNX 导出** → 走「普通模型」流程。
- 模型是动辄数十 GB 的 LLM，只做**权重量化 + 跑分评估（PPL）**，希望用配置文件可复现地批量实验 → 走「大模型」流程。

---

## 2. 支持平台与版本要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（x86_64，推荐 Ubuntu 20.04 及以上） |
| Python | ≥ 3.8（已验证 3.8 ~ 3.12） |
| PyTorch | ≥ 1.13.0（普通模型流程必需） |
| CUDA | 普通模型流程：GPU 训练 / QAT 推荐；大模型流程：CUDA 工具链用于编译 `llama.cpp`（CPU 亦可运行但较慢） |
| 核心依赖 | `numpy>=1.20`、`scipy>=1.7`、`onnx>=1.11`、`onnxruntime>=1.13`、`pillow>=8.0`、`bokeh>=2.4`、`jsonschema>=3.2`（随安装包自动安装） |

> ⚠️ **外部依赖（普通模型 + Haste GRU 场景）**：`QuantGRU` 来自上游
> [CX9898/quant-gru-pytorch](https://github.com/CX9898/quant-gru-pytorch)，包含 CUDA / C++ 扩展，
> 需从源码编译安装，**本安装包不打包该模块**。仅当你的模型使用 Haste GRU 时才需要它，
> 详见 [`quant-gru-pytorch/`](../quant-gru-pytorch/)。

---

## 3. 安装

### 3.1 安装 Python 包（两类流程都需要）

使用发布的 wheel 包安装：

```bash
pip install rx_met-*.whl
```

验证安装成功：

```bash
python -c "import aimet_torch; import aimet_llama; print('aimet_rx OK')"
```

### 3.2 编译 REEX 版 llama.cpp（仅大模型流程需要）

大模型流程依赖本仓库内置的 REEX 版 `llama.cpp`（新增 block-64 量化族与定点 Psum 截断）。首次使用前需编译一次：

```bash
cd aimet_rx/llama.cpp

# GPU 构建（推荐）
cmake -B build_cuda -DGGML_CUDA=ON
cmake --build build_cuda --config Release -j

# 或 CPU 构建
cmake -B build_cpu
cmake --build build_cpu --config Release -j
```

验证编译成功：

```bash
./build_cuda/bin/llama-cli --help
./build_cuda/bin/llama-quantize --help
```

> 编译产物默认位于 `llama.cpp/build_cuda/bin/`，大模型流程的 JSON 配置默认指向该路径。

---

## 4. 环境变量与路径配置

| 变量 / 路径 | 作用 | 适用流程 |
|-------------|------|----------|
| `CUDA_VISIBLE_DEVICES` | 指定可见 GPU（多卡 LLM 跑分常用，如 `0,1,2,3`） | 大模型 |
| `REEX_Q64_PSUM_BITS` | 定点 Psum 截断到 B 位（`<=0` 或不设 = 关闭）；改值无需重新量化或重编译 | 大模型 |
| `LD_LIBRARY_PATH` | 指向 `llama.cpp/build_cuda/bin`，确保动态库可加载（JSON 中由 `binaries.ld_library_path` 自动注入） | 大模型 |
| 运行目录 | 普通模型 demo 须 `cd aimet_rx/examples/` 后运行；大模型 demo 须 `cd aimet_rx/` 后运行 | 两者 |

---

## 5. 使用方法（一）：普通模型

本章基于 `examples/quick_start.py`（SpeechCommands + MRNN 关键词识别），演示 AIMET v2 的完整量化流程。该脚本的 `main()` 内联了所有标准 AIMET API，可直接照抄到自己的工程。

### 5.1 完整流程总览

```
FP 训练 → prepare_model → QuantizationSimModel + 混合精度位宽
  → compute_encodings 校准 (PTQ)
  → apply_power_of_2_workflow (NPU 友好的 Po2 量化)
  → freeze_quantizer_parameters + QAT 微调
  → 保存 state_dict + ONNX + .encodings 三件套
  → 重建 sim → load_state_dict → load_quantizer_encodings → 精度对比
```

### 5.2 运行示例脚本

```bash
cd aimet_rx/examples
python quick_start.py
```

脚本会依次打印 8 个步骤的耗时与精度汇总（浮点 / PTQ / Po2 / QAT / 重载）。

### 5.3 关键步骤代码（照抄模板）

**步骤 1：准备模型并创建量化模拟器（基础配置）**

```python
import aimet_torch.v2 as aimet
from aimet_torch import model_preparer
from aimet_torch.v2 import quantsim
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth

# prepare_model 把 forward 中的 functional 调用重构成可被 sim 抓取的 nn.Module
prepared_model = model_preparer.prepare_model(model)

sim = quantsim.QuantizationSimModel(
    prepared_model,
    dummy_input=dummy_input,
    quant_scheme="percentile",                 # min_max / tf / tf_enhanced / percentile
    config_file="config/mrnn_quantsim_config_custom_mixed_precision_v2.json",  # 基础配置
    default_output_bw=8,
    default_param_bw=8,
)
sim.set_percentile_value(99.99)                 # 仅 percentile 方案生效
```

**步骤 2：应用混合精度位宽（阶段配置，可选）**

```python
apply_mixed_precision_bitwidth(
    sim.model,
    config_file="config/quick_start_full_quant.json",  # 阶段配置：逐层指定位宽 / 对称性
    verbose=True,
)
```

**步骤 3：校准（PTQ）**

```python
with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
    for idx, (x, _) in enumerate(calib_loader):
        if idx >= 100:        # 校准批次数
            break
        sim.model(x.to(device))
```

**步骤 4：Power-of-2 量化（NPU 友好，可选）**

```python
from aimet_torch.utils_rx import apply_power_of_2_workflow

apply_power_of_2_workflow(
    sim.model, method="round", tolerance=0.02,
    align_bias_scale=True, verbose=True,
)
```

**步骤 5：QAT 微调（可选）**

```python
from aimet_torch.utils_rx import freeze_quantizer_parameters, set_train_mode_freeze_bn

freeze_quantizer_parameters(sim.model, verbose=True, freeze_bn_affine=True)
# 之后用普通 PyTorch 训练循环微调，但用 set_train_mode_freeze_bn(sim.model) 替代 model.train()
```

**步骤 6：导出部署产物（ONNX + encodings）**

```python
from export_onnx_and_encodings.export_onnx_json import export_onnx_json

sim.model.cpu().eval()      # 导出要求模型在 CPU
onnx_path, enc_path = export_onnx_json(
    sim, export_dir="output", filename_prefix="my_model",
    dummy_input_shape=(1, 16000, 1), opset=18,
)
torch.save({"model": sim.model.state_dict()}, "output/my_model_qat.pth")
```

**步骤 7：重新加载验证（生产侧复现）**

```python
from aimet_torch.staged_quantization_utils import load_quantizer_encodings

# 用与训练时完全一致的参数重建 sim 后：
fresh_sim.model.load_state_dict(state, strict=False)   # 1. 加载权重
load_quantizer_encodings(fresh_sim.model, load_path=enc_path)  # 2. 加载量化参数
```

> ⚠️ **一致性要点**：重建 `sim` 时 `prepare_model` 的 `stateless_modules_to_preserve`、
> `config_file`、`default_*_bw`、量化方案必须与训练侧**完全一致**，否则 quantizer 名称错位会导致加载失败。

### 5.4 两类配置文件的区别

普通模型采用**双层配置**，详细字段说明见 [`Quant_config.md`](./Quant_config.md)：

| 配置 | 传入时机 | 控制内容 | 命名规则 |
|------|----------|----------|----------|
| **基础配置（JSON 1）** | 创建 `QuantizationSimModel` 时 | 是否量化、per-channel、对称性 | ONNX 算子名（如 `Conv`、`Gemm`、`Add`） |
| **阶段配置（JSON 2）** | `apply_mixed_precision_bitwidth` 时 | 各层位宽、输入/输出对称性 | AIMET 量化后类名（如 `QuantizedConv2d`、`QuantizedLinear`） |

> 进阶：当精度一次性全量化会崩溃时，可使用**多阶段量化**（逐步启用 Conv → GRU → BN…），
> 以及 **Haste GRU 专用配置 / 校准**，完整流程见 [`Quant_config.md`](./Quant_config.md) 第「多阶段量化配置」节。

---

## 6. 使用方法（二）：大模型

本章基于 `aimet_llama` 模块 + 内置 REEX 版 `llama.cpp`，用**一个 JSON 文件**驱动「imatrix 校准 → 量化 → 跑分（PPL）」全流程，结果可复现、可审计、可版本管理。底层 block-64 量化细节见 [`REEX_Q64_USAGE.md`](../llama.cpp/REEX_Q64_USAGE.md)。

### 6.1 前置准备：HF 模型转 FP16 GGUF

`llama-quantize` 的输入必须是 GGUF，因此先把 Hugging Face 模型转为 FP16 GGUF：

```bash
cd aimet_rx/llama.cpp
python convert_hf_to_gguf.py /path/to/hf_model \
  --outtype f16 --outfile /path/to/model-f16.gguf
```

完整链路为：`HF → model-f16.gguf → model-量化.gguf`。

### 6.2 编写 JSON 配置

以 REEX block-64 量化为例（`examples/config/qwen3_reex_q4_k_64.json`）：

```json
{
  "schema_version": "1.0",
  "experiment": {
    "name": "qwen3_reex_q4_k_64",
    "output_dir": "./runs/qwen3_reex_q4_k_64"
  },
  "binaries": {
    "llama_quantize":   "./llama.cpp/build_cuda/bin/llama-quantize",
    "llama_imatrix":    "./llama.cpp/build_cuda/bin/llama-imatrix",
    "llama_perplexity": "./llama.cpp/build_cuda/bin/llama-perplexity",
    "ld_library_path":  "./llama.cpp/build_cuda/bin"
  },
  "model":       { "gguf_fp16_path": "/path/to/model-f16.gguf" },
  "calibration": { "enabled": false },
  "quantization": {
    "default_type":     "Q4_K_64",
    "output_quantized": "model-Q4_K_64.gguf",
    "use_imatrix":      false,
    "n_threads":        16
  },
  "evaluation": {
    "enabled": true,
    "perplexity": {
      "dataset_file":   "/path/to/wikitext-2-raw/wiki.test.raw",
      "context_length": 512,
      "n_gpu_layers":   99,
      "chunks":         8,
      "reex_psum_bits": 8
    }
  }
}
```

关键字段：

- `quantization.default_type`：量化类型，可填标准类型（`Q4_K_M` 等）或 REEX block-64 类型（见 6.4）。
- `evaluation.perplexity.reex_psum_bits`：跑分时把定点 Psum 截断到 B 位；`null` 或 `<=0` 关闭。
- 完整字段与 CLI 映射见 [`aimet_llama/README.md`](../aimet_llama/README.md)。

### 6.3 运行流程

```bash
cd aimet_rx

# 1) 先预览将要执行的 CLI（不产生任何副作用）
python examples/llm_quick_start.py examples/config/qwen3_reex_q4_k_64.json --dry-run

# 2) 端到端运行（imatrix → quantize → perplexity）
python examples/llm_quick_start.py examples/config/qwen3_reex_q4_k_64.json
```

每次运行会在 `runs/<experiment_name>/` 下生成可复现产物：`resolved_config.json`（解析后的完整配置）、`manifest.json`（输入/二进制/产物的 SHA-256 与耗时）、各阶段日志、量化后的 GGUF、以及人类可读的 `experiment_report.md`。

### 6.4 REEX block-64 量化类型

内置 `llama.cpp` 在标准 ggml 类型之外，新增了全 block=64 对齐的量化族，可像普通类型一样填入 `default_type` / `output_tensor_type` / `tensor_type_overrides`：

```
Q4_0_64  Q5_0_64  Q8_0_64  Q8_1_64        （Legacy，每 64 元素一个 scale）
Q4_1_64  Q5_1_64
Q2_K_64  Q3_K_64  Q4_K_64  Q5_K_64  Q6_K_64    （K-quant，超块 256 / 子块 64）
Q2_K_64S Q4_K_64S Q5_K_64S                      （K-quant 对称版）
```

各类型的位宽、对称性、量化/反量化公式见 [`REEX_Q64_USAGE.md`](../llama.cpp/REEX_Q64_USAGE.md) 第 1 节。

### 6.5 定点 Psum 位宽截断（硬件保真）

REEX 运行时可模拟有限位宽硬件，把 64 元素整数 Psum 截断到 B 位，**CPU 与 GPU 数值对齐**。除在 JSON 的 `reex_psum_bits` 配置外，也可直接用环境变量驱动原生命令：

```bash
# 不截断（默认）
./build_cuda/bin/llama-perplexity -m model-Q4_K_64.gguf -ngl 99 -f wiki.test.raw -c 512 --chunks 8

# 截断到 8 位
REEX_Q64_PSUM_BITS=8 ./build_cuda/bin/llama-perplexity -m model-Q4_K_64.gguf -ngl 99 -f wiki.test.raw -c 512 --chunks 8
```

要点：

- `-ngl 99` 走 GPU，`-ngl 0` 走 CPU，二者数值对齐（截断也对齐）。
- 截断粒度统一为 **64 元素整数 Psum**；只截断整数 Psum，min 修正项不截断。
- 改 `B` 无需重新量化或重编译，单个量化 GGUF 即可扫不同位宽。
- 多卡跑分用 `CUDA_VISIBLE_DEVICES=0,1,2,3` + `-ngl 99`；**勿用 `--split-mode row`**，它会破坏 64 元素 Psum 对齐。

### 6.6 实测参考精度

`qwen2.5-0.5b`（wikitext-2，`-c 512 --chunks 8`，f16 基线 PPL = 15.55）部分结果：

| 类型 | PPL | 类型 | PPL |
|------|-----|------|-----|
| Q8_0_64 | 15.70 | Q5_K_64 | 16.19 |
| Q6_K_64 | 15.72 | Q4_K_64 | 18.29 |
| Q5_0_64 | 16.63 | Q2_K_64 | 25.24 |

经验：8/6/5-bit 近无损；4-bit 退化约 6%；3-bit 勉强可用；2-bit 在 MoE 上易崩溃（需 imatrix）。更多实测见 [`REEX_Q64_USAGE.md`](../llama.cpp/REEX_Q64_USAGE.md) 第 3、4 节。

---

## 7. 常见问题（FAQ）

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `import quant_gru` 失败 | 未安装外部 `QuantGRU` 扩展 | 仅 Haste GRU 模型需要，按 [`quant-gru-pytorch/`](../quant-gru-pytorch/) 从源码编译安装 |
| 重载后精度与训练侧差异大 | 重建 sim 的配置/`stateless_modules_to_preserve` 与训练时不一致 | 保证两侧完全一致（见 5.3 一致性要点） |
| `load_state_dict` 报缺失/多余键 | 量化模型含 quantizer 参数 | 使用 `load_state_dict(..., strict=False)` |
| `llama-quantize` 找不到 block-64 类型 | 用了上游 llama.cpp 而非内置 REEX 版 | 必须使用 `aimet_rx/llama.cpp` 编译产物 |
| GPU 与 CPU 跑分结果不一致 | 误用 `--split-mode row` | 改用默认 layer split，多卡仍数值对齐 |
| 动态库加载失败 | `LD_LIBRARY_PATH` 未指向 build 目录 | JSON 中设 `binaries.ld_library_path`；手动跑命令时先 `export LD_LIBRARY_PATH=...` |

---

## 8. 卸载与回滚

- **卸载 Python 包**：

```bash
pip uninstall rx-met
```

- **回滚到旧版本**：安装对应版本的 wheel 即可，例如 `pip install rx_met-<旧版本>-py3-none-any.whl`。
- **清理大模型流程产物**：删除 `aimet_rx/runs/` 下对应实验目录与 `llama.cpp/build_cuda/` 编译目录即可（删除 build 目录后需重新编译）。

---

## 9. 变更记录

| 版本 | 日期 | 变更项 | 责任人 |
|------|------|--------|--------|
| 2026.06-r1 | 2026-06 | 首次发布：普通模型与大模型两条使用路径快速上手 | （待填写） |
