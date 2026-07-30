# rx-met @VERSION@-cpu 使用说明（纯 CPU）

本发布包为 **CPU 版本**：不需要 NVIDIA Driver / GPU。大模型量化与 PPL 可在 CPU 上运行（较慢）。

**不可用**：QuantGRU 与 `examples/quick_start.py` 小模型流程（依赖 CUDA）。小模型请使用 GPU 发布包。

本发布包包含：

```text
rx-met-@VERSION@-cpu/
├── README.md
├── SHA256SUMS
├── env.example
├── scripts/
│   ├── setup.sh
│   └── rx-met-shell.sh
├── rx-met-@VERSION@-cpu-image.tar
└── examples/
```

## 环境要求

- Linux x86_64
- Docker（无需 NVIDIA Container Toolkit）

## 1. 首次初始化

在解压后的发布包目录执行：

```bash
cd rx-met-@VERSION@-cpu/
./scripts/setup.sh
```

脚本会：校验包、导入镜像、创建旁路 `workspace/`、拷贝 `examples/`、生成 `workspace/.env`。

编辑 `../workspace/.env`，按实际路径填写：

```bash
MODELS_DIR=/data/models
DATASETS_DIR=/data/datasets
```

宿主机目录建议组织为：

```text
/data/models/
├── Qwen3.5-35B-A3B/
└── ...

/data/datasets/
├── evaluation.txt
└── ...
```

> 启动后映射为：`$MODELS_DIR` → `/models`，`$DATASETS_DIR` → `/datasets`，`workspace` → `/workspace`。  
> 不做 PPL 时可将 `DATASETS_DIR` 留空。

## 2. 启动容器

```bash
./scripts/rx-met-shell.sh
```

进入后工作目录为 `/workspace`。也可直接跑命令：

```bash
./scripts/rx-met-shell.sh -- rx-met --help
```

## 3. 大模型量化

编辑 `$WORKSPACE/examples/config/llm_quant.json`（`$WORKSPACE` 即旁路 `workspace/`）：

```json
{
  "model": "/models/Qwen/Qwen3.5-35B-A3B",
  "quant": "Q4_0_64",
  "eval": {
    "dataset": "/datasets/evaluation.txt"
  },
  "hw_export": true,
  "output": "/workspace/runs/Qwen3.5-35B-A3B-q4-0-64"
}
```

字段填写规则：

- `model`：填写 `/models/` 下的模型目录。
- `quant`：填写目标量化类型，例如 `Q4_K_64`、`Q4_0`。
- `output`：填写 `/workspace/runs/` 下的输出目录；不填时默认写入当前工作目录下的 `runs/`。
- `eval.dataset`：填写 `/datasets/` 下的评测文件；不需要 PPL 时可删除整个 `eval` 字段。
- JSON 中只填写容器路径，不填写宿主机绝对路径。

> 配置字段说明见 `examples/config/README.md`。  
> CPU 上 imatrix / PPL 会明显更慢；如需加快可减小 `calib.chunks` / `eval.chunks`。

在容器内：

```bash
rx-met --dry-run /workspace/examples/config/llm_quant.json
rx-met /workspace/examples/config/llm_quant.json
```

或在宿主机一条命令执行：

```bash
./scripts/rx-met-shell.sh -- \
  rx-met /workspace/examples/config/llm_quant.json
```
