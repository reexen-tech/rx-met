# rx-met @VERSION@ 使用说明

本发布包包含：

```text
rx-met-@VERSION@/
├── README.md
├── SHA256SUMS
├── env.example
├── scripts/
│   ├── setup.sh
│   └── rx-met-shell.sh
├── rx-met-@VERSION@-image.tar
└── examples/
```

## 环境要求

- Linux x86_64
- 满足 CUDA 12.8 运行要求的 NVIDIA Driver
- Docker
- NVIDIA Container Toolkit

## 1. 首次初始化

在解压后的发布包目录执行：

```bash
cd rx-met-@VERSION@/
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
├── speech_commands_v0.02/
└── ...
```

> 启动后映射为：`$MODELS_DIR` → `/models`，`$DATASETS_DIR` → `/datasets`，`workspace` → `/workspace`。
> 不做 PPL、也不跑小模型数据时，可将 `DATASETS_DIR` 留空。

## 2. 启动容器

```bash
./scripts/rx-met-shell.sh
```

## 3. 大模型示例

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

## 4. 小模型示例

将 `speech_commands_v0.02` 放在宿主机 `$DATASETS_DIR` 下，容器内路径为 `/datasets/speech_commands_v0.02`。

在容器内：

```bash
cd /workspace/examples
python3 quick_start.py
```

脚本会写入宿主机的 `workspace/examples/model_fp.pth` 和
`workspace/examples/output/`。
