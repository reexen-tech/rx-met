# rx-met @VERSION@ 使用说明

本发布包包含：

```text
rx-met-@VERSION@/
├── README.md
├── SHA256SUMS
├── rx-met-@VERSION@-image.tar
└── examples/
```

## 环境要求

- Linux x86_64
- 满足 CUDA 12.8 运行要求的 NVIDIA Driver
- Docker
- NVIDIA Container Toolkit

## 1. 导入镜像并初始化环境

```bash
mkdir -p workspace && cd workspace
mv /path/rx-met-@VERSION@-release.tar.gz .
tar -xzf rx-met-@VERSION@-release.tar.gz
sha256sum -c SHA256SUMS
docker load -i rx-met-@VERSION@-image.tar
docker run --rm --gpus all rx-met:@VERSION@ rx-met --help
WORKSPACE="$(pwd)"
RELEASE_DIR="$WORKSPACE/rx-met-@VERSION@"
mkdir -p "$WORKSPACE/runs"
```

## 2. 大模型量化

### 2.1 设置模型目录

模型可以统一放在宿主机的任意目录，例如：

```text
/data/models/
├── Qwen3.5-35B-A3B/
├── Qwen3-32B/
└── ...
```

设置模型根目录。将 `/data/models` 替换为实际路径：

```bash
MODELS_DIR="$(cd /data/models && pwd)"
```

Docker 启动后，`$MODELS_DIR` 会映射为容器内的 `/models`。

### 2.2 配置 JSON

编辑 `$RELEASE_DIR/examples/config/llm_quant.json`：

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
- `output`：填写 `/workspace/runs/` 下的输出目录, 不填默认在当前目录。
- `eval.dataset`：填写 `/workspace/` 下的评测文件路径, 不需要 PPL 评测时，可以删除整个 `eval` 字段。
- JSON 中填写容器路径，不填写宿主机绝对路径。

> 配置字段说明见 `examples/config/README.md`。

### 2.3 启动并进入容器

```bash
test -r "$WORKSPACE/datasets/evaluation.txt"

docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e USER="$(id -un)" \
  -v "$WORKSPACE:/workspace" \
  -v "$MODELS_DIR:/models:ro" \
  -w /workspace \
  rx-met:@VERSION@ \
  bash
```

工作目录和模型目录已分别映射为容器内的 `/workspace` 和 `/models`。
`/workspace` 可写，模型目录 `/models` 保持只读。

### 2.4 在容器内使用

先检查配置和执行计划：

```bash
rx-met --dry-run \
  "/workspace/rx-met-@VERSION@/examples/config/my_config.json"
```

确认无误后运行：

```bash
rx-met "/workspace/rx-met-@VERSION@/examples/config/my_config.json"
```

## 3. 小模型示例

### 3.1 设置数据目录

设置宿主机上的 `speech_commands_v0.02` 目录。将路径替换为实际位置：

```bash
HOST_SPEECH_COMMANDS="$(
  cd /path/to/speech_commands_v0.02 && pwd
)"
test -r "$HOST_SPEECH_COMMANDS/validation_list.txt"
test -r "$HOST_SPEECH_COMMANDS/testing_list.txt"
```

Docker 启动后，`$HOST_SPEECH_COMMANDS` 会映射为容器内的
`/data/speech_commands`。

### 3.2 启动并进入容器

```bash
SMALL_MODEL_WORK="$WORKSPACE/small-model-work"
mkdir -p "$SMALL_MODEL_WORK"
cp -an "$RELEASE_DIR/examples/." "$SMALL_MODEL_WORK/"

docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e USER="$(id -un)" \
  -e RX_MET_SPEECH_COMMANDS_ROOT=/data/speech_commands \
  -v "$SMALL_MODEL_WORK:/work" \
  -v "$HOST_SPEECH_COMMANDS:/data/speech_commands:ro" \
  -w /work \
  rx-met:@VERSION@ \
  bash
```

工作目录和数据目录已分别映射为容器内的 `/work` 和
`/data/speech_commands`。`/work` 可写，数据目录保持只读。

### 3.3 在容器内使用

```bash
python3 quick_start.py
```

脚本会写入宿主机的 `small-model-work/model_fp.pth` 和
`small-model-work/output/`。
