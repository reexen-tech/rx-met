# rx-met @VERSION@ 离线使用说明

本发布包包含：

```text
rx-met-@VERSION@/
├── README.md
├── SHA256SUMS
├── rx-met-@VERSION@-image.tar
└── examples/
```

## 1. 环境要求

- Linux x86_64
- 满足 CUDA 12.8 运行要求的 NVIDIA Driver
- Docker
- NVIDIA Container Toolkit

宿主机不需要安装 CUDA Toolkit、Python、PyTorch 或 rx-met。

## 2. 校验并导入镜像

在解压后的 `rx-met-@VERSION@` 目录执行：

```bash
sha256sum -c SHA256SUMS
docker load -i rx-met-@VERSION@-image.tar
docker run --rm --gpus all rx-met:@VERSION@ rx-met --help
```

## 3. 大模型量化

### 3.1 设置工作目录

假设宿主机上有一个工作目录，里面放置解压后的 rx-met 工具包、输出目录和可选的评测数据：

```text
workspace/
├── rx-met-@VERSION@/
├── runs/
└── datasets/
    └── evaluation.txt
```

在终端进入 `workspace`，设置环境变量并创建输出目录。将第一行替换为实际路径：

```bash
cd /path/to/workspace
WORKSPACE="$(pwd)"
RELEASE_DIR="$WORKSPACE/rx-met-@VERSION@"

mkdir -p "$WORKSPACE/runs"
test -d "$RELEASE_DIR/examples"
```

### 3.2 设置模型目录

模型可以统一放在宿主机的任意目录，例如：

```text
/data/models/
├── Qwen3-30B-A3B-Instruct-2507/
├── Qwen3-32B/
└── ...
```

设置模型根目录。将 `/data/models` 替换为实际路径：

```bash
MODELS_DIR="$(cd /data/models && pwd)"
test -d "$MODELS_DIR/Qwen3-30B-A3B-Instruct-2507"
```

Docker 启动后，`$MODELS_DIR` 会映射为容器内的 `/models`。

### 3.3 配置 JSON

复制一份配置：

```bash
cp \
  "$RELEASE_DIR/examples/config/qwen3_reex_q4_k_64.json" \
  "$RELEASE_DIR/examples/config/my_config.json"
```

编辑 `$RELEASE_DIR/examples/config/my_config.json`：

```json
{
  "model": "/models/Qwen3-30B-A3B-Instruct-2507",
  "quant": "Q4_K_64",
  "output": "/workspace/runs/qwen3-30b-a3b-instruct-2507-q4-k-64",
  "eval": {
    "dataset": "/workspace/datasets/evaluation.txt"
  }
}
```

字段填写规则：

- `model`：填写 `/models/` 下的模型目录，例如
  `/models/Qwen3-30B-A3B-Instruct-2507`。
- `quant`：填写目标量化类型，例如 `Q4_K_64`、`Q4_0`。
- `output`：填写 `/workspace/runs/` 下的输出目录。
- `eval.dataset`：填写 `/workspace/` 下的评测文件路径，例如
  `/workspace/datasets/evaluation.txt`。
- 不需要 PPL 评测时，可以删除整个 `eval` 字段。
- JSON 中填写容器路径，不填写宿主机绝对路径。

### 3.4 启动并进入容器

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

### 3.5 在容器内使用

先检查配置和执行计划：

```bash
rx-met --dry-run \
  "/workspace/rx-met-@VERSION@/examples/config/my_config.json"
```

确认无误后运行：

```bash
rx-met "/workspace/rx-met-@VERSION@/examples/config/my_config.json"
```

结束后退出容器：

```bash
exit
```

配置字段说明见 `examples/config/README.md`。

## 4. 小模型示例

本章可以单独执行，不依赖第 3 章中设置的环境变量。先进入
`workspace`，再复制一份可写的 Example。将第一行替换为实际路径：

```bash
cd /path/to/workspace
WORKSPACE="$(pwd)"
RELEASE_DIR="$WORKSPACE/rx-met-@VERSION@"
EXAMPLE_WORK="$WORKSPACE/examples-work"

test -d "$RELEASE_DIR/examples"
test -d "$EXAMPLE_WORK" || cp -a "$RELEASE_DIR/examples" "$EXAMPLE_WORK"
```

设置宿主机上的 `speech_commands_v0.02` 目录。将路径替换为实际位置：

```bash
HOST_SPEECH_COMMANDS="$(
  cd /path/to/speech_commands_v0.02 && pwd
)"
test -r "$HOST_SPEECH_COMMANDS/validation_list.txt"
test -r "$HOST_SPEECH_COMMANDS/testing_list.txt"
```

启动并进入容器：

```bash
docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e USER="$(id -un)" \
  -e RX_MET_SPEECH_COMMANDS_ROOT=/data/speech_commands \
  -v "$EXAMPLE_WORK:/examples" \
  -v "$HOST_SPEECH_COMMANDS:/data/speech_commands:ro" \
  -w /examples \
  rx-met:@VERSION@ \
  bash
```

进入容器后运行：

```bash
python3 quick_start.py
```

路径对应关系：

```text
宿主机 $HOST_SPEECH_COMMANDS  →  容器 /data/speech_commands
宿主机 $EXAMPLE_WORK         →  容器 /examples
```

重新运行时无需再次复制 `examples-work`。脚本会写入
`examples-work/model_fp.pth` 和 `examples-work/output/`。

## 5. 常用检查

```bash
docker run --rm --gpus all \
  rx-met:@VERSION@ \
  python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 输出 True 则正常

docker run --rm rx-met:@VERSION@ \
  python3 -c "import aimet_torch, quant_gru, rx_met_llm; print('imports OK')"

# 输出 imports OK 则正常
```
