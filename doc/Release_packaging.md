# rx-met 打包 / 发布 / 使用

| 阶段           | 谁关心                      | 本文章节 |
| -------------- | --------------------------- | -------- |
| **打包** | 研发 / CI，产出构建制品     | §1      |
| **发布** | 研发 / CI，构建并导出镜像   | §2      |
| **使用** | 客户，`docker run` 跑量化 | §3      |

**交付策略**：单镜像全栈（大模型 `rx-met` + 小模型 PyTorch/aimet），外加可直接查看和修改的 `examples/` 与客户 `README.md`。三者组成一个离线 Release Bundle。**客户 Release 不含 wheel / native**（仅在 CI 构建镜像时使用）。

本文中的 `${VERSION}` 均取自仓库根目录 `VERSION`：

```bash
VERSION="$(tr -d '[:space:]' < VERSION)"
```

---

## 1. 打包（CI 构建制品）

Docker Builder 阶段在隔离的 CUDA 12.8 devel 环境中产出以下 **镜像构建制品**；它们只在多阶段构建中传给 Runtime 阶段，**不随客户 Release 单独下发**。

```
ci-artifacts/rx-met-${VERSION}/
├── wheels/
│   ├── rx_met-${VERSION}-py3-none-any.whl  # 含 rx_met_llm、aimet_torch、rx-met CLI
│   └── quant_gru-1.0.11-cp310-cp310-linux_x86_64.whl
├── native/
│   └── rx-met-native-${VERSION}-cuda12.8-linux_x86_64.tar.gz
└── docker/
    └── Dockerfile
```

仓库内脚本（本地 / CI 调用）：

| 脚本 | 作用 |
| ---- | ---- |
| `scripts/package_native.sh` | Builder 内编译 llama.cpp 并打 native tar |
| `scripts/build_wheel.sh` | Builder 内构建 `rx-met` wheel |
| `scripts/build_quant_gru_wheel.sh` | Builder 内构建 Python 3.10/CUDA `quant_gru` wheel |
| `scripts/release_build.sh` | 多阶段 `docker build` → 验收 → `docker save` |
| `scripts/verify_image.sh` | §3.6 镜像 smoke test |

正式构建环境统一为 CUDA 12.8 Builder 镜像：由其提供 Toolkit/nvcc、CMake、Python 3.10 和 PyTorch 2.9.1+cu128。构建机只需 Docker，客户运行机不需要这些编译工具。

### 1.1 各制品含义

| 路径                  | 类型            | 用途                                          |
| --------------------- | --------------- | --------------------------------------------- |
| `wheels/*.whl`      | Python 包（`rx-met`、`quant_gru`） | Dockerfile 中 `pip install` |
| `native/*.tar.gz`   | 预编译 C++/CUDA | Dockerfile 中解压到`/opt/rx-met`            |
| `examples/`         | 客户示例        | 不进入镜像；打包到外层 Release Bundle       |
| `docker/Dockerfile` | 镜像定义        | 构建运行时镜像                                |

### 1.2 `native/*.tar.gz` 内部

```
rx-met-native-${VERSION}-cuda12.8-linux_x86_64.tar.gz
├── bin/     # llama-quantize, llama-imatrix, convert_hf_to_gguf.py, ...
└── lib/     # libggml*.so
```

**CI 打 native 包**（也可用 `scripts/package_native.sh`）：

```bash
cmake -S llama.cpp -B build_release \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_USE_REEX_Q64=ON \
  -DGGML_USE_REEX=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_TOOLS_INSTALL=ON
cmake --build build_release -j
cmake --install build_release --prefix /tmp/rx-met-native-stage

tar -C /tmp/rx-met-native-stage -czf rx-met-native-${VERSION}-cuda12.8-linux_x86_64.tar.gz bin lib
```

### 1.3 Dockerfile（运行时镜像）

见仓库 `docker/Dockerfile`。这是多阶段 Dockerfile：

1. `builder`：基于 CUDA 12.8 devel，编译 llama.cpp、quant_gru 并构建 wheels；
2. `runtime`：基于 CUDA 12.8 runtime，只复制 Builder 产物，不包含 nvcc、CMake、编译器、源码和中间文件。

build context 为仓库根目录，由 `.dockerignore` 排除已有 build、模型、数据集和实验输出。

基于 **CUDA 12.8 runtime**（含 cuBLAS 等 user-space 库），**不是** devel / Toolkit。PyTorch、aimet、quant_gru 和 native 在镜像构建时装入；`examples/` 不进入镜像。运行用户固定为非 root UID/GID `10001`，默认工作目录为 `/work`。

镜像内固定 **`RX_MET_HOME=/opt/rx-met`**；`rx_met_llm` 从此处解析 `bin/`、`lib/`。

---

## 2. 发布（客户制品）

客户 Release 是一个外层压缩包（及其校验文件）：

```
rx-met-${VERSION}-release.tar.gz
rx-met-${VERSION}-release.tar.gz.sha256
```

解压后：

```text
rx-met-${VERSION}/
├── README.md
├── SHA256SUMS
├── rx-met-${VERSION}-image.tar
└── examples/
```

**体积参考**（linux/amd64，单镜像全栈）：

| 项                        | 大约大小              |
| ------------------------- | --------------------- |
| 镜像（`docker images`） | **6–8 GB**     |
| Release Bundle（`tar.gz`） | **约 6.9 GB**（参考值） |

主要组成：CUDA 12.8 runtime（含 cuBLAS）、PyTorch、Python 依赖、llama native bin/lib；外层另含 examples 和 README。

**CI 导出**（或 `./scripts/release_build.sh`）：

```bash
./scripts/release_build.sh
# 等价于：docker build → verify_image → docker save
#       → 组装 image.tar + examples + README → tar.gz
```

构建机有本地 HTTP 代理时可显式传入；脚本会为 Docker build 启用 host network，使容器内的 `127.0.0.1` 指向构建宿主机：

```bash
RX_MET_BUILD_PROXY=http://127.0.0.1:7890 ./scripts/release_build.sh
```

PyTorch CUDA 12.8 wheel 仍使用官方 `download.pytorch.org`；清华/阿里 PyPI 镜像通常不提供该 wheel，不作为替代源。

大于 4 GB 时可分卷：`split -b 3900M rx-met-${VERSION}-release.tar.gz rx-met-${VERSION}-release.part`

### 2.1 宿主机前提（客户 IT）

| 组件                               | 位置   | 说明                                    |
| ---------------------------------- | ------ | --------------------------------------- |
| **NVIDIA Driver**            | 宿主机 | 必须；版本需满足 CUDA 12.8 runtime 要求 |
| **nvidia-container-toolkit** | 宿主机 | 必须；离线环境需单独摆渡安装包          |
| **CUDA Toolkit**（nvcc 等）  | 不需要 | 运行预编译 rx-met**不必**安装     |
| **模型 / 评测数据**          | 挂载卷 | 不打进镜像                              |

---

## 3. 使用（客户日常）

### 3.1 导入镜像

```bash
sha256sum -c rx-met-${VERSION}-release.tar.gz.sha256
tar xzf rx-met-${VERSION}-release.tar.gz
cd rx-met-${VERSION}
sha256sum -c SHA256SUMS
docker load -i rx-met-${VERSION}-image.tar
# 分卷时先执行：
# cat rx-met-${VERSION}-release.part* > rx-met-${VERSION}-release.tar.gz
```

### 3.2 环境说明

| 项                        | 说明                                          |
| ------------------------- | --------------------------------------------- |
| **`RX_MET_HOME`** | 镜像内已设为`/opt/rx-met`，客户一般无需修改 |
| 二进制                    | `/opt/rx-met/bin/*`，用户 JSON 不配置       |
| 示例                      | Release Bundle 的`examples/`，运行时挂载    |
| GPU                       | `docker run --gpus all`                     |
| 数据                      | `-v /data:/data` 挂载模型、数据集、config   |

### 3.3 大模型

将 `model`、`eval.dataset` 改为容器挂载路径下的实际位置：

```bash
mkdir -p runs
docker run --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/examples:/examples:ro" \
  -v /data:/data:ro \
  -v "$PWD/runs:/output" \
  -w /output \
  rx-met:${VERSION} \
  rx-met /examples/config/qwen3_reex_q4_k_64.json
```

若 config 在宿主机，挂载后指定容器内路径：

```bash
docker run --gpus all \
  -v "$PWD/examples:/examples:ro" \
  -v /data:/data:ro \
  rx-met:${VERSION} \
  rx-met /examples/config/my_config.json
```

**用户 JSON（最小）**：

```json
{
  "model": "/data/models/Qwen3-30B-A3B",
  "quant": "Q4_K_64",
  "eval": { "dataset": "/data/eval/wiki.test.raw" }
}
```

| 字段             | 含义                         |
| ---------------- | ---------------------------- |
| `model`        | HuggingFace 模型目录（必填） |
| `quant`        | 量化类型（必填）             |
| `eval.dataset` | PPL 数据集；省略则不评估     |

输出默认在容器工作目录 `./runs/<auto>/`；上例将容器工作目录映射到宿主机 `runs/`。可选字段见 Release Bundle 内的 `examples/config/README.md`。

### 3.4 小模型

```bash
cp -a examples examples-work
docker run --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/examples-work:/examples" \
  -v /data:/data:ro \
  -w /examples \
  rx-met:${VERSION} \
  python quick_start.py
```

无 CLI；复制后修改脚本中的模型与数据路径，量化 JSON 见 `config/quick_start_full_quant.json`。

### 3.5 两个 Example 对照

|      | 大模型                           | 小模型                                    |
| ---- | -------------------------------- | ----------------------------------------- |
| 入口 | `rx-met <config.json>`         | `python quick_start.py`                 |
| 配置 | `/examples/config/`（挂载）   | `/examples/config/`（挂载）+ Python   |
| 场景 | HF 模型目录 → 量化 GGUF         | 自定义 PyTorch（MRNN KWS）                |

### 3.6 验收建议

```bash
docker run --rm --gpus all rx-met:${VERSION} rx-met --help
docker run --rm --gpus all rx-met:${VERSION} python -c "import torch; print(torch.cuda.is_available())"
```

---

## 4. 三阶段对照表

|          | CI 制品（§1）      | 客户 Release（§2）           | 使用（§3）                               |
| -------- | ------------------- | ----------------------------- | ----------------------------------------- |
| Python   | `wheels/*.whl`    | 已在镜像层内                  | 容器内`rx-met` / `import aimet_torch` |
| 二进制   | `native/*.tar.gz` | 已在镜像`/opt/rx-met`       | 框架自动找，JSON 不配置                   |
| 镜像     | `docker build`    | Bundle 内 `rx-met-${VERSION}-image.tar` | `docker load` → `docker run`    |
| 示例     | 源码树`examples/` | Bundle 内 `examples/`         | 运行时挂载到 `/examples`             |
| 环境变量 | Dockerfile 写入     | `RX_MET_HOME=/opt/rx-met`   | 一般无需改                                |

---

## 附录 A：离线部署检查清单

1. 校验并解压 `rx-met-${VERSION}-release.tar.gz`
2. 在解压目录执行 `sha256sum -c SHA256SUMS`
3. `docker load -i rx-met-${VERSION}-image.tar`
4. 确认宿主机 Driver 版本满足 CUDA 12.8 runtime（见 NVIDIA 兼容表）
5. 确认已安装 `nvidia-container-toolkit`
6. `docker run --rm --gpus all rx-met:${VERSION} rx-met --help`
7. 准备挂载目录：examples、模型、评测集和输出目录
8. 按 Bundle 内 README 或 §3.3 / §3.4 运行
