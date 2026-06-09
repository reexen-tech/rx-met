# Docker 挂载 datasets 与 models 目录

当在容器内使用 `datasets`（评测数据集）和 `models/shared_models`（共享模型）时，需在**宿主机**上把对应目录挂进容器；若宿主机路径有变动，只需改挂载参数后重新启动容器。

## 1. 挂载约定（容器内路径）

| 容器内路径 | 用途 |
|------------|------|
| `/mnt/data8t/share/datasets` | 评测数据集（对应仓库内 `datasets` 符号链接目标） |
| `/mnt/data8t/share/models`   | 共享模型（对应 `models/shared_models` 符号链接目标） |

宿主机上的**实际路径**由你自己决定，下面用环境变量表示，方便修改。

## 2. 使用环境变量（推荐）

在宿主机定义两个变量，指向**当前** datasets 与 models 所在目录，例如：

```bash
# 宿主机上执行，路径按实际修改
export LLAMA_DATASETS_HOST=/新路径/share/datasets
export LLAMA_MODELS_HOST=/新路径/share/models
```

之后所有 `docker run` / 脚本都使用这两个变量。

## 3. docker run 示例

**仅挂载 datasets + models（容器内已有代码或只关心数据）：**

```bash
docker run -it --rm \
  -v "${LLAMA_DATASETS_HOST:-/mnt/data8t/share/datasets}:/mnt/data8t/share/datasets:ro" \
  -v "${LLAMA_MODELS_HOST:-/mnt/data8t/share/models}:/mnt/data8t/share/models:ro" \
  -v "$(pwd):/workspace/llama.cpp" \
  -w /workspace/llama.cpp \
  <你的镜像> \
  bash
```

**若容器内需要可写：** 去掉 `:ro`。

进入容器后，在 `/workspace/llama.cpp` 下执行一次符号链接（首次或路径变过时）：

```bash
./scripts/remount_datasets_models.sh /mnt/data8t/share/datasets /mnt/data8t/share/models
```

这样 `datasets` 和 `models/shared_models` 会指向刚挂载进来的目录。

## 4. 挂载目录变了怎么改

1. **只改宿主机路径**  
   修改并导出环境变量后，用**新的** `docker run` 启动容器（同上，仍挂到容器内 `/mnt/data8t/share/datasets` 和 `/mnt/data8t/share/models`）：
   ```bash
   export LLAMA_DATASETS_HOST=/新位置/datasets
   export LLAMA_MODELS_HOST=/新位置/models
   # 再执行上面的 docker run
   ```
2. **容器内符号链接**  
   若容器内目标路径没变（仍是 `/mnt/data8t/...`），只需重新启动容器并挂载新路径，**不必**再跑 `remount_datasets_models.sh`。  
   若容器内目标路径也改了，进容器后执行：
   ```bash
   ./scripts/remount_datasets_models.sh <新容器内_datasets路径> <新容器内_models路径>
   ```

## 5. 一键脚本（宿主机）

在 **llama.cpp 仓库根目录**下执行（宿主机）：

```bash
# 挂载目录变了：只改这两个环境变量为新路径，再运行
export LLAMA_DATASETS_HOST=/你的新路径/datasets
export LLAMA_MODELS_HOST=/你的新路径/models
./scripts/docker_run_with_mounts.sh
```

可选：传入镜像名，例如 `./scripts/docker_run_with_mounts.sh ghcr.io/ggml-org/llama.cpp:full-cuda`。  
脚本默认使用 `LLAMA_DATASETS_HOST`、`LLAMA_MODELS_HOST`（未设置时用 `/mnt/data8t/share/datasets` 与 `/mnt/data8t/share/models`），以后目录变了只需改环境变量或脚本内默认值即可。
