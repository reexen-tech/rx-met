# rx-met runtime 与应用包发布

rx-met 对外发布三个固定环境版本的 runtime 镜像，以及每个产品版本的三个应用包。
用户运行包内 install.sh，之后直接按 README 使用 docker run。发布包没有 run.sh、env.sh 或 internal/。

## 生命周期与版本

- `VERSION` 定义产品版本；三个项目 wheel 都使用 `<VERSION>+<variant>`。
- `docker/variants.json` 定义环境版本、平台、运行依赖和 GPU 架构。
- `python3 scripts/dependencies.py lock` 生成 requirements 和 Bake 配置。
- 已发布的环境版本不可覆盖；依赖或系统环境变化应创建新环境版本。
- build-env 内部使用；runtime-env 不安装项目代码。

## 准备环境

```bash
./scripts/environment/build.sh --no-export
```

此操作与产品发布独立。产品构建要求对应 build-env 和 runtime-env 已存在，缺失时直接失败，
不会联网重建同名环境。基础镜像首次构建仍需网络；应保留并复用验证过的镜像身份和归档。
原有 environment/export.sh、load.sh 保留为内部 build/runtime 成对搬运工具。

单独导出用户需要的 runtime：

```bash
./scripts/environment/export_runtime.sh cu118 cu126 cu130
```

默认输出到 `.release/runtime-images/<环境版本>/`，每个变体一个 `.tar.gz` 和相应 JSON。
归档只包含按 image ID 保存的 runtime，不携带 build-env，也不在导入时覆盖本地 tag。
已存在的归档必须与当前镜像身份及 SHA-256 完全匹配，禁止静默覆盖。

## 构建应用发布包

```bash
./scripts/release/build.sh cu118 cu126 cu130
```

构建流程：校验并提取当前版本 CHANGELOG → 检查固定环境 → 导出/复用 runtime 归档 → 编译三个 wheel →
生成 manifest 和 install.sh → 在纯 runtime 内离线安装并做无 GPU 自检 → 打包应用。
默认并行编译数为 8，可用 `--jobs N` 修改。

默认离线交付目录：

```text
.release/export/v<VERSION>/
├── rx-met-v<VERSION>-<环境版本>-cu118-linux-amd64.tar.gz
├── rx-met-v<VERSION>-<环境版本>-cu126-linux-amd64.tar.gz
├── rx-met-v<VERSION>-<环境版本>-cu130-linux-amd64.tar.gz
├── rx-met-v<VERSION>-<环境版本>-cu118-linux-amd64/   # 可直接验收的解压目录
├── ...
├── images/
│   ├── rx-met-runtime-env-<环境版本>-cu118-linux-amd64.tar.gz
│   ├── ...
│   └── 各归档的JSON身份记录
├── README.md
├── CHANGELOG.md
└── SHA256SUMS
```

应用 tar.gz 不重复携带 runtime；首次离线交付同时提供 images/，后续只提供新应用包。
将应用解压到 images/ 的同级目录后，执行包内 `./install.sh --offline` 即可。
`--with-runtime` 可将相应镜像归档直接附加到各应用包的 images/，适合单文件首次交付。
应用包包含 install.sh、README.md、CHANGELOG.md、manifest.json、SHA256SUMS、wheels/、examples/ 和可选 images/。
仓库根目录 CHANGELOG 保留完整历史，发布前将确认纳入当前版本的 Unreleased 条目归入对应版本。
发布脚本按 `VERSION` 精确提取该版本标题及全部小节，保留原发布日期；三个应用包和交付总目录
使用同一份提取结果，不包含其他版本或 Unreleased。缺失、重复或空版本记录会在构建开始前报错。
安装后保留 README、CHANGELOG 和 manifest，三者均参与安装完整性校验。

常用发布参数：

- `--output-dir DIR`：交付目录。
- `--runtime-dir DIR`：可复用的 runtime 归档目录。
- `--install-prefix DIR`：构建时验收的安装根目录，默认 `.release/installed`。
- `--runtime-base-url https://...`：预先发布的 runtime 归档所在目录；默认空，先交付离线。
- `--skip-build`：仅在 wheel 源码及 build/runtime 身份未变时重新组装。
- `--replace`：替换本机尚未发布的应用归档，不覆盖已有安装或 runtime；修改后的候选包请配合新的 `--install-prefix`。

发布身份记录包含源码 revision（脏工作区显式标记 dirty）、wheel 源码摘要、交付文件源码摘要、
build/runtime image ID、Python ABI、精确依赖版本、GPU 架构、最低驱动声明和 wheel 校验和。
交付文件摘要只纳入提取后的当前版本 CHANGELOG；修改历史版本或 Unreleased 不会影响该摘要。
运行始终使用指定 image ID。下载地址及预期校验值在生成安装器时固定，不使用 latest。

## 验收

```bash
./scripts/release/verify_bundle.sh --gpu --gpu-device 0 cu118 cu126 cu130
./scripts/release/verify_kws_example.sh \
  --dataset-dir /path/to/speech_commands_v0.02 --gpu-device 0 cu118 cu126 cu130
./scripts/release/verify_onnx_ptq_example.sh \
  --model /path/to/mobilenetv2.onnx --dataset-dir /path/to/calib --gpu-device 0 cu118 cu126 cu130
```

以上命令在非 root、断网、应用目录只读的纯 runtime 中执行。根文件系统只读，临时文件写入 tmpfs，
结果写入独立输出目录。`--install-prefix DIR` 可选择待验收安装根目录。
无 GPU 构建验收不能替代真实 GPU 验收；声明的最低驱动仍需对应硬件环境单独验证。

发布前还需覆盖离线归档导入、重复安装、错误环境、文件损坏、失败恢复、安装目录迁移及版本切换。
用户如何挂载应用、工作区、数据集，以及运行两个 examples，详见发布包 README。
