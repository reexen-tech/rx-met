# rx-met 应用包发布

产品 VERSION 保持独立；公共 GPU 环境使用 rx-quant-env v1，详见 [公共环境](Shared_environment.md)。
两个仓库独立保存公共环境定义及发布身份记录，无跨仓库运行依赖。

## 日常发布

本机准备好发布记录指定的 build/runtime 镜像后执行：

```bash
./scripts/release/build.sh cu118 cu126 cu130
```

新构建机器可直接指定已归档的环境目录，两个参数可以指向同一目录：

```bash
./scripts/release/build.sh cu126 \
  --build-env-dir /path/to/environment-images \
  --runtime-env-dir /path/to/environment-images
```

本地已有匹配镜像时直接复用；缺少时校验对应归档及其 `.json` 身份记录，再执行 `docker load`。
同名镜像的身份不符时直接报错。两个目录参数用于复用环境，不会重新构建镜像；只有 `--with-images` 会自动导出缺失的 runtime 归档。

脚本核对公共环境配置和精确镜像 ID，在对应 build 环境中生成三个项目 wheel，
组装应用包，并在纯 runtime 中实际离线安装、自检。默认不生成 images/，不执行 docker save。
输出目录为 `.release/export/rx-quant-env-v1/v<VERSION>/`。每个应用包默认在独立临时目录
完成安装验收，成功或失败后自动清理，不受已有安装目录影响。

未指定 `--replace` 时，启动前检查所有所选变体；同名应用包目录或压缩包已存在则立即退出，
不会调用 Docker、编译 wheel 或导出镜像。

应用包包含 install.sh、README、当前版本 CHANGELOG、manifest、SHA256SUMS、wheels、examples。
用户运行 install.sh 后按 README 使用原生 Docker 命令，不提供 run.sh。
每个变体的应用包只生成对应的一套启动命令；安装器核对镜像 ID 并准备可读的版本标签，用户无需填写长 SHA-256。

## 首次离线交付

本机已有匹配的 build/runtime 镜像时，一条命令生成应用包及镜像附件：

```bash
./scripts/release/build.sh --with-images cu126
```

省略 `cu126` 时处理全部三个 CUDA 变体。归档默认缓存于 `.release/runtime-images/<环境版本>/`：
已有完整归档则校验复用，缺失则从本机固定镜像 ID 自动导出。删除 `.release` 后也可重新执行，
无需先手动导出镜像。Docker 中的环境镜像和构建缓存不会随 `.release` 删除；脚本不会重建已发布环境。

需要复用共享盘归档时，加上 `--runtime-env-dir /path/to/public-runtime-archives`。
带 `--with-images` 时，该目录不存在会自动创建；已有归档损坏或身份不符时会报错，不会覆盖。
缺少本机环境镜像时，仍需通过两个环境目录参数导入已发布镜像。

交付目录布局如下，应用包内部不嵌入镜像：

```text
v<VERSION>/
├── 应用包.tar.gz
├── 解压后的应用包/
├── images/
│   ├── rx-quant-env-runtime-v1-cu126-linux-amd64.tar.gz
│   ├── rx-quant-env-runtime-v1-cu126-linux-amd64.tar.gz.json
│   └── rx-quant-env-runtime-v1-cu126-linux-amd64.tar.gz.sha256
├── README.md
├── CHANGELOG.md
└── SHA256SUMS
```

`--with-images` 是唯一的镜像附件开关。归档通过硬链接或复制放入 `images/`。
首次交付同时提供应用包和 `images/`；后续已安装相同环境的用户只需应用包。

### 镜像身份与文件校验

应用包只固定公共环境配置和精确镜像 ID，不记录 runtime 归档文件的 SHA256 或大小。
同一镜像重新导出的归档，即使文件 SHA256 改变，也可以用于同一个应用包；已发布的旧安装器需要重新打包才能采用此规则。

归档实际 SHA256 单独保存在同名 `.sha256`、`.json` 和交付 `SHA256SUMS` 中。
安装器支持同名 `.sha256`（一行 SHA256 或标准 sha256sum 格式），以及归档同目录 `SHA256SUMS` 中对应文件的记录。
提供了匹配的文件校验记录时必须通过校验；没有记录时仍检查 gzip 完整性和镜像身份、内容。
应用包自己的 SHA256SUMS 和安装后的 INSTALL_SHA256SUMS 继续校验应用文件。

安装器检查本次 `docker load` 返回的每个镜像身份，再使用固定 runtime 中的 Python 验证归档内的
Docker/OCI 配置、清单和层摘要；不会执行归档中的其他镜像。若本机已有匹配 runtime，则可在导入前完成内容验证。
无需在宿主机安装 Python；本机原有的正确镜像不能使错误的显式 `--runtime-image` 通过校验。

配置 `--runtime-base-url` 后，安装器下载对应归档，并尝试下载 `<归档URL>.sha256`。
校验文件返回 404/410 时按无校验文件处理，网络错误、其他 HTTP 错误或校验失败会停止安装。
下载文件只有通过镜像身份和内容验证后才进入缓存。下载地址可以提供同一固定镜像的其他导出文件，
配套校验文件应与实际归档一致；更新归档不要求重新生成应用包。

独立 `scripts/environment/export_runtime.sh` 和 `load.sh` 使用相同的镜像身份规则，
允许重新导出文件的 SHA256 改变，保留公共环境首次发布的身份记录。目录结构和环境版本保持原样。

## 参数与版本记录

- `--build-env-dir DIR`：缺少本地 build 镜像时，从该目录导入；未指定时要求镜像已在本机。
- `--runtime-env-dir DIR`：runtime 归档的导入、复用及自动导出目录，默认 `.release/runtime-images/<环境版本>/`。
- `--with-images`：附带镜像到交付目录 `images/`，缺少归档时自动导出；默认关闭。
- `--jobs N`：编译并行数，默认 8。
- `--output-dir DIR`：应用交付目录。
- `--install-prefix DIR`：指定并保留安装验收目录；默认使用临时目录并自动清理。
- `--skip-build`：源码和 build/runtime 身份完全匹配时复用已有编译产物，重新组装交付文档。
- `--replace`：替换未发布候选应用包；默认临时安装验收，不覆盖已有安装目录或公共环境。
- `--runtime-base-url https://...`：预留固定归档下载地址，当前留空、离线交付。

根 CHANGELOG 保留历史；打包前按 VERSION 校验并提取唯一且非空的当前版本，
应用包、交付总目录和安装目录均只保留当前版本内容。已安装内容参与完整性校验。

## 验证

需要保留安装结果用于 GPU 或示例验证时，构建时显式指定验收目录。
已有不同内容的目录不会被覆盖，请使用新目录：

```bash
./scripts/release/build.sh cu126 --replace --install-prefix .release/installed-validation
./scripts/release/verify_bundle.sh --install-prefix .release/installed-validation --gpu --gpu-device 0 cu126
```

构建时自检使用非 root、断网、只读应用挂载；GPU 验证单独运行。发布前还需运行真实 examples，
检查默认 runtime 复用、归档导入、重复安装、失败恢复、目录迁移、升级和回滚。
