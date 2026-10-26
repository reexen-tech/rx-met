# rx-met 应用包发布

产品 VERSION 保持独立；公共 CPU/GPU 环境使用 rx-quant-env v1，详见 [公共环境](Shared_environment.md)。
两个仓库独立保存公共环境定义和版本标签约定，无跨仓库运行依赖。

## 日常发布

本机准备好对应版本标签的 build/runtime 镜像后执行：

```bash
./scripts/release/build.sh cu118 cu126 cu130
```

省略变体时默认仅构建 `cu118`、`cu126`、`cu130`；CPU 必须显式指定 `cpu`。
环境构建、runtime 导出及 Docker bake 默认组遵循相同规则。可同时指定 CPU/CUDA，
例如 `./scripts/release/build.sh cu126 cpu`。

CPU 变体使用普通 Ubuntu、CPU PyTorch 和 ONNX Runtime，支持校准与 PTQ，不需要
NVIDIA Driver 或 Container Toolkit：

```bash
./scripts/environment/build.sh --no-export cpu
./scripts/release/build.sh cpu
./scripts/release/verify_bundle.sh --no-gpu cpu
```

CPU 的三个项目 wheel 使用 `+cpu` 版本后缀；构建原生库时设置
`RX_MET_ENABLE_CUDA=0`，算子扩展不链接 CUDA。KWS CPU 验证加载现成浮点检查点并
指定 `--ptq-only`，两个示例的固定输入及制品对照见 [示例说明](../examples/README.md)。

新构建机器可直接指定已归档的环境目录，两个参数可以指向同一目录：

```bash
./scripts/release/build.sh cu126 \
  --build-env-dir /path/to/environment-images \
  --runtime-env-dir /path/to/environment-images
```

本地已有兼容版本标签时直接复用；缺少时校验并加载对应归档，不要求 `.json` 身份记录。
实际环境的版本、角色、变体或平台不符时直接报错。两个目录参数用于复用环境，不会重新构建镜像；只有 `--with-env` 会自动导出缺失的 runtime 归档。

脚本核对环境元数据、Python ABI 和实际运行依赖，在对应 build 环境中生成三个项目 wheel，
组装应用包，并在纯 runtime 中实际离线安装、自检。默认不生成公共环境交付目录，不执行 docker save。
输出目录为 `.release/export/rx-met-v<VERSION>/`。每个应用包默认在独立临时目录
完成安装验收，成功或失败后自动清理，不受已有安装目录影响。

产品交付目录可以存在，脚本只检查本次所选变体的目标应用归档是否已存在。
未指定 `--replace` 且任一目标归档重名时，在调用 Docker、编译或导出前一次性报告所有冲突。
不同变体可分批追加到同一产品目录，例如先构建 CPU 并附带环境，再追加默认 CUDA 包：

```bash
./scripts/release/build.sh cpu --with-env
./scripts/release/build.sh
```

上述命令要求本机已准备好对应的 build/runtime 镜像。第二次构建保留 CPU 归档及其校验文件，
追加三个 CUDA 归档；未带 `--with-env` 不会补齐 CUDA 环境附件。

使用 `--replace` 时，仅替换本次所选变体的目标归档及其 `.sha256`。
本次全部目标先在暂存区完成构建和安装验收，再发布到产品目录；构建或发布失败时保留或恢复原有产物。
未选择的归档、已解压的应用目录及目录内其他内容保留，已有安装目录和公共环境不受此参数影响。
首次成功发布到旧版产品目录时，会清理原外层 `README.md`、`CHANGELOG.md`、`release.json`
和聚合 `SHA256SUMS`；应用包内及已解压应用目录中的同名文件保留。
已有归档缺少独立 `.sha256` 时会补齐；旧聚合清单中有对应校验值的，先核对再迁移，
校验不匹配则停止发布并保留原目录。
展开的应用及临时验收内容位于 `.release/staging/`，完成后自动清理。
产品发布和环境导出的互斥锁位于 `.release/locks/`，不写入 `export/` 或环境交付目录。
锁按输出目录的规范绝对路径区分，构建结束自动释放，锁文件保留以供后续复用。
自定义输出目录时优先复用上层的 `.release/locks/`；否则在输出父目录建立 `.release/locks/`。
路径包含 `export/` 时，工作目录放在其外层，嵌套的交付目录也不会包含锁文件。

应用包包含 install.sh、README、当前版本 CHANGELOG、manifest、SHA256SUMS、wheels、examples。
用户运行 install.sh 后按 README 使用原生 Docker 命令，不提供 run.sh。
每个变体的应用包只生成对应的一套启动命令；安装器核对环境元数据并准备可读的版本标签，用户无需填写长 SHA-256。

### Wheel 命名

三个项目的发行包名及 wheel 文件名前缀使用以下固定形式：

| 发行包名（METADATA Name） | Wheel 文件名前缀 |
| --- | --- |
| `rx-met` | `rx_met-` |
| `quant-gru` | `quant_gru-` |
| `quant-lstm` | `quant_lstm-` |

wheel 格式将发行包名中的连字符转为下划线，各字段之间使用连字符，例如
`quant_gru-1.1.0+cpu-cp310-cp310-linux_x86_64.whl`。Python 导入继续使用
`quant_gru`、`quant_lstm` 等模块名。

发布校验同时检查包内发行名和标准 wheel 文件名。修改命名声明后，通过正常构建重新
生成 wheel 及应用包，使 METADATA、RECORD 和发布校验值对应同一份产物。

## 首次离线交付

本机已有匹配的 build/runtime 镜像时，一条命令生成应用包及镜像附件：

```bash
./scripts/release/build.sh --with-env cu126
```

省略变体时仅处理 CUDA 变体，`--with-env` 也不会隐式加入 CPU；交付 CPU 包及环境时执行
`./scripts/release/build.sh --with-env cpu`。归档默认缓存于 `.release/runtime-images/<环境版本>/`：
已有完整归档则校验复用，缺失则从本机版本标签 自动导出。删除 `.release` 后也可重新执行，
无需先手动导出镜像。Docker 中的环境镜像和构建缓存不会随 `.release` 删除；脚本不会重建已发布环境。

需要复用共享盘归档时，加上 `--runtime-env-dir /path/to/public-runtime-archives`。
带 `--with-env` 时，该目录不存在会自动创建；已有归档损坏时会报错，不会覆盖。
缺少本机环境镜像时，仍需通过两个环境目录参数导入已发布镜像。

交付目录布局如下，产品与公共环境独立，应用包内部不嵌入镜像：

```text
.release/export/
├── rx-met-v<VERSION>/
│   ├── rx-met-v<VERSION>-cu126-linux-amd64.tar.gz
│   └── rx-met-v<VERSION>-cu126-linux-amd64.tar.gz.sha256
└── rx-quant-env-v1/                  # 仅 --with-env 导出或复用
    ├── rx-quant-env-runtime-v1-cu126-linux-amd64.tar.gz
    ├── rx-quant-env-runtime-v1-cu126-linux-amd64.tar.gz.json
    ├── rx-quant-env-runtime-v1-cu126-linux-amd64.tar.gz.sha256
    ├── README.md
    ├── manifest.json
    └── SHA256SUMS
```

`--with-env` 只附带所选变体的公共运行环境，不附带构建环境镜像。
归档通过硬链接或复制放入产品目录平级的 `<环境名>-<环境版本>/`；已有环境校验复用、
按变体补齐，不覆盖其他变体或版本。不带此参数时已有环境目录保持原样。

首次离线交付同时提供产品目录和对应环境目录；后续已安装相同环境的用户只需产品目录。
在产品目录内解压所选应用包并运行包内 `install.sh`，安装器自动查找平级环境目录；
也支持将解压后的应用目录与环境目录直接平级放置，或使用 `--runtime-image FILE` 显式指定归档。
产品目录只生成应用归档及各自的 `.sha256`，不再生成外层 README、CHANGELOG、
`release.json` 或聚合 `SHA256SUMS`。版本、变体和环境要求由每个应用包内的
`manifest.json` 描述，使用说明和当前版本日志分别保存在包内 README、CHANGELOG 中，
包内 `SHA256SUMS` 校验应用文件。所需公共环境的完整归档文件名也列在包内 README 中。

分发或解压前，在应用归档所在目录检查对应的独立校验文件，例如 1.1.0 的 cu126 包：

```bash
sha256sum -c rx-met-v1.1.0-cu126-linux-amd64.tar.gz.sha256
```

公共环境目录继续维护 README、manifest、导出元数据和校验清单；应用归档校验不绑定环境归档文件的字节校验值。

### 环境标签与文件校验

应用包通过版本标签要求公共环境，不绑定本地镜像 ID、构建脚本摘要或归档文件的 SHA256。
镜像 ID 自动记录在构建资料中，供问题排查和编译缓存判断使用。

环境归档实际 SHA256 单独保存在同名 `.sha256`、导出元数据 `.json` 和环境交付 `SHA256SUMS` 中。
安装器支持环境归档同名 `.sha256`（一行 SHA256 或标准 sha256sum 格式），以及归档同目录
`SHA256SUMS` 中对应文件的记录。有校验记录时必须通过检查；没有记录时检查 gzip 完整性，
由 Docker 加载并验证镜像内容。应用文件和已安装内容继续独立校验。

安装器验证本次实际导入镜像的环境名称、版本、runtime 角色、CUDA/CPU 变体和平台，
再补齐公共版本标签。安装中的容器使用本次解析出的同一个本地镜像，避免标签变化影响单次安装。
宿主机无需 Python。相同环境版本下重新构建的兼容镜像可以复用，错误角色或变体会被拒绝。

配置 `--runtime-base-url` 后，安装器下载对应归档，并尝试下载 `<归档URL>.sha256`。
校验文件返回 404/410 时按无校验文件处理，网络错误、其他 HTTP 错误或校验失败停止安装。
下载通过加载和环境检查后才写入缓存。更新兼容环境的归档不要求重新生成应用包。

独立环境导出和加载命令采用相同的版本标签约定，不要求仓库内存在镜像身份记录。

## 参数与版本记录

- `--build-env-dir DIR`：缺少本地 build 镜像时，从该目录导入；未指定时要求镜像已在本机。
- `--runtime-env-dir DIR`：运行环境归档的导入、复用及自动导出缓存，默认 `.release/runtime-images/<环境版本>/`。
- `--with-env`：向产品目录平级的环境目录附带所选变体的运行环境镜像，缺少归档时自动导出；默认关闭。
- `--jobs N`：编译并行数，默认 8。
- `--output-dir DIR`：最终产品交付目录；环境附件放在其父目录下的 `<环境名>-<环境版本>/`。
- `--install-prefix DIR`：指定并保留安装验收目录；默认使用临时目录并自动清理。
- `--skip-build`：源码和 build/runtime 身份完全匹配时复用已有编译产物，重新组装交付文档。
- `--replace`：验收通过后，仅替换本次所选变体的同名应用归档及其 `.sha256`；保留未选择的归档、已解压应用目录及其他内容，不覆盖已有安装目录或公共环境。
- `--runtime-base-url https://...`：预留固定归档下载地址，当前留空、离线交付。

根 CHANGELOG 保留历史；打包前按 VERSION 校验并提取唯一且非空的当前版本，
应用包和安装目录内的 CHANGELOG 均只保留当前版本内容，产品外层不再放置 CHANGELOG。
已安装内容参与完整性校验。

## 验证

需要保留安装结果用于 GPU 或示例验证时，构建时显式指定验收目录。
已有不同内容的目录不会被覆盖，请使用新目录：

```bash
./scripts/release/build.sh cu126 --replace --install-prefix .release/installed-validation
./scripts/release/verify_bundle.sh --install-prefix .release/installed-validation --gpu --gpu-device 0 cu126
```

构建时自检使用非 root、断网、只读应用挂载；GPU 验证单独运行。发布前还需运行真实 examples，
检查默认 runtime 复用、归档导入、重复安装、失败恢复、目录迁移、升级和回滚。
