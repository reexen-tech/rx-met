# rx-met 应用包发布

本文面向构建和交付 rx-met 应用包的维护者，命令均在仓库根目录执行。
安装和运行现成应用包见[用户使用指南](User_guide.md)。

发布流程为：准备 build/runtime 环境 → 构建并完成离线安装自检 → 验证目标设备和真实示例
→ 交付应用包。接收方缺少匹配的本机 runtime 镜像时，离线交付还需提供 runtime 归档。
Linux x86_64 构建宿主机需有 Python 3（3.10 或更新版本）、Docker 和 Buildx；构建包含本机 bind mount
安装验收，因此需通过 Unix socket 访问本机 Docker daemon。GPU 验收另需匹配的
NVIDIA 驱动及 Container Toolkit。接收方运行安装器需要本机 Docker daemon、Bash、
GNU coreutils、GNU findutils、`flock` 和加载归档时使用的 `gzip`，宿主机无需 Python；
下载 runtime 时还需 `curl`。远程 Docker context 不支持安装器所需的本机路径挂载。
产品 `VERSION` 与公共环境版本独立；环境定义、版本及兼容镜像复用规则以
[公共环境](Shared_environment.md#版本与复用)为准。

## 日常发布

本机准备好对应版本标签的 build/runtime 镜像后，选择要发布的变体。以下以 `cu126`
为例，保留安装目录供下一节验证；已有同名归档时需显式加 `--replace`：

```bash
./scripts/release/build.sh cu126 --install-prefix .release/installed-validation
```

省略变体时默认仅构建 `cu118`、`cu126`、`cu130`；CPU 必须显式指定 `cpu`。
环境构建、runtime 导出及 Docker bake 默认组遵循相同规则。可同时指定 CPU/CUDA，
例如 `./scripts/release/build.sh cu126 cpu`。

CPU 变体支持校准与 PTQ，不需要 NVIDIA Driver 或 Container Toolkit。
发布 CPU 时使用以下流程替代上面的 CUDA 构建：

```bash
./scripts/environment/build.sh cpu
./scripts/release/build.sh cpu --install-prefix .release/installed-validation
./scripts/release/verify_bundle.sh --install-prefix .release/installed-validation --no-gpu cpu
```

CPU 的三个项目 wheel 使用 `+cpu` 版本后缀；构建原生库时设置
`RX_MET_ENABLE_CUDA=0`，算子扩展不链接 CUDA。KWS CPU 验证加载现成浮点检查点并
指定 `--ptq-only`，两个示例的固定输入及制品对照见 [示例说明](../examples/README.md)。
Torch 版本更新时须重新编译这三个 wheel，不能沿用旧版本环境下的原生扩展。

新构建机器可在构建命令中指定已归档的环境目录，替代上面的本地环境构建命令。
两个参数可以指向同一目录：

```bash
./scripts/release/build.sh cu126 \
  --build-env-dir /path/to/environment-images \
  --runtime-env-dir /path/to/environment-images \
  --install-prefix .release/installed-validation
```

对每个角色，本地已有兼容版本标签时复用镜像；没有该标签时，从对应目录校验并加载归档。
镜像的环境版本、角色、变体或平台不符时直接报错。两个目录参数不会触发镜像构建；
发布命令只有带 `--with-env` 才会自动导出缺失的 runtime 归档。

脚本核对环境元数据、Python ABI 和实际运行依赖，在对应 build 环境中生成三个项目 wheel，
组装应用包，并在纯 runtime 中实际离线安装、自检。默认不生成公共环境交付目录，不执行 docker save。
输出目录为 `.release/export/rx-met-v<VERSION>/`。未指定 `--install-prefix` 时，每个应用包
在独立临时目录完成安装验收，成功或失败后自动清理，不受已有安装目录影响。
指定该参数时保留安装结果，已有不同内容的目录不会被覆盖；需要重建时应使用新安装目录。
从 1.1.1 起，产品目录生成 `README.md`，说明文件用途、当前已有变体的选择条件，以及环境准备、解压和安装步骤。
增量追加变体时会更新该说明，已有归档及其独立校验文件保留。

应用包包含 install.sh、README、当前版本 CHANGELOG、manifest、SHA256SUMS、wheels、examples。
用户运行 install.sh 后按 README 使用原生 Docker 命令，不提供 run.sh。
每个变体的应用包只生成对应的一套启动命令；安装器核对环境元数据并准备可读的版本标签，用户无需填写长 SHA-256。

## 验证

使用日常发布命令保留的安装目录继续验证 CUDA 变体：

```bash
./scripts/release/verify_bundle.sh --install-prefix .release/installed-validation --gpu --gpu-device 0 cu126
```

构建时自检使用宿主调用者的 UID/GID、断网和只读应用挂载；
非 root 验收需要由非 root 用户运行构建或验收命令。成功安装不代表目标 GPU 已经通过验证。
发布前还需完成真实数据示例，并检查默认 runtime 复用、归档导入、重复安装、失败恢复、
目录迁移、升级和回滚。CUDA 真实数据验收继续使用同一安装目录，例如：

```bash
./scripts/release/verify_kws_example.sh --install-prefix .release/installed-validation \
  --dataset-dir /path/to/speech_commands_v0.02 cu126
./scripts/release/verify_onnx_ptq_example.sh --install-prefix .release/installed-validation \
  --model /path/to/mobilenetv2-12.onnx --dataset-dir /path/to/calibration-samples cu126
```

ONNX 验证入口读取与模型匹配的 `.npy` / `.npz` 校准样本目录。模型与样本准备、CPU 固定检查点
与校准数据验证见[示例说明](../examples/README.md)。
涉及新环境基线时，还应完成[产品验收](Shared_environment.md#产品验收)。

## 首次离线交付

本机已有匹配的 build/runtime 镜像时，一条命令生成应用包及镜像附件：

```bash
./scripts/release/build.sh --with-env cu126 --install-prefix .release/installed-validation
```

本次需要 runtime 附件时，使用该命令替代日常构建命令，再执行前述验证。
同名归档已存在时需加 `--replace`；不同内容的安装目录不会被覆盖，应换用新目录。
省略变体时仅处理 CUDA 变体，`--with-env` 也不会隐式加入 CPU；交付 CPU 包及环境时执行
`./scripts/release/build.sh --with-env cpu`。归档默认缓存于 `.release/runtime-images/<环境版本>/`：
已有完整归档则校验复用，缺失则从本机版本标签自动导出。删除 `.release` 后也可重新执行，
无需先手动导出镜像。Docker 中的环境镜像和构建缓存不会随 `.release` 删除；脚本不会重建已发布环境。

需要复用共享盘归档时，加上 `--runtime-env-dir /path/to/public-runtime-archives`。
已有归档损坏时会报错，不会覆盖。目录尚不存在时，`--with-env` 可从本机已有的匹配 runtime
导出并创建目录；本机也没有 runtime 时，须先准备镜像，或指定已有归档的目录供构建脚本导入。
本机缺少 build 镜像时同理，可指定 `--build-env-dir` 加载已有归档。

交付目录布局如下，产品与公共环境独立，应用包内部不嵌入镜像：

```text
.release/export/
├── rx-met-v<VERSION>/
│   ├── README.md
│   ├── rx-met-v<VERSION>-cu126-linux-amd64.tar.gz
│   └── rx-met-v<VERSION>-cu126-linux-amd64.tar.gz.sha256
└── rx-quant-env-202610/              # 仅 --with-env 导出或复用
    ├── rx-quant-env-runtime-202610-cu126-linux-amd64.docker.tar.gz
    ├── rx-quant-env-runtime-202610-cu126-linux-amd64.docker.tar.gz.json
    ├── rx-quant-env-runtime-202610-cu126-linux-amd64.docker.tar.gz.sha256
    └── README.md
```

`--with-env` 只附带所选变体的公共运行环境，不附带构建环境镜像。
开发交付可另用 `./scripts/environment/export.sh --role build` 导出 build 镜像；
`--role all` 同时导出两种角色。
归档通过硬链接或复制放入产品目录平级的 `<环境名>-<环境版本>/`；已有环境校验复用、
按变体补齐，不覆盖其他变体或版本。不带此参数时已有环境目录保持原样。

首次离线交付同时提供产品目录和对应环境目录；后续已安装相同环境的用户只需产品目录。
在产品目录内解压所选应用包并运行包内 `install.sh`，安装器自动查找平级环境目录；
也支持将解压后的应用目录与环境目录直接平级放置，或使用 `--runtime-image FILE` 显式指定归档。
产品目录生成通用 README、应用归档及各自的 `.sha256`，不再生成外层 CHANGELOG、
`release.json` 或聚合 `SHA256SUMS`。版本、变体和环境要求由每个应用包内的
`manifest.json` 描述，使用说明和当前版本日志分别保存在包内 README、CHANGELOG 中，
包内 `SHA256SUMS` 校验应用文件。所需公共环境的完整归档文件名也列在包内 README 中。

分发或解压前，在应用归档所在目录检查对应的独立校验文件，将 `<VERSION>` 替换为
本次产品版本：

```bash
sha256sum -c 'rx-met-v<VERSION>-cu126-linux-amd64.tar.gz.sha256'
```

公共环境目录保留 README、每个镜像的环境说明 `.json` 及独立 `.sha256`，不生成外层
`manifest.json` 和 `SHA256SUMS`。环境说明记录对应镜像的实际系统和主要组件版本，
以及完整的 Python、系统依赖清单。镜像归档、`.sha256` 和 `.json` 的用途见目录内 README；应用包内的
`SHA256SUMS` 保留，用于校验应用文件。

## 环境标签与文件校验

环境版本和兼容条件见[版本与复用](Shared_environment.md#版本与复用)。
本节说明应用安装器如何校验环境归档和应用文件。
应用包通过版本标签要求公共环境，不绑定本地镜像 ID、构建脚本摘要或归档文件的 SHA256。
镜像 ID 自动记录在构建资料中，供问题排查和编译缓存判断使用。

环境归档实际 SHA256 保存在同名 `.sha256` 和导出元数据 `.json` 中。
这两个文件均为可选辅助材料；安装器可只使用 Docker 镜像归档，
不读取归档 `.json`，也不要求 `.sha256` 必须存在。应用包自身的 manifest 和包内校验清单仍需完整保留。
安装器支持环境归档同名 `.sha256`（一行 SHA256 或标准 sha256sum 格式），以及归档同目录
旧版 `SHA256SUMS` 中对应文件的记录。有校验记录时必须通过检查；没有记录时检查 gzip 完整性，
由 Docker 加载并验证镜像内容。应用文件和已安装内容继续独立校验。

安装器验证本次实际导入镜像的环境名称、版本、runtime 角色、CUDA/CPU 变体和平台，
再补齐公共版本标签。安装中的容器使用本次解析出的同一个本地镜像，避免标签变化影响单次安装。
宿主机无需 Python。相同环境版本下重新构建的兼容镜像可以复用，错误角色或变体会被拒绝。

配置 `--runtime-base-url` 后，安装器下载对应归档，并尝试下载 `<归档URL>.sha256`。
校验文件返回 404/410 时按无校验文件处理，网络错误、其他 HTTP 错误或校验失败停止安装。
下载通过加载和环境检查后才写入缓存。更新兼容环境的归档不要求重新生成应用包。

独立环境导出和加载采用相同的版本标签约定，命令与旧环境保留规则见
[公共环境归档](Shared_environment.md#离线导出与加载)。

## 参数与版本记录

- `--build-env-dir DIR`：缺少本地 build 镜像时，从该目录导入；未指定时要求镜像已在本机。
- `--runtime-env-dir DIR`：运行环境归档的导入、复用及自动导出缓存，默认 `.release/runtime-images/<环境版本>/`。
- `--with-env`：向产品目录平级的环境目录附带所选变体的运行环境镜像，缺少归档时自动导出；默认关闭。
- `--jobs N`：编译并行数，默认 8。
- `--output-dir DIR`：最终产品交付目录；环境附件放在其父目录下的 `<环境名>-<环境版本>/`。
- `--install-prefix DIR`：指定并保留安装验收目录；默认使用临时目录并自动清理。
- `--skip-build`：源码和 build/runtime 身份完全匹配时复用已有编译产物，重新组装交付文档。
- `--replace`：验收通过后，仅替换本次所选变体的同名应用归档及其 `.sha256`；保留未选择的归档、已解压应用目录及其他内容，不覆盖已有安装目录或公共环境。
- `--runtime-base-url https://...`：为应用包设置公共 runtime 归档的 HTTPS 下载地址。

`--runtime-base-url` 未指定时读取 `RX_MET_RUNTIME_BASE_URL`；两者均未设置时使用离线归档。

根 CHANGELOG 保留历史；打包前按 VERSION 校验并提取唯一且非空的当前版本，
应用包和安装目录内的 CHANGELOG 均只保留当前版本内容，产品外层不再放置 CHANGELOG。
已安装内容参与完整性校验。

## 追加、替换与失败恢复

产品交付目录可以存在，脚本只检查本次所选变体的目标应用归档是否已存在。
未指定 `--replace` 且任一目标归档重名时，在调用 Docker、编译或导出前一次性报告所有冲突。
不同变体可分批追加到同一产品目录，例如先构建 CPU 并附带环境，再追加默认 CUDA 包：

```bash
./scripts/release/build.sh cpu --with-env
./scripts/release/build.sh
```

上述命令要求本机已准备好对应的 build/runtime 镜像。第二次构建保留 CPU 归档及其校验文件，
追加三个 CUDA 归档；未带 `--with-env` 不会补齐 CUDA 环境附件。

构建命令使用 `--replace` 时，仅替换本次所选变体的目标归档及其 `.sha256`。
本次全部目标先在暂存区完成构建和安装验收，再发布到产品目录；构建或发布失败时保留或恢复原有产物。
未选择的归档、已解压的应用目录及目录内其他内容保留，已有安装目录和公共环境不受此参数影响。
成功发布时会重新生成外层 `README.md`，并清理旧版外层 `CHANGELOG.md`、`release.json`
和聚合 `SHA256SUMS`；应用包内及已解压应用目录中的同名文件保留。
已有归档缺少独立 `.sha256` 时会补齐；旧聚合清单中有对应校验值的，先核对再迁移，
校验不匹配则停止发布并保留原目录。

应用包组装及未指定 `--install-prefix` 时的安装验收使用 `.release/staging/`，完成后自动清理。
整批应用归档另在产品输出目录的父目录中暂存，成功后替换产品目录，正常完成或失败恢复后清理。
产品目录发布和环境附件发布的互斥锁位于 `.release/locks/`，不写入交付目录。
锁按输出目录的规范绝对路径区分，构建结束自动释放，锁文件保留以供后续复用。
自定义输出目录时优先复用上层的 `.release/locks/`；否则在输出父目录建立 `.release/locks/`。
路径包含 `export/` 时，这些发布锁放在其外层，嵌套的交付目录也不会包含发布锁文件。
镜像归档导出还在归档缓存目录保留 `.<归档文件名>.lock`，用于防止同一归档被并发生成；
直接把环境导出到交付目录时，该隐藏锁文件也会位于其中，不属于安装所需材料。

## Wheel 命名

三个项目的发行包名及 wheel 文件名前缀使用以下固定形式：

| 发行包名（METADATA Name） | Wheel 文件名前缀 |
| --- | --- |
| `rx-met` | `rx_met-` |
| `quant-gru` | `quant_gru-` |
| `quant-lstm` | `quant_lstm-` |

wheel 格式将发行包名中的连字符转为下划线，各字段之间使用连字符，例如
`quant_gru-1.1.1+cpu-cp310-cp310-linux_x86_64.whl`。Python 导入继续使用
`quant_gru`、`quant_lstm` 等模块名。

发布校验同时检查包内发行名和标准 wheel 文件名。修改命名声明后，通过正常构建重新
生成 wheel 及应用包，使 METADATA、RECORD 和发布校验值对应同一份产物。
