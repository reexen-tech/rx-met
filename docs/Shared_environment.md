# 公共 CPU/GPU 环境

rx-met 的 CPU/GPU 环境定义和构建脚本由本仓库维护，通过版本标签复用兼容镜像。
环境版本独立于产品 VERSION。

## 版本与复用

公共环境名为 `rx-quant-env`，当前环境版本为固定字符串 `202610`，支持 cpu、cu118、cu126、cu130。
环境版本在 `docker/variants.json` 的 `environment_version` 中统一维护，格式为 `YYYYMM`。
它表示环境定义发布的月份，重复构建或跨月构建都保持不变；产品或算子升级也不改变环境版本。
调整基础镜像、Python、CUDA、PyTorch、ORT 等环境定义后，再改为调整当月的版本。

公共 CPU 环境统一使用 PyTorch `2.9.1+cpu`、torchvision `0.24.1+cpu` 和
torchaudio `2.9.1+cpu`，ONNX Runtime 保持 `1.23.2`。该组合已完成 rx-met
三个 wheel 的源码构建和安装，并通过 42 项 CPU PTQ/ONNX 测试。
切换 Torch 版本后必须从源码重新编译原生扩展，不能复用旧 Torch 构建的 wheel。
每个变体提供两个标签：

- `rx-quant-env-build:202610-<变体>`：编译器、开发依赖和 ORT 构建头文件。
- `rx-quant-env-runtime:202610-<变体>`：运行依赖，供已安装的应用挂载运行。

`docker/variants.json` 定义 Python、CUDA、Torch、ONNX Runtime 和精确依赖。
环境构建入口优先复用已有版本标签，缺少对应角色的镜像时才构建。
消费环境时检查环境名称、版本、build/runtime 角色、CPU/CUDA 变体和 linux/amd64 平台；
环境验收还检查实际包版本、导入、pip check 和可选 GPU 运算。

镜像 ID 和构建来源摘要用于追踪，不作为兼容镜像复用的前提。
重新构建的镜像 ID 可以不同，兼容性由环境元数据和实际依赖验证。
目录移动、文档或构建脚本调整不会使已有兼容镜像失效。
一个月份版本对应一套固定的环境定义。依赖、ABI 或工具链变化时使用新的环境版本，
并重新完成环境与应用包验收。真实构建时间、来源提交和镜像 ID/摘要独立记录，环境版本不从当前时间生成。

## 构建材料

ORT 离线头文件位于 `docker/environment/third_party/`：

- `onnxruntime/`：固定的 1.23.2 头文件。
- `onnxruntime-prepared/`：1.20.1 和 1.27.0 的已准备头文件。

每份材料保留来源、许可证、版本和 SHA256SUMS，准备时检查完整性。
rx-met 本地 AIMET 编译也使用这里的 1.23.2 头文件，支持通过
`RX_MET_ONNXRUNTIME_ROOT` 指定其他目录。
公共 build 镜像统一把所需头文件准备到 `/opt/rx-quant-env/onnxruntime`。

镜像构建使用 `.release/environment-context/<环境版本>/<变体>/` 下的独立上下文，
不把产品 VERSION、pyproject 或业务源码装入公共环境。
构建来源摘要可通过下面的命令查看，它不参与已有环境的兼容性判断：

```bash
python3 scripts/environment/common.py digest
```

## 构建与验收

```bash
./scripts/environment/build.sh --gpu cu126
./scripts/environment/verify.sh --gpu cu126
```

未指定变体时仅处理 `cu118`、`cu126`、`cu130`。CPU 需显式指定：

```bash
./scripts/environment/build.sh cpu
./scripts/environment/verify.sh cpu
```

`--wheelhouse-dir DIR` 可以指定本地依赖缓存，按 cpu/cu118/cu126/cu130 分目录。
缓存中的 wheel 必须与依赖定义匹配。build 不自动导出镜像。

暂存依赖时，`onnxsim==0.7.0` 的已核对源码会应用退出引用修复：将
`_model_executor` 的类型注解延迟为字符串，避免 `typing` 缓存间接保留原生 executor。
该修复由 `scripts/environment/patch_onnxsim_wheel.py` 检查源文件 SHA-256 后执行，
为 wheel 添加 `1rxmet` build tag，重写 `WHEEL` / `RECORD`，并更新依赖缓存校验清单；
包版本和原生扩展保持不变。未知源码会报错，不能静默套用补丁。

Torch `2.10.0` / `2.10.0+cu130`、`2.8.0` / `2.8.0+cu126` 的已核对源码还会应用
FX 生成源码缓存的生命周期修复。自动生成的唯一文件名由编译代码对象持有，最后一个代码对象
及其 traceback 释放后，才回收对应源码；仍存活的函数、生成器和 traceback 可以继续查阅源码。
显式文件名和 profiler 内容哈希保留 Torch 原有缓存语义。fork 子进程保守保留继承的源码快照，
子进程新编译的源码继续正常回收，避免在 fork 前引入跨组件锁顺序。
`scripts/environment/patch_torch_wheel.py` 同样校验源文件和修复模块的 SHA-256，添加
`1rxmet` build tag 并重写 wheel 元数据，不修改原生库。其他 Torch 版本明确报告未覆盖，
不因此宣称已经解决这些版本的 FX 缓存增长。

环境验收在独立进程中按 Torch、onnxsim 的顺序导入，执行三次真实常量折叠并检查
ONNX 和 ORT 输出。等待解释器正常退出后，还会检查完整 stderr 中的 nanobind 泄漏告警；
退出码为 0 不足以通过此项检查。
受覆盖的 Torch 版本还检查 FX 缓存随重复追踪释放，以及存活代码和 traceback 的源码可用性。
已有镜像的复用规则不变，源码修改不会修补现有镜像；受影响的旧镜像会在新增验收中报错。
发布时须按环境版本规则构建并验收修复后的环境，不能用产品运行时修改依赖私有状态代替。

## 离线导出与加载

```bash
./scripts/environment/export_runtime.sh cu126
./scripts/environment/load.sh --directory /path/to/rx-quant-env-202610 cu126

# 开发构建环境可独立导出、加载；all 同时处理 build 和 runtime。
./scripts/environment/export.sh --role all --output-dir /path/to/rx-quant-env-202610 cu126
./scripts/environment/load.sh --role all --directory /path/to/rx-quant-env-202610 cu126
```

导出和加载省略变体时同样仅处理 CUDA；CPU 使用显式参数，例如
`./scripts/environment/export_runtime.sh cpu`。

导出默认写入 `.release/runtime-images/202610/`，`--role` 默认值为 `runtime`。
Docker 归档名称为 `rx-quant-env-<角色>-202610-<变体>-linux-amd64.docker.tar.gz`，
例如 `rx-quant-env-build-202610-cu126-linux-amd64.docker.tar.gz`。
独立校验和元数据文件分别在完整归档名后追加 `.sha256` 和 `.json`；目录清单保留各变体、各角色的记录。
产品应用归档继续使用 `.tar.gz`，Docker 镜像归档使用 `.docker.tar.gz`，加载方式为 `docker load`。
这些文件属于构建输出，不写回仓库配置。

导入不要求 `.json` 身份记录。存在文件校验清单时检查其 SHA-256；
Docker 负责加载镜像内容，导入后检查实际镜像的环境元数据。
已有兼容版本标签时直接复用。
旧的无标签归档可以根据本次 docker load 返回的镜像引用验证并补齐版本标签。

产品发布使用 `--with-env` 时，将 runtime 归档放入产品目录平级的
`<环境名>-<环境版本>/`，后续应用发布可以复用该目录。
内部构建还可以通过 `--build-env-dir DIR` 与 `--runtime-env-dir DIR` 导入环境。
用户通常直接运行应用包 install.sh；应用包要求环境版本标签，不锁定某次构建的镜像 ID。

已有归档经过完整性检查后复用，损坏归档报错，不自动覆盖。
兼容环境的重新导出文件可以继续用于同一应用包，归档文件校验值独立维护。
旧 `v1` 镜像和归档保留；新版本通过正常构建和导出生成，使标签、内部环境元数据和归档名一致。
不能仅将旧镜像重新打标签作为新环境版本。
完整交付命令见 [应用包发布](Release_packaging.md)。

## 产品验收

环境验证完成后，还需验收 rx-met 的 ONNX PTQ 和 GRU/LSTM 流程。
应用包验收需检查离线安装、非 root、只读挂载、重复安装、失败恢复和错误环境拒绝。
