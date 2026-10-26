# 公共 CPU/GPU 环境

rx-met 与 rx-lm-quant 各自维护公共环境定义和脚本，不读取另一个仓库的文件。
两个产品通过版本标签复用兼容环境，环境版本独立于产品 VERSION。

## 版本与复用

公共环境名为 `rx-quant-env`，当前环境版本为 `v1`，支持 cpu、cu118、cu126、cu130。
每个变体提供两个标签：

- `rx-quant-env-build:v1-<变体>`：编译器、开发依赖和 ORT 构建头文件。
- `rx-quant-env-runtime:v1-<变体>`：运行依赖，供已安装的应用挂载运行。

`docker/variants.json` 定义 Python、CUDA、Torch、ONNX Runtime 和精确依赖。
环境构建入口优先复用已有版本标签，缺少对应角色的镜像时才构建。
消费环境时检查环境名称、版本、build/runtime 角色、CPU/CUDA 变体和 linux/amd64 平台；
环境验收还检查实际包版本、导入、pip check 和可选 GPU 运算。

镜像 ID 和构建来源摘要用于追踪，不作为跨仓库复用的前提。
两边分别构建的镜像 ID 可以不同，也不需要同步镜像身份记录。
目录移动、文档或构建脚本调整不会使已有兼容镜像失效。
依赖、ABI 或工具链的兼容性要求变化时使用新的环境版本，并分别验收两个产品。

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

## 离线导出与加载

```bash
./scripts/environment/export_runtime.sh cu126
./scripts/environment/load.sh --directory /path/to/rx-quant-env-v1 cu126
```

导出和加载省略变体时同样仅处理 CUDA；CPU 使用显式参数，例如
`./scripts/environment/export_runtime.sh cpu`。

导出默认写入 `.release/runtime-images/v1/`，包含带版本标签的 runtime 归档、
独立的 `.sha256` 校验文件、用于追踪的 `.json` 元数据和目录清单。
这些文件属于构建输出，不写回仓库配置。

导入不要求 `.json` 身份记录。存在文件校验清单时检查其 SHA-256；
Docker 负责加载镜像内容，导入后检查实际镜像的环境元数据。
已有兼容版本标签时直接复用。
旧的无标签归档可以根据本次 docker load 返回的镜像引用验证并补齐版本标签。

产品发布使用 `--with-env` 时，将 runtime 归档放入产品目录平级的
`<环境名>-<环境版本>/`，两个产品可以复用该目录。
内部构建还可以通过 `--build-env-dir DIR` 与 `--runtime-env-dir DIR` 导入环境。
用户通常直接运行应用包 install.sh；应用包要求环境版本标签，不锁定某次构建的镜像 ID。

已有归档经过完整性检查后复用，损坏归档报错，不自动覆盖。
兼容环境的重新导出文件可以继续用于同一应用包，归档文件校验值独立维护。
完整交付命令见 [应用包发布](Release_packaging.md)。

## 产品验收

环境验证不能替代产品验收：rx-met 验证 ONNX PTQ、GRU/LSTM；
LM 验证真实模型转换、量化、imatrix/PPL 和支持的算法路径。
两个产品均需检查离线安装、非 root、只读挂载、重复安装、失败恢复和错误环境拒绝。
