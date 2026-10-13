# rx-quant 公共 GPU 环境

rx-met 与 rx-lm-quant 各自维护环境定义和脚本，不读取另一个仓库的文件。
两个仓库引用同一批已发布镜像，公共环境版本独立于产品 VERSION。

| 变体 | 构建镜像 | 运行镜像 |
| --- | --- | --- |
| cu118 | rx-quant-env-build:v1-cu118 | rx-quant-env-runtime:v1-cu118 |
| cu126 | rx-quant-env-build:v1-cu126 | rx-quant-env-runtime:v1-cu126 |
| cu130 | rx-quant-env-build:v1-cu130 | rx-quant-env-runtime:v1-cu130 |

## 环境内容与身份

`docker/variants.json` 的 environment_name、environment_version、build_packages、common_packages 和
variants 定义公共 GPU 环境。CPU 专用字段和产品 VERSION 不参与公共环境摘要。
运行依赖包括两个产品使用的第三方库；编译器、CUDA Toolkit、ORT 头文件等位于 build 镜像。
两个产品的代码、定制 llama.cpp、算子和示例由各自应用包交付。

`docker/environment-images.json` 保存发布环境的公共配置摘要、各变体的 build/runtime 镜像 ID、
runtime 归档名称和 SHA-256。产品打包核对本地配置、镜像标签、精确镜像身份和运行依赖。
安装器按镜像 ID 复用或导入环境，不依赖 latest 或可变标签。

公共摘要包含环境定义、Dockerfile、构建辅助脚本和 vendored ORT 头文件。对比两仓库输出：

```bash
python3 scripts/environment/common.py digest
```

公共部分发生变化时，两仓库分别同步文件，并核对摘要。只有同一摘要和同一份发布身份记录，
才能作为同一个公共环境版本消费；不同机器分别构建出的同名镜像不会自动视为同一份发布环境。

## 准备新环境版本

初次发布公共环境时，在选定的一个仓库完成构建与验收：

```bash
./scripts/environment/build.sh --gpu cu126
./scripts/environment/export_runtime.sh cu126
```

环境构建使用 `.release/environment-context/<环境版本>/<变体>/` 下的独立上下文，
不把产品 VERSION、pyproject 或产品源码装入公共环境。可用 `--wheelhouse-dir DIR` 指定本地
wheel 缓存根目录，其下按 cu118/cu126/cu130 分目录；脚本校验 wheel 与公共 lock 精确匹配。
ORT 1.20.1/1.27.0 所需头文件保存在 native/third_party/onnxruntime-prepared，
从已验证的旧环境原样恢复，附来源、许可证和 SHA256SUMS；构建时校验缓存，不重复下载大型源码归档。

导出默认写入 `.release/runtime-images/v1/`，同时创建或补全 `docker/environment-images.json`。
将该身份记录独立保存到另一个仓库，在同一批镜像上完成两个产品的验收。正式发布前需完成全部目标变体。
已发布版本禁止覆盖；现有镜像和归档不匹配时直接报错。需要升级依赖或工具链时，使用新环境版本
及对应的新身份记录，保留旧版本。升级前先将当前 docker/environment-images.json
归档为 docker/environment-records/<旧环境版本>.json，再构建和导出新版本，生成新的当前记录。已有发布记录时，环境 build 复用该记录的精确镜像；
其他构建机器应先导入或拉取这批内部 build 镜像，不能用本地重建结果覆盖发布身份。

## 导出、加载和共享

只交付 runtime，build 镜像供内部构建使用。runtime 归档采用 gzip，每个变体一份：

```text
images/
├── rx-quant-env-runtime-v1-cu118-linux-amd64.tar.gz
├── rx-quant-env-runtime-v1-cu126-linux-amd64.tar.gz
├── rx-quant-env-runtime-v1-cu130-linux-amd64.tar.gz
├── 各归档的 .json 身份记录
├── manifest.json
├── README.md
└── SHA256SUMS
```

首次离线交付可将 images/ 放在应用包旁；两个工具复用同一份归档。
加载匹配本仓库发布记录的 runtime：

```bash
./scripts/environment/load.sh --directory /path/to/images cu126
```

用户通常直接运行应用包 install.sh，由安装器自动完成上述检查和导入。
正式下载地址未配置时使用离线归档；以后可在产品发布时配置固定 HTTPS 地址。

## 验收边界

环境验证检查精确包版本、pip check、Torch/ONNX Runtime 导入和可选 GPU 运算。
环境检查不能替代产品验收：rx-met 运行 ONNX PTQ、GRU/LSTM；LM 运行真实模型转换、量化、imatrix/PPL
及支持的算法路径。两个产品都需验证离线安装、非 root、只读挂载、重复安装和错误身份拒绝。
