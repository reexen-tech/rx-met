# CUDA / PyTorch 镜像兼容性矩阵

调研日期：2026-09-09；ONNX Runtime GPU 修订：2026-09-11

实施状态：本文的推荐矩阵已在 `docker/docker-bake.hcl`、一份参数化多阶段
Dockerfile 和变体依赖锁中落地。ONNX Runtime 已按 CUDA 变体切换为 GPU wheel；
完整三镜像构建、GPU 冒烟和最低驱动实机验收仍须在发布环境执行。

## 范围

本文为 rx-met 自有 Docker 镜像选择第一版 Linux x86_64 构建矩阵，前提如下：

- cu118/cu126 使用 Ubuntu 22.04 + CPython 3.10；
- cu130 使用 Ubuntu 24.04 + CPython 3.12，以满足 CUDA 13 ORT wheel 的 Python 要求；
- 宿主机安装 NVIDIA Container Toolkit；
- 每个 CUDA 变体产出一个独立镜像，不在单个镜像中安装多套 CUDA；
- 基础镜像使用 NVIDIA 官方 CUDA 镜像，PyTorch 使用官方 wheel。

在这个方案中，宿主机不需要安装与容器一致的 CUDA Toolkit。容器提供 CUDA
用户态库，宿主机提供 NVIDIA 驱动。因此，“CUDA 11.8 机器”应根据宿主机驱动
和 GPU 计算能力定义，不能只根据宿主机的 `nvcc --version` 输出判断。

## 推荐初始矩阵

以下矩阵同时满足 PyTorch 与 ONNX Runtime GPU 的官方 CUDA 兼容范围。

| 变体 | NVIDIA 构建镜像 | NVIDIA 最终基础镜像 | Python | PyTorch 组合 | ONNX Runtime GPU | Linux 严格最低驱动 |
| --- | --- | --- | --- | --- | --- | --- |
| `cu118` | `11.8.0-cudnn8-devel-ubuntu22.04` | `11.8.0-cudnn8-runtime-ubuntu22.04` | 3.10 | 2.7.1 / 0.22.1 / 2.7.1 | 1.20.1（CUDA 11.8 / cuDNN 8） | `520.61.05` |
| `cu126` | `12.6.3-devel-ubuntu22.04` | `12.6.3-base-ubuntu22.04` | 3.10 | 2.8.0 / 0.23.0 / 2.8.0 | 1.23.2（CUDA 12 / cuDNN 9） | `560.35.05` |
| `cu130` | `13.0.3-devel-ubuntu24.04` | `13.0.3-base-ubuntu24.04` | 3.12 | 2.10.0 / 0.25.0 / 2.10.0 | 1.27.0（CUDA 13 / cuDNN 9） | `580.126.20` |

上表驱动下限采用各 CUDA 发行版对应的严格版本，避免默认依赖 CUDA 小版本
兼容模式。NVIDIA 还给出了系列级小版本兼容下限：CUDA 11.x 为驱动 450，
CUDA 12.x 为驱动 525，CUDA 13.x 为驱动 580。较低的系列级下限存在功能和
PTX 限制，在实际驱动分支完成测试之前，不应作为默认客户兼容性承诺。

构建入口使用 NVIDIA 官方 NGC registry。核对时，上述 NGC tag 与 Docker Hub
上对应 `nvidia/cuda` tag 的 manifest list digest 一致；选择 NGC 只改变 registry
入口，不改变基础镜像内容。

### Torch 版本选择依据

- CUDA 11.8：PyTorch 2.7.1 是官方列表中最后一个提供 cu118 wheel 的版本。
  官方存在 CPython 3.10 x86_64 wheel，平台标签为 `manylinux_2_28`。
- CUDA 12.6：PyTorch 2.8.0 官方提供 cu126 wheel，并且最接近项目当前使用的
  Torch 接口。
- CUDA 13.0：PyTorch 2.9.0 是第一个提供 cu130 wheel 的官方版本。第一版镜像
  建议优先使用 PyTorch 2.10.0，因为它依赖 CUDA runtime 13.0.96，与 NVIDIA
  CUDA 13.0.3 镜像完全对应，并且有匹配的 torchvision 和 torchaudio 版本。

不存在同时覆盖 cu118、cu126 和 cu130 的单一官方 PyTorch 版本：cu118 支持
截止于 PyTorch 2.7.1，而 cu130 从 PyTorch 2.9.0 开始。如果强制三个 CUDA
变体使用同一个 Torch 版本，就需要自行从源码编译 PyTorch，不建议在第一版
方案中采用。

### cu130 备选组合

`torch==2.9.1`、`torchvision==0.24.1`、`torchaudio==2.9.1` 也是有效的
官方 cu130 组合，并且比 Torch 2.10 更接近当前 Torch 2.8 代码。该 wheel 使用
CUDA runtime 13.0.48，而 NVIDIA CUDA 13.0.3 镜像提供 13.0.96。应分别构建
并运行一次 rx-met/QuantGRU 验证；除非 Torch 2.10 出现明显接口回归，否则
优先选择 2.10.0。

## NVIDIA 镜像内容

NVIDIA 支持标签中，CUDA 11.8/12.6 提供 Ubuntu 22.04 变体，CUDA 13.0 提供
Ubuntu 24.04 变体；按需要选择 `base`、`runtime`、`devel` 或 `cudnn-*`。

NVIDIA 官方 Dockerfile 中的重要软件版本如下：

| CUDA 镜像 | cudart | cuBLAS | cuSPARSE | NCCL | `cudnn-*` 镜像中的 cuDNN |
| --- | --- | --- | --- | --- | --- |
| 11.8.0 | 11.8.89 | 11.11.3.6 | 11.7.5.86 | 2.15.5 | 8.9.6.50 |
| 12.6.3 | 12.6.77 | 12.6.4.1 | 12.5.4.2 | 2.23.4 | 9.5.1.17 |
| 13.0.3 | 13.0.96 | 13.1.1.3 | 12.6.3.3 | 2.28.3 | 9.14.0.64 |

PyTorch 与 ONNX Runtime 对 cuDNN 的要求并不完全相同：

- CUDA 11.8 镜像为 cuDNN 8.9.6，Torch 2.7.1 依赖 cuDNN 9.1.0；
- CUDA 12.6 镜像为 cuDNN 9.5.1，Torch 2.8.0 依赖 cuDNN 9.10.2；
- CUDA 13.0 镜像为 cuDNN 9.14.0，Torch 2.10.0 依赖 cuDNN 9.15.1。

cu126/cu130 的 builder 使用普通 `devel`，最终镜像使用普通 `base`，由 PyTorch
wheel 提供 cuDNN 9。cu118 是例外：ORT 1.20.1 CUDA 11 wheel 要求 cuDNN 8，故
使用 NVIDIA `cudnn8-devel/runtime`；PyTorch wheel 的 cuDNN 9 与系统 cuDNN 8
通过不同 soname 共存。构建过程必须为安装后的
`nvidia/*/lib` 目录设置稳定的动态库搜索路径；AIMET 编译需要的 `cudnn.h`
来自同一 Torch 组合锁定的 cuDNN wheel，不能使用版本不同的镜像内 headers。
最终通过 `ldd` 和真实 CUDA 冒烟测试进行验证。

如果最终阶段改用 NVIDIA `runtime` 镜像，会额外带入另一套 cuBLAS、
cuSPARSE 和 NCCL。除非测试证明较小的 `base` 镜像无法满足 rx-met 原生库
加载要求，否则不建议使用 `runtime`。

NVIDIA CUDA 标签存在支持周期，旧标签可能在新版本发布后被移除。正式发布
构建应解析并记录镜像 digest，在内部镜像仓库镜像这些基础镜像，并归档最终
rx-met 镜像。只有可变 tag 不能构成可复现的构建输入。

## 公共非 CUDA 软件版本

第一版镜像建议沿用项目已有的已知可运行版本，不要在切换镜像方案的同时升级
所有依赖：

| 软件 | 初始版本 |
| --- | --- |
| 操作系统 | cu118/cu126 为 Ubuntu 22.04；cu130 为 Ubuntu 24.04 |
| Python | cu118/cu126 为 CPython 3.10；cu130 为 CPython 3.12 |
| CMake / Ninja / GCC / Eigen | 对应 Ubuntu 官方软件包 |
| setuptools | 70.2.0 |
| wheel | 0.45.1 |
| pybind11 | 2.13.6 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| ONNX | 1.17.0 |
| ONNX Runtime GPU | cu118=1.20.1；cu126=1.23.2；cu130=1.27.0 |
| Pillow | 11.3.0 |
| onnxsim | 0.7.0 |
| tqdm | 4.67.1 |

ONNX Runtime GPU 必须按 CUDA 主版本分别锁定。cu118 从 Microsoft CUDA 11 专用
feed 获取 1.20.1；cu126 从 PyPI 获取 1.23.2；cu130 从 PyPI 获取 1.27.0。
AIMET custom-op 也必须用对应 ORT C API 头文件编译，不能让 1.20 runtime 加载按
1.23 API 编译的库。官方源码归档按 SHA-256 固定，cu126 复用仓库内 vendored
1.23.2 头文件。

以上版本只是第一版锁定候选，不是完整依赖集合。旧发布流程还固定了
`librosa==0.11.0` 和 `soundfile==0.13.1`，但当前两个 quick start 都没有
导入它们。除非 `quick_start_kws.py` 以外的受支持音频流程需要，否则不应
继续放入新镜像。

## 仓库依赖清单

实施前的 `requirements.txt` 是面向旧通用容器的 wheel 依赖声明，不是干净镜像的
完整环境定义。原文件注释明确指出 Docker 构建会重写依赖以匹配
旧通用容器（`requirements.txt:1-9`、
`scripts/internal/prepare_packaging.py:59-69`）。rx-met 开始自行维护镜像后，应从
正式支持的入口反推依赖并进行完整锁定。

### 打包和原生构建依赖

| 范围 | 依赖要求 | 代码证据 |
| --- | --- | --- |
| Python ABI | Linux x86_64 上的 CPython 3.10 / 3.12 | `pyproject.toml`、`setup.py` |
| Wheel 构建后端 | setuptools、wheel | `pyproject.toml:1-3` |
| AIMET 原生构建 | CMake >=3.19、C++17 编译器、目标 Python 开发 headers、pybind11、Eigen3；启用 CUDA 时还需要 CUDA Toolkit | `native/aimet/CMakeLists.txt:1-19` |
| QuantGRU 原生构建 | CMake、CUDA Toolkit/nvcc、OpenMP、cuBLAS、cudart | `quant-gru/CMakeLists.txt:39-40,108-109` |
| QuantGRU Python 扩展 | 构建扩展之前必须安装 CUDA 版 Torch | `quant-gru/pytorch/pyproject.toml:1-6`、`quant-gru/pytorch/setup.py:9-59` |
| Wheel 平台 | Linux x86_64、CPython 3.10 / 3.12 | `setup.py` |

编译器、headers、CMake、Ninja、pybind11 和 Python 开发包只应保留在 builder
阶段。最终镜像只需要 CPython、锁定后的 Python 环境、rx-met 和 QuantGRU
wheel、它们依赖的原生动态库、examples 以及验证元数据。

### 实施前声明的运行时依赖

根 wheel 当时声明了 NumPy、SciPy、ONNX、onnxsim、ONNX Runtime、Torch、
torchvision 和 Pillow（`requirements.txt:2-9`）。这不足以在干净环境完成
安装和运行，因为部分 vendored AIMET 模块在导入阶段还会直接导入其他软件包。

### KWS quick start

直接依赖包括 NumPy、Torch、torchaudio、SciPy、tqdm、rx-met/AIMET 和
QuantGRU（`examples/quick_start_kws.py:44-69`）。

该示例通过 `scipy.io.wavfile` 读取 WAV
（`examples/quick_start_kws.py:175-187`），通过
`torchaudio.functional.resample` 重采样
（`examples/quick_start_kws.py:245,255`），不直接使用 librosa 或
soundfile。

递归导入 AIMET 后至少还会增加以下依赖：

- `torchvision`：由 `aimet_torch.utils` 导入
  （`aimet_torch/utils.py:73`）；
- `PyYAML` 和 `packaging`：由 `aimet_torch.onnx_utils` 导入
  （`aimet_torch/onnx_utils.py:48-54`）；
- `safetensors`：由 QuantSim 基础实现导入
  （`aimet_torch/_base/quantsim.py:64-68`）；
- `tqdm` 和 `Bokeh`：在 `aimet_common.utils` 加载时导入
  （`aimet_common/utils.py:53-60`）；
- Bokeh 可视化模块：由 `aimet_torch.v2` 提前导入
  （`aimet_torch/v2/__init__.py:37-43`）；
- `jsonschema`：QuantSim 量化配置 JSON 校验时使用
  （`aimet_common/quantsim_config/json_config_importer.py:37-44`）。

因此，`packaging`、`PyYAML`、`safetensors`、`bokeh`、
`jsonschema` 和 `tqdm` 都是当前 KWS 导入链的镜像依赖，尽管其中大部分
未写入 `requirements.txt`。最终锁文件确定前，必须在干净环境中验证完整
导入链。

### ONNX PTQ quick start

入口脚本导入 rx-met ONNX PTQ facade
（`examples/onnx_ptq_quick_start.py:38-49`）。核心执行依赖为 NumPy、ONNX、
ONNX Runtime、packaging、tqdm、jsonschema，以及原生
`_libpymo`、`libquant_info`、`libaimet_onnxrt_ops` 动态库。pipeline
在 `aimet_onnx/rx_ptq/pipeline.py:159-172` 中创建 ONNX Runtime session。

发布示例默认要求 `CUDAExecutionProvider`；只有显式设置
`RX_MET_ONNX_PTQ_DEVICE=cpu` 时才使用 CPU。provider 缺失会直接报错，不静默回退。

导入 `aimet_onnx.rx_ptq` 时会先执行 `aimet_onnx/__init__.py`。当前
initializer 会提前导入 QuantSim、Adaround、Sequential MSE 和 QuantAnalyzer
（`aimet_onnx/__init__.py:47-55`）。该导入链还会在加载阶段需要 Torch、
scikit-learn、psutil、Bokeh 和 tqdm，例如
`aimet_onnx/quant_analyzer.py:45-52` 和
`aimet_onnx/adaround/adaround_optimizer.py:43`。除非把 initializer 改为
延迟导入，否则这些都应视为导入阶段的必需依赖。

调研时还发现一个与 CUDA 矩阵无关的示例/API 字段不一致问题：pipeline 返回
`aimet_encodings`，quick start 却读取 `rxmet_encodings`。本次实施已统一为
`aimet_encodings`，ONNX quick start 现已进入镜像验收门禁。

### ONNX PTQ 数据准备

`examples/prepare_onnx_ptq_data.py` 使用 NumPy 和 pyarrow 读取 Parquet
（`examples/prepare_onnx_ptq_data.py:34,42-50`），使用 Pillow 解码和缩放
图像（`examples/prepare_onnx_ptq_data.py:53-59,74-91`），并使用 ONNX 固定
模型 batch 维度（`examples/prepare_onnx_ptq_data.py:166-185`）。

OpenCV 只是在 Pillow 不可用时的备选方案
（`examples/prepare_onnx_ptq_data.py:61-71,98-107`），该脚本不需要 Torch
或 CUDA。

这个脚本可以作为独立的宿主机数据准备工具。如果产品要求它在发布镜像内运行，
则应添加固定版本的 pyarrow 和 Pillow；除非明确承诺支持备用解码路径，否则
不要同时安装 OpenCV。数据集和模型文件应通过 volume 挂载，不应打进镜像层。

### 可选的 vendored AIMET 功能

提交 `719fa371` 在对齐旧通用容器时从 `requirements.txt` 删除了 Jinja2、
PyYAML、hvplot、onnx2torch、osqp、pandas、psutil、scikit-learn、tqdm、
Bokeh 和 jsonschema。不能不加区分地把它们全部放回镜像：

| 范围 | 软件包 |
| --- | --- |
| 当前 quick start 和导入链 | packaging、PyYAML、safetensors、tqdm、Bokeh、jsonschema，以及前述直接依赖 |
| ONNX/Torch 特定功能 | 实验性 Torch 导出使用 onnxscript；实验性 AdaScale 使用 onnx2torch |
| 高级分析和优化 | scikit-learn、pandas、osqp、psutil、hvplot、Jinja2 |
| 示例和数据准备 | torchaudio、SciPy、pyarrow、Pillow |
| 两个已检查 quick start 均未使用 | librosa、soundfile |

最终运行时锁文件取决于镜像只承诺支持两个 quick start，还是承诺支持全部
vendored AIMET API。必须使用干净环境递归导入测试，才能把上表转化为最终
锁文件。

## Torch 2.8 是否是全局要求

不是。仓库历史表明，全局版本下限是为了对齐旧通用容器，而不是因为所有受支持
rx-met 路径同时开始使用 Torch 2.8 API。提交 `719fa371` 将根依赖声明从
`torch>=1.13.0` 改为 `torch>=2.8.0`，并在
`scripts/internal/prepare_packaging.py:59-69` 中加入旧通用容器专用依赖重写。

源码中唯一明确的 2.8 检查位于实验性 ExportedProgram API：
`aimet_torch/v2/experimental/export/__init__.py:22-26`。

KWS 导出器使用 `dynamo=False` 的 legacy `torch.onnx.export`
（`aimet_torch/rx_export/export_onnx_json.py:223-276`），现有 AIMET ONNX
wrapper 也拒绝 `dynamo=True`（`aimet_torch/onnx.py:297-300`）。

因此，Torch 2.7.1 是 cu118 镜像合理的候选版本，但尚未通过当前源码树的完整
兼容性验证。cu118 的产品契约应明确不支持实验性 ExportedProgram，并重点验证
常规 KWS、QuantGRU、legacy ONNX 导出和 AIMET v2 量化。

不能直接把全局 metadata 下限降下来就结束：需要使用变体专用 constraints 或
构建时生成的 wheel metadata，使 pip 接受 cu118 环境，同时保留明确的功能
边界。

## 可复现版本锁定策略

所有软件输入都可以并且应该在 Docker 构建阶段固定：

1. 将每个 NVIDIA `devel` 和 `base` tag 解析为 digest，并同时保存 tag 和
   digest，因为 tag 可变，而且旧 CUDA tag 可能被删除。
2. 维护一份公共 Python lock 和三份 CUDA 变体 constraints。每个变体固定精确
   的 Torch/torchvision/torchaudio 组合和对应的 `nvidia-*` 传递 wheel；
   build target 另外固定官方 PyTorch index URL。
3. 发布镜像使用精确的 `==` 版本和 wheel hash，不直接安装当前宽泛的
   `>=` requirements。
4. 基础设施允许时固定 Ubuntu APT snapshot；至少需要在镜像 SBOM 或构建
   manifest 中记录最终安装的系统软件版本。
5. 在镜像 label 或配套 release manifest 中记录 rx-met 源码 revision、构建
   wheel hash、基础镜像 digest、架构列表、编译器版本和最终 `pip freeze`。
6. 构建依赖和下载的源码归档只保留在 builder 阶段；最终阶段只复制已验证的
   wheel 和必需动态库。

公共 lock 可以避免三个镜像的非 CUDA 依赖无意漂移；三份 constraints 只表达
真实存在的 CUDA/Torch 差异。每个变体都应单独执行依赖解析，最终镜像必须通过
`pip check`。

## 环境镜像归档与复用

NVIDIA 基础镜像体积较大，正式构建不应每次从公网重复拉取。构建流程应增加
“本地 Docker、环境镜像归档、官方 registry”三级查找机制：

1. 先检查本地 Docker 中是否已有 manifest 指定的镜像 ID 和上游 digest；
2. 本地不存在时，从调用方提供的归档目录读取，先验证 SHA-256，再执行
   `docker load`；
3. 未提供归档时，才从 NVIDIA 官方 registry 拉取并构建环境镜像；
4. 首次拉取并验证后，在本地磁盘生成归档和清单，再按部署规范上传；
5. 后续构建设置 `--pull=false`，使用已经验证的本地镜像。

环境归档的维护应遵守部署环境的存储规范，建议至少满足以下约束：

- 避免在共享存储上直接执行 Docker 构建、解压镜像或高频写入；
- 所有归档先在本地磁盘生成并校验，再一次性上传；
- 上传阶段使用 `.partial` 后缀，同一目录只允许一个写入者；
- 校验成功后再原子重命名，消费者不得读取上传中的文件；
- 文件名只能使用 ASCII 字母、数字、点、下划线和连字符；
- 共享存储不是唯一备份，正式构建输入还需保留镜像仓库或其他可恢复副本；
- 归档第三方 NVIDIA 镜像前，必须确认其许可证允许团队内部存储和使用。

### 稳定环境镜像归档

首套稳定 GPU 环境版本为 `deps-v1`。环境与产品使用独立版本轴：requirements、
Torch、CUDA、Python、Ubuntu 或工具链发生变化时才递增环境版本；普通 AIMET、
QuantGRU 和产品 `VERSION` 变化不触发环境重建。

共享存储中的归档目录可以按以下结构组织：

```text
/path/to/environment-images/
`-- deps-v1/
    |-- rx-met-environment-deps-v1-cu118-linux-amd64.tar.zst
    |-- rx-met-environment-deps-v1-cu126-linux-amd64.tar.zst
    |-- rx-met-environment-deps-v1-cu130-linux-amd64.tar.zst
    |-- environment-manifest.json
    |-- README.md
    `-- SHA256SUMS
```

每个新归档包含同一 CUDA 变体的 build-env 和 runtime-env 两个 tag。逻辑上仍是
六个环境镜像，但归档目录只有三个镜像文件，且同一 Docker archive 可以复用公共
layer。产品镜像不进入环境归档。

### 发布时的环境解析

`release_build.sh` 不再负责安装第三方环境，只执行以下决策：

- 本机已有完整环境对时直接验证并复用；
- 通过 `--environment-dir` 或重复的 `--environment-archive` 提供归档时，调用独立
  加载器验证 SHA-256 并执行 `docker load`；
- 本机和归档都没有时，默认调用 `build_environment_images.sh`；
- 只存在 build/runtime 其中一个时拒绝继续，防止环境版本错配；
- 显式归档损坏或缺失时不回退公网构建。

环境构建、验证、加载和导出分别由独立脚本负责。导出脚本生成经过 `docker save`、
zstd 压缩和 SHA-256 校验的可搬运文件，但不自动上传；归档的复制或移动由维护人
手工执行，发布脚本不拥有共享目录写入职责。

## Dockerfile 结构决策

环境和产品使用两份职责独立的参数化 Dockerfile，但不按 CUDA 版本复制文件：

- `Dockerfile.environment` 只在依赖环境变化时执行，产出 build-env/runtime-env；
- `Dockerfile` 只处理当前仓库源码，产出三个产品镜像。

以下内容在三个 CUDA 变体中保持一致：

- APT 软件包和 builder 工具链；
- AIMET 和 QuantGRU 原生构建流程；
- Torch CUDA 组合以外的 Python 依赖；
- wheel 安装、examples、labels、entrypoint 和验证流程。

只有受控矩阵数据不同：

| 参数 | cu118 | cu126 | cu130 |
| --- | --- | --- | --- |
| NVIDIA devel/runtime 镜像 | 11.8.0 cuDNN 8 / Ubuntu 22.04 | 12.6.3 / Ubuntu 22.04 | 13.0.3 / Ubuntu 24.04 |
| Python | 3.10 | 3.10 | 3.12 |
| Torch/vision/audio 锁定版本 | 2.7.1 / 0.22.1 / 2.7.1 | 2.8.0 / 0.23.0 / 2.8.0 | 2.10.0 / 0.25.0 / 2.10.0 |
| ONNX Runtime GPU | 1.20.1 | 1.23.2 | 1.27.0 |
| CMake CUDA 架构 | `80;86;89;90` | `80;86;89;90` | `80;86;89;90;120` |
| Torch CUDA 架构列表 | `8.0;8.6;8.9;9.0` | `8.0;8.6;8.9;9.0` | `8.0;8.6;8.9;9.0;12.0` |
| 发布时声明的严格最低驱动 | 520.61.05 | 560.35.05 | 580.126.20 |

CUDA 13 已移除 Maxwell、Pascal 和 Volta 的离线编译器及库支持，因此 cu130
镜像不能声明支持 Turing 之前的 GPU。当前全局架构列表
`80;86;89;90;120` 也不能复用于 cu118 或 cu126，因为 CUDA 11.8 和 12.6
不能编译 `sm_120`。

应将上述映射放入一份机器可读的构建矩阵，例如 `docker-bake.hcl` 或受版本
控制的 shell/JSON 配置。构建过程必须拒绝 CUDA 11.8 + Torch 2.10 或 CUDA
11.8 + `sm_120` 这类无效组合。

最终产品仍是三个独立 tag，不制作一个包含三套 CUDA 的通用运行镜像。拆分依据
是环境与产品的生命周期不同，而不是 CUDA 版本不同。

## 已落地决策与待确认事项

- 第一版镜像以两个 quick start、它们的导入链和 ONNX 数据准备脚本为契约；
  不承诺全部实验性 vendored AIMET API。
- 固定架构列表前，需要确认具体支持的 GPU 型号，而不能只确认 CUDA 版本。
- cu130 第一版选择 Torch 2.10.0；仍需通过 GPU 实机测试确认，再决定是否回退
  到 2.9.1。
- 旧版基础镜像和 wheelhouse 缓存格式不与稳定环境镜像归档混用；环境归档生成后
  应在另一台服务器验证加载与发布流程。
- 对外声明使用严格发行版驱动下限；若要采用 CUDA 系列兼容下限，必须另行测试。
- `prepare_onnx_ptq_data.py` 已放入运行镜像，encoding 字段不一致已修复。

## 必需验收测试

每个变体都应执行以下测试：

1. 在对应 NVIDIA `devel` 镜像中构建 rx-met 原生模块和 QuantGRU，且构建时
   不要求可见 GPU。
2. 在最终镜像中校验 Python、Torch、torchvision、torchaudio、
   `torch.version.cuda` 和各软件包版本。
3. 对 `_libpymo`、`libquant_info`、`libaimet_onnxrt_ops`、
   `gru_interface_binding` 和 `libgru_quant_shared` 执行 `ldd`，拒绝
   存在未解析库或跨 CUDA 主版本链接。
4. 运行真实的 CUDA QuantGRU forward/backward 冒烟测试，不能只验证 import。
5. 运行最小 AIMET CUDA 量化操作和禁止 CPU 回退的 ONNX Runtime CUDA 推理。
6. 执行 `pip check`，并确认镜像中只安装目标 CUDA 主版本的软件包。
7. 在发布声明的最低驱动宿主机上验证镜像。只在现代 r580+ 驱动上测试三个
   镜像，不能证明较旧驱动兼容性声明成立。

## 一手资料

- [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)
- [PyTorch 官方 cu118 wheel 索引](https://download.pytorch.org/whl/cu118/torch/)
- [PyTorch 官方 cu126 wheel 索引](https://download.pytorch.org/whl/cu126/torch/)
- [PyTorch 官方 cu130 wheel 索引](https://download.pytorch.org/whl/cu130/torch/)
- [Torch 2.7.1 cu118 CPython 3.10 wheel metadata](https://download-r2.pytorch.org/whl/cu118/torch-2.7.1%2Bcu118-cp310-cp310-manylinux_2_28_x86_64.whl.metadata)
- [Torch 2.8.0 cu126 CPython 3.10 wheel metadata](https://download-r2.pytorch.org/whl/cu126/torch-2.8.0%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl.metadata)
- [Torch 2.10.0 cu130 CPython 3.12 wheel metadata](https://download-r2.pytorch.org/whl/cu130/torch-2.10.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl.metadata)
- [ONNX Runtime CUDA Execution Provider 兼容矩阵](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime CUDA 11 官方 Python feed](https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/)
- [ONNX Runtime GPU 1.23.2](https://pypi.org/project/onnxruntime-gpu/1.23.2/)
- [ONNX Runtime GPU 1.27.0](https://pypi.org/project/onnxruntime-gpu/1.27.0/)
- [NVIDIA CUDA 容器支持标签](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/doc/supported-tags.md)
- [NVIDIA CUDA 容器源码](https://gitlab.com/nvidia/container-images/cuda)
- [NVIDIA CUDA 小版本兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [CUDA 11.8 发行说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html)
- [CUDA 12.6 Update 3 发行说明](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html)
- [CUDA 13.0 Update 3 发行说明](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html)
