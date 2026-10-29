# 测试

项目级回归使用标准库 `unittest`，统一入口是 `python3 -m tests`，不新增 pytest 依赖。
算子仓库内部及原生 C++ 专项测试仍由各自目录维护。

## 目录与职责

| 目录 | 覆盖内容 |
| --- | --- |
| `common/` | 公共整数网格、编译器编码协议、运行记录和测试入口 |
| `torch/` | FX 准备、可量化模块、数值稳定性及 Torch 导出 |
| `onnx/` | 配置、数据/控制端口、校准输入、bias、Po2、导出与恢复 |
| `recurrent/` | GRU/LSTM 梯度、状态、存储及原生编码导出 |
| `integration/` | 跨框架整数域、Percentile、Po2 工作流和日志 |
| `examples/` | 小模型示例流程、数据准备及验证工具 |
| `release/` | 依赖、环境、wheel 修复、离线构建、安装与交付 |
| `support/` | 公共模型、输入夹具、编码断言与路径，不收集测试 |

顶层只保留包入口和本说明。测试用例不再从另一个测试模块导入测试类充当夹具。
公共路径从 `tests.support.paths` 读取；入口同时为子进程配置相同的源码路径。

## 运行

在仓库根目录执行。先查看测试模块，列举操作不会导入 Torch 或原生库：

```bash
python3 -m tests --list
python3 -m tests onnx recurrent --list
```

选择一个或多个测试组：

```bash
python3 -m tests common release
python3 -m tests onnx recurrent -v
python3 -m tests examples -k kws
python3 -m tests torch --failfast
```

不指定组或指定 `all` 时运行全部自动化测试：

```bash
python3 -m tests
```

`-k` 按完整测试名称匹配，可重复传入；未匹配任何用例会返回非零退出码。
失败或导入错误同样返回非零，不会自动忽略。原生 CPU/GPU 能力要求由测试自身声明并报告跳过。

也可使用标准 unittest 指定模块、类或方法：

```bash
python3 -m unittest -v tests.onnx.test_input_quantization
python3 -m unittest -v tests.examples.test_kws_example.KwsExampleConfigurationTest
python3 -m unittest discover -s tests -t . -v
```

使用包入口，不要将 `tests/` 加入 `PYTHONPATH`，以免其中的 `torch/`、`onnx/`
或 `examples/` 被误当作产品模块。

## 环境

`common` 和 `release` 不需要 GPU，也不运行真实训练、下载或 Docker 构建。
其余组应在已安装 RX-MET、QuantGRU、QuantLSTM 及匹配依赖的 runtime 中执行。
CPU/GPU 使用同一入口，GPU 全量回归需要匹配的 CUDA wheel 和可用设备。

测试源码应与所加载的原生扩展版本匹配。普通 Python 检查通过不代表原生运行库、
CUDA backward 或设备精度已被验证；报告须同时记录运行环境、失败和跳过项。

## 真实数据验收

需要 Speech Commands、浮点检查点和 CUDA 的往返流程不属于自动化测试收集，
保留在 `scripts/validation/verify_kws_ptq_roundtrip.py`：

```bash
python3 scripts/validation/verify_kws_ptq_roundtrip.py \
  --rnn-type gru \
  --fp-checkpoint /path/to/model_fp_kws.pth \
  --data-dir /datasets/speech_commands_v0.02
```

该命令固定波形和标签，提取外置 MFCC，执行 Torch PTQ/Po2、ONNX 再校准和恢复。
两次均不执行 QAT，检查量化前后浮点权重不变。需要原生算子的 ORT 桥接扩展。

输出包含 ONNX、JSON、逐样本 logits、`comparison.json` 和摘要。
差异会如实记录并返回 1；`--report-only` 只改变退出策略，不把差异标为通过。
已有输出目录拒绝覆盖。两轮产物、校准网格或浮点累加不同，不应仅凭最终精度相近认定逐值一致。

KWS 专用产物检查可单独执行：

```bash
python3 -m tests.support.kws_artifacts \
  --onnx /path/to/model.onnx --encodings /path/to/model.json
```

这只检查已整理的静态图和编码规则，不代替编译或设备推理。
发布镜像中的真实数据验收继续使用 `scripts/release/verify_*_example.sh`。

## 整理与维护

- 保留修复之后仍有回归价值的测试，不按提交时间或文件年龄删除。
- 数值非有限值、Log 零码和奇异梯度合入 `torch/test_numerical_stability.py`。
- GRU 截断链式法则、eval backward 和零点梯度合入 `recurrent/test_gru_autograd.py`。
- ONNX 输入精度、控制端口及 bias 分别按行为合并；原 `alignment/remaining` 杂项按配置、
  bias、Po2、校准输入和恢复拆分。
- KWS 卷积和 MFCC 精度回归合入示例测试；编译器数学等价与导出测试独立保留。
- 已删除只有自测调用的历史报告路径解析工具及其 6 个自测，保留真实往返报告的编码比较测试。

新增长期用例放入对应测试组，复用 `support` 中的普通辅助对象并用 `addCleanup`
管理临时资源。一次性排查脚本、模型、真实数据及运行产物不提交到本目录；
可复用的手工验收工具放入 `scripts/validation`。
