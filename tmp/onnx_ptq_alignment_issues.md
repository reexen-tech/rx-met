# ONNX PTQ 与 Torch 示例的待修复问题

日期：2026-10-20。分析基线：`0446ccd`。

核对入口：`examples/quick_start_kws.py` 与 `examples/quick_start_onnx_ptq.py`。
前者已有较多使用验证，但不将旧实现的所有行为直接视为正确规范。
第一轮修复第 1 至 4 项并完成验证；本轮继续修复第 5 至 8 项。
第 9 项的完整验证集评估和第 10 项保留为后续工作。

## 修复进度

| 编号 | 优先级 | 问题 | 状态 |
| --- | --- | --- | --- |
| 1 | P1 | 输入配置覆盖权重和 bias 配置 | 已修复，回归通过 |
| 2 | P1 | bias 对齐缺少范围检查，覆盖用户禁用意图 | 已修复，回归通过 |
| 3 | P1 | 无效 scale 被关闭后仍可能通过 Po2 和覆盖率检查 | 已修复，回归通过 |
| 4 | P1 | 重新加载验证没有恢复 encodings | 已修复，真实模型验证通过 |
| 5 | P2 | 非对称 Po2 的整数 offset 与 min/max 不一致 | 待后续修复 |
| 6 | P2 | 相邻层共用 ONNX 张量量化器，配置冲突静默覆盖 | 待后续修复 |
| 7 | P2 | JSON2 类型映射、字段支持和未知类型检查不完整 | 待后续修复 |
| 8 | P2 | 校准输入全部强制转换成 float32 | 待后续修复 |
| 9 | P2 | 只评估首个校准样本，NaN 可能被当作验证成功 | 已拒绝 NaN/Inf；完整验证集待后续处理 |
| 10 | P2 | Torch 的 default_bitwidth 未转换成实际读取的字段 | 待后续修复 |

## 1. 输入配置覆盖参数配置

- 位置：`src/aimet_onnx/rx_ptq/mixed_precision.py` 的 `_quantizers_for_op` 和 `_apply_layer_config`。
- 原因：遍历 `op.inputs` 时把权重和 bias initializer 也收进了激活输入；先设置参数配置，再被输入配置覆盖。
- 容器复现：Gemm 设置 weight=16-bit symmetric、bias=32-bit symmetric、input=8-bit asymmetric，最终三者都是 8-bit asymmetric。相同 Torch Linear 配置保持正确。
- 默认全 8-bit、对称配置掩盖了其中一部分影响。
- 修复：适配层明确区分参数与激活，并保留激活输入槽位，避免缺少 quantizer 时改变 `input_bitwidths` 的索引意义。
- 验收：真实 Gemm、MatMul+Add 和多输入算子上，输入设置不改变参数设置，禁用量化仍能覆盖该层参数。
- 本轮实现：参数定位先于激活输入定位，参数不进入激活列表，缺少 quantizer 的激活槽保留为 None。
- 新发现并修复：底层 `_get_matmul_add_bias_idx` 能找到 bias 位置，但 ConnectedGraph 原来没有登记 Add 的 bias 参数，导致 `_get_weight_and_bias` 仍返回 None。已补齐登记，真实 MatMul+Add 的配置和恢复测试通过。

## 2. bias 对齐缺少安全检查和禁用状态区分

- 位置：`src/aimet_onnx/rx_ptq/po2.py` 的 `disable_onnx_bias_quantizers`、`align_onnx_bias_scale`。
- 范围复现：bias=0.5，Sx*Sw 约 6.82e-7，需要整数码约 733136，超过 INT16 的 32767；ONNX 接受，Torch 拒绝。
- 状态复现：明确禁用 b1，校准临时关闭列表只包含 b2；对齐后 b1 仍被重新启用。
- 其它差异：ONNX 不验证 bias 位宽 2..32、不拒绝多通道输入 scale、不校验输出通道数量，写回时没有统一强制 signed symmetric；逐层修改也可能在后续失败时留下部分修改。
- 修复：共用位宽、scale 和整数可表示性检查；框架适配读取真实 bias 与 scale。记录临时关闭集合，只恢复集合中的 bias；全部候选验证通过后再写回。
- 验收：INT16/INT32 溢出拒绝，非法位宽/NaN/Inf/不匹配通道拒绝；用户禁用保持禁用；零 bias 可用；后层检查失败时前层 bias 不改变。
- 本轮实现：`aimet_common.power_of_2.validate_bias_bitwidth/validate_bias_values` 供两框架共用；ONNX 记录临时关闭的 bias 名称，检查真实 initializer 的可表示性，暂存全部候选，最后写回 signed symmetric 网格。
- 验证包含 JSON1 明确禁用 bias 的配置，以及不执行 bias 对齐时的完整通用 PTQ 流程。

## 3. 无效 scale 被静默关闭，覆盖率无法发现丢失

- 位置：`po2.py:align_sim_encodings_to_po2` 和 `compiler_encodings.py:CompilerEncodingConverter.convert`。
- 容器复现：将必需权重 w 的 scale 设为 0，Po2 关闭 w，`all_power_of_2=True`；导出缺少 w 的 encoding，覆盖率仍为 2/2。
- 原因：Po2 将异常状态转成 disabled；导出仅检查仍启用的量化器能否映射，没有验证异常前的量化要求。
- 修复：无效 scale 直接报错，预检全部启用的整数 encoding 后再修改；导出验证量化器状态未意外丢失。显式禁用与合法图优化保持原有语义。
- 验收：0、负数、NaN、Inf 和未初始化 encoding 均失败且不关闭量化器；必需 encoding 缺失时不能报告覆盖成功；正常模型可以导出。
- 本轮实现：新增 ONNX `validation.validate_quantizer_state`，供 Po2、导出和恢复共用；Po2 先检查全部启用的整数 encoding，导出前检查状态快照和待恢复 bias，失败时不写产物。
- `coverage.enabled_quantizer_names` 记录实际启用集合；后续意外禁用或删除其中量化器时拒绝导出。该检查基于量化状态要求，不将所有 initializer 都强制量化，显式禁用仍合法。

## 4. 重新加载验证只验证了浮点图

- 位置：`examples/quick_start_onnx_ptq.py` 步骤 8；通用 `pipeline.run_onnx_ptq` 也只有 clean ONNX 对比。
- 当前行为：导出 clean ONNX 后与原始 FP32 对比，没有重建 QuantSim 和恢复 encoding；encoding 损坏、缺失或偏移错误均可能漏检。
- Torch 行为：重建模型和 QuantSim，恢复权重与 encoding，对比保存前后的量化输出，并比较完整测试集精度。
- 修复：重建 clean ONNX QuantSim，加载保存的原生 encoding，冻结恢复值；比较导出前后的量化输出。重新导出 compiler encoding 并与已保存的 sidecar 比较，避免只验证原生文件而漏掉部署文件。
- 验收：正常产物恢复一致；原生 encoding 和 compiler sidecar 的缺失/篡改都被拒绝；验证不依赖导出前 sim 的对象状态。clean ONNX 浮点图验证仍保留。
- 本轮实现：`pipeline.verify_onnx_ptq_artifacts` 从 clean ONNX 重建独立 sim，加载并冻结原生 encoding，检查启用集合，重新生成 compiler payload 并与已保存 sidecar 比较，再检查量化输出。示例和通用 `run_onnx_ptq` 共用此实现。
- clean ONNX 不一致由警告改为错误；新 metadata 字段为 `restored_quantized_vs_pre_export`。恢复临时目录在验证后自动清理。

## 5. 非对称 Po2 网格不一致

- 位置：`po2.py:align_encoding_values`。
- 原因：公共范围计算得到小数 offset，ONNX 将 offset 取整后未同步重算 min/max。
- 复现：旧 scale=0.01、offset=-13、min=-0.13、max=2.42；Po2 后 scale=0.015625、offset=-8，却仍写 min=-0.13，整数网格应为 -0.125。Torch 的有效网格为 -0.125。
- 后续方案：统一整数网格数值规则，包括 strict/unsigned symmetric；不直接复制当前标量公式。

## 6. 相邻层配置冲突

- 位置：`mixed_precision.py:_apply_layer_config`。
- 复现：前层输出要求 16-bit、后层输入要求 8-bit，两者实际引用同一 quantizer，最终为 8-bit。
- 后续方案：在 ONNX 适配层检测共享张量上的请求冲突，明确报错或定义插入重定量化的支持方式；不可依靠遍历顺序决定。

## 7. JSON2 支持和错误检查不完整

- `QuantizedLinear` 同时映射 Gemm 和 MatMul，但 `QuantizedMatMul` 没有映射，专门配置被忽略。
- BatchNorm、Softmax、Log、Clamp 等部分类型映射缺失；ONNX 忽略 `param_bitwidth/param_symmetric`。
- ONNX 未区分“已注册但当前图没有的类型”和“拼写错误的未知类型”；Torch 对后者报错。
- `per_channel_quantization` 在 Torch 文档中宣称支持，但 JSON2 的两个实现均没有处理，是共同缺口。
- 名称匹配优先级当前实现相同：精确名称 > 类型 > 通配符 > 默认。Torch 部分说明与实现不一致，不能据此认定两框架优先级不同。
- `QuantizedPad` 的整数形状操作与 ONNX 空间 Pad 不等价，不应机械映射。
- 后续方案：共用配置解析、规范化和校验；类型识别和参数/激活定位保留框架适配。

## 8. 校准输入 dtype 丢失

- 位置：`calibration.py:_validate_feed`。
- 复现：INT64 输入 `[1,2]` 被转换成 float32，不能正确用于 Gather 等索引输入；BOOL/FLOAT16 也受影响。
- 后续方案：读取 ONNX 输入的 shape 和 dtype，依据声明验证或转换；静态 shape 限制属于当前明确支持范围。
- 额外风险：全部校准数据一次性读入内存，多输入 NPY 配对仍含 IMU/audio 特例；后续按需求扩展。

## 9. 评估覆盖和非有限值

- 入口仅使用 `feeds[0]`，未使用准备脚本产出的 200 个 val 样本，也没有任务精度指标。
- `pipeline.compare_outputs` 不拒绝 NaN/Inf；NaN RMSE 与 0 的大小比较为 False，旧示例可能打印通过。
- 后续方案：通用比较器检查有限值与误差容差；示例按任务实现验证集精度，ImageNette 需要正确处理标签与 ImageNet 输出的映射。
- 第 4 项恢复验证必须拒绝 NaN/Inf，因此这部分会随本轮修复提前处理；完整验证集评估仍留待后续。

## 10. Torch 默认位宽不生效

- 位置：`aimet_torch/utils_rx.py`，默认分支直接使用 `default_bitwidth`，后续读取的却是 `weight_bitwidth/output_bitwidth`。
- 复现：初始 8-bit，JSON 仅指定 `default_bitwidth={weight:16,output:16}`；Torch 仍为 8，ONNX 为 16。
- 后续方案：共用默认字段规范化，不能把现有 Torch 缺陷作为期望行为复制到 ONNX。

## 接口核对结论

| 环节 | 当前判断 |
| --- | --- |
| JSON1 和 Percentile | 两边均设置同一份 JSON1 和 99.99；统计内核的数值等价性未验证 |
| 混合精度 | 配置规则适合共用；类型、参数和量化器定位需要适配 |
| 校准 | 保留各框架执行方式，共用参数和结果检查规范 |
| Po2 | 已共用选 scale 的部分函数，网格计算与失败策略仍不一致 |
| bias | Sb=Sx*Sw 公式相同，安全检查与状态管理需要对齐 |
| QAT/冻结 BN | ONNX PTQ 不执行合理，不需要模拟 Torch 训练流程 |
| GRU/LSTM | Torch 原生循环算子有内部配置；ONNX 当前忽略，不代表能力等价 |
| 导出 | `0446ccd` 已共用 compiler encoding 规范化，公开名沿用 export_onnx_json |
| 恢复验证 | 应统一“恢复量化状态后输出一致”的契约，保留框架专属恢复实现 |

## 本轮验证记录

运行环境：`rx-quant-env-runtime:v1-cu126`，容器使用已安装的 native 库，Python 源码由 `/repo/src` 覆盖；没有重新打包或更新用户安装目录。

### 自动回归

78 项 unittest 全部通过，入口为：

- `tests/test_onnx_ptq_alignment.py`
- `tests/test_onnx_encoding_export.py`
- `tests/test_bias_alignment_range.py`
- `tests/test_compiler_encodings.py`
- `tests/test_torch_export_validation.py`

覆盖不同参数/激活位宽、缺失输入槽、Gemm 与 MatMul+Add、用户禁用 bias、INT16/INT32 溢出、后层失败不部分写回 bias、无效 scale、状态丢失、原生 encoding 删除/篡改、compiler encoding 删除、量化输出变化、NaN/Inf，以及通用 PTQ 的 bias 对齐开关。

日志：`.release/validation/onnx-ptq-alignment/tests.log`。

### 真实 MobileNetV2

直接运行修改后的 `examples/quick_start_onnx_ptq.py`，读取已有预处理模型和 200 个校准样本：

| 检查 | 结果 |
| --- | --- |
| 校准样本 | 200 |
| 已映射/已启用量化器 | 276/276 |
| 全部整数 scale 为 Po2 | True |
| 参数 zero_point | 全部为 0 |
| clean ONNX 与原始 FP32 的最大误差 | 0 |
| 恢复后的量化输出与保存前的最大误差 | 0 |
| 重新生成的 compiler encodings 与保存文件 | 完全一致 |

产物与日志：`.release/validation/onnx-ptq-alignment/`，包括 `run.log`、模型、两种 encodings 和 metadata。

### 剩余限制

- 第 5 至 10 项按本文件记录继续处理；第 9 项仅提前修复恢复验证依赖的非有限值检查。
- 未重新训练完整 KWS，也未新增 MobileNetV2 的任务精度基准；Torch 相关回归已通过。
- 当前恢复使用底层支持的原生 encoding 0.6.1/1.0.0 格式，真实运行使用默认 1.0.0；2.0.0 的恢复不在本轮支持范围。
- 前四项修复按参数配置、bias/网格校验、恢复验证分阶段提交，并补充根目录 CHANGELOG。
- 本轮没有修改 `packaging/` 或 `rx-lm-quant`，未重新打包或安装。
