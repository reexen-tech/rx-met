# ONNX PTQ 与 Torch 示例的待修复问题

日期：2026-10-20。分析基线：`0446ccd`。

核对入口：`examples/quick_start_kws.py` 与 `examples/quick_start_onnx_ptq.py`。
前者已有较多使用验证，但不将旧实现的所有行为直接视为正确规范。
原始第 1 至 10 项及后续补查的第 12、13 项均已修复并完成回归验证。
第 11 项仅保留 Po2 精度评估结果：按用户要求，精度变化本身不视为实现缺陷，不作为待修任务。

## 修复进度

| 编号 | 优先级 | 问题 | 状态 |
| --- | --- | --- | --- |
| 1 | P1 | 输入配置覆盖权重和 bias 配置 | 已修复，回归通过 |
| 2 | P1 | bias 对齐缺少范围检查，覆盖用户禁用意图 | 已修复，回归通过 |
| 3 | P1 | 无效 scale 被关闭后仍可能通过 Po2 和覆盖率检查 | 已修复，回归通过 |
| 4 | P1 | 重新加载验证没有恢复 encodings | 已修复，真实模型验证通过 |
| 5 | P2 | 非对称 Po2 的整数 offset 与 min/max 不一致 | 已修复，共用网格规则，回归通过 |
| 6 | P2 | 相邻层共用 ONNX 张量量化器，配置冲突静默覆盖 | 已修复，冲突在写入前报错 |
| 7 | P2 | JSON2 类型映射、字段支持和未知类型检查不完整 | 已修复，共用配置规则，回归通过 |
| 8 | P2 | 校准输入全部强制转换成 float32 | 已修复，真实多输入模型验证通过 |
| 9 | P2 | 只评估首个校准样本，NaN 可能被当作验证成功 | 已修复，完整验证集和分类精度验证通过 |
| 10 | P2 | Torch 的 default_bitwidth 未转换成实际读取的字段 | 已修复，两框架默认位宽回归通过 |
| 11 | - | 真实 MobileNetV2 的 Po2 精度变化 | 评估记录，不作为缺陷 |
| 12 | P1 | 显式关闭 GRU/LSTM 量化后仍被要求提供编码，导出失败 | 已修复，6 种状态组合通过 |
| 13 | P1 | ONNX 产物未显式保存中间张量维度，编译器无法直接读取 | 已修复，104 个中间张量全部有固定维度 |

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
- 修复：`aimet_common.power_of_2.recompute_min_max_for_new_scale` 同时供 Torch/ONNX 使用，先取整并约束 affine offset，再重算真实范围，确保零可表示。
- ONNX INT8 strict signed symmetric 使用 254 个间隔，unsigned symmetric 保留完整 255 个间隔；原生重载同步修正 strict 范围，避免恢复后多出一个码位。
- 验证：普通非对称、逐通道、strict/unsigned、scale 缩小时的零点约束，以及真实 QuantSim 导出恢复均通过。Torch 原来重复的范围计算已删除。

## 6. 相邻层配置冲突

- 位置：`mixed_precision.py:_apply_layer_config`。
- 复现：前层输出要求 16-bit、后层输入要求 8-bit，两者实际引用同一 quantizer，最终为 8-bit。
- 修复：公共 `QuantizerConfigPlan` 收集设置，ONNX 适配层在实际写入前检查全部共享量化器请求。
- 两处显式的位宽、对称性、启用状态或逐通道要求不一致时报错，消息包含来源层和字段；默认值仅作为回退，不覆盖显式要求。
- 验证：反转遍历顺序仍报同类冲突，失败后量化器不改变；一致的两处请求和显式覆盖默认值可正常应用。
- 当前选择报错，不自动插入重定量化节点。需要不同输入/输出网格的同一条边仍需图级适配，不能静默选择其中一个。

## 7. JSON2 支持和错误检查不完整

- `QuantizedLinear` 同时映射 Gemm 和 MatMul，但 `QuantizedMatMul` 没有映射，专门配置被忽略。
- BatchNorm、Softmax、Log、Clamp 等部分类型映射缺失；ONNX 忽略 `param_bitwidth/param_symmetric`。
- ONNX 未区分“已注册但当前图没有的类型”和“拼写错误的未知类型”；Torch 对后者报错。
- `per_channel_quantization` 在 Torch 文档中宣称支持，但 JSON2 的两个实现均没有处理，是共同缺口。
- 名称匹配优先级当前实现相同：精确名称 > 类型 > 通配符 > 默认。Torch 部分说明与实现不一致，不能据此认定两框架优先级不同。
- `QuantizedPad` 的整数形状操作与 ONNX 空间 Pad 不等价，不应机械映射。
- 修复：公共 `aimet_common.mixed_precision` 负责读取 JSON2、字段/类型/位宽校验和精确名称 > 类型 > 通配符的匹配；框架保留类型识别及量化器写回。
- ONNX 补齐本文列出的 MatMul、BatchNorm、Softmax、Log、Clamp 及常见归一化映射；Conv 按权重维度区分 1d/2d/3d。MatMul 专属配置优先，Linear 仅在没有专属配置且权重静态时回退适配。
- 同一 ONNX 类型的别名配置冲突会报错，注释不参与比较，精确层名仍可覆盖类型配置。
- 已知映射但图中没有的类型可以忽略；Var/Snake2d 没有单节点等价适配，记录于 `unsupported_types` 并在详细输出中说明。未识别或未支持的其它类型报错，不再静默吞掉配置。
- `param_bitwidth/param_symmetric` 应用于普通 weight/bias 之外的参数；逐通道设置用于权重，两框架均在校准前实际调整。Torch 保留 percentile 等分析器参数；无法安全支持的量化器或通道布局明确报错。
- 验证：真实 Log/Clip/Softmax 和 BatchNorm 图、Conv 维度、MatMul 专属配置、别名冲突、仓库实际 JSON2、两框架逐通道设置，以及 KWS 原有精度回归通过。

## 8. 校准输入 dtype 丢失

- 位置：`calibration.py:_validate_feed`。
- 复现：INT64 输入 `[1,2]` 被转换成 float32，不能正确用于 Gather 等索引输入；BOOL/FLOAT16 也受影响。
- 修复：保留 `inspect_model_inputs` 的 shape 返回格式，新增 `inspect_model_input_dtypes`；通用入口将声明的 dtype 传给校准读取器。
- INT64/BOOL/FLOAT16 保持模型要求的类型；浮点到整数不静默截断，整数缩窄先检查范围，浮点转换溢出及非有限输入报错。只提供 shape 的旧调用保留文件自身 dtype；标量输入保持标量形状。
- 验证：真实 Gather+Where 模型同时使用 INT64 索引、BOOL mask、FLOAT16 数组，NPZ 加载和 ONNX Runtime 推理通过；整数范围、浮点溢出、FLOAT64 到 FLOAT32、标量和旧调用分别有回归覆盖。
- 静态 shape、数值/布尔类型为当前明确支持范围；字符串及其它未支持的输入类型报错。
- 额外风险：全部校准数据一次性读入内存，多输入 NPY 配对仍含 IMU/audio 特例；后续按需求扩展。

## 9. 评估覆盖和非有限值

- 原入口仅使用 `feeds[0]`，未使用准备脚本产出的 200 个 val 样本，也没有任务精度指标；NaN RMSE 可能被旧判断当作通过。
- 修复：公共 `aimet_common.evaluation` 负责有限值检查、跨 batch 的累计数值误差和分类精度。RMSE 按元素数累计平方误差，避免直接平均各 batch RMSE。
- ONNX `evaluate_onnx_dataset` 为整个数据集复用一个 session；示例对完整 val 分别评估 FP32、PTQ、Po2 和恢复后的结果，clean ONNX 也检查全部样本。
- 准备脚本将验证样本标签按文件名保存到 manifest。ImageNette 的 0..9 映射到完整 ImageNet-1k 输出索引 `[0,217,482,491,497,566,569,571,574,701]`，没有把 1000 类模型缩成 10 类后报告精度。
- 验证标签依文件名对齐，检查缺失、重复或多余文件。旧 manifest 缺少标签时明确要求重跑准备脚本，两步示例流程保持不变。
- `verify_onnx_ptq_artifacts` 保留原来的单样本调用形式，并支持整个验证集逐样本恢复检查；分类正确数差异达到 0.5 个百分点时失败，使用整数计数避免阈值浮点误差。
- 通用 `run_onnx_ptq` 新增可选 validation_dir/validation_labels。用户没有独立验证集时检查全部校准数据，并在 metadata 中明确记录 evaluation_source=calibration；自定义数据没有标签时仍可检查数值误差，不虚构任务精度。
- 示例默认读取 calib 同级的 val，也支持 `RX_MET_ONNX_PTQ_VAL`；不执行模型专属预处理。
- 验证覆盖后续样本的 NaN、恢复输出变化、不同 batch 大小的加权误差、session 复用、标签顺序和单独验证集的全部样本计数。

## 10. Torch 默认位宽不生效

- 位置：`aimet_torch/utils_rx.py`，默认分支直接使用 `default_bitwidth`，后续读取的却是 `weight_bitwidth/output_bitwidth`。
- 复现：初始 8-bit，JSON 仅指定 `default_bitwidth={weight:16,output:16}`；Torch 仍为 8，ONNX 为 16。
- 修复：两框架共用 `aimet_common.mixed_precision.default_layer_config`，将 weight/output 转成实际读取的 weight_bitwidth/input_bitwidth/output_bitwidth；output 同时作为已有激活输入、输出的默认位宽。
- 保留精确名称、类型、通配符的既有优先级；默认值不会重新创建用户明确禁用的 Torch 输入槽。
- 验证：初始 8-bit 时同时指定 16-bit、仅指定权重、仅指定激活，以及禁用输入的行为均通过真实 Torch/ONNX 小模型回归。

## 11. 评估记录：MobileNetV2 的 Po2 精度变化

- 完整验证集评估结果：FP32 79.00%、PTQ 75.50%、Po2 51.50%、恢复后 51.50%，均为 200 个独立验证样本的完整 1000 类 Top-1。
- Po2 比 PTQ 下降 24.00 个百分点。Po2 scale 受约束，可能偏离最优 scale；用户明确接受由此造成的精度变化，要求关注实现及量化规则是否正确。本项不再作为缺陷或待修任务。
- 恢复前后逐样本输出误差为 0，compiler 编码一致。与第 5 至 8 项完成时的 encodings 文件 SHA256 完全相同，表明完整评估暴露了已有精度损失。
- 编码 SHA256：`d96775d6576468d892b6aa1aca7c76ccb80c9f71150589d653cee37a1ff39e68`。
- 保留各阶段精度、网格校验和恢复一致性记录；本轮不调整 Po2 策略，也不按验证集调参。

## 12. 显式关闭循环层量化后导出失败

- 用例：`test_native_recurrent_state.NativeRecurrentStateTest.test_disabled_stage_roundtrip_and_onnx`。
- 原因：编码收集正确跳过 `use_quantization=False` 的原生 GRU/LSTM，但最终 ONNX 校验没有收到这个状态，对所有 GRU/LSTM 无条件要求激活、权重和 bias 编码。
- 修复前复查：关闭 GRU、关闭 LSTM、同时关闭，分别在校准前或校准后关闭，共 6 个子用例均以 missing encoding 报错。
- 修复：`export_onnx_json` 从 QuantSim 捕获禁用模块名及 ONNX 类型，通过可信上下文传给最终校验；原生导出适配层将关闭的循环层记录在既有 `excluded_layers` 字段中。
- 最终校验只豁免名称和类型均匹配的已禁用循环层的量化编码要求，仍检查标准 ONNX 节点恰好存在一个；关闭且未校准的循环层也可按浮点路径导出。
- `excluded_layers` 文件内容不能代替 QuantSim 的真实状态。启用层不能出现在排除名单中；禁用层必须有对应排除记录，且不能残留量化编码。
- 保留普通计算节点、启用循环层、参数编码和 ONNX 结构的原有检查；公开 `export_onnx_json` 参数及字符串元组返回值不变。
- 验证：原 6 个组合全部通过；两种循环层都关闭时，使用相同公开权重构建 PyTorch 标准 GRU/LSTM，并与 ONNX Runtime CPU 输出比较，`atol=1e-6, rtol=1e-5` 通过。
- 负向测试覆盖启用层伪造排除名单（编码保留或删除）、禁用层缺少排除记录、残留编码及循环节点改名，均正确拒绝导出。

## 接口核对结论

| 环节 | 当前判断 |
| --- | --- |
| JSON1 和 Percentile | 两边均设置同一份 JSON1 和 99.99；统计内核的数值等价性未验证 |
| 混合精度 | 已共用解析、校验和匹配；类型、参数和量化器定位保留适配 |
| 校准与评估 | 保留框架执行方式；ONNX 已检查独立验证集、任务精度和全部恢复输出 |
| Po2 | 已共用选 scale 和整数网格计算，ONNX 无效 scale 直接拒绝 |
| bias | 已共用安全检查，各框架适配读取真实参数及写回 |
| QAT/冻结 BN | ONNX PTQ 不执行合理，不需要模拟 Torch 训练流程 |
| GRU/LSTM | Torch 原生循环算子有内部配置；ONNX 当前忽略，不代表能力等价 |
| 导出 | `0446ccd` 已共用 compiler encoding 规范化，公开名沿用 export_onnx_json |
| 恢复验证 | 已检查恢复量化状态后的输出一致，保留框架专属恢复实现 |

## 13. ONNX 中间张量维度未保存

- 原始下载、预处理输入和旧导出模型的 `graph.value_info` 均为 0；有维度的是模型输入和最终输出，量化校准通过运行时张量收集统计，并不要求文件事先保存中间维度。
- 当前 MobileNetV2 共 105 个节点、104 个中间输出；标准 ONNX 形状推导可补出大部分信息，但末尾 Reshape_103 的输出张量 472 缺少 rank/shape。
- 修复：`aimet_common.onnx_shapes` 先执行标准形状推导，信息不足时复用 ONNX Runtime 的符号形状推导，不猜测未知 rank，不用单个校准样本固化动态维度。
- 只将 tensor type/shape 标注写回原图，保留计算节点、属性、连接、初始化张量和外部权重引用；覆盖标量、零长度张量、动态符号维度和控制流子图。
- ONNX PTQ 的 `export_onnx_json` 自动补全并保存，再检查磁盘中的实际产物；`verify_onnx_ptq_artifacts` 在重建 QuantSim 前独立检查已保存维度和 ONNX 结构，不自动修复被删除的标注。
- 缺少 dtype、rank、维度或不能推导的中间输出时导出/校验报错；动态维度可显式保留符号。`onnx_shapes` 返回并记录 tensor_count/intermediate_tensors/symbolic_dimensions。
- 新增 `tests/test_onnx_shapes.py`，覆盖元数据之外的内容不变、形状子图/Reshape、动态维度、标量、零长度、控制流、外部权重、无法推导时拒绝，以及删除已保存标注后的产物回读失败。
- 151 项相关回归通过，包含原 142 项和新增 9 项。日志：`.release/validation/onnx-ptq-shapes/tests.log`。
- 真实模型结果：value_info 从 0 条补到 104 条；tensor_count=106、intermediate_tensors=104、symbolic_dimensions=0，Reshape_103 输出 `[1,1280]`。
- 与旧导出相比，节点及其顺序、初始化张量、输入输出定义均完全一致，compiler encodings 文件也完全一致。200 个验证样本上的 clean ONNX 和量化恢复输出误差均为 0。
- 新验证产物：`.release/validation/onnx-ptq-shapes/mobilenetv2_ptq.onnx`，配套 encodings、metadata 和 run.log 均位于同一目录；未覆盖用户现有模型。
- 本轮只实现维度补全和产物检查，前面讨论的示例主线精简仍暂停，未修改 `packaging/` 或用户安装目录。

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

- 本文记录的功能缺陷已修复；第 11 项按用户要求只作为精度评估记录。
- 未重新训练完整 KWS；Torch 相关回归已通过，MobileNetV2 的独立验证集精度已记录。
- 当前恢复使用底层支持的原生 encoding 0.6.1/1.0.0 格式，真实运行使用默认 1.0.0；2.0.0 的恢复不在本轮支持范围。
- 前四项修复按参数配置、bias/网格校验、恢复验证分阶段提交，并补充根目录 CHANGELOG。
- 本轮没有修改 `packaging/` 或 `rx-lm-quant`，未重新打包或安装。

## 第 5 至 8 项验证记录

- 107 项 unittest 全部通过，包含前一轮回归，新增 `tests/test_onnx_ptq_remaining.py`，以及 `test_kws_power_precision`、`test_kws_conv_precision` 两个原有 KWS 精度回归。
- 日志：`.release/validation/onnx-ptq-alignment/tests-followup.log`。
- MobileNetV2 使用仓库实际 JSON2 和 200 个校准样本再次完整运行成功；覆盖率 276/276，全部 scale 为 Po2，恢复量化输出的最大绝对误差和 RMSE 都为 0，compiler encoding 对比通过。
- 真实模型产物：`.release/validation/onnx-ptq-alignment/followup/`。
- 已分阶段提交：`0b97a40` 参数配置、`f54b832` bias/网格校验、`9b87454` 恢复验证、`2aea946` Po2 整数网格、`14747fc` JSON2 规则和冲突处理；最后一组提交输入 dtype、回归测试和本文更新。
- 所有修复均补充根目录 `CHANGELOG.md`。未推送，未重新打包或安装。

## 第 9、10 项验证记录

- 129 项 unittest 全部通过，包括新增 `tests/test_ptq_evaluation.py` 和准备脚本、前序量化及 KWS 回归。
- 日志：`.release/validation/onnx-ptq-final/tests.log`。
- 从原始 ONNX/parquet 重新准备 200 个 calib 和 200 个 val，保留不重叠的分层抽样及正确 ImageNet-1k 标签。
- 完整示例运行结果：FP32 79.00%、PTQ 75.50%、Po2/恢复后 51.50%；覆盖率 276/276，全部整数 scale 为 Po2，clean ONNX 和量化恢复均在全部 200 个验证样本上误差为 0。
- 产物与 metadata：`.release/validation/onnx-ptq-final/`。原始输入和用户原有预处理目录均未覆盖。
- 默认位宽修复提交：`852b1e1`；完整验证集评估单独提交，均补充 CHANGELOG。
- 未修改 `packaging/`、未推送、未重新打包或安装。

## 禁用循环层修复验证记录

- 本轮将完整 `test_native_recurrent_state` 模块加入前一轮 129 项回归，合计 142 项相关测试通过，包含原来失败的 6 个状态组合。
- 129 项通过的旧记录没有覆盖该模块；本轮补齐覆盖后再确认修复。
- 日志：`.release/validation/onnx-ptq-final/disabled-recurrent-full-regression.log`。
- 单独修复禁用状态与最终导出检查的衔接；不修改 Po2 量化策略，不修改 `packaging/` 或用户安装目录。
