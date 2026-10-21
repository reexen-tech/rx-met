# 统一编译器编码处理与导出入口名称

- 状态：已接受
- 日期：2026-10-20

## 背景

Torch 的 legacy 导出已定制为直接写入自身的加性 offset。ONNX 的 legacy 编码仍使用
无符号索引域的 TfEncoding.offset。直接复制并重命名字段，使对称 INT8 参数在
编译器 JSON 中错误地出现 zero_point=-128。两侧还重复维护了 dtype、指数及数组处理。

## 决策

两侧公开入口统一命名为 export_onnx_json，分别保留在 aimet_torch.rx_export 和
aimet_onnx.rx_ptq。为兼容长期使用的接口，Torch 保留原参数、默认值及两个字符串路径的
元组返回值；ONNX 返回包含 onnx、encodings、aimet_encodings 和 coverage 的字典。
旧 ONNX 名称 export_onnx_compiler_artifacts 保留原参数及字典返回值。

公共格式及纯数据校验放在 aimet_common.compiler_encodings，不导入 Torch 或 ONNX Runtime。
前端分别负责模型导出、节点映射和源网格语义转换；GRU/LSTM 专用处理继续留在 Torch 路径。
原有通用处理改为调用公共实现。

2026-10-21 零点约定更新：编译器当前只支持对称量化，统一新导出为 schema_version=4，
所有记录采用标准减性零点 real=scale*(q-zero_point)，替代普通算子的旧加性约定。
Torch 前端源 offset 在公共模块中取反；ONNX 前端根据目标整数下界转换为 qmin-offset；
原生 GRU/LSTM 已使用标准零点，保持原值。实际 scale、整数网格和量化计算不变。
不将非对称量化点强制设为零，也不因此增加编译器对非对称量化的支持。

公共层按 schema 版本识别旧文件：缺少版本或 schema 1/2/3 时，仅将普通算子的
zero_point 取反；原生 GRU/LSTM 记录保持原值。schema 4 可重复规范化，未知版本拒绝。
Torch 编译器编码重载先复用该规范化，再恢复内部网格；原始阶段文件的 additive offset
保持原样。公共层统一验证普通及原生记录的 dtype、数值、数组、整数域及真实范围。
循环算子专用字段和容器仍由算子库维护，独立阶段文件继续使用原有 schema 3。

2026-10-21 编码格式更新：普通层记录保留源量化器的 enc_type，原生算子接口为全部
整数记录生成 n，并将顶层 input/output 统一为端口编号字典。内部量化点的 output
继续使用单元素列表；加载端兼容旧列表及缺少 n 的文件。n 使用已有舍入指数规则，
不改变 scale 或实际整数网格。

ONNX Po2 工作流要求 scale 和 n 严格匹配。Torch 既有导出仍允许 affine scale，保留
其舍入指数行为，避免迁移入口时改变已有导出前提。

## 验证

- 等价的 Torch 与 ONNX 网格生成相同的编译器数值字段。
- 覆盖 INT8/16/32、非对称、无符号、单通道和多通道，以及嵌套循环层编码。
- 通过真实 QuantSim 导出检查 clean ONNX、参数零点及新旧入口兼容性。
- 继续执行 PyTorch 图覆盖、量化器缺失及 GRU/LSTM 导出回归。
- 非零零点覆盖全部 UINT8 整数码的实数等价、signed/unsigned 及多位宽网格，
  并验证真实 Torch 导出、新旧版本重载、重复规范化和错误符号拒绝。
