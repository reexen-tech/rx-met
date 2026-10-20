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

为兼容现有编译器，普通算子的 zero_point 保持加性含义：real=scale*(q+zero_point)。它不是标准
ONNX Q/DQ 的减性零点。ONNX 前端根据实际符号、对称模式和整数下界转换 TfEncoding.offset，
不将所有量化点强制设为零。公共层验证 dtype、有效数字、通道数组、整数域及真实范围。
原生 GRU/LSTM 保留算子库定义的字段、列表和网格约定；公共层校验其基础数值，
不重写专用编码或推测其零点符号。普通层格式化在写入原生循环编码之前完成。

ONNX Po2 工作流要求 scale 和 n 严格匹配。Torch 既有导出仍允许 affine scale，保留
其舍入指数行为，避免迁移入口时改变已有导出前提。

## 验证

- 等价的 Torch 与 ONNX 网格生成相同的编译器数值字段。
- 覆盖 INT8/16/32、非对称、无符号、单通道和多通道，以及嵌套循环层编码。
- 通过真实 QuantSim 导出检查 clean ONNX、参数零点及新旧入口兼容性。
- 继续执行 PyTorch 图覆盖、量化器缺失及 GRU/LSTM 导出回归。
