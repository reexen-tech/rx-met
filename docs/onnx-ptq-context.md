# ONNX PTQ Compiler Adapter Context

只覆盖 ONNX 直量化这条线，不是整个仓库的术语表。
用来区分 AIMET 原生格式、Torch 导出格式和编译器部署格式。

## Artifacts

**Clean ONNX**:
移除 AIMET 仿真量化节点后的部署模型图；它保留模型的真实输入输出契约，量化参数由外部 encodings 文件提供。
_Avoid_: QDQ ONNX, AIMET graph

**Raw AIMET encodings**:
由 `aimet_onnx` 直接导出的、用于调试和重载复现的原生量化参数表示。
_Avoid_: compiler encodings, deployment encodings

**Compiler encodings**:
目标编译器直接消费的节点级量化参数表示；它不是 AIMET encoding version 的别名。
_Avoid_: raw encodings, AIMET encodings

## Quantization

**Calibration manifest**:
描述校准样本、模型输入名、样本配对关系和文件校验值的可复现输入清单。
_Avoid_: random calibration, sample folder

**Po2 scale**:
与 Torch QAT 相同的 `cover_range` 选择结果，实际写入 quantizer 的 `1/2^n` scale；`n`、范围必须与该 scale 一致。
_Avoid_: JSON-only Po2, always-floor Po2, exponent-only encoding

**Bias scale**:
Conv/Gemm/MatMul 的 bias 不参与统计校准；Po2 之后写入 `Sb = Sx * Sw`，位宽默认 INT32（与 RX mixed-precision 一致）。AIMET ONNX 近零 bias 通道会得到 `scale=0`，校准期 fake-quant 会把后续图打成 Inf。
_Avoid_: official INT32 concretize, int8 bias after analytic scale, statistical-only bias for conv/gemm

**Compiler coverage**:
所有 enabled quantizer 都能映射到 compiler encodings 的 node input/output 或 parameter 条目，并且不存在 orphan key。
_Avoid_: supported-only coverage, partial coverage
