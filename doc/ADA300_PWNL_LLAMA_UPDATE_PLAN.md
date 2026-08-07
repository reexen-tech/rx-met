# ADA300 PWNL LUT 在 llama.cpp 中的更新与实验方案

> 状态：讨论结论已合入，待文档审核
>
> 本文档定义实施范围、算子验收和模型实验方案。文档审核通过前不修改实现、不生成 GGUF、不启动实验。
>
> 本项目禁止自动提交：任何阶段都不得由执行者擅自运行 `git commit`。

## 1. 方案变化分析

对照以下两个只读依据：

- `/home/chengxing.zou.srv/projects/abc_lut/lut_fp/test_ada300_bxc_16seg.py`
- `/home/chengxing.zou.srv/projects/abc_lut/doc/ada300_pwnl_spec.md`，其中 sigmoid 以 §4.7 为准

当前 llama.cpp 实现与最新方案存在以下差异：

| 算子 | llama.cpp 当前实现 | 最新方案 | 本次动作 |
|---|---|---|---|
| `exp` | 在 `[-20, 0]` 上直接查 `e^x`；小于下界返回 0；所有正输入被 clamp 到 0 | `x -> u=x*log2(e) -> (k,f)`，只在 `[0,1)` 查 `2^f`，再按 `2^k` 恢复 | 删除旧上下界和正输入 clamp，实现 exp2 range reduction，覆盖正负输入 |
| `sin` | 旧表覆盖 `[-pi, pi]`，周期映射后直接查表 | 任意输入折叠到 `[0, pi/2]`，只查第一象限 sin 表并恢复符号 | 更新表、定义域、象限折叠和边界处理 |
| `cos` | 通过旧 sin 表做相位平移 | `cos(x)=sin(x+pi/2)`，复用新的 `[0,pi/2]` sin 表 | 不增加 cos 表，更新为新 sin 折叠路径 |
| `sigmoid` | 独立 `[-6,6]` 表并 clamp 输入 | §4.7：`reciprocal(1 + exp(-x))` | 删除独立 sigmoid 表和 clamp，复用新版 exp 与 normalized reciprocal |
| `reciprocal/rsqrt/sqrt/log` | 已有 normalized mixed-FP16 路径 | 数学路径基本不变，但固化系数必须与最新 16 段导出一致 | 全量比较，任何不一致都整表更新 |
| `power_2` | production 使用直接 `x*x` | 本次明确忽略 | 保持现状，不接 LUT、不做跨语言 golden、不主动清理其遗留接口或数据 |

因此，本次不是只替换四张系数表。`exp`、`sin/cos`、`sigmoid` 的前处理、查表语义和后处理均发生变化；其他实际使用的 LUT 也必须核对最新导出系数。

## 2. 已确认决策

1. sigmoid 严格使用 §4.7 的 `1/(1+exp(-x))`，不使用独立 sigmoid LUT。
2. sigmoid 支持任意输入，不保留 `[-6,6]` clamp，也不引入 `exp(-abs(x))`、正负分支或半表折叠。
3. exp 使用新版 exp2 range reduction，支持正输入；删除所有与新方案矛盾的旧范围限制。
4. llama.cpp 中 production mixed-FP16 使用的固化阈值和系数必须与最新 16 段导出逐项比较；数值不一致时更新整张表。
5. `abc_lut` 仓库完全只读。本次不修改其脚本、文档、JSON 或其他内容。
6. 在 llama.cpp tests 下增加 Python adapter，导入 `test_ada300_bxc_16seg.py` 的构件并生成无 clamp golden。
7. 最新 production 路径是 mixed-FP16。第一阶段只实现和测试 mixed-FP16。
8. 未使用的 full-FP16/full-BF16 LUT 接口、表和测试可以删除；`power_2` 是例外，保持现状。
9. 本次只支持 16 段。编译期显式禁止任何非 16 段配置；31 段不更新、不测试、不可编译启用。
10. `power_2` 不属于本次 LUT 更新，继续直接 `x*x`，不与 Python normalized power_2 比较。
11. 第一阶段沿用并改造 `tests/test-reex-lut.cpp`，不另建绕过 production interface 的 C++ 算法实现。
12. reciprocal/rsqrt 已有 CUDA production helper；增加 test-only CUDA launch adapter 直接调用这些 helper，不复制其算法。
13. Python 生成唯一输入与 golden 输出；C++/CUDA 读取完全相同的输入 bit，并分别输出结果文件。
14. 结果文件使用 IEEE-754 十六进制 TSV。NaN 检查语义和类别，不比较 payload bit。
15. 跨语言结果报告 exact-match rate。由 CPU/CUDA 指令差异导致的有限值偏差允许 `MaxULP <= 1`，特殊值类别必须一致。
16. 第一阶段通过并提交审核报告后才能进入第二阶段；不需要保留失败实现作为备用路径。
17. 第二阶段使用 LUT OFF 和 LUT ON 两个独立 build，除 LUT 开关外配置一致。
18. 两个模型都测试 F16、Q8_0_64、Q4_0_64，并与 F16 原始基线对比。
19. 两个数据集都执行完整 PPL、KLD 和 Same top-1 实验，固定 `ctx=4096`、`chunks=32`。
20. 不为 CCI2 生成前缀、截断副本或其他预处理文件；正式命令直接使用用户指定的原始文件路径。
21. 第二阶段没有预设质量通过阈值。先完成完整结果报告，再审核 PPL/KLD/Top-1 是否正常。
22. 除 crash、NaN、CUDA error 或模型损坏等无法继续的错误外，不因中途某项指标看起来异常而缩减实验矩阵。
23. TSV、日志、KLD baseline 等中间产物全部保留在本地，不自动删除。
24. 禁止擅自执行 commit、push 或其他发布操作。

## 3. 目标与非目标

### 3.1 目标

- 将 ADA300 最新的 16 段 `exp2`、四分之一周期 sin/cos、组合 sigmoid 方案移植到 llama.cpp CPU/CUDA production mixed-FP16 路径。
- 删除与新算法冲突的 exp/sigmoid clamp、旧 direct 表和旧 trig 定义域。
- 更新所有 production mixed-FP16 路径实际消费且与最新导出不一致的 LUT 数据。
- 建立只读消费 `abc_lut` 导出物的确定性 header 生成与 manifest 流程。
- 建立 Python、C++ CPU、CUDA 三方 bit-level 算子门禁。
- 在两个 Qwen3.5 模型、三种权重和两个数据集上完成 LUT OFF/ON 对照实验。

### 3.2 非目标

- 不修改 `/home/chengxing.zou.srv/projects/abc_lut`。
- 不设计、更新或验证 31 段方案。
- 不重新设计 `power_2`；不对比 Python normalized power_2。
- 不修改 Q4_0_64/Q8_0_64 量化算法或 GEMM kernel。
- 不启用 `GGML_REEX_FP16_PIPELINE` 或 `GGML_REEX_GEMM`。
- 不扩展到任务准确率、生成质量或多模态 mmproj 实验。
- 不在本方案中提前定义模型质量合格线。

## 4. Production module 边界

外部调用 interface 保持在统一 REEX LUT module：

```text
ggml_sin_lut_mixed_fp16_f32_REEX(x)
ggml_cos_lut_mixed_fp16_f32_REEX(x)
ggml_exp_lut_mixed_fp16_f32_REEX(x)
ggml_sigmoid_lut_mixed_fp16_f32_REEX(x)
ggml_silu_lut_mixed_fp16_f32_REEX(x)
```

RoPE、softmax、SiLU、GELU quick、MoE gating、softplus 等调用方不应自行展开 LUT 定义域、range reduction 或 sigmoid 组合逻辑。

```text
callers
   |
   v
REEX mixed-FP16 interface
   |
   +-- CPU production implementation
   |
   +-- CUDA production device helper
```

test-only CUDA adapter 只解决 reciprocal/rsqrt 缺少公开 ggml unary graph op 的测试可达性问题：

```text
golden input bits
   -> test-only launch adapter
   -> production ggml_cuda_*_lut_mixed_fp16_reex helper
   -> CUDA output bits
```

adapter 不得复制 segment 查找、MAC、normalized reconstruction 或特殊值逻辑。

## 5. 新运行时算法

### 5.1 Mixed-FP16 舍入合同

以 `test_ada300_bxc_16seg.py` 调用的 `infer_with_float_lut(..., precision="mixed_fp16")` 为可执行依据：

```text
输入、threshold、b、c：FP32
segment compare：FP32，右闭阈值
product = FP32(b * x)
sum     = FP32(product + c)
core    = FP16(sum) -> FP32
range reconstruction / 组合：按 runner 的 FP32/FP64 顺序执行
```

关键点：FP16 舍入发生在每次 LUT 核心 `b*x+c` 的输出处。不能把它误写成“整个组合算子只在最终输出统一截断一次”。exp、sigmoid 和 normalized 算子都必须复现 runner 的实际舍入点和组合顺序。

Python 使用非融合的乘法与加法顺序。CPU/CUDA 应使用显式中间值保持该顺序；若编译器或硬件指令仍造成差异，按 `MaxULP <= 1` 门禁处理并报告 exact-match rate。

### 5.2 exp：exp2 range reduction

对有限输入 `x`：

```text
u = x * log2(e)
k = floor(u)
f = u - k
if f >= 1:
    f = 0
    k = k + 1
core = LUT_exp2(f)           # f in [0, 1)，core 按 mixed-FP16 舍入
y = core * 2^k
```

要求：

- 表从旧 `[-20,0]` direct `e^x` 改为 `[0,1)` 上的 `2^f`。
- 删除 `EXP_MIN`、`EXP_MAX`、小输入短路和正输入 clamp。
- range reduction 常量、运算类型、`f` 边界和 reconstruction 顺序与 Python adapter 一致。
- 正输入不再错误地返回 `exp(0)`。
- `k` 使用足够宽的表示或在 reconstruction 前显式判断指数范围，禁止极大有限输入在 float-to-int 转换时溢出或产生未定义行为。
- reconstruction 的数学上溢/下溢按 IEEE 语义得到 `+Inf`/`0`；这不是旧的经验区间 clamp。
- 特殊值语义单独覆盖 `NaN`、`+Inf`、`-Inf`。

### 5.3 sin/cos：折叠到第一象限

sin：

```text
theta = x mod (2*pi)                  # [0, 2*pi)
q = floor(theta / (pi/2))             # 0, 1, 2, 3
r = theta - q*(pi/2)
u = r            if q is even
    pi/2 - r     if q is odd          # [0, pi/2]
sign = +1        if q in {0,1}
       -1        if q in {2,3}
y = sign * LUT_sin_halfpi(u)
```

cos 通过 `x + pi/2` 复用同一流程和同一张 sin 表。

要求：

- 删除旧 `[-pi,pi]` sin 表及旧 clamp/映射。
- 不新增 cos 表。
- CPU/CUDA 使用同一常量、边界规则和运算顺序。
- 覆盖 `n*pi/2`、`n*pi`、`2*n*pi`、负周期、大输入和 RoPE 代表范围。

### 5.4 sigmoid：§4.7 组合路径

```text
e_neg = exp(-x)
denom = FP32(1 + e_neg)
y = reciprocal(denom)
```

即：

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

要求：

- 不做 sigmoid 输入 clamp。
- 不使用独立 sigmoid table。
- 不使用 `abs(x)`、半表、`1-y` 恢复或正负分支公式。
- exp 和 reciprocal 分别在自身 LUT core 输出位置执行 mixed-FP16 舍入。
- `[-6,6]` 只用于复现 §4.7 的 16 段精度指标，不是 production 定义域。
- Python adapter 不得直接调用 runner 中仍带旧 clamp 的 `sigmoid_composed_eval`；应复用其 `exp_eval` 和 `normalized_eval(..., "reciprocal")` 构件，无 clamp 地组装 §4.7 公式。
- 极大有限输入不通过修改输入实现饱和：`exp(-x)` 的自然上溢/下溢及 reciprocal reconstruction 应分别产生接近 0/1 的结果。
- 特殊值必须满足：`sigmoid(+Inf)=1`、`sigmoid(-Inf)=0`、`sigmoid(+0/-0)=0.5`、`sigmoid(NaN)=NaN`。

### 5.5 SiLU 与组合调用

```text
silu(x) = x * sigmoid(x)
```

SiLU 作为派生算子进入第一阶段 mixed-FP16 golden，验证真实模型使用的组合路径。还需审计 SwiGLU、GELU quick、softplus、softmax、MoE gating 等调用点，确保它们通过统一 REEX interface 获得新版 exp/sigmoid 行为。

### 5.6 power_2

`power_2` 明确排除在本次变更之外：

```text
power_2(x) = x * x
```

- production 继续直接乘法。
- 不导入或比较 Python normalized power_2 LUT。
- 不为它新增 CUDA adapter 或跨语言 golden。
- 不以清理名义主动修改其现有接口、表或本地回归；现有直接 `x*x` 回归可继续运行。

## 6. 固化数据与 16 段限制

### 6.1 唯一数据源

`abc_lut` 最新 16 段导出 JSON 是阈值和系数的只读真源。生成工具放在 llama.cpp 内，只读取这些文件并生成：

```text
ggml-reex-lut-data.h
ggml-reex-lut-data.manifest.json
```

manifest 至少记录：

- abc_lut commit id 和 dirty status。
- runner 路径与 SHA-256。
- 每个输入 JSON 的相对路径和 SHA-256。
- 段数，固定为 16。
- 每张表的函数语义、定义域和精度路径。
- 生成 header 的 SHA-256。

### 6.2 比较和更新范围

| 逻辑表 | 最新来源/动作 | mixed-FP16 16 段 |
|---|---|---:|
| sin/cos | `sin_halfpi`，cos 共享 | 比较并按需更新 |
| exp | `exponential_exp2` | 比较并按需更新 |
| sigmoid | 删除独立表，改为 exp + reciprocal | 删除 |
| reciprocal | `reciprocal_normalized` | 比较并按需更新 |
| rsqrt | `rsqrt_normalized` | 比较并按需更新 |
| sqrt | `sqrt_normalized` | 比较并按需更新 |
| log | `log_normalized` | 比较并按需更新 |
| power_2 | 不参与本次流程 | 不比较、不更新 |

production mixed-FP16 使用的 FP32 threshold/b/c cast 到 C++ `float` 后必须 bit-identical。发现任意差异时更新整张表，不手工挑选系数。

### 6.3 删除 legacy 精度路径

- 删除未被 production 使用的 full-FP16/full-BF16 LUT 函数、声明、host tables、CUDA tables、复制逻辑和对应测试。
- 删除范围不包含 `power_2`，避免把被明确排除的算子带入本次改动。
- 删除前再次用静态搜索确认没有 production caller；若发现 caller，将其迁移到 mixed-FP16 统一 interface，不保留双实现。

### 6.4 显式禁用 31 段

只允许：

```text
GGML_LUT_NUM_SEGMENTS_REEX == 16
```

CMake 和公共 header 至少一处必须对其他值明确报错，不能静默 fallback 到 16，也不能继续允许 `-DGGML_LUT_NUM_SEGMENTS_REEX=31`。旧 31 段数据可以保持不动，但不得被编译选择，本次不生成、不更新、不测试。

## 7. llama.cpp 预计修改面

| 文件/区域 | 计划修改 |
|---|---|
| `ggml/include/reex/ggml-reex-lut-data.h` | 更新 16 段 mixed-FP16 数据；加入 exp2/halfpi 语义；删除独立 sigmoid 数据 |
| `ggml/include/ggml-reex-lut.h` | 只允许 16 段；删除 legacy 精度声明和旧 exp/sigmoid/trig 常量 |
| `ggml/src/ggml-reex-lut.cpp` | 更新 host 初始化；删除 legacy 和 sigmoid table 初始化/复制 |
| `ggml/src/ggml-cpu/reex/reex_lut.cpp` | 实现第一象限 sin/cos 折叠 |
| `ggml/src/ggml-cpu/reex/reex_lut_direct.cpp` | 实现 exp2 reduction 和无 clamp 组合 sigmoid |
| `ggml/src/ggml-cpu/reex/reex_lut_normalized.cpp` | 对齐最新系数、舍入点和 reciprocal 内部复用；power_2 不动 |
| `ggml/src/ggml-cuda/reex/reex_lut.cuh` | CUDA 对等实现，删除旧 clamp 和独立 sigmoid table；power_2 不动 |
| `ggml/src/ggml-cuda/unary.cu` | 更新 table copy 和调用，删除 legacy/sigmoid copy |
| `tests/test-reex-lut.cpp` | 改造成 TSV 驱动的 mixed-FP16 CPU/CUDA 精度门禁 |
| llama.cpp tests 下 Python adapter | 只读导入 abc_lut runner，生成输入和无 clamp golden |
| test-only CUDA adapter | 使 reciprocal/rsqrt production device helper 可被 golden 测试直接调用 |
| 实验脚本 | 固定 build、模型、数据集、参数、运行矩阵、日志和 manifest |

最终修改面以静态 caller/table 审计为准；不因文件表格遗漏而保留与新算法冲突的旧逻辑。

## 8. 第一阶段：LUT 算子验证

第一阶段只处理 LUT 实现、数据和算子测试。不生成 GGUF，不运行模型实验。

### 8.1 复用 `test-reex-lut.cpp`

保留并改造现有 target、CTest 接线、CPU backend 初始化和 CUDA graph 测试框架：

- 删除固定 linspace 作为正式 golden 输入的做法。
- 删除 full-FP16/full-BF16 指标路径。
- 从 Python TSV 读取输入 bit。
- 分别输出 CPU/CUDA TSV。
- 正式模式下，缺文件、解析错误、CUDA 不可用、skip 或 fallback 均返回非零。
- 原 MAE/MaxErr/Cosine 统计可以保留为辅助报告，但不能替代 bit-level 门禁。

### 8.2 Python adapter

adapter 位于 llama.cpp tests 内，运行时将 abc_lut 加入只读 import path：

```text
abc_lut/lut_fp/test_ada300_bxc_16seg.py
   -> 导入 LUT loader、exp_eval、sin_like_eval、normalized_eval 等构件
   -> 无 clamp 组装 sigmoid = reciprocal(1 + exp(-x))
   -> 生成唯一输入与 golden TSV
```

禁止：

- 修改或覆盖 abc_lut 文件。
- 调用带旧 sigmoid clamp 的 helper 作为 golden。
- 在 Python 和 C++ 两端分别重建测试输入。
- 为了让比较通过而在 adapter 内复制另一套近似算法。

### 8.3 TSV 协议

```text
# reex-lut-bit-v1 segments=16 precision=mixed_fp16
op<TAB>case_id<TAB>input_bits<TAB>output_bits<TAB>class
exp<TAB>000001<TAB>3f800000<TAB>...	finite
```

- bit 字段是 8 位小写十六进制 IEEE-754 binary32 pattern。
- Python 是唯一 case 生成方。
- C++/CUDA 必须保持相同 op、case_id、输入 bit 和行顺序。
- NaN 比较 `isnan`/类别，不比较 payload。
- 比较器报告首个差异及每算子的 exact-match、MaxULP、特殊值结果。

建议产物：

```text
llama.cpp/tests/test-reex/results/ada300_pwnl_lut_ops/
  manifest.json
  python-golden.tsv
  cpp-cpu.tsv
  cpp-cuda.tsv
  compare-summary.json
  metrics-summary.json
  phase1-report.md
  logs/
```

这些文件全部保留，不自动删除。

### 8.4 覆盖矩阵

| 算子 | CPU mixed-FP16 | CUDA mixed-FP16 | 说明 |
|---|---:|---:|---|
| exp | 是 | 是 | 新 exp2 reduction |
| sigmoid | 是 | 是 | 无 clamp exp + reciprocal |
| sin/cos | 是 | 是 | 共享 half-pi sin table |
| reciprocal/rsqrt | 是 | 是 | CUDA 通过 test-only adapter 调用 production helper |
| sqrt/log | 是 | 是 | normalized production path |
| SiLU | 是 | 是 | `x * sigmoid(x)` 派生路径 |
| power_2 | 否 | 否 | 明确排除；保持直接 `x*x` |

只测试 16 段和 mixed-FP16。

### 8.5 Case 集

每个适用算子的输入由 Python 一次性生成，包含：

1. runner 当前的 1000 点常规采样。
2. 每个 LUT threshold 的精确值及 `nextafter` 左右邻点。
3. exp 的 `n*ln2`、正输入、上溢/下溢附近输入。
4. sigmoid 的 `+/-0`、`+/-6`、`+/-20`、`+/-100` 和扩展有限输入，不 clamp。
5. sin/cos 的 `n*pi/2`、多周期、负周期、大输入和 RoPE 代表值。
6. normalized 算子的 `2^n`、左右邻点、normal/subnormal 和有效域边界。
7. 固定 seed 的 4096 个 float32 bit-pattern，按算子合法域筛选。
8. `NaN`、`+Inf`、`-Inf`、正负零等特殊值语义测试。

随机 seed、case 顺序、筛选规则和输入文件 SHA-256 写入 manifest。

### 8.6 第一阶段门禁

必须同时满足：

1. 16 段 production mixed-FP16 threshold/b/c 与源 JSON bit-identical。
2. Python adapter 自检、TSV schema 和 manifest 校验通过。
3. C++ CPU 和 CUDA 都消费 Python 生成的同一输入 bit。
4. 每个有限输出相对 Python golden 的 `MaxULP <= 1`。
5. 每个算子报告 exact-match 数量和比例，不隐藏 1 ULP 差异。
6. 正负零、Inf 和 NaN 等特殊值类别符合 golden；NaN payload 除外。
7. CUDA 正式运行没有 skip、CPU fallback 或未覆盖 case。
8. 最新 runner 的 16 段 mixed-FP16 质量指标可复现；sigmoid 在 `[-6,6]` 上单独报告 §4.7 的 MeanULP/MaxULP，并报告无 clamp 扩展域结果。
9. 生成独立第一阶段报告，列出所有命令、版本、SHA-256、exact rate、ULP 和失败 case。

第一阶段未通过或报告未审核时，停止，不进入第二阶段。不需要额外保留旧实现作为运行时 fallback。

## 9. 第二阶段：双 Build

在容器 `rx-met-zcx` 中构建两个独立目录：

| 目录 | 共同配置 | 唯一差异 |
|---|---|---|
| `build_cuda_q64_nolut` | Release、CUDA、Q64 ON、FP16 pipeline OFF、REEX GEMM OFF | `GGML_USE_REEX=OFF` |
| `build_cuda_q64_lut` | Release、CUDA、Q64 ON、FP16 pipeline OFF、REEX GEMM OFF | `GGML_USE_REEX=ON` |

显式配置：

```text
-DCMAKE_BUILD_TYPE=Release
-DGGML_CUDA=ON
-DGGML_USE_REEX_Q64=ON
-DGGML_REEX_FP16_PIPELINE=OFF
-DGGML_REEX_GEMM=OFF
-DGGML_LUT_NUM_SEGMENTS_REEX=16
-DGGML_USE_REEX=OFF/ON
```

两个 build 编译 `llama-perplexity`、`llama-cli` 和 `test-backend-ops`。LUT build 还必须使用与第一阶段相同的配置构建并通过 `test-reex-lut`。`llama-quantize` 从 no-LUT build 生成一次即可。

除 LUT 开关及其必然产生的相关定义外，比较两个 `CMakeCache.txt`，发现其他差异即停止正式实验。

## 10. GGUF 生成

### 10.1 模型和权重

```text
/mnt/data1/share/models/Qwen/Qwen3.5-35B-A3B/
/mnt/data1/share/models/Qwen/Qwen3.5-9B/
```

每个模型生成：

```text
F16
Q8_0_64
Q4_0_64
```

原始 HF checkpoint 直接转换为 F16 GGUF，不做 SmoothQuant 或其他重参数化。Q8/Q4 从同一份 F16 GGUF 分别量化，禁止二次量化。

### 10.2 输出位置

```text
Qwen3.5-35B-A3B/GGUF/
  Qwen3.5-35B-A3B-F16.gguf
  Qwen3.5-35B-A3B-Q8_0_64.gguf
  Qwen3.5-35B-A3B-Q4_0_64.gguf

Qwen3.5-9B/GGUF/
  Qwen3.5-9B-F16.gguf
  Qwen3.5-9B-Q8_0_64.gguf
  Qwen3.5-9B-Q4_0_64.gguf
```

转换前做 converter dry-run；转换后记录 tensor 类型统计、文件大小和 SHA-256。

## 11. 第二阶段：端到端实验

### 11.1 数据集与固定参数

正式数据集：

```text
/mnt/data1/share/datasets/model_evaluation/CCI2-Data/data/cci2-00000-of-00178.raw
/mnt/data1/share/datasets/model_evaluation/wikitext-103-v1/test-00000-of-00001.raw
```

两者均为原始 UTF-8 文本，直接通过 `llama-perplexity -f` 使用。CCI2 不生成前缀或截断副本；接受工具先读取/tokenize 整个 2.884 GB 文件，再按 `--chunks 32` 选择评测 chunk 的行为。

| 参数 | 值 |
|---|---|
| 容器 | `rx-met-zcx` |
| ctx | `4096` |
| chunks | `32`，对每个数据集独立应用 |
| batch | `512` |
| ngl | `99`，显存允许时全 GPU |
| build | `build_cuda_q64_nolut` / `build_cuda_q64_lut` |

LUT ON/OFF 必须使用相同模型文件、数据集路径、ctx、chunks、batch、ngl、GPU 和非 LUT 环境变量。

### 11.2 主 PPL 矩阵

每个模型、每个数据集执行：

| 权重 | LUT OFF | LUT ON |
|---|---:|---:|
| F16 | 是 | 是 |
| Q8_0_64 | 是 | 是 |
| Q4_0_64 | 是 | 是 |

总计：

```text
2 models * 2 datasets * 3 weights * 2 builds = 24 primary PPL combinations
```

每次记录 PPL、统计误差、tok/s、wall time、峰值显存、退出码，以及 NaN/Inf、CUDA error、模型加载和 backend 状态。

### 11.3 KLD 与 Same top-1

每个模型、每个数据集都执行两类比较：

1. 所有候选相对 `F16 + LUT OFF`，衡量总误差。
2. 每个 `LUT ON` 相对同权重 `LUT OFF`，隔离 LUT 增量。

`llama-perplexity` 输出名 `Same top p` 的实现实际比较 argmax token 是否一致。报告统一写成 `Same top-1 (argmax match)`，避免误解为 top-p sampling。

由于同一次运行不能同时读取一个 KLD baseline 并生成另一个 baseline，完整矩阵预计为：

```text
10 runs per (model, dataset)
2 models * 2 datasets * 10 = 40 total model runs
```

40 次中已包含 24 个主 PPL 组合，不是额外再加 40 次。

Qwen3.5 两个模型的词表均为 248320，且 tokenizer SHA-256 相同。按 `ctx=4096, chunks=32` 估算：

```text
one .kld baseline ~= 30.3 GiB
12 retained baselines ~= 363.6 GiB
```

所有 baseline、日志和汇总均保留，不自动清理。执行前检查目标盘空间，单个文件完成后记录实际大小和 SHA-256。

### 11.4 报告指标，不预设质量线

结果表至少包含：

| model | dataset | weight | build | PPL | PPL vs F16/no-LUT | PPL vs same-weight/no-LUT | Mean KLD vs F16 | Mean KLD vs same weight | Same top-1 | tok/s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|

本阶段不使用预设 PPL ratio、KLD 或 Same top-1 门限提前判定成败。完整实验结束后先输出报告，再由审核决定指标是否合格以及是否需要定位或补充实验。

若某项结果异常但程序仍可正常运行，记录异常并继续其他组合。出现 crash、NaN、CUDA error、模型损坏等无法获得有效结果的情况时保留现场日志，并按依赖范围停止受影响运行，不擅自改变参数或减少矩阵后宣称完成。

## 12. 产物位置与保留策略

### 12.1 第一阶段

```text
/home/chengxing.zou.srv/projects/aimet_rx/llama.cpp/tests/test-reex/results/ada300_pwnl_lut_ops/
```

保存 TSV、manifest、比较 JSON、报告和完整日志。

### 12.2 第二阶段

小型日志和跨模型汇总：

```text
/home/chengxing.zou.srv/projects/aimet_rx/llama.cpp/tests/test-reex/results/ada300_pwnl_lut_qwen35/
```

大型 GGUF、KLD baseline 和每模型运行产物：

```text
<model>/GGUF/ada300_pwnl_eval/
  cci2/
  wikitext103/
```

最终实验报告：

```text
/home/chengxing.zou.srv/projects/aimet_rx/doc/ADA300_PWNL_LLAMA_EXPERIMENT_REPORT.md
```

repo 内生成目录加入 `.gitignore`；模型目录本身位于 repo 外。任何产物都不自动删除。禁止自动 commit。

## 13. 可复现性记录

第一、二阶段分别记录：

- abc_lut 和 aimet_rx commit id、dirty status。
- 输入脚本、JSON、header、TSV、GGUF、数据集、KLD 文件 SHA-256。
- Python、编译器、CMake、CUDA、driver 版本。
- GPU 型号和容器标识。
- 完整 CMake 命令及关键 CMakeCache 差异。
- 所有运行命令、环境变量、开始/结束时间和退出码。
- CPU/CUDA LUT 实际命中证明，禁止把 fallback 结果当作 LUT 结果。
- 日志和结果文件的绝对路径。

实验脚本必须使用显式参数并在运行前打印最终解析配置，避免环境变量静默改变口径。

## 14. 实施顺序与审核点

### 第一阶段：LUT 算子验证

1. 在 llama.cpp tests 下实现只读 Python adapter 和 TSV/manifest 协议；不修改 abc_lut。
2. 对最新 16 段 mixed-FP16 数据做全量 bit 比较，生成 llama.cpp header/manifest。
3. 显式禁止 31 段，删除未使用 full-FP16/full-BF16 LUT 路径；power_2 不动。
4. 更新 CPU/CUDA 的 exp、sin/cos、sigmoid 和其他不一致 LUT 数据。
5. 改造 `test-reex-lut.cpp`，增加 reciprocal/rsqrt test-only CUDA adapter。
6. 在 `build_cuda_q64_lut` 中运行全部 bit-level、特殊值和函数精度测试。
7. 生成第一阶段审核报告并停止，等待审核。

### 第二阶段：模型端到端实验

8. 第一阶段审核通过后，完成 LUT/no-LUT 两个一致配置 build 和 backend 冒烟测试。
9. 为两个模型生成或验证 F16/Q8_0_64/Q4_0_64 GGUF。
10. 直接使用两个原始数据集，以 `ctx=4096, chunks=32` 完成 24 个 PPL 组合和两类 KLD/Top-1 对比，共预计 40 次模型运行。
11. 保留所有临时结果、KLD baseline 和日志。
12. 生成完整第二阶段实验报告并停止，等待质量审核。

任何阶段均不得自动执行 `git commit`。
