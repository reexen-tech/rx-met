# reex-gemm-datagen — 设计与说明

GPU 上的 **量化 GEMM 采数工具**:对固定形状的 `C = A @ Wᵀ` 在不同量化组合下生成
一整套可供 **硬件 model 验证** 的数据(量化前 FP16 源、量化后权重/激活块、GPU 结果、
CPU golden、自描述 meta.json),并且 **计算路径与数据排布都尽量贴合硬件**。

- 计算 **struct-faithful**:权重以 `llama.cpp/reex` 原生 block 结构存储,激活按 CPU 那种
  "一个 scale + 连续 int8" 的量化组存储;GPU kernel 与 CPU golden **共用同一套整数 MAC**。
- 数据 **按 §4.1 tiling 排布**:权重/激活在量化块粒度上分块,结果在元素粒度上分块;
  FP16 源数据也按同样的 tile 顺序落盘,和量化块逐元素对齐。

---

## 1. 总体流水线(镜像硬件三级:量化单元 → MAC → 输出转换)

```
datagen_fill            生成 A[M,K]、W[N,K] 随机数(A 已 round 到 act_in 源精度,W round 到 fp16)
   │
   ├─ wt.encode         [stage-0 离线] 权重量化为 reex 原生 block(roundf)
   │   └─ reorder       搬成 §4.1 块-tile 顺序                      → weight_blocks
   │
   ├─ act_quant_run_host [stage-1 在片量化单元, device kernel]
   │                     A → act_blocks {f16 d; intX qs[agroup]}(amax→scale(fp16)→向偶数→clamp)
   │
   ├─ gemm_run_host     [stage-2 通用 GEMM, device kernel] 读 act_blocks + weight_blocks
   │                     块内 int32 MAC → fp32 折算/累加 → 在片 OutConv(convert.cuh)
   │                     → C_gpu(fp32 acc,供校验)+ 11 种 dtype 输出
   │                       (F16/BF16/E4M3/I16/I8/I6/I4/U16/U8/U6/U4)
   │
   ├─ golden_cpu_*      CPU 整数 golden(共用 qmac, double)→ C_ref;与 C_gpu 逐元素比对(~1e-5)
   ├─ golden_dequant_*  reex dequant + 浮点 matmul 独立交叉校验(psum=0 时)
   │
   └─ dump_case         落盘:act_src(原生 dtype,模拟器输入)+ weight_src(fp16)
                        + weight_blocks + act_blocks(芯片中间产物)+ 11 输出 + golden + meta.json
                        K-quant 权重落盘前重打包成 HW bit-pack;目录名带时间戳
```

> **硬件对应**:模拟器输入 = `act_src_<DT>.bin` + `weight_blocks.bin`;输出与 GPU 采数的 11 份
> `output_<DT>.bin` 比对。激活量化(stage-1)与输出类型转换(OutConv)都在片完成,GPU 采数镜像之。
> CPU golden(double)仅验正确性,**不要求逐 bit**。设计决策依据见 §12。

---

## 2. 配置轴(正交)

`reex_layout.h`(共享库 `reex-hw-layout`,原 `case.h`)定义三条正交配置轴 + 一个 case 结构:

| 轴 | 取值 |
|---|---|
| `Family`   | `Kquant` / `Legacy` / `IntBlock` |
| `ActDType` | `F32` / `F16` / `BF16` / `E5M2` / `E4M3`(激活**源精度**,量化前 round;`--actin`) |
| `A_bits`   | `16` / `8` / `4`(激活整数位宽;容器 A16=int16,A8/A4=int8;`--abits`) |
| `OutDType` | `F16`/`BF16`/`E4M3` / `I16`/`I8`/`I6`/`I4` / `U16`/`U8`/`U6`/`U4`(共 11 种,**每个 case 一次性全产出**) |

`GemmCase`(默认值,里程碑固定形状):

```
M=8192, N=2048, K=2048      A_bits=8   out=全6种   act_in=F16
psum_bits=0(整数 Psum 不截断)  seed=1234
```

**支持的计算模式**(A_bits × W_bits,合法组合):
`A8×W8`、`A16×W8`、`A16×W4`、`A8×W4`、`A4×W4`(量化路径 W8=`q8_0_64`/`q8_1_64s`,
W4=`q4_0_64`;K-quant 另支持 W2/W3/W4/W5/W6,见下表)。纯整数 IntBlock 路径见 §11。

**权重量化类型注册表**(`wquant.cpp`,新增类型只加一行):

| name | family | W_bits | has_min | block 结构 | Kt | scale_bits | encoder |
|---|---|---|---|---|---|---|---|
| `Q6_K_64`  | Kquant | 6 | 否 | `block_q6_K_64`  | 256 | 8 | `quantize_row_q6_K_64_ref` |
| `Q5_K_64S` | Kquant | 5 | 否 | `block_q5_K_64S` | 256 | 6 | `quantize_row_q5_K_64S_ref` |
| `Q4_K_64S` | Kquant | 4 | 否 | `block_q4_K_64S` | 256 | 6 | `quantize_row_q4_K_64S_ref` |
| `Q3_K_64`  | Kquant | 3 | 否 | `block_q3_K_64`  | 256 | 6 | `quantize_row_q3_K_64_ref` |
| `Q2_K_64S` | Kquant | 2 | 否 | `block_q2_K_64S` | 256 | 4 | `quantize_row_q2_K_64S_ref` |
| `q8_0_64`  | Legacy | 8 | 否 | `block_q8_0_64`  | 64  | 0 | `quantize_row_q8_0_64_ref` |
| `q8_1_64s` | Legacy | 8 | 否 | `block_q8_1_64`(带 `s=d·Σq`) | 64 | 0 | `quantize_row_q8_1_64_ref` |
| `q4_0_64`  | Legacy | 4 | 否 | `block_q4_0_64`(nibble) | 64 | 0 | `quantize_row_q4_0_64_ref` |
| `INT`      | IntBlock | CLI | 否 | bit-pack `[16×16]`(位宽/符号由 CLI 定) | 16 | 0 | —(纯整数路径,见 §11) |
| `W8_16`    | IntBlock | 8 | 否 | 裸 int8 `[16×16]`(旧入口) | 16 | 0 | —(同上) |

> `q8_1_64s` 对称量化 W8,额外携带行和 `s`(仅为对齐 HW 数据格式,对称点积不使用)。
> **对称 block-64(Legacy `q4_0_64`/`q8_0_64`,K-quant `Q2/Q4/Q5_K_64S`、`Q3_K_64`/`Q6_K_64`)的量化码与
> sub_scale 均按二进制补码有符号整数存储**(无 `-mid`/`+bias` 零点偏置),反量化即符号扩展。

> `scale_bits` 是 K-quant **HW 落盘时 sub_scale 的位宽**;Legacy 没有 sub_scale,置 0。
> `IntBlock` 是**纯整数、无 scale** 的独立路径(`intgemm.cu`),不走量化/反量化,详见 §11。

---

## 3. Tiling(§4.1)

tile = `[Mt 行 × Nt 列(N) × Kt 列(K)]`。权重/激活在 **量化块粒度** 分块(`Kt == 量化块的 K 元素数`),
结果在 **元素粒度** 分块。

| Family | `Mt,Nt,Kt` | 激活 tile `[Mt×Kt]` | 权重 tile `[Nt×Kt]` | agroup | wgroup |
|---|---|---|---|---|---|
| Kquant   | `1,64,256`  | `[1,256]`  | `[64,256]` | 256 | 64 |
| Legacy   | `4,64,64`   | `[4,64]`   | `[64,64]`  | 64  | 64 |
| IntBlock | `16,16,16`  | `[16,16]`  | `[16,16]`  | 16  | 16 |

**槽位公式**(`reex_layout.h`,均返回扁平 slot;`Xtiles = X / Xt`):

- 权重块(块间 `(kt,nt)` 行主序,块内 N 列主序):
  `weight_block_slot(n,sb) = (sb*Ntiles + n/Nt)*Nt + n%Nt`,`sb = k/Kt`
- 激活组块(块间 `(mt,kt)` 行主序,块内行主序;`agroup==Kt` 时每 tile 行 1 组):
  `act_group_slot(m,kg) = (m/Mt*Ktiles + kg)*Mt + m%Mt`,`kg = k/Kt`
- 结果元素(块间 `(mt,nt)` 列主序,块内行主序):
  `result_tiled_index(m,n) = (n/Nt*Mtiles + m/Mt)*(Mt*Nt) + (m%Mt)*Nt + n%Nt`
- FP16 源(每个量化块展开成 Kt 个 fp16,与量化块对齐):
  - 激活:`slot(m,k) = act_group_slot(m, k/Kt)*Kt + k%Kt`
  - 权重:`slot(n,k) = weight_block_slot(n, k/Kt)*Kt + k%Kt`

---

## 4. 数据表示

### 4.1 权重量化块 `weight_blocks.bin`
- **Legacy(对称)**:reex 原生 struct,scale 在前,直接落盘,无需重打包:
  - `q8_0_64 = { f16 d; int8 qs[64] }`(66B),`w = d·q`。
  - `q8_1_64s = { f16 d; f16 s; int8 qs[64] }`(68B),`w = d·q`,`s=d·Σq`(HW 字段,点积不用)。
  - `q4_0_64 = { f16 d; nibble qs[64] }`(34B),`w = d·q`(q 为**有符号 4-bit 补码** [−8,7]),nibble 交错:元素 `e<32`→`qs[e]&0xF`,`e≥32`→`qs[e−32]>>4`,`d=−max/8`。
- **K-quant**:落盘前重打包成 **HW 连续 LSB-first bitstream**(不字节对齐):
  ```
  glb_scale(fp16,16b) + 4 × [ sub_scale(scale_bits 位, 有符号补码) + 64 × code(W_bits 位) ]
  ```
  - `Q5_K_64S`:`16 + 4×(6 + 64×5) = 1320 bit = 165 B/块`,sub_scale=int6、code=5bit。
  - `Q6_K_64`:sub_scale=int8、code=6bit。
  - 实现见 `wquant.cpp`:`kq_extract_block`(解出 codes + 有符号 sub_scale + glb)、
    `BitWriter`(逐位写)、`wquant_repack_hw`。

### 4.2 激活量化块 `act_blocks.bin`
一个连续量化组一个块,容器宽度随 `A_bits`(A16=int16,A8/A4=int8):
```
{ f16 d ; intX qs[agroup] }   X: A16→i16, A8/A4→i8
  Legacy agroup=64:  A16→130B, A8/A4→66B     Kquant agroup=256: A8→258B
```
每组一个 scale,`qmax = 2^(A_bits-1)-1`(A16=32767/A8=127/A4=7),
`scale = amax/qmax`,`q = clamp(round(v/scale), ±qmax)`(`datagen.cpp::quantize_act`)。

### 4.3 源数据 `act_src_<DT>.bin` / `weight_src_f16.bin`
量化 **前** 的源,按 §4.1 tile 序落盘(每个量化块对应的 Kt 个元素与该块对齐)。
- **激活**按 `act_in` 的 **原生 dtype** 存:`F32`=4B、`F16`/`BF16`=2B、`E5M2`/`E4M3`=1B
  (`E5M2`=1-5-2 bias15,`E4M3`=1-4-3 bias7、无 Inf;RNE,见 `fp8.h`)。文件名含 dtype。
- **权重**统一存 fp16(`weight_src_f16.bin`)。
HW 既可吃源数据自己量化,也可直接吃量化块。

### 4.4 输出与 golden
输出类型本质是 **"算完一次,最后做一次类型转换"**:GEMM 主体只算一遍(fp32 实值 `C[M,N]`),
然后对 **全部 11 种** `OutDType` 各转换一份落盘(同一 case 目录,输入文件不重复)。
**全程无 scale,仅饱和**(转换实现见 `convert.cuh`,按 `OutKind` 分派)。

- 浮点(`F16`/`BF16`):直接 fp cast(round-to-nearest);`F16` 上溢 → ±Inf。
- `E4M3`(fp8 1-4-3,bias 7,**无 Inf 编码**):RNE + **饱和到 ±448**(超出不溢出成 Inf),1 字节容器。
- 有符号整数(`I16`/`I8`/`I6`/`I4`):**硬件 FP→INT 直转** = `round-half-to-even`(半数向偶数)
  后**饱和**到两补码范围 `[-(qmax+1), qmax]`(I16 `[-32768,32767]`、I8 `[-128,127]`、I6 `[-32,31]`、
  I4 `[-8,7]`)。`I6`/`I4` 用 **int8 容器**承载(符号扩展)。
- 无符号整数(`U16`/`U8`/`U6`/`U4`):RNE 后**饱和**到 `[0, 2ⁿ−1]`(U16 `[0,65535]`、U8 `[0,255]`、
  U6 `[0,63]`、U4 `[0,15]`),**负值钳到 0**。`U6`/`U4` 用 **uint8 容器**承载(零扩展)。
- 均**不除任何 scale**(`convert.cuh`,与运行时 FP 舍入模式无关)。
- `output_<DT>.bin`:结果 tile 序(`result_tiled_index`)。
- `golden_f32.bin`:CPU golden(实值),f32,顺序同 output。

---

## 5. 计算(中间精度)

GPU kernel(`gemm.cu`)与 CPU 整数 golden(`reference.cpp`)**共用 `qmac.cuh`** 的整数 MAC,
逐子块流程(以 K-quant 256 超块 / 4 个 64 子块为例):

```
sumi(int32) = Σ_{e=0..63} w_code · a_qs[e]             ← int×int8 精确整数累加
                                                          (w_code 为有符号补码,符号扩展;无 -mid 偏置)
sumi        = reex_q64_psum_trunc_b(sumi, psum_bits)    ← 整数 Psum 截断(保留高 B 有效位;B=0 关闭)
acc(fp32)  += d_glb · sub_scale · a_d · (float)sumi     ← fp32 反量化折算 + 累加(sub_scale 亦有符号补码)
C[m,n]      = Σ_subblock acc → round 到 fp16
```
- Legacy:`sumi = Σ_{j=0..63} w_code·a_qs[j]` → trunc → `w.d·a.d·sumi`(q4_0:`w_code` 为有符号 4-bit 补码)。
- **中间精度固定:块内整数 Psum = INT32,块间累加 = FP32**(`acc`)。GPU 在 fp32 累加,CPU golden 在
  double 累加(故二者 ~1e-5 量级差,以 **GPU 输出为准**)。
- 各模式每 64 块的整数 Psum 上界均 ≤ INT32:A8×W8 ~21bit、A16×W8 ~29bit、A16×W4 ~25bit、
  A8×W4 ~17bit、A4×W4 ~13bit(故无需 int64;窄 HW 累加器可用 `--psum B` 模拟)。

**双 golden 校验**:
1. 整数 golden(与 GPU 同一 MAC,支持 psum 截断)→ 与 GPU 逐元素比对(`max_abs/max_rel/mse` 写入 meta)。
2. dequant golden(`reex::dequantize_row_*` + 浮点 matmul,仅 `psum=0` 有效)→ 独立交叉校验前若干行。

---

## 6. 产物清单(每个 case 一个时间戳目录)

`<out>/<case_name>-YYYYMMDD-HHMMSS/`:

| 文件 | 含义 | dtype | 顺序 |
|---|---|---|---|
| `act_src_<DT>.bin`   | 量化前激活 A[M,K](原生 act_in dtype) | F32/F16/BF16/E5M2/E4M3 | §4.1 tile `[Mt×Kt]` |
| `weight_src_f16.bin` | 量化前权重 W[N,K]   | f16     | §4.1 tile `[Nt×Kt]` |
| `weight_blocks.bin`  | 量化权重(送 HW)   | 见 §4.1 | 块-tile 序 |
| `act_blocks.bin`     | 量化激活(送 HW)   | `{f16 d; intX qs}` 块 | 块-tile 序 |
| `output_<DT>.bin`    | GPU 结果 C[M,N](11 份,**无 scale**,cast/RNE+饱和) | F16/BF16/E4M3/I16/I8/I6/I4/U16/U8/U6/U4 | 结果 tile 序 |
| `golden_f32.bin`     | CPU 整数 golden     | f32     | 同 output |
| `meta.json`          | 自描述元数据        | —       | — |

`meta.json` 对每个 `.bin` 给出 `total_bytes` / `block_bytes` / `byte_layout` /
`block_index`(或 `elem_index`)反推公式,以及 gemm/quant/tiling/verify 段,可直接对照 de-tile。

---

## 7. 模块结构

**目录布局**(内部工具,头文件与实现并放在 `src/`,**不单独建 `include/`**):

```
tools/reex-gemm-datagen/
├── CMakeLists.txt        # 构建定义(源文件路径均以 src/ 为前缀)
├── .gitignore            # 排除 output/ 产物
├── docs/                 # DESIGN.md(单一设计文档)
├── scripts/              # 一键脚本:build.sh / gen_{legacy,kquant,int,all}.sh(_common.sh 为共享库)
├── src/                  # 全部 C++/CUDA 源码与头文件(.cpp/.cu/.h/.cuh)
├── tests/                # check_q5k64s_hw.py 等独立校验脚本(不参与 C++ 构建)
└── output/               # 生成产物(每 case 一个时间戳目录,git 忽略)
```

> `src/` 内的 `#include "..."` 均为相对引用,编译器按源文件所在目录解析即可命中(无需额外
> `target_include_directories`);外部头(`ggml.h`、`reex/...`)由 `ggml` target 的 include 目录提供。

> **共享库**:类型注册表 / §4.1 tiling+槽位公式 / reorder / K-quant HW 重打包
> 已抽到独立库 **`reex-hw-layout`**(`tools/reex-hw-convert/`),datagen 链接复用之
> (单一实现,无漂移)。对应文件 `reex_layout.h`(原 `case.h`)与 `wquant.{h,cpp}`
> 现位于 `tools/reex-hw-convert/src/`,datagen 通过该库的 `PUBLIC` include 目录引用。
> CPU-only 权重转换器 CLI `reex-hw-convert`(native/fp32/random → HW `weight_blocks.bin`
> + `meta.json`,`-DREEX_HW_CONVERT=ON`,不依赖 CUDA)与其字节级校验
> `tests/check_wconvert.py` 同在 `tools/reex-hw-convert/`。

下表文件位于 `src/`(除标注属共享库者):

| 文件 | 职责 |
|---|---|
| `reex_layout.h` *(共享库)* | 配置轴、`TilingSpec`/`tiling_for`、§4.1 槽位公式、激活块访问(原 `case.h`) |
| `datagen.{h,cpp}`   | 输入生成(A→act_in 源,W→fp16);`quantize_act` 为 host 备用(主路径用 device) |
| `fp8.h`             | E5M2/E4M3 round/encode/decode(`RGD_HD`,host+device;encode 供 E4M3 输出与激活源量化共用) |
| `convert.cuh`       | **OutConv 黑盒**(`RGD_HD`):fp32/int32 acc → 11 种 dtype 字节;按 `OutKind` 分派(float cast / E4M3 饱和 / 有符号补码饱和 / 无符号饱和);GEMM/IntBlock/dumper 共用 |
| `actquant.{cuh,cu}` | **stage-1 在片量化** device kernel(A → act_blocks) |
| `wquant.{h,cpp}` *(共享库)* | 权重量化类型注册表、encode、reorder、K-quant HW 重打包(BitWriter) |
| `qmac.cuh`          | host+device 共享整数 MAC / dot 黑盒(镜像 reex `vec_dot_*`)+ Psum 截断 |
| `gemm.{cuh,cu}`     | 通用 GEMM kernel(按 qtype/A_bits dispatch)+ 在片 OutConv 产 6 输出 + host 包装 |
| `reference.{h,cpp}` | CPU 整数 golden + dequant golden + 比对 |
| `dumper.{h,cpp}`    | 落盘(输出由 kernel 产出,dumper 仅写)+ 自描述 meta.json + 时间戳目录 |
| `intgemm.{cuh,cu}`  | **IntBlock 纯整数 GEMM 路径**(独立 kernel,仅共享 OutConv,见 §11) |
| `main.cpp`          | CLI 串联整个流水线(IntBlock 走 §11 分支) |

`tests/check_q5k64s_hw.py`(**不参与构建**):NumPy 独立校验(解 HW bit-pack、de-tile 源、复算比对 golden;权重码按有符号补码符号扩展,对齐 §12.8.1)。

---

## 8. 构建与运行

仅在 `REEX_GEMM_DATAGEN=ON` 时构建(依赖 `GGML_CUDA` + `GGML_USE_REEX_Q64`)。

```bash
cmake -S . -B build_cuda_q64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_USE_REEX_Q64=ON -DGGML_REEX_FP16_PIPELINE=ON -DGGML_USE_REEX=ON \
  -DREEX_GEMM_DATAGEN=ON -DREEX_GEMM_DATAGEN_CUDA_ARCH=120
cmake --build build_cuda_q64 -j --target reex-gemm-datagen
```

运行(CLI,`main.cpp`):

```bash
# --out DIR --wtype NAME --abits 16|8|4 --actin F32|F16|BF16|E5M2|E4M3 --psum B
#           --seed S --M m --N n --K k --check-rows R
#           --wbits 8|6|5|4|3|2 --asign i|u --wsign i|u   (仅 IntBlock/--wtype INT 路径)
./build_cuda_q64/bin/reex-gemm-datagen --wtype q8_0_64  --out output/datagen
./build_cuda_q64/bin/reex-gemm-datagen --wtype Q5_K_64S --out output/datagen
# Legacy 计算模式示例:
./build_cuda_q64/bin/reex-gemm-datagen --wtype q8_1_64s --abits 16 --actin E4M3  # A16×W8
./build_cuda_q64/bin/reex-gemm-datagen --wtype q4_0_64  --abits 16               # A16×W4
./build_cuda_q64/bin/reex-gemm-datagen --wtype q4_0_64  --abits 8                # A8×W4
./build_cuda_q64/bin/reex-gemm-datagen --wtype q4_0_64  --abits 4                # A4×W4
# 纯整数(§11):--wtype INT,位宽/符号由 --abits/--wbits/--asign/--wsign 指定
./build_cuda_q64/bin/reex-gemm-datagen --wtype INT --abits 8 --asign i --wbits 4 --wsign u  # A I8 × W U4
```

校验:

```bash
python3 tools/reex-gemm-datagen/tests/check_q5k64s_hw.py output/datagen/<case-timestamp>
```

> 注:`psum_bits` 目前在 `main.cpp` 内固定为 0(无 CLI 开关),需要截断时改这里或加开关重产。

---

## 9. 扩展指南

- **新增权重量化类型**:在 `wquant.cpp` 的 `g_registry` 加一行(name/family/W_bits/has_min/
  block 结构/Kt/scale_bits/encoder);若是 K-quant,在 `kq_extract_block` 补该类型的位布局,
  并在 `qmac.cuh` 加对应 dot、在 `gemm.cu`/`reference.cpp` 加 dispatch 分支。
- **新增 tiling**:改 `tiling_for`(`Mt,Nt,Kt,wgroup,agroup`);槽位公式自动套用。
- **新激活/输出精度**:`ActDType`/`OutDType` 已留位,在 `datagen`/`dumper` 补对应转换即可。

---

## 10. 与硬件 model 对齐时需确认

1. **Psum 截断位宽**:当前 `B=0`(不截断);若 HW 按 B 位截整数累加,需对齐并重产(golden 随之变)。
2. **HW 读哪份输入**:吃 `*_src_f16`(自己量化,需对齐量化算法/舍入)或吃 `*_blocks`(跳过量化分歧)。
3. **反量化折算/累加**:当前 `(d·sc)·ad·sumi` 逐块 fp32 累加;HW 若定点或不同结合顺序/累加位宽会有末位差。
4. **输出舍入**:fp32→fp16 舍入模式需一致。
5. **排布**:输出/块都是 tile 序(非自然序),比对前按 meta.json 公式 de-tile。

---

## 11. IntBlock — 纯整数 GEMM(无 scale)

与量化路径**完全并行的独立路径**(`intgemm.cu`),输入/权重/输出**都是裸 INT 数值,不带任何 scale**。
`main.cpp` 检测到 `family == IntBlock` 时直接走这条路,不经过量化/反量化/qmac。
入口 `--wtype INT`,位宽与符号由 CLI 指定:`--abits/--wbits`(16/8/6/5/4/3/2)、`--asign/--wsign`(`i`=有符号补码 / `u`=无符号)。

### 11.1 数据
- 输入随机生成为整数,**全范围**(不做 float 量化):
  - 有符号:`[-2^(b-1), 2^(b-1)-1]`(如 INT8 `[-128,127]`、INT4 `[-8,7]`);
  - 无符号:`[0, 2^b-1]`(如 UINT8 `[0,255]`、UINT4 `[0,15]`)。
- **存储:按各自位宽 bit-pack 紧密存放**(`put_bits`,LSB-first;sub-byte 不字节对齐;
  有符号取补码低 b 位)。容器不再固定 1 字节/元素。
- 排布:block-wise `[16×16]`,元素粒度 tile(`Mt=Nt=Kt=16`):
  - **激活块** `[16 M-行 × 16 K-列]`,**块内行主序、块间 `(mt,kt)` 行主序**:
    `act_elem_slot(m,k) = ((m/Mt*Ktiles + k/Kt)*Mt + m%Mt)*Kt + k%Kt`
  - **权重块** `[16 N-行 × 16 K-列]`,**块内列(N)主序、块间 `(kt,nt)` 行主序**:
    `weight_elem_slot(n,k) = ((k/Kt*Ntiles + n/Nt)*Kt + k%Kt)*Nt + n%Nt`
  - bit-pack 的 slot 序即上式;落盘 `act_blocks.bin` / `weight_blocks.bin`(纯 INT,无 scale)。

### 11.2 计算
```
C[m,n] = Σ_{k=0..K-1} A[m,k] · W[n,k]     ← INT32 逐步饱和累加(每步累加后 clamp 回 int32;可选 psum 截断)
```
GPU kernel(`kernel_int_gemm`)与 CPU golden 都是这套**步进式饱和 INT32 累加**(用 `long long`
做积与累加、每步饱和回 `int32`),**逐元素精确相等(max_abs=0)**。结果(已饱和的 int32)按全部
**11 种** `OutDType` **直接 cast/饱和(无 scale)**:浮点 = `fp16/bf16((float)C)`(FP16 上溢→Inf);
`E4M3` = RNE+饱和 ±448;有符号整数 = `sat(C, [-(qmax+1), qmax])`(I16 `[-32768,32767]`…I4 `[-8,7]`,
I6/I4 用 int8 容器);无符号整数 = `sat(C, [0, 2ⁿ−1])`(负值→0,U6/U4 用 uint8 容器)。
与量化路径输出规则一致(此处源已是 int32,整数转换无需 round)。

> ⚠ **范围饱和**:大 K 累加后 `|C|` 可达 ~1e6,远超 fp16 / 各整数容器范围,逐 dtype 转换会大面积饱和。
> 因此整数路径**额外固定输出一份 `output_i32.bin`**(已步进饱和的 INT32)作为真正的"输出INT数值";
> 其余 11 份 `output_<DT>.bin` 仅作容器演示。

### 11.3 产物(`<case>-时间戳/`)
| 文件 | 含义 | dtype |
|---|---|---|
| `weight_blocks.bin` | 裸 INT 权重 `[16×16]` 块(按 `--wbits` bit-pack) | bit-packed |
| `act_blocks.bin`    | 裸 INT 激活 `[16×16]` 块(按 `--abits` bit-pack) | bit-packed |
| `output_i32.bin`    | **饱和 INT32 输出**(首选,无逐 dtype clamp) | int32 |
| `output_<DT>.bin`   | 由饱和 i32 cast/clamp 的 11 份输出(**会饱和**) | F16/BF16/E4M3/I16/I8/I6/I4/U16/U8/U6/U4 |
| `golden_f32.bin`    | CPU 整数 golden 转 f32(|C|<2²⁴ 精确;首选 i32) | f32 |
| `meta.json`         | 自描述(整数语义、`--asign/--wsign` 范围) | — |

> 注:整数路径无量化、无 scale,所以**没有** `*_src_f16.bin` / `out_scale_*`(块本身就是 INT 源数据)。
> case 名格式:`int-A{I|U}{abits}-W{I|U}{wbits}-psum{B}`(`I`=signed/`U`=unsigned,如 `int-AI8-WU4-psum0`)。

---

## 12. 设计决策记录(为什么)

> 本节是「为什么这么定」的依据(原 `REDESIGN_PLAN.md` §7/§8 已并入此处);上文各节是「现在是什么」。

**核心架构决策(初版定稿,均已确认)**

1. **激活在线量化 / 权重离线量化**:激活由芯片第一级在片量化(stage-1 device kernel 吃 `act_src`),
   权重离线参考量化(roundf)。经 llama.cpp `from_float → wdata → vec_dot` 流程佐证:独立量化 kernel → 通用 GEMM kernel。
2. **激活 block 不带 sum**:全对称量化,点积用不上行和(仅 `q8_1_64s` 为对齐 HW 数据格式额外携带 `s`,计算不使用)。
3. **量化/输出舍入统一为半数向偶数(RNE)**;scale 存 **fp16**;容器宽度 A16=int16、A8/A4=int8。
4. **输出仅饱和、无 scale**:INT=两补码全范围 `[-(qmax+1),qmax]`;UINT=`[0,2ⁿ-1]`(负值→0);E4M3=RNE+饱和±448;F16/BF16=RNE。
5. **中间精度**:块内 INT32、块间 FP32;**不要求**与 CPU 逐 bit(故不关 FMA,golden 用 double 仅验正确性,~1e-5)。
6. **IntBlock 路径独立**:裸整数、tiling[16,16]、INT32 精确,**仅共享 OutConv**,不强行并入量化大 kernel。
7. **dispatch 策略**:运行期 tag + kernel 内 `switch`(权重类型 / A_bits / 输出),线程一致分支不发散。
8. **黑盒接口**(GPU/CPU golden 共用,均 `RGD_HD`):`ActReader`(读 `{f16 d; intX q}`)、`WeightCodec`(解码 block→码+scale)、
   `Dot`(块内 MAC + psum 截断 + fp32 折算)、`OutConv`(fp32 acc→11 dtype 字节)。对应实现见 §7 模块表。

**后续增补(初版之后确认并实现)**

8.1. **对称量化改有符号补码**:对齐 reex 上游 commit `f95e0c8`——所有对称 block-64 量化码与 sub_scale 改为
   二进制补码有符号整数(去掉 `-mid`/`+bias` 零点),反量化即符号扩展;`qmac.cuh`/`wquant.cpp`/`reference.cpp` 同步(`q64_unpack4x*_s`)。
8.2. **K-quant 扩展 W2/W3/W4**:新增 `Q2_K_64S`(scale_bits=4)、`Q3_K_64`、`Q4_K_64S`(scale_bits=6),含 dot/GPU kernel/dispatch/HW 重打包。
8.3. **IntBlock 泛化**:`--wtype INT` + `--abits/--wbits/--asign/--wsign`;支持有符号补码(全范围)与无符号;权重块内列(N)主序;
   bit-pack 紧密存放;累加为步进式饱和 INT32;case 名 `int-A{I|U}b-W{I|U}b-psum{B}`。
8.4. **输出类型扩到 11 种**:新增 `E4M3`(RNE+饱和±448、无 Inf)与 `U16/U8/U6/U4`(饱和 `[0,2ⁿ-1]`,负值→0);`convert.cuh` 按 `OutKind` 分派。
8.5. **目录归整**:源码全部移入 `src/`(CMake 路径加前缀,内部工具不建 `include/`);`check_q5k64s_hw.py` 移入 `tests/`
   并修正权重码解码为有符号补码符号扩展(对齐 8.1,否则跑当前 Q5_K_64S 会 FAIL)。
