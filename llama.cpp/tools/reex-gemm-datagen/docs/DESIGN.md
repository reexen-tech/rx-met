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
   │                     → C_gpu(fp32 acc,供校验)+ 6 种 dtype 输出(F16/BF16/I16/I8/I6/I4)
   │
   ├─ golden_cpu_*      CPU 整数 golden(共用 qmac, double)→ C_ref;与 C_gpu 逐元素比对(~1e-5)
   ├─ golden_dequant_*  reex dequant + 浮点 matmul 独立交叉校验(psum=0 时)
   │
   └─ dump_case         落盘:act_src(原生 dtype,模拟器输入)+ weight_src(fp16)
                        + weight_blocks + act_blocks(芯片中间产物)+ 6 输出 + golden + meta.json
                        K-quant 权重落盘前重打包成 HW bit-pack;目录名带时间戳
```

> **硬件对应**:模拟器输入 = `act_src_<DT>.bin` + `weight_blocks.bin`;输出与 GPU 采数的 6 份
> `output_<DT>.bin` 比对。激活量化(stage-1)与输出类型转换(OutConv)都在片完成,GPU 采数镜像之。
> CPU golden(double)仅验正确性,**不要求逐 bit**。详见 `docs/REDESIGN_PLAN.md`。

---

## 2. 配置轴(正交)

`case.h` 定义三条正交配置轴 + 一个 case 结构:

| 轴 | 取值 |
|---|---|
| `Family`   | `Kquant` / `Legacy` / `IntBlock` |
| `ActDType` | `F32` / `F16` / `BF16` / `E5M2` / `E4M3`(激活**源精度**,量化前 round;`--actin`) |
| `A_bits`   | `16` / `8` / `4`(激活整数位宽;容器 A16=int16,A8/A4=int8;`--abits`) |
| `OutDType` | `F16` / `BF16` / `I16` / `I8` / `I6` / `I4`(输出精度,**每个 case 一次性全产出**) |

`GemmCase`(默认值,里程碑固定形状):

```
M=8192, N=2048, K=2048      A_bits=8   out=全6种   act_in=F16
psum_bits=0(整数 Psum 不截断)  seed=1234
```

**支持的计算模式**(A_bits × W_bits,合法组合):
`A8×W8`、`A16×W8`、`A16×W4`、`A8×W4`、`A4×W4`(W8=`q8_0_64`/`q8_1_64s`,W4=`q4_0_64`)。

**权重量化类型注册表**(`wquant.cpp`,新增类型只加一行):

| name | family | W_bits | has_min | block 结构 | Kt | scale_bits | encoder |
|---|---|---|---|---|---|---|---|
| `Q6_K_64`  | Kquant | 6 | 否 | `block_q6_K_64`  | 256 | 8 | `quantize_row_q6_K_64_ref` |
| `Q5_K_64S` | Kquant | 5 | 否 | `block_q5_K_64S` | 256 | 6 | `quantize_row_q5_K_64S_ref` |
| `q8_0_64`  | Legacy | 8 | 否 | `block_q8_0_64`  | 64  | 0 | `quantize_row_q8_0_64_ref` |
| `q8_1_64s` | Legacy | 8 | 否 | `block_q8_1_64`(带 `s=d·Σq`) | 64 | 0 | `quantize_row_q8_1_64_ref` |
| `q4_0_64`  | Legacy | 4 | 否 | `block_q4_0_64`(nibble) | 64 | 0 | `quantize_row_q4_0_64_ref` |
| `W8_16`    | IntBlock | 8 | 否 | 裸 int8 `[16×16]` | 16 | 0 | —(纯整数路径,见 §11) |

> `q8_1_64s` 对称量化 W8,额外携带行和 `s`(仅为对齐 HW 数据格式,对称点积不使用)。

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

**槽位公式**(`case.h`,均返回扁平 slot;`Xtiles = X / Xt`):

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
  - `q4_0_64 = { f16 d; nibble qs[64] }`(34B),`w = d·(q−8)`,nibble 交错:元素 `e<32`→`qs[e]&0xF`,`e≥32`→`qs[e−32]>>4`,`d=max/−8`。
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
然后对 **全部 6 种** `OutDType` 各转换一份落盘(同一 case 目录,输入文件不重复)。**全程无 scale。**

- 浮点(`F16`/`BF16`):直接 fp cast(round-to-nearest)。
- 整数(`I16`/`I8`/`I6`/`I4`):**硬件 FP→INT 直转** = `round-half-to-even`(banker's,半数向偶数)
  后**饱和**到两补码范围 `[-(qmax+1), qmax]`(I16 `[-32768,32767]`、I8 `[-128,127]`、I6 `[-32,31]`、
  I4 `[-8,7]`);即 `q = sat(rne(C))`,**不除任何 scale**(`dumper.cpp::rne_lround`,与运行时
  FP 舍入模式无关)。`I6`/`I4` 用 **int8 容器**承载(符号扩展)。
- `output_<DT>.bin`:结果 tile 序(`result_tiled_index`)。
- `golden_f32.bin`:CPU golden(实值),f32,顺序同 output。

---

## 5. 计算(中间精度)

GPU kernel(`gemm.cu`)与 CPU 整数 golden(`reference.cpp`)**共用 `qmac.cuh`** 的整数 MAC,
逐子块流程(以 K-quant 256 超块 / 4 个 64 子块为例):

```
sumi(int32) = Σ_{e=0..63} (w_code - mid) · a_qs[e]      ← int×int8 精确整数累加(mid: Q5=16, Q6=32)
sumi        = reex_q64_psum_trunc_b(sumi, psum_bits)    ← 整数 Psum 截断(保留高 B 有效位;B=0 关闭)
acc(fp32)  += d_glb · sub_scale · a_d · (float)sumi     ← fp32 反量化折算 + 累加
C[m,n]      = Σ_subblock acc → round 到 fp16
```
- Legacy:`sumi = Σ_{j=0..63} w_code·a_qs[j]` → trunc → `w.d·a.d·sumi`(q4_0:`w_code=q−8`)。
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
| `output_<DT>.bin`    | GPU 结果 C[M,N](6 份,**无 scale**,FP→INT round+饱和) | F16/BF16/I16/I8/I6/I4 | 结果 tile 序 |
| `golden_f32.bin`     | CPU 整数 golden     | f32     | 同 output |
| `meta.json`          | 自描述元数据        | —       | — |

`meta.json` 对每个 `.bin` 给出 `total_bytes` / `block_bytes` / `byte_layout` /
`block_index`(或 `elem_index`)反推公式,以及 gemm/quant/tiling/verify 段,可直接对照 de-tile。

---

## 7. 模块结构

| 文件 | 职责 |
|---|---|
| `case.h`            | 配置轴、`TilingSpec`/`tiling_for`、§4.1 槽位公式、激活块访问 |
| `datagen.{h,cpp}`   | 输入生成(A→act_in 源,W→fp16);`quantize_act` 为 host 备用(主路径用 device) |
| `fp8.h`             | E5M2/E4M3 round/encode/decode(decode 为 `RGD_HD`) |
| `convert.cuh`       | **OutConv 黑盒**(`RGD_HD`):fp32/int32 acc → dtype 字节;向偶数+饱和;GEMM/IntBlock/dumper 共用 |
| `actquant.{cuh,cu}` | **stage-1 在片量化** device kernel(A → act_blocks) |
| `wquant.{h,cpp}`    | 权重量化类型注册表、encode、reorder、K-quant HW 重打包(BitWriter) |
| `qmac.cuh`          | host+device 共享整数 MAC / dot 黑盒(镜像 reex `vec_dot_*`)+ Psum 截断 |
| `gemm.{cuh,cu}`     | 通用 GEMM kernel(按 qtype/A_bits dispatch)+ 在片 OutConv 产 6 输出 + host 包装 |
| `reference.{h,cpp}` | CPU 整数 golden + dequant golden + 比对 |
| `dumper.{h,cpp}`    | 落盘(输出由 kernel 产出,dumper 仅写)+ 自描述 meta.json + 时间戳目录 |
| `intgemm.{cuh,cu}`  | **IntBlock 纯整数 GEMM 路径**(独立 kernel,仅共享 OutConv,见 §11) |
| `main.cpp`          | CLI 串联整个流水线(IntBlock 走 §11 分支) |
| `check_q5k64s_hw.py`| NumPy 独立校验(解 HW bit-pack、de-tile 源、复算比对 golden) |

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
./build_cuda_q64/bin/reex-gemm-datagen --wtype q8_0_64  --out output/datagen
./build_cuda_q64/bin/reex-gemm-datagen --wtype Q5_K_64S --out output/datagen
./build_cuda_q64/bin/reex-gemm-datagen --wtype W8_16    --out output/datagen   # 纯整数(§11)
# Legacy 计算模式示例:
./build_cuda_q64/bin/reex-gemm-datagen --wtype q8_1_64s --abits 16 --actin E4M3  # A16×W8
./build_cuda_q64/bin/reex-gemm-datagen --wtype q4_0_64  --abits 16               # A16×W4
./build_cuda_q64/bin/reex-gemm-datagen --wtype q4_0_64  --abits 8                # A8×W4
./build_cuda_q64/bin/reex-gemm-datagen --wtype q4_0_64  --abits 4                # A4×W4
```

校验:

```bash
python3 tools/reex-gemm-datagen/check_q5k64s_hw.py output/datagen/<case-timestamp>
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

### 11.1 数据
- 输入随机生成为整数:`A[M,K] ∈ [-(2^(A_bits-1)-1), +]`、`W[N,K] ∈ [-(2^(W_bits-1)-1), +]`
  (A8/W8 → `[-127,127]`),**不做 float 量化**。
- 排布:block-wise `[16×16]`,元素粒度 tile(`Mt=Nt=Kt=16`):
  - 激活块 = `[16 M-行 × 16 K-列]`,`act_elem_slot(m,k) = ((m/Mt*Ktiles + k/Kt)*Mt + m%Mt)*Kt + k%Kt`
  - 权重块 = `[16 N-行 × 16 K-列]`,`weight_elem_slot(n,k) = ((k/Kt*Ntiles + n/Nt)*Nt + n%Nt)*Kt + k%Kt`
  - 块内行主序;落盘即 `act_blocks.bin` / `weight_blocks.bin`(纯 int8,无 scale)。
- 第一版只注册 `W8_16`(INT8 裸字节)。INT2/4/6 后续:存储按 `W_bits` bit-pack(有符号补码),
  kernel 加按位 unpack 即可,框架已留位。

### 11.2 计算
```
C[m,n] = Σ_{k=0..K-1} A[m,k] · W[n,k]     ← int8×int8 → int32 精确累加(可选 psum 截断)
```
GPU kernel(`kernel_int_gemm`)与 CPU golden 都是这套整数累加,**逐元素精确相等(max_abs=0)**。
输出按全部 6 种 `OutDType` **直接 cast/饱和(无 scale)**:浮点 = `fp16/bf16((float)C)`;
整数 = `sat(C, [-(qmax+1), qmax])`(I16 `[-32768,32767]`、I8 `[-128,127]`、I6 `[-32,31]`、
I4 `[-8,7]`,I6/I4 用 int8 容器)。与量化路径的 FP→INT 规则一致(此处源已是 int32,无需 round)。

> ⚠ **范围饱和**:A8×W8 在 K=2048 上累加,`|C|` 可达 ~1.3e6,远超 fp16 上限 65504、
> 也远超各整数容器范围,直接转换会大面积饱和。因此整数路径**额外固定输出一份精确的 `output_i32.bin`**
> 作为真正的"输出INT数值";其余 `output_<DT>.bin` 仅作容器演示。

### 11.3 产物(`<case>-时间戳/`)
| 文件 | 含义 | dtype |
|---|---|---|
| `weight_blocks.bin` | 裸 INT 权重 `[16×16]` 块 | int8 |
| `act_blocks.bin`    | 裸 INT 激活 `[16×16]` 块 | int8 |
| `output_i32.bin`    | **精确整数输出**(无饱和,首选) | int32 |
| `output_<DT>.bin`   | 直接 cast/clamp 的 6 份输出(**会饱和**) | F16/BF16/I16/I8/I6/I4 |
| `golden_f32.bin`    | CPU 整数 golden(等于 i32) | f32 |
| `meta.json`         | 自描述(整数语义) | — |

> 注:整数路径无量化、无 scale,所以**没有** `*_src_f16.bin` / `out_scale_*`(块本身就是 INT 源数据)。
> case 名格式:`W8_16-A8W8-psum0`。
