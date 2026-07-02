# reex-gemm-datagen — REEX 量化 GEMM 采数工具（GPU）

在固定形状的 `C = A @ Wᵀ` 上，对不同量化组合生成一整套供**硬件 model 验证**的数据：量化前 FP16/源精度输入、量化后权重/激活块、GPU 结果（11 种 dtype）、CPU golden、以及自描述 `meta.json`。计算路径与数据排布都尽量贴合硬件。

- 计算 **struct-faithful**：权重用 llama.cpp/reex 原生 block，激活按“一个 scale + 连续 intX”量化组，GPU kernel 与 CPU golden 共用同一套整数 MAC。
- 数据按 **§4.1 tiling** 排布：权重/激活在量化块粒度分块，结果在元素粒度分块。
- **需要 CUDA**（GPU 采数）。权重排布 / K-quant HW 重打包等原语复用共享库 `reex-hw-layout`（与 CPU 转换工具 `reex-hw-convert` 同源）。

> 设计与格式细节（tiling、槽位公式、HW bitstream、决策记录）见 **[`docs/DESIGN.md`](docs/DESIGN.md)**。本文只讲**怎么用**。

---

## 1. 构建

需要 `GGML_CUDA` + `GGML_USE_REEX_Q64`：

```bash
cmake -S . -B build_cuda_q64 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_USE_REEX_Q64=ON -DGGML_REEX_FP16_PIPELINE=ON -DGGML_USE_REEX=ON \
  -DREEX_GEMM_DATAGEN=ON -DREEX_GEMM_DATAGEN_CUDA_ARCH=120
cmake --build build_cuda_q64 -j --target reex-gemm-datagen
# 产物：build_cuda_q64/bin/reex-gemm-datagen
```

或用一键脚本（自动配置 + 构建，`REEX_GEMM_DATAGEN_CUDA_ARCH` 默认 120 / RTX5090）：

```bash
bash tools/reex-gemm-datagen/scripts/build.sh
```

> `REEX_GEMM_DATAGEN_CUDA_ARCH`：CUDA 架构（Ada=89，Blackwell/RTX5090=120）。

---

## 2. 用法与参数

```
reex-gemm-datagen [--out DIR] [--wtype NAME] [--abits 16|8|4]
                  [--actin F32|F16|BF16|E5M2|E4M3] [--psum B] [--seed S]
                  [--M m --N n --K k] [--check-rows R]
                  [--wbits 8|6|5|4|3|2] [--asign i|u] [--wsign i|u]   # 仅 IntBlock/--wtype INT
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--out DIR`   | 输出根目录（每个 case 一个时间戳子目录） | `output/datagen` |
| `--wtype NAME`| 权重量化类型（见 §3） | `q8_0_64` |
| `--abits N`   | 激活整数位宽：`16`（int16 容器）/`8`/`4`（int8 容器） | `8` |
| `--actin DT`  | 激活**源精度**（量化前 round）：`F32/F16/BF16/E5M2/E4M3` | `F16` |
| `--psum B`    | 整数 Psum 截断位宽（`0`=不截断） | `0` |
| `--seed S`    | 随机种子 | `1234` |
| `--M/--N/--K` | GEMM 形状 | `8192/2048/2048` |
| `--check-rows R` | dequant golden 校验的行数 | `256` |
| `--wbits/--asign/--wsign` | **仅** `--wtype INT`（纯整数路径）：权重位宽 / 激活符号 / 权重符号 | — |

**合法计算模式**（A_bits × W_bits）：`A8×W8`、`A16×W8`、`A16×W4`、`A8×W4`、`A4×W4`（K-quant 另支持 W2/W3/W5/W6）。

示例：
```bash
reex-gemm-datagen --wtype q8_0_64  --out output/datagen                 # A8×W8
reex-gemm-datagen --wtype q4_0_64  --abits 16                           # A16×W4
reex-gemm-datagen --wtype Q5_K_64S --out output/datagen                 # K-quant W5
reex-gemm-datagen --wtype INT --abits 8 --asign i --wbits 4 --wsign u   # 纯整数 A I8 × W U4
```

---

## 3. 权重量化类型

| 类型 | family | W_bits | 说明 |
|---|---|---|---|
| `q8_0_64` / `q8_1_64s` / `q4_0_64` | Legacy | 8/8/4 | block-64，scale-first，直接落盘 |
| `Q6_K_64` / `Q5_K_64S` / `Q4_K_64S` / `Q3_K_64` / `Q2_K_64S` | Kquant | 6/5/4/3/2 | 256 超块 / 64 子块，落盘前重打包成 HW bitstream |
| `INT` / `W8_16` | IntBlock | CLI 指定 | 纯整数 GEMM（无 scale），独立路径，见 DESIGN §11 |

> 新增类型只在共享库 `reex-hw-layout` 的 `wquant.cpp` registry 加一行；K-quant 还需补 `kq_extract_block` 位布局 + `qmac.cuh` dot + `gemm.cu`/`reference.cpp` dispatch。

---

## 4. 一键脚本

`scripts/`（`_common.sh` 提供 `BUILD_DIR/OUT/M/N/K` 等 env 覆盖 + `run_quant_sweep`）：

| 脚本 | 作用 |
|---|---|
| `build.sh`      | 配置并构建 `reex-gemm-datagen` |
| `gen_legacy.sh` | 生成 Legacy 组（q8_0_64 / q8_1_64s / q4_0_64 的 act_in×A_bits sweep） |
| `gen_kquant.sh` | 生成 K-quant 组 |
| `gen_int.sh`    | 生成纯整数 IntBlock 组 |
| `gen_all.sh`    | 全部生成 |

```bash
bash tools/reex-gemm-datagen/scripts/gen_legacy.sh
# env 覆盖示例：OUT=output/my M=4096 N=1024 K=1024 bash .../gen_legacy.sh
```

---

## 5. 产物（每个 case 一个时间戳目录 `<out>/<case>-YYYYMMDD-HHMMSS/`）

| 文件 | 含义 | dtype | 顺序 |
|---|---|---|---|
| `act_src_<DT>.bin`   | 量化前激活 A[M,K]（源精度） | F32/F16/BF16/E5M2/E4M3 | §4.1 tile `[Mt×Kt]` |
| `weight_src_f16.bin` | 量化前权重 W[N,K] | f16 | §4.1 tile `[Nt×Kt]` |
| `weight_blocks.bin`  | 量化权重（送 HW） | 见 DESIGN §4.1 | 块-tile 序 |
| `act_blocks.bin`     | 量化激活（送 HW） | `{f16 d; intX qs}` | 块-tile 序 |
| `output_<DT>.bin`    | GPU 结果 C[M,N]，**11 份**（无 scale，cast/RNE+饱和） | F16/BF16/E4M3/I16/I8/I6/I4/U16/U8/U6/U4 | 结果 tile 序 |
| `golden_f32.bin`     | CPU golden（实值） | f32 | 同 output |
| `meta.json`          | 自描述元数据（各 `.bin` 的 `block_bytes`/`byte_layout`/反推公式 + gemm/quant/tiling/verify 段） | — | — |

> **模拟器对应**：输入 = `act_src_<DT>.bin` + `weight_blocks.bin`；输出与 11 份 `output_<DT>.bin` 比对。所有块/输出都是 **tile 序**，比对前按 `meta.json` 公式 de-tile。

---

## 6. 校验

```bash
python3 tools/reex-gemm-datagen/tests/check_q5k64s_hw.py output/datagen/<case-timestamp>
```
NumPy 独立复算（解 HW bit-pack、de-tile 源、复算比对 golden）。

---

## 7. 相关

- **权重排布转换（离线、CPU-only）**：见 [`../reex-hw-convert/README.md`](../reex-hw-convert/README.md)——把 GGUF/native 权重转成本工具 `weight_blocks.bin` 同款 HW 排布，不需要 CUDA。
- **设计文档**：[`docs/DESIGN.md`](docs/DESIGN.md)。
