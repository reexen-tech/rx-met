# reex-hw-convert — REEX 权重 HW 排布转换工具（CPU-only）

把**单个权重张量** `W[N,K]` 从 **native 布局**（GGUF / `quantize_row_*_ref` 产出的行主序 reex block）转换成**硬件模拟器可用的排布**（与 `reex-gemm-datagen` 落盘的 `weight_blocks.bin` 语义一致），并附带自描述 `meta.json`。

- **纯离线转换**，不触碰 llama.cpp 推理路径（计算图 / vec_dot / MUL_MAT kernel 全不动）。
- **不依赖 CUDA**：转换是纯 CPU 活。
- 复用 `reex-gemm-datagen` 已验证的 `reorder`/`repack` 原语（同一份实现，无漂移）——两者共用共享库 `reex-hw-layout`。

> 当前阶段：**仅支持 Legacy block-64 量化**（K-quant 因格式可能变动暂缓；底层机制已通用，将来格式稳定即可放开）。

---

## 1. 支持的量化类型（Legacy block-64）

| `--wtype` | 说明 | native 块字节 | HW 块字节 |
|---|---|---|---|
| `q8_0_64`  | 对称 W8，`{d; qs[64]}` | 66 | 66 |
| `q8_1_64s` | 对称 W8 带行和 `s`，`{d; s; qs[64]}` | 68 | 68 |
| `q4_0_64`  | 对称 W4（nibble），`{d; qs[32]}` | 34 | 34 |
| `q4_1_64`  | 非对称 W4（d+min） | 36 | 36 |
| `q5_0_64`  | 对称 W5（low4+qh） | 42 | 42 |
| `q5_1_64`  | 非对称 W5（d+min） | 44 | 44 |

Legacy 转换 = **整块 memcpy 重排**：块内 reex 结构体字节不变，只按 §4.1 tiling 把块换到新槽位。因此 HW 块字节 == native 块字节。

---

## 2. 布局说明（native → HW）

矩阵约定 `W[N,K]`：`N`=输出维（`ne1`），`K`=输入维（`ne0`，内存连续）。每 `Kt=64` 个 K 元素为一个量化块，每行有 `K/Kt` 个块。

- **native**：行主序，块索引 `native[n*(K/Kt) + sb]`，`sb = k/Kt`。
- **HW（§4.1 tile 序）**：块间 `(kt,nt)` 行主序、tile 内沿 N 列主序：
  ```
  weight_block_slot(n, sb) = (sb*Ntiles + n/Nt)*Nt + n%Nt      Ntiles = N/Nt, Nt = 64
  ```
  即每个 tile `[Nt=64 行 × Kt=64 列]` 装 64 个原子量化块（每行一块），块内字节保持不变。

> 只有块**之间**按列主序换位；块**内部**（某行的 64 个 K 码/结构体字段）不重排。真正的元素级块内重排只发生在纯整数 IntBlock 路径，与 Legacy 无关。

---

## 3. 构建

CPU-only，仅需 `GGML_USE_REEX_Q64`：

```bash
cmake -S . -B build_cpu_wconvert -DCMAKE_BUILD_TYPE=Release \
  -DGGML_USE_REEX_Q64=ON -DREEX_HW_CONVERT=ON -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build_cpu_wconvert -j --target reex-hw-convert
# 产物：build_cpu_wconvert/bin/reex-hw-convert
```

运行时需能找到 `libggml*.so`：
```bash
export LD_LIBRARY_PATH="$PWD/build_cpu_wconvert/bin:$LD_LIBRARY_PATH"
```

---

## 4. 用法与参数

```
reex-hw-convert --wtype NAME --shape N,K
                ( --in FILE | --in-fp32 FILE | --random )
                [--seed S] [--out-dir DIR]
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--wtype NAME` | Legacy 类型名（见 §1） | 必填 |
| `--shape N,K`  | `N`=输出行(ne1)，`K`=输入维(ne0)。要求 `N%64==0 && K%64==0` | 必填 |
| `--in FILE`    | **native 量化 bytes**（GGUF 抽出的原样，行主序 reex struct）→ 仅重排 | 三选一 |
| `--in-fp32 FILE` | **fp32 权重**（`N*K` 个 float，行主序）→ 先量化再重排 | 三选一 |
| `--random`     | 确定性随机生成 fp32（同 datagen 权重配方）→ 量化再重排 | 三选一 |
| `--seed S`     | `--random` 的随机种子 | `1234` |
| `--out-dir DIR`| 输出目录 | `output/wconvert` |

**输入模式怎么选：**
- 已有 GGUF 抽出的量化块 → `--in`（无精度损失，纯搬运）。
- 只有 fp32 权重、想让工具量化 → `--in-fp32`。
- 只是想造测试数据 / 对齐排布 → `--random`。

示例：
```bash
# 1) native 量化 bin（从 GGUF 抽出的 tensor 原始字节）
reex-hw-convert --wtype q8_0_64 --shape 2048,2048 --in weight_native.bin --out-dir out/attn_q

# 2) fp32 权重
reex-hw-convert --wtype q4_0_64 --shape 2048,2048 --in-fp32 w.f32 --out-dir out/attn_q

# 3) 随机
reex-hw-convert --wtype q5_1_64 --shape 128,256 --random --seed 1234 --out-dir out/demo
```

---

## 5. 产物

`--out-dir` 下生成三个文件：

| 文件 | 含义 | 排布 |
|---|---|---|
| `weight_blocks.bin` | **送硬件的量化权重**（HW 排布） | §4.1 块-tile 序（`weight_block_slot`） |
| `weight_native.bin` | 本次使用的 native 量化字节（复现/校验用；`--in` 模式即输入原样） | 行主序 `native[n*(K/Kt)+sb]` |
| `meta.json`         | 自描述元数据 | — |

`meta.json` 字段（节选）：
```jsonc
{
  "gemm":   { "N": 2048, "K": 2048 },
  "weight": { "wtype": "q8_0_64", "family": "Legacy", "W_bits": 8,
              "native_block_bytes": 66, "block_bytes": 66, "n_blocks": 65536,
              "elems_per_block": 64, "repacked": false, "byte_layout": "..." },
  "tiling": { "Mt": 4, "Nt": 64, "Kt": 64, "wgroup": 64, "Ntiles": 32 },
  "index":  { "weight_block_slot(n,sb)": "(sb*Ntiles + n/Nt)*Nt + n%Nt", "sb": "k / Kt" },
  "files":  { "weight_blocks.bin": {...}, "weight_native.bin": {...} }
}
```
模拟器按 `index` 里的公式即可 de-tile 反查。

---

## 6. 校验

`tests/check_wconvert.py`：**类型无关的字节级**校验——对每个 `(n, sb)` 比较
`weight_blocks.bin[weight_block_slot(n,sb)]` 与 `weight_native.bin[n*(K/Kt)+sb]` 的原始字节，
Legacy 期望**逐字节完全相等**（因为就是整块重排）。

```bash
python3 tools/reex-hw-convert/tests/check_wconvert.py out/demo
# wtype=q5_1_64  N=128 K=256  blocks=512  block_bytes=44  mismatches=0
# PASS
```

---

## 7. 边界与后续

- `N%64 != 0` 或 `K%64 != 0` → 报错（暂不做 padding）。
- 非 Legacy（K-quant / IntBlock）→ CLI 拒绝（底层机制已通用，格式稳定后放开即可自动接上 HW bitstream repack）。
- **后续可扩展**：`--in-gguf --tensor <name>` 直接从 GGUF 抽块并推断 `wtype/N/K`；MoE 3D 逐专家；整包 GGUF→HW-GGUF。

底层原语与 §4.1 tiling 的完整定义见 `../reex-gemm-datagen/docs/DESIGN.md`。
