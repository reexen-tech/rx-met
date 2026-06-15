# REEX block-64 量化改动清单

为对齐硬件,新增 block size = 64 的量化类型(隔离在 `reex/` 与 `#ifdef GGML_USE_REEX_Q64` 下)。

- Legacy(block=64):`Q4_0_64`、`Q4_1_64`、`Q5_0_64`、`Q5_1_64`、`Q8_0_64`、`Q8_1_64`
- K-quant(superblock=256,subblock=64):`Q4_K_64`、`Q2_K_64`、`Q3_K_64`、`Q5_K_64`、`Q6_K_64`
- K-quant 对称版(superblock=256,subblock=64,有符号子块 scale、无 min,`w = d·scale·(q−mid)`):
  - `Q2_K_64S`(2-bit,子块 scale **int4** 有符号,mid=2)
  - `Q4_K_64S`(4-bit,子块 scale **int6** 有符号,mid=8)
  - `Q5_K_64S`(5-bit,子块 scale **int6** 有符号,mid=16)
  - 子块 scale 位宽与各自非对称版一致(Q2_K:4 / Q4_K、Q5_K:6);超块 scale 为 fp16 `d`

---

## 新增文件

| 文件 | 功能 |
|---|---|
| `ggml/include/reex/ggml-reex-q64-common.h` | 11 种类型的 block 结构体与常量、位打包/解包辅助 |
| `ggml/include/reex/ggml-reex-q64.h` | base 侧参考量化/反量化函数声明 |
| `ggml/src/ggml-reex-q64.c` | base 侧参考量化/反量化实现 |
| `ggml/src/ggml-cpu/reex/reex_q64_quants.h` | CPU 侧 `from_float` / `vec_dot` 声明 |
| `ggml/src/ggml-cpu/reex/reex_q64_quants.c` | CPU 侧 `from_float` / `vec_dot` 标量实现 |
| `ggml/src/ggml-cuda/reex/reex_q64_dequant.cuh` | CUDA legacy 反量化 kernel |
| `ggml/src/ggml-cuda/reex/reex_q64_vecdotq.cuh` | CUDA legacy MMVQ vec_dot |
| `ggml/src/ggml-cuda/reex/reex_q64_mmq.cuh` | CUDA legacy MMQ load_tiles |
| `ggml/src/ggml-cuda/reex/reex_q64_kquant_dequant.cuh` | CUDA K-quant 反量化 kernel |
| `ggml/src/ggml-cuda/reex/reex_q64_kquant_vecdotq.cuh` | CUDA K-quant MMVQ vec_dot(decode) |
| `ggml/src/ggml-cuda/reex/reex_q64_kquant_mmq.cuh` | CUDA K-quant MMQ load_tiles(prefill) |
| `ggml/src/ggml-cuda/reex/reex_q64_kquant_getrows.cuh` | CUDA K-quant get_rows(gather+反量化) |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q*_64.cu` | 11 种类型的 MMQ kernel 显式模板实例 |

---

## 修改文件

| 文件 | 改动 / 功能 |
|---|---|
| `ggml/include/ggml.h` | `ggml_type` 枚举新增 11 种类型,更新 `GGML_TYPE_COUNT` |
| `ggml/src/ggml.c` | `type_traits` 表项 + `ggml_quantize_chunk` 分支 |
| `ggml/src/ggml-quants.c` | `validate_row_data` 分支 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `type_traits_cpu` 表项 |
| `ggml/src/ggml-cpu/ops.cpp` | CPU `get_rows` 分支 |
| `ggml/src/ggml-cuda/common.cuh` | CUDA `type_traits`(qk/qr/qi) |
| `ggml/src/ggml-cuda/convert.cu` | `to_fp16` / `to_fp32` 反量化分发 |
| `ggml/src/ggml-cuda/getrows.cu` | CUDA `get_rows` 分发(legacy + K-quant) |
| `ggml/src/ggml-cuda/mmvq.cu` | MMVQ 分发(vec_dot / vdr / switch_type) |
| `ggml/src/ggml-cuda/mmq.cu` | MMQ 分发 + `should_use_mmq` |
| `ggml/src/ggml-cuda/mmq.cuh` | MMQ tile 尺寸 / `mmq_type_traits` / ds_layout |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `supports_op`、`mul_mat` 路径分发、`get_rows` 类型白名单、fusion 门控 |
| `include/llama.h` | `llama_ftype` 新增 11 项 |
| `src/llama-quant.cpp` | ftype→ggml_type 映射 + tensor 回退逻辑 |
| `src/llama-model-loader.cpp` | ftype 字符串/类型映射 |
| `tools/quantize/quantize.cpp` | `llama-quantize` 注册 11 种格式 |
| `gguf-py/gguf/constants.py` | GGUF 类型 / ftype / quant size |
| `ggml/CMakeLists.txt`、`ggml/src/CMakeLists.txt`、`ggml/src/ggml-cpu/CMakeLists.txt`、`ggml/src/ggml-cuda/CMakeLists.txt` | 编译新增 `reex/` 源文件与模板实例 |

---

## 功能覆盖

| 路径 | Legacy(6) | K-quant(5) |
|---|---|---|
| CPU `from_float` / `to_float` / `vec_dot` | ✅ | ✅ |
| CUDA 反量化(cuBLAS) | ✅ | ✅ |
| CUDA MMVQ(decode) | ✅ | ✅(标量) |
| CUDA prefill(大 batch) | ✅ 分块 MMVQ | ✅ 分块 MMVQ |
| CUDA `get_rows` | ✅ | ✅ |
| 量化工具链 / GGUF | ✅ | ✅ |

> prefill 不再走 MMQ:block-64 类型在 `ggml_cuda_mul_mat` 被强制路由到 CPU 对齐的 MMVQ(见下文),MMQ 内核虽仍编译但不被这些类型使用。
>
> 对称 K-quant(`Q2_K_64S`/`Q4_K_64S`/`Q5_K_64S`)功能与 5 个非对称 K-quant 一致(CPU/反量化/MMVQ decode+prefill/get_rows/工具链/GGUF),复用 q8_K 激活、走对称分支(无 min 修正项),并因 prefill 统一走 MMVQ 而无需 MMQ 实例。子块有符号 scale 用 signed-biased 打包(int6→value+32 复用 `q64_pack4x6`;int4→value+8 新增 `q64_pack4x4`),反量化 `w = d·scale·(q−mid)`。已验证 GPU≡CPU(默认 PPL 仅小数点后第 3–4 位差异;`REEX_Q64_PSUM_BITS` 截断下同样对齐)。

未做(性能/质量项,不影响功能):imatrix、CPU SIMD、MMVQ fusion、其他后端(Metal/Vulkan/SYCL)。

---

## 定点 Psum 精度截断(可选)

模拟有限位宽硬件:每个 scale 组的整数 Psum(`q_w*q_a` 累加器)在乘 scale 前,按目标有效位宽 `B` 从 LSB 侧截断(丢 `nbits-B` 位,移位量折进反量化),符号保留、按幅值向零截断。默认关闭。

- 配置:环境变量 `REEX_Q64_PSUM_BITS`(`<=0`/未设 = 关闭)
- 作用粒度:统一为 **64 元素整数 Psum**(legacy 每块;K-quant 每子块 64 元素 `acc`,min 修正项不截断)

| 文件 | 改动 / 功能 |
|---|---|
| `ggml/include/reex/ggml-reex-q64-common.h` | `reex_q64_psum_trunc_b(psum, B)` 纯整数截断(host+device) |
| `ggml/include/reex/ggml-reex-q64.h` | `reex_q64_psum_bits()` 声明 |
| `ggml/src/ggml-reex-q64.c` | `reex_q64_psum_bits()` 读环境变量(缓存) |
| `ggml/src/ggml-cpu/reex/reex_q64_quants.c` | CPU 11 个 vec_dot 在 Psum 上应用截断 |
| `ggml/src/ggml-cuda/mmvq.cu` | `__constant__ reex_q64_psum_bits_dev` + `reex_q64_sync_psum_bits()`(按设备缓存同步 B) |
| `ggml/src/ggml-cuda/reex/reex_q64_vecdotq.cuh`、`reex_q64_kquant_vecdotq.cuh` | 11 个 MMVQ vec_dot 截断合并后的 64 元素 Psum |

覆盖:CPU 全 11 种 + CUDA decode/prefill(均走 MMVQ)全 11 种,统一 **64 元素 Psum** 粒度。CPU 与 GPU 端到端 PPL 对齐(差异 ≤ ~0.2%,fp16 scale 存储/累加顺序所致)。

---

## GPU 激活量化与 CPU 对齐(decode + prefill 均走 MMVQ)

上游 MMVQ 把激活量化成 `block_q8_1`(每 **32** 元素一个 scale),与 CPU 不一致(CPU legacy 每 64、K-quant 每 256 一个 scale),导致 GPU 的整数 Psum 只能到 32 元素。现让 GPU 激活与 CPU 对齐:**保持 `block_q8_1` 内存布局**,但同一 group 内所有 q8_1 子块共享同一 scale,从而 vec_dot 可把两半部分和合并成一个 **64 元素整数 Psum**(与 CPU 完全一致,包括零点偏移折进整数乘积、Psum 截断)。

- legacy:每 **64** 元素一个 scale(对齐 `q8_0_64`/`q8_1_64`)
- K-quant:每 **256** 超块一个 scale(对齐 `q8_K`)

- `ds.y = d*sum(quantized)`,对应 CPU min 修正项

**prefill 也对齐**:MMVQ 内核每次启动只支持 `ncols<=8`。`ggml_cuda_mul_mat` 把 block-64 类型(非 split、F32)强制路由到 MMVQ 并关闭 MMQ;`ggml_cuda_mul_mat_vec_q` 把 dst 列按 ≤8 分块多次启动同一(已对齐的)vec_dot。

| 文件 | 改动 / 功能 |
|---|---|
| `ggml/src/ggml-cuda/reex/reex_q64_quantize.cuh` | 新增:per-group(64/256)共享 scale 的激活量化 kernel + launcher(`block_q8_1` 布局) |
| `ggml/src/ggml-cuda/mmvq.cu` | `ggml_cuda_mul_mat_vec_q`:11 种 REEX 类型改用上述量化器(legacy→64,K-quant→256);大 batch 时按 ≤8 列分块循环启动 vec_dot |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_mul_mat`:block-64 类型(非 split、F32)强制 `use_mul_mat_vec_q=true`、`use_mul_mat_q=false` |
| `ggml/src/ggml-cuda/reex/reex_q64_vecdotq.cuh` | legacy 6 个 vec_dot 改写为标量逐元素、合并 64 元素 Psum;`vdr=qi`(每块 1 线程) |
| `ggml/src/ggml-cuda/reex/reex_q64_kquant_vecdotq.cuh` | K-quant 5 个 vec_dot 合并 `sumi0+sumi1` 为 64 元素 Psum |

代价:legacy MMVQ `vdr` 由 2 改为 `qi`(每块 1 线程而非 4/8),且 prefill 放弃 MMQ tiling 改用分块 MMVQ,GPU 吞吐下降——换取与 CPU 逐算子的数值一致性(REEX 路径以硬件保真为目标)。
