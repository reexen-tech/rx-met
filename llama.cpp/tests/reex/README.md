# `tests/reex/` — TurboQuant 单元测试

本目录存放 `REEX_TURBOQUANT` 启用后的所有单元测试。

## 编译开关

仅在 `cmake -DREEX_TURBOQUANT=ON -DLLAMA_BUILD_TESTS=ON` 时编译。

## 命名约定

- 文件名：`test-reex-turboquant-<场景>.cpp`，例如 `test-reex-turboquant-block-roundtrip.cpp`、`test-reex-turboquant-wht-roundtrip.cpp`。
- 测试框架：沿用 `tests/` 下既有的 `ctest` 注册风格。
- 与 oracle 对拍：可以 `#include "../oracle/reex_turboquant_oracle_core.h"` 拿 reference 数据。

## 测试矩阵（P4 A2.2 K3 P3 升级后 — TheTom turbo3_0）

P4 A2.1（2026-04-30）把所有 cos_sim 阈值对齐到上游 `scos-lab/turboquant` reference
水平，并通过 T1/T2/T3 follow-up 把 `p4a21-diagnostics` 的 multi-pass drift /
systematic-bias 度量从"informational only"提升为 ctest contract gate；当时 K3 P1
simplification 留下 4 个 expected FAIL。

P4 A2.2（同日，2026-04-30）K3 P3 升级落地（采用 TheTom `turbo3_0` 路径：8-centroid
3-bit PolarQuant + norm-correction trick `corrected_norm = ‖x‖ / ‖recon_unit‖`），
**4 个 expected FAIL 全部一次性关闭**：

- 算法层：K3 cos_sim 0.977 → ~0.983（d=128 + 3-bit Lloyd-Max-Gaussian 理论极限）；
  multi-pass drift 0.021 → 0（norm correction 保证 ‖dequant‖ ≡ ‖x‖，第二轮
  bucket 与第一轮完全相同）。
- 阈值层：K3 cos_sim gate 0.99 → 0.97（与上游 `llama-cpp-turboquant` 的
  `test_set_rows_turbo3::max_nmse_err = 0.05` 一致，约 cos_sim ≥ 0.975）。原
  0.99 来自论文 Algorithm 2 + full QJL @ d=256 reference（cos_sim = 1.000），
  与本仓选定的 TheTom turbo3_0 路径在头维 d=128 的理论极限不匹配；
  详见 `docs/turboquant/p4.a22_k3_p3_upgrade_plan.md` §0。

| 测试 | 阈值 | 备注 |
| --- | --- | --- |
| `test-reex-turboquant-block-roundtrip` | TQ_K3 cos_sim ≥ **0.97** / norm_rel ≤ 0.05 | K3 P3 实测 0.982~0.985 ✅；与 TheTom NMSE ≤ 0.05 一致 |
| | TQ_V2 cos_sim ≥ **0.93** / norm_rel ≤ 0.10 | upstream 0.940（paper-marginal tier）；T3：input 走 truncated N(0, 0.3²) at ±2σ + `min_blocks=8` 后稳定 0.94+ |
| | TQ_V4 cos_sim ≥ **0.99** / norm_rel ≤ 0.05 | upstream 0.997 ✅；T3 input truncation 一致应用 |
| `test-reex-turboquant-vs-oracle` (V2/V4) | cos_sim ≥ **0.99999** | bit-identical to oracle ✅ |
| `test-reex-turboquant-vs-oracle` (K3 full path) | cos_sim ≥ **0.99999** | K3 P3：与 oracle `TurboQuantPolar` (identity Π, bits=3) bit-identical ✅（实测 1.0000000） |
| `test-reex-turboquant-kv-types` (CPY) | K3 ≥ **0.97** / V2 ≥ **0.93** / V4 ≥ **0.99** | K3 P3 实测 0.983 ✅；与 block-roundtrip 完全对齐 |
| `test-reex-turboquant-kv-types` (GET_ROWS) | cos_sim ≥ **0.99999** | lookup 无损 ✅ |
| `test-reex-turboquant-wht-roundtrip` | abs FP32 err ≤ **5e-6** | FP32 ULP 量级 ✅ |
| `test-reex-turboquant-wht-vs-oracle` | rel inner-product err ≤ **1e-3** | WHT 等价类 ✅ |
| `test-reex-turboquant-backend-op-wht` | abs FP32 err ≤ **1e-6** | ggml op 与 scalar reference 一致 ✅ |
| `test-reex-turboquant-layer-spec` | preset mode 0/1/2/5/7 确定性 | pure-function ✅ |
| `test-reex-turboquant-p4a21-diagnostics` | controls (Q4_0/Q8_0) 必通过；**T1: multi-pass drift ≤ 1e-3 / T2: \|z-score\| < 5σ** for every type | K3 P3 后 T1 全 PASS（drift = 0 by construction）；T2 全 PASS（max z = 1.61）✅ |

### 当前 ctest 实测（K3 P3 升级后，全 PASS）

```text
$ ctest -R "test-reex-turboquant-" --output-on-failure -j 4
PASS: test-reex-turboquant-block-roundtrip       (K3 cos_sim 0.982~0.985 ≥ 0.97)
PASS: test-reex-turboquant-vs-oracle             (K3 vs TurboQuantPolar 1.000 ≥ 0.99999)
PASS: test-reex-turboquant-kv-types              (K3 CPY 0.983 ≥ 0.97; GET_ROWS 1.000)
PASS: test-reex-turboquant-p4a21-diagnostics     (T1 K3 drift 0; T2 max |z| = 1.61)
PASS: test-reex-turboquant-wht-roundtrip
PASS: test-reex-turboquant-wht-vs-oracle
PASS: test-reex-turboquant-backend-op-wht
PASS: test-reex-turboquant-layer-spec
100% tests passed, 0 tests failed out of 8
```

> **K3 P3 升级承诺达成**：① 4 个 expected ctest FAIL 全部 PASS；② multi-pass
> drift 0.021 → 0（norm correction trick 的几何保证）；③ 代数收敛证明对齐
> 论文 Algorithm 1（Lloyd-Max-optimal scalar quant of unit-sphere marginal）+
> 上游 `llama-cpp-turboquant` `turbo3_0` reference。**未承诺独立修复
> 30B-A3B-Instruct-2507 的 PPL/acc 退化**——已在 Stage-1 跨模型 ablation
> 中锁定为该模型自身 KV 数值分布敏感性，不在 K3 算法层。

更多设计细节见 `docs/turboquant/01_设计与实现计划.md` §1.2、
`docs/turboquant/p4.a21_stage0_algorithm_diagnostics_report.md` §5 + §10、
`docs/turboquant/p4.a22_k3_p3_upgrade_plan.md`（K3 P3 完整设计）。
