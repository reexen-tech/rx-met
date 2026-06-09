# 基于 llama.cpp 的量化开发与官方同步指南

在保留内部量化算子等开发的同时，与官方 llama.cpp 长期同步的推荐做法。

---

## 一、目标

- **内部功能**：在 llama.cpp 上开发/修改量化相关算子（如 ggml、llama-quant、tools/quantize 等）。
- **与官方同步**：能定期把官方最新代码合入，减少冲突、保留己方修改。

---

## 二、推荐分支策略

```
                    upstream/main (官方)
                         │
                         │ 定期 merge/rebase
                         ▼
    origin/main (小组主分支，尽量与官方对齐)
                         │
                         │ 开发时从 main 拉分支
                         ▼
    origin/dev/quant-ops (或 feature/xxx)  ← 内部量化开发
                         │
                         │ 完成功能后 merge 回 main
                         ▼
    origin/main (合并了内部功能 + 后续再与 upstream 同步)
```

**原则**：

1. **main**：小组主分支，定期从 `upstream/main` 合并；合并前尽量保证 main 是「官方 + 已验收的内部功能」。
2. **功能分支**：所有「新增/修改量化算子」在 `dev/quant-ops` 或 `feature/量化描述` 上做，不要直接在 main 上大改。
3. **同步顺序**：先让 main 追上官方（`git merge upstream/main`），再把 main 合入功能分支（或 rebase 功能分支到 main），在功能分支上解决冲突，最后把功能分支合回 main。

这样冲突主要集中在「功能分支与官方」之间，且集中在你们改动的文件上，便于排查。

---

## 三、代码组织（减少冲突、便于同步）

### 3.1 尽量「新增文件」，少改官方文件

| 类型 | 建议 | 说明 |
|------|------|------|
| **新增量化类型/算子** | 优先放在 ggml 侧的新文件，如 `ggml/src/ggml-quants-custom.c`、`ggml/include/ggml-quants-custom.h` | 官方更新 `ggml.c` / `ggml-quants.c` 时不会直接覆盖你的实现 |
| **llama 侧量化入口** | 若官方已有 `llama-quant.cpp`，尽量只做最小改动（如新 ftype 映射、新选项）；大逻辑可抽到 `llama-quant-impl.cpp`（自建）再由 `llama-quant.cpp` 调用 | 合并 upstream 时只需在少量位置处理冲突 |
| **tools/quantize** | 新量化工具或新参数可放在新文件（如 `quantize_ops.cpp`），在 `quantize.cpp` 里做最小集成 | 同上 |
| **必须改动的官方文件** | 在 [OUR_CHANGES.md](#四、维护内部变更清单-our_changesmd) 里逐项列出「文件 + 修改目的 + 大概位置」 | 每次拉官方后重点检查这些文件的冲突 |

### 3.2 用 CMake 选项区分「内部扩展」（可选）

若希望内部代码可与官方完全隔离编译，可在 CMake 中加选项，例如：

```cmake
# 在 ggml/CMakeLists.txt 或 src/CMakeLists.txt 中
option(LLAMA_QUANT_CUSTOM "Build custom quantization operators" ON)
if(LLAMA_QUANT_CUSTOM)
    # 只编译你们的 ggml-quants-custom.c、llama-quant-impl.cpp 等
endif()
```

这样官方未包含的源文件只在开启选项时参与编译，方便以后把补丁单独发给上游或在不同版本间切换。

### 3.3 文档与脚本

- 小组内部文档（如 `docs/LLAMA_CPP_*_CN.md`、`docs/LLAMA_CPP_目录架构与文件说明_CN.md`）建议放在单独目录或统一前缀，便于识别并在合并时重点检查。
- 同步脚本、内部用脚本放在 `scripts/` 下（如 `sync_upstream_official.md`、下面的 `sync_and_merge.sh`），不要改官方 CI 脚本。

---

## 四、维护内部变更清单 (OUR_CHANGES.md)

在仓库根目录或 `docs/` 下维护一份 **OUR_CHANGES.md**（或 `PATCHES.md`），列出：

1. **新增文件**（相对仓库根）：路径 + 一句话说明。
2. **修改的官方文件**：路径 + 修改目的 + 行号或函数名（便于冲突时快速定位）。
3. **CMake/配置**：若在 CMakeLists 中增加了新源文件或选项，注明文件和选项名。

示例：

```markdown
# 内部变更清单（与官方 llama.cpp 的差异）

## 新增文件
- `ggml/src/ggml-quants-custom.c` / `ggml/include/ggml-quants-custom.h`：自定义量化算子 xxx
- `src/llama-quant-impl.cpp`：量化入口扩展，供 llama-quant.cpp 调用
- `docs/LLAMA_CPP_*.md`：内部中文文档

## 修改的官方文件
- `src/llama-quant.cpp`：增加 ftype 映射、调用 llama-quant-impl（约 Lxx–Lyy）
- `src/CMakeLists.txt`：链接 llama-quant-impl.cpp
- `ggml/CMakeLists.txt`：编译 ggml-quants-custom.c

## 同步时重点检查
合并 upstream 后，优先解决上述文件的冲突，再跑测试。
```

每次合并官方后，可据此快速检查冲突和回归。

---

## 五、与官方同步的推荐流程

### 5.1 一次性配置

```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git
```

### 5.2 日常：在功能分支上开发

```bash
git checkout main
git pull origin main
git checkout -b dev/quant-ops    # 或 git checkout dev/quant-ops
# ... 开发、提交 ...
git push origin dev/quant-ops
```

### 5.3 定期：把官方合入 main，再合入功能分支

```bash
# 1. 更新 main 为「官方最新 + 当前小组 main」
git fetch upstream
git checkout main
git merge upstream/main          # 有冲突在 main 上解决
git push origin main

# 2. 把 main 合入功能分支（在功能分支上解决冲突）
git checkout dev/quant-ops
git merge main                   # 或 git rebase main
# 若有冲突，按 OUR_CHANGES.md 逐个文件处理
git push origin dev/quant-ops
```

### 5.4 功能验收后合回 main

```bash
git checkout main
git merge dev/quant-ops
git push origin main
```

之后继续从 main 拉新功能分支，重复 5.2–5.4。

---

## 六、可选：半自动同步脚本

在 `scripts/sync_and_merge.sh` 中实现：拉取 upstream、合并到 main、若有冲突则列出冲突文件并提示解决（见下一节）。平时执行该脚本即可完成「官方 → main」的一步，再手动把 main 合入功能分支。

---

## 七、冲突处理建议

- **仅我们改动的文件**：保留我方逻辑，再视情况把官方同文件的合理改动手工合入。
- **官方也改动的文件**：以官方行为为主，把我们的修改作为「增量」迁到新代码上（必要时用 `git show upstream/main:path/to/file` 对照）。
- **CMakeLists**：冲突多为新增源文件列表；合并两边的新增项，去掉重复即可。

---

## 八、小结

| 做法 | 说明 |
|------|------|
| **分支** | main 追官方；功能在 dev/quant-ops（或 feature/*）上做，再合回 main |
| **代码** | 新增算子/逻辑尽量新文件；必须改的官方文件记入 OUR_CHANGES.md |
| **同步** | 先 `merge upstream/main` 到 main，再 `merge main` 到功能分支，在功能分支解冲突 |
| **清单** | 维护 OUR_CHANGES.md，列出新增/修改文件，便于每次合并时重点检查 |

这样可以在保证内部量化开发持续进行的前提下，与官方 llama.cpp 长期同步，并把冲突范围控制在有限文件内。

---

## 九、快速参考

| 需求 | 操作 |
|------|------|
| 仅同步官方到 main（不关心分支策略） | 见 [scripts/sync_upstream_official.md](../scripts/sync_upstream_official.md) |
| 一键把 upstream 合并到当前 main | `./scripts/sync_and_merge.sh`（需已配置 upstream） |
| 查看/维护「我们改了哪些」 | 编辑 [OUR_CHANGES.md](../OUR_CHANGES.md) |
| 完整分支策略 + 代码组织 + 冲突处理 | 本文档第二～八节 |
