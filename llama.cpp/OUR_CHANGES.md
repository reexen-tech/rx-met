# 内部变更清单（与官方 llama.cpp 的差异）

用于与官方同步时快速定位「我们改了什么」，便于解决冲突和回归测试。

**维护约定**：每次新增/删除/修改与官方不同的文件或逻辑时，更新本清单。

---

## 新增文件

| 路径 | 说明 |
|------|------|
| `docs/LLAMA_CPP_LEARNING_GUIDE_CN.md` | 内部学习与架构说明（中文） |
| `docs/LLAMA_CPP_ARCHITECTURE_CN.md` | 内部架构文档（中文） |
| `docs/LLAMA_CPP_目录架构与文件说明_CN.md` | 内部目录与文件说明（中文） |
| `docs/DEVELOPMENT_FORK_SYNC.md` | 开发与官方同步指南 |
| `docs/docker-mount-datasets-models.md` | 内部 Docker 挂载说明 |
| `scripts/sync_upstream_official.md` | 同步官方仓库步骤说明 |
| `scripts/sync_and_merge.sh` | 半自动同步脚本（可选） |
| （示例）`ggml/src/ggml-quants-custom.c` | 自定义量化算子时可在此添加 |

---

## 修改的官方文件

| 路径 | 修改目的 | 大致位置/备注 |
|------|----------|----------------|
| （暂无；若修改了官方文件请在此列出） | 例如：增加 ftype、调用自定义量化 | 行号或函数名 |

---

## CMake / 配置

| 路径 | 修改内容 |
|------|----------|
| （若在 CMakeLists 中新增源文件或选项，请在此列出） | 例如：添加 llama-quant-impl.cpp |

---

## 同步时重点检查

合并 `upstream/main` 后，优先处理上述「修改的官方文件」中的冲突，然后执行：

- 编译：`cmake --build build`
- 量化相关测试（若有）：如 `tests/test-quantize-*` 或你们自己的测试

更新日期：按需填写。
