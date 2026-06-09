# 将小组版 llama.cpp 同步到官方最新版

当前仓库 `origin` 指向小组内网（如 `192.168.30.203`），需把**官方** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 作为 `upstream` 拉取并合并。

**若要在保留内部开发（如量化算子）的前提下长期与官方同步**，请参阅 **[docs/DEVELOPMENT_FORK_SYNC.md](../docs/DEVELOPMENT_FORK_SYNC.md)**，内含分支策略、代码组织与冲突处理建议。

---

## 一、一次性配置（只需做一次）

在仓库根目录执行：

```bash
cd /home/llq/workspace/llama.cpp

# 添加官方仓库为 upstream（若已添加过会报错，可忽略）
git remote add upstream https://github.com/ggml-org/llama.cpp.git

# 确认
git remote -v
# 应看到 origin → 小组仓库，upstream → github.com/ggml-org/llama.cpp
```

## 二、每次要同步到官方最新时

任选一种方式即可。

### 方式 A：merge（保留小组提交历史，推荐）

```bash
cd /home/llq/workspace/llama.cpp

git fetch upstream
git checkout main
git merge upstream/main

# 若有冲突，解决后：
# git add .
# git commit -m "Merge upstream/main, resolve conflicts"

# 推送到小组仓库
git push origin main
```

### 方式 B：rebase（历史更线性，小组提交会“挪”到最新官方之上）

```bash
cd /home/llq/workspace/llama.cpp

git fetch upstream
git checkout main
git rebase upstream/main

# 若有冲突，解决后：
# git add .
# git rebase --continue

# 推送到小组仓库（若之前已 push 过 main，需要强制推送）
git push origin main
# 若提示被拒绝：git push origin main --force-with-lease
```

## 三、快速一条龙（merge 方式）

```bash
cd /home/llq/workspace/llama.cpp && \
git fetch upstream && \
git checkout main && \
git merge upstream/main && \
echo "合并完成。若有冲突请手动解决后 push。无冲突可执行: git push origin main"
```

## 四、若未配置过 upstream

先执行第一节，再执行第二节或第三节。

## 五、查看与官方差异（不合并）

```bash
git fetch upstream
git log main..upstream/main --oneline   # 官方比本仓多哪些提交
git log upstream/main..main --oneline   # 本仓比官方多哪些提交
git diff main upstream/main --stat      # 文件级差异
```
