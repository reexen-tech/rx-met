#!/usr/bin/env bash
# 将官方 llama.cpp (upstream) 合并到当前仓库的 main 分支。
# 用法：
#   ./scripts/sync_and_merge.sh           # 合并到当前分支（需当前分支为 main）
#   ./scripts/sync_and_merge.sh main      # 明确指定合并到 main
#
# 使用前请确保已配置： git remote add upstream https://github.com/ggml-org/llama.cpp.git

set -e
TARGET_BRANCH="${1:-main}"

echo "[sync_and_merge] 目标分支: $TARGET_BRANCH"
if ! git rev-parse --verify "$TARGET_BRANCH" >/dev/null 2>&1; then
    echo "[sync_and_merge] 错误: 分支 '$TARGET_BRANCH' 不存在"
    exit 1
fi

if ! git remote get-url upstream >/dev/null 2>&1; then
    echo "[sync_and_merge] 未配置 upstream，请先执行："
    echo "  git remote add upstream https://github.com/ggml-org/llama.cpp.git"
    exit 1
fi

echo "[sync_and_merge] 正在 fetch upstream ..."
git fetch upstream

CURRENT=$(git branch --show-current)
if [ "$CURRENT" != "$TARGET_BRANCH" ]; then
    echo "[sync_and_merge] 当前分支为 $CURRENT，将 checkout 到 $TARGET_BRANCH 再合并"
    git checkout "$TARGET_BRANCH"
fi

echo "[sync_and_merge] 正在 merge upstream/main 到 $TARGET_BRANCH ..."
if git merge upstream/main --no-edit; then
    echo "[sync_and_merge] 合并成功。可执行 git push origin $TARGET_BRANCH 推送到小组仓库。"
else
    echo "[sync_and_merge] 合并产生冲突，请解决后执行："
    echo "  git add . && git commit"
    echo "冲突文件列表："
    git diff --name-only --diff-filter=U
    exit 1
fi
