# 发布制品输入

本目录保存发布制品使用的模板和静态输入。构建结果写入 `.release/export/`。

## 目录说明

- `release-bundle/README.md.in` 是镜像和导出目录中的使用说明模板。
- `release-bundle/CHANGELOG.md.in` 是镜像和导出目录中的变更记录模板。

`scripts/release/build.sh` 复制模板并替换版本占位符。Git 忽略
`.release/export/` 中的生成结果。
