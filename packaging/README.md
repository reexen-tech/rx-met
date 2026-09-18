# 发布制品输入

本目录保存发布制品使用的模板和静态输入，不保存构建结果。

## 目录说明

- `release-bundle/README.md.in` 是镜像和导出目录中的使用说明模板。
- `release-bundle/CHANGELOG.md.in` 是镜像和导出目录中的变更记录模板。

`scripts/release/build.sh` 复制模板并替换版本占位符。生成结果写入被 Git 忽略的
`.release/export/`，不应提交到本目录。
