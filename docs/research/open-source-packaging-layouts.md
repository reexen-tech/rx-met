# 开源仓库中的发布打包目录组织

调研日期：2026-09-18

## 结论

下面的 rx-met 布局符合常见开源仓库的职责分离方式：

```text
scripts/release_build.sh              # 维护者直接调用的发布入口
scripts/internal/                     # 发布流程的内部实现
packaging/release-bundle/*.md.in      # 被复制或渲染进制品的输入模板
.release/export/                      # Git 忽略的生成物
```

开源社区没有规定这些目录必须分别叫 `scripts/`、`packaging/` 和
`.release/`。真正普遍的原则是区分三类内容：版本控制中的制品输入、执行构建或发布
的工具，以及不纳入版本控制的生成物。`packaging/` 表示“如何组成发布制品”时，
不要求发布入口也必须放入其中。

## 官方仓库实例

| 项目 | 仓库中的组织方式 | 与 rx-met 的对应关系 |
| --- | --- | --- |
| [Qualcomm AIMET 2.17.0 `packaging/`](https://github.com/qualcomm/aimet/tree/0e679b705818df7f408971e34693b8098f0971a9/packaging) | `pypi_readme.md.in`、`setup.py.in`、`MANIFEST.in.in`、安装说明和依赖配置均放在 `packaging/`；构建入口仍在仓库根目录的 [`buildntest.sh`](https://github.com/qualcomm/aimet/blob/0e679b705818df7f408971e34693b8098f0971a9/buildntest.sh)。 | 与 rx-met 最接近，也与其 AIMET 2.17 基线直接相关：模板属于 `packaging/`，入口脚本无需跟随模板移动。AIMET 的 `packaging/` 也包含若干打包工具，因此它证明的是职责命名合理，而不是“目录内只能有静态文件”。 |
| [Kubernetes `build/`](https://github.com/kubernetes/kubernetes/tree/3551fa84d739d7e0ceb1954025d61c43e504e97f/build) | [`build/release.sh`](https://github.com/kubernetes/kubernetes/blob/3551fa84d739d7e0ceb1954025d61c43e504e97f/build/release.sh) 是发布构建入口；官方 [`build/README.md`](https://github.com/kubernetes/kubernetes/blob/3551fa84d739d7e0ceb1954025d61c43e504e97f/build/README.md) 明确说明先暂存到 `_output/release-stage`，再输出到 `_output/release-tars`。 | 名称不是 `packaging/`，并且脚本与构建定义同放 `build/`；但清楚区分受管的发布逻辑与忽略的暂存/产物目录，和 `.release/export/` 的角色一致。 |
| [PyTorch `scripts/release/`](https://github.com/pytorch/pytorch/tree/f7a710f51a2acab5c9d455584e15eeb52d831bc9/scripts/release) 与 [`tools/packaging/`](https://github.com/pytorch/pytorch/tree/f7a710f51a2acab5c9d455584e15eeb52d831bc9/tools/packaging) | 分支切割等发布行为放在 `scripts/release/`；wheel 构建工具放在 `tools/packaging/build_wheel.py`。 | 说明大型项目通常按调用者和职责拆分发布工具，并不存在所有发布相关内容必须归入一个顶层 `release/` 的规则。它没有与 rx-met 完全相同的静态模板目录，因此只能作为命名多样性的例子。 |
| [Electron `script/release/`](https://github.com/electron/electron/tree/a88c70b1b0f435439b7a167cbef5e7aae4018919/script/release) 与 [`build/templates/`](https://github.com/electron/electron/tree/a88c70b1b0f435439b7a167cbef5e7aae4018919/build/templates) | GitHub Release 查询、上传等行为放在 `script/release/`；构建使用的 `.tmpl` 输入单独放在 `build/templates/`。 | 与 rx-met 的“脚本留在 `scripts/`、模板移入专用输入目录”原则一致。Electron 还将 zip manifest 放在 `script/zip_manifests/`，说明具体目录名和层级会服从项目自身构建体系。 |

## 对 rx-met 的判断

将现有 `release/README.md` 和 `release/ChangeLog.md` 改为：

```text
packaging/release-bundle/README.md.in
packaging/release-bundle/CHANGELOG.md.in
```

比保留顶层 `release/` 更准确，因为这两个文件是制品输入模板，而不是发布命令、
GitHub Release 配置或生成结果。`.in` 后缀也能明确它们还需要复制或变量替换。

`scripts/release_build.sh` 应继续作为稳定、易发现的维护者入口；只有被它调用且不希望
用户直接依赖的实现才放入 `scripts/internal/`。`.release/export/` 继续作为 Git 忽略的
输出目录也合理。若未来希望采用更普遍的工具默认名，可把输出改成 `dist/`，但这不是
本次结构合规性的必要条件。

以上链接均指向项目官方仓库；除明确标注 AIMET 2.17.0 外，其他链接固定到调研当日的
提交，检索于 2026-09-18。
