# 应用发布模板

`release-bundle/install.sh.in` 是用户唯一的安装入口。发布过程将镜像身份、下载信息及安装自检代码
内嵌进 install.sh，不交付内部辅助脚本。README 模板记录原生 docker run 的挂载、GPU 和权限参数。

应用包包含 README、CHANGELOG、三个 wheel、examples、manifest、校验清单和可选 images/。
仓库根目录的 `CHANGELOG.md` 保留完整历史；应用包和交付总目录仅包含 `VERSION` 对应的版本内容。
发布开始时校验该版本唯一且非空，保留原发布日期及全部小节，排除其他版本和 Unreleased。
安装后原样保留该版本的 CHANGELOG，与 README 一并参与完整性校验。
构建入口为 `scripts/release/build.sh`；详见 `docs/Release_packaging.md`。
