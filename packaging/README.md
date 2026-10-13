# 应用发布模板

release-bundle/install.sh.in 是唯一用户安装入口。打包时内嵌镜像身份和自检代码。
README.md.in 提供安装、数据准备和原生 Docker 挂载运行流程；CHANGELOG 从根目录提取当前版本。
公共 runtime 可由本地镜像、离线 images/ 或配置的固定 HTTPS 地址提供。
