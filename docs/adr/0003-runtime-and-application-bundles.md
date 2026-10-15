# 0003: 独立 runtime 与应用 bundle

状态：已接受。

## 背景

每次通过 docker save 发布完整产品镜像会重复交付大型第三方依赖。现有构建已经有
独立 build-env/runtime-env，并能生成 rx-met、QuantGRU、QuantLSTM 三个 wheel。

## 决策

保留三个变体的固定环境镜像，产品版本发布三个仅含项目 wheel 的应用 bundle。
环境版本与产品版本独立；发布时复用精确镜像身份，不自动重建缺失环境。

用户唯一安装入口为 install.sh。它在匹配的 runtime 中断网安装 wheel 到独立版本目录，
通过资源与原生库检查后才完成安装。runtime 可来自本机、离线 images/ 附件或已配置的
固定 HTTPS 地址。首次交付提供离线归档，下载地址留空。

运行采用 README 中的原生 Docker 命令，显式挂载应用、工作区、数据集并配置 GPU 和权限。
应用挂载位置固定为 /opt/rx-met，Python 搜索目录为 /opt/rx-met/python。
不提供 run.sh、独立 env.sh、宿主机 rx-met 命令或用户侧 internal/ 目录。

## 影响

升级和回滚通过选择版本目录及匹配的 runtime、启动新容器完成。旧目录不会被原地覆盖。
项目默认配置必须随 wheel 提供，用户数据及运行输出写入应用目录之外。
验收对象为纯 runtime + 外部只读安装目录，包含非 root、离线、资源定位和真实 GPU examples。
预安装目录包及完整产品镜像不作为第一阶段的额外交付格式。
