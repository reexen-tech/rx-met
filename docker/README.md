# Docker 构建配置

`Dockerfile.environment` 与另一个产品仓库保持公共配方一致，由 `scripts/environment/build.sh`
使用独立上下文构建 rx-quant-env-build/runtime。消费镜像时按版本标签检查环境名称、角色、变体和平台，源码摘要仅供追踪。
`Dockerfile` 只输出三个项目 wheel，由 release/build.sh 组装应用包。
variants.json 定义精确依赖，requirements 和 docker-bake.hcl 由 scripts/dependencies.py lock 生成。
公共 GPU 环境构建不通过产品 Bake 目标运行；产品编译使用 Bake。

公共 GPU 环境的 ORT 构建材料位于 `environment/third_party/`，详见 [目录说明](environment/README.md)。
