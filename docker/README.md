# Docker 构建配置

`Dockerfile.environment` 与另一个产品仓库保持公共配方一致，由 `scripts/environment/build.sh`
使用独立上下文构建 rx-quant-env-build/runtime。环境身份见 environment-images.json。
`Dockerfile` 只输出三个项目 wheel，由 release/build.sh 组装应用包。
variants.json 定义精确依赖，requirements 和 docker-bake.hcl 由 scripts/dependencies.py lock 生成。
公共 GPU 环境构建不通过产品 Bake 目标运行；产品编译使用 Bake。
