# Docker 构建配置

| 文件或目录 | 职责 |
| --- | --- |
| `Dockerfile.environment` | 公共 CPU/GPU build/runtime 环境，使用独立上下文构建 |
| `Dockerfile` | 编译 rx-met、QuantGRU、QuantLSTM 三个项目 wheel |
| `variants.json` | 环境名、环境版本、变体和精确依赖的定义 |
| `requirements/`、`docker-bake.hcl` | 由 `scripts/dependencies.py lock` 生成的依赖锁和产品构建配置 |

公共环境由 `scripts/environment/build.sh` 构建；产品编译使用 Bake，应用包由
`scripts/release/build.sh` 组装。环境的版本、复用、归档命名和操作步骤统一见
[公共环境说明](../docs/Shared_environment.md)，应用发布见
[应用包发布](../docs/Release_packaging.md)。

公共 CPU/GPU 环境的 ORT 构建材料位于 `environment/third_party/`，详见 [目录说明](environment/README.md)。
