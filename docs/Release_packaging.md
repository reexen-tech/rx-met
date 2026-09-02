# rx-met 打包 / 发布

小模型软件包：预编译 wheel + examples，装进官方 `ada200_docker`。**客户包不含 Docker 镜像。**

版本约定：

| 名称 | 来源 | 格式 | 例子 |
| ---- | ---- | ---- | ---- |
| 组件版本 | 仓库 `VERSION` | `MAJOR.MINOR.PATCH` | `1.0.0` |
| wheel | 同上 | `MAJOR.MINOR.PATCH`（不能带 `v`） | `1.0.0` |
| 共享盘目录 | 同上 | `vMAJOR.MINOR.PATCH` | `v1.0.0` |
| 软件包文件名 | SDK 日期 | `vYYMMDD` | `v260902` |

## 两套脚本

| 脚本 | 作用 |
| ---- | ---- |
| `scripts/build_dev_image.sh` | 构建团队镜像 `ada200_docker:rx-met-dev` |
| `scripts/release_build.sh` | 在该镜像里编 wheel，组装客户 `tar.gz` |
| `scripts/verify_bundle.sh` | 在官方 `ada200_docker:latest` 里安装并 smoke |

`build_dev_image.sh`：

1. 已有 `ada200_docker:latest`（或 `ada200_docker`）→ `docker/Dockerfile` 以它为底图，只加编译工具
2. 没有统一 Docker → 先用 `docker/Dockerfile.ada200` 造运行时底图，再用同一份 `docker/Dockerfile` 加编译工具
3. 只打 `ada200_docker:rx-met-dev`，**不会**覆盖官方 `ada200_docker:latest`

`release_build.sh` 若找不到 `ada200_docker:rx-met-dev`，会提示先运行 `./scripts/build_dev_image.sh`。

```bash
# 官方统一 Docker 建议先导入
# docker load -i /mnt/data2/reexen_release/ADA200/docker/v1.0.0/artifacts/ada200_docker_v1.tar.gz
./scripts/build_dev_image.sh
./scripts/release_build.sh
./scripts/verify_bundle.sh
```

可用 `RX_MET_RELEASE_DATE=260902` 覆盖软件包日期标签。已发布的组件版本目录不可覆盖；升高版本时改 `VERSION`。

产物：`.release/export/ada200-rx-met-vYYMMDD-linux_x86_64.tar.gz`

`examples/` 按整目录拷贝（去掉 `__pycache__` / `output`）。当前入口是 `quick_start_kws.py` 和 `onnx_ptq_quick_start.py`。正式发布先落到 `tmp_test_data/`，再移到 `reexen_release/ADA200/rx-met/v1.0.0/`。
