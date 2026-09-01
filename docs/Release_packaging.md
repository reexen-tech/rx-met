# rx-met 打包 / 发布

小模型软件包：预编译 wheel + examples，装进 `ada200_docker`。**客户包不含 Docker 镜像**。

版本约定：

| 名称 | 格式 | 例子（2026-08-31） |
| ---- | ---- | ------------------ |
| `DATE_TAG` | `vYYMMDD` | `v260831` |
| `VERSION`（PEP 440 / wheel） | `YY.M.D` | `26.8.31` |

可用 `RX_MET_RELEASE_DATE=260831` 覆盖当天日期。

## 脚本

| 脚本 | 作用 |
| ---- | ---- |
| `scripts/build_aimet_native.sh` | 编译 AIMET native `.so` |
| `scripts/build_wheel.sh` | 打 `rx-met` wheel |
| `scripts/build_quant_gru_wheel.sh` | 打 `quant_gru` wheel |
| `scripts/download_extra_wheels.sh` | 下载 torchaudio / librosa 等离线依赖 |
| `scripts/release_build.sh` | Docker builder 导出 wheels 并组装客户包 |
| `scripts/verify_bundle.sh` | 在 `ada200_docker:latest` 里安装并 smoke |

```bash
./scripts/release_build.sh
./scripts/verify_bundle.sh
```

产物：`.release/export/ada200-rx-met-vYYMMDD-linux_x86_64.tar.gz`
