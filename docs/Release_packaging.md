# rx-met 应用包发布

产品 VERSION 保持独立；公共 GPU 环境使用 rx-quant-env v1，详见 [公共环境](Shared_environment.md)。
两个仓库独立保存公共环境定义及发布身份记录，无跨仓库运行依赖。

## 日常发布

本机准备好发布记录指定的 build/runtime 镜像后执行：

```bash
./scripts/release/build.sh cu118 cu126 cu130
```

脚本核对公共环境配置和精确镜像 ID，在对应 build 环境中生成三个项目 wheel，
组装应用包，并在纯 runtime 中实际离线安装、自检。默认不生成 images/，不执行 docker save。
输出目录为 `.release/export/rx-quant-env-v1/v<VERSION>/`，安装验收目录默认为 `.release/installed/`。

应用包包含 install.sh、README、当前版本 CHANGELOG、manifest、SHA256SUMS、wheels、examples。
用户运行 install.sh 后按 README 使用原生 Docker 命令，不提供 run.sh。

## 首次离线交付

公共 runtime 已在环境发布时单独导出。需要随应用附带镜像时：

```bash
./scripts/release/build.sh --with-images --runtime-dir /path/to/public-runtime-archives cu126
```

此参数校验并将已有归档硬链接或复制到交付目录 images/，不会构建或重新导出镜像。
也可直接向用户提供共享的 images/；已安装过另一个产品相同公共环境的机器可复用镜像。
`--with-runtime` 保留为把归档嵌入每个应用包的可选方式，默认关闭。

## 参数与版本记录

- `--jobs N`：编译并行数，默认 8。
- `--output-dir DIR`：应用交付目录。
- `--install-prefix DIR`：此次安装验收的根目录。
- `--skip-build`：源码和 build/runtime 身份完全匹配时复用已有编译产物，重新组装交付文档。
- `--replace`：替换未发布候选应用包；不覆盖已有安装目录或公共环境，修改后的候选使用新安装目录。
- `--runtime-base-url https://...`：预留固定归档下载地址，当前留空、离线交付。

根 CHANGELOG 保留历史；打包前按 VERSION 校验并提取唯一且非空的当前版本，
应用包、交付总目录和安装目录均只保留当前版本内容。已安装内容参与完整性校验。

## 验证

```bash
./scripts/release/verify_bundle.sh --install-prefix .release/installed --gpu --gpu-device 0 cu126
```

构建时自检使用非 root、断网、只读应用挂载；GPU 验证单独运行。发布前还需运行真实 examples，
检查默认 runtime 复用、归档导入、重复安装、失败恢复、目录迁移、升级和回滚。
