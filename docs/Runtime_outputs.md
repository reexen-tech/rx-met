# 运行目录与日志

RX-MET 在库内部自动创建运行记录，用户无需初始化 logger 或修改现有调用参数。
传给 RX 导出接口的 `output_dir` / `export_dir` 表示输出根目录，每次任务使用独立子目录：

```text
<output_root>/
  20261021T063025123456Z/
    run.log
    run.json
    config/
    exports/
      <prefix>.onnx
      <prefix>.encodings
      <prefix>.aimet.encodings
      <prefix>.meta.json
    checkpoints/
    _work/
```

`config/`、`exports/`、`checkpoints/` 和 `_work/` 按需创建。
执行 ID 使用 UTC 微秒时间戳，冲突时追加 `-1`、`-2`，通过原子创建目录防止覆盖。
同一次完整流程中的校准、导出和恢复验证复用运行目录；重复调用完整流程会创建新目录。

## 输出文件

| 路径 | 内容 |
| --- | --- |
| `run.log` | 详细运行日志，独立于终端 verbose 设置 |
| `run.json` | 状态、开始结束时间、工具与依赖版本、构建来源、配置快照和产物校验值 |
| `config/` | JSON 配置快照；配置内容改变时保留另一份快照 |
| `exports/` | 最终模型、编码及配套产物 metadata |
| `checkpoints/` | KWS 浮点和 QAT 检查点 |
| `_work/` | 本次流程由工具创建的中间文件 |

ONNX PTQ 的 `.aimet.encodings` 用于恢复量化状态，随最终产物保留。
工具版本记录在 `run.json` 的 `tool` 中；编码文件中的 `version` / `schema_version`
仍表示编码格式版本，不作为 RX-MET 软件版本。
源码运行在 Git 可用时记录提交和修改状态，发布包运行记录包内清单的构建信息。

导出接口的参数、返回类型和返回字典键保持不变，返回的路径指向实际的 `exports/` 文件。
加载时使用返回路径，示例产物位置见终端输出；不再假设模型直接位于根目录。
显式指定的单个保存文件路径仍被尊重，例如阶段编码或 `RX_MET_KWS_FP_MODEL`。

## 默认位置

- KWS：`./output/quick_start_kws/<gru|lstm>/<execution-id>/`。
- ONNX PTQ 示例：`./output/onnx_ptq_quick_start/<execution-id>/`。
- 尚未知输出根目录的底层调用：`./output/<execution-id>/`，可用 `RX_MET_OUTPUT_DIR` 覆盖。

底层调用先产生的诊断日志会随第一次 RX 导出迁移到用户指定的根目录。
每次创建运行记录时终端显示日志位置，迁移时显示最终位置。
尚未指定根目录且当前目录不可写时，早期日志暂存在系统临时目录，导出时一并迁移；
用户显式指定的目录不可写则直接报错。

ONNX 演示准备脚本生成的 `models/` 和 `datasets/` 继续位于输出根目录，供多次任务复用，
不会复制到每个时间戳目录。准备和量化仍按原来的两步命令执行。

## 日志与失败处理

`verbose=False` 保持终端的简略行为，详细诊断仍写入 UTF-8 日志；`verbose=True`
同时在终端显示详情。AIMET 各日志区域也接入同一个文件，终端级别与文件级别独立。
完整流程捕获 Python stdout/stderr、警告和异常堆栈；进度条刷新字符不写入文件。
原生 C++/CUDA 直接写入操作系统文件描述符的文本不经过 Python 日志通路。

任务开始即写入 `status=running`。成功后记录产物路径、大小和 SHA-256，
清理本次 `_work/`。
失败时保留日志、运行记录和中间文件，记录 `status=failed`，继续抛出原来的异常；
已有成功运行不被失败任务替换。历史任务不会自动删除。

仅导入包不会创建运行目录。直接使用独立 GRU/LSTM wheel 而未安装 RX-MET 时，
算子保持原有终端行为，不增加对 AIMET 的强制依赖。
