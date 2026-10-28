# 运行目录与日志

RX-MET 在库内部自动创建运行记录，用户无需初始化 logger 或修改现有调用参数。
传给 RX 导出接口的 `output_dir` / `export_dir` 表示输出根目录，默认每次任务使用独立子目录：

```text
<output_root>/
  20261027_105308_820401/
    run.log
    run.json
    config/
    exports/
      <prefix>.onnx
      <prefix>.json
      <prefix>.encodings
      <prefix>.aimet.encodings
      <prefix>.meta.json
    checkpoints/
    _work/
```

`config/`、`exports/`、`checkpoints/` 和 `_work/` 按需创建。
执行 ID 使用北京时间（UTC+8）微秒时间戳，格式为 `YYYYMMDD_HHMMSS_ffffff`，
不依赖宿主机或容器的默认时区。冲突时追加 `-1`、`-2`，通过原子创建目录防止覆盖。
同一次完整流程中的校准、导出和恢复验证复用运行目录；重复调用完整流程会创建新目录。
`run.json` 的 `started_at`、`finished_at` 保留带时区的完整 ISO 8601 时间，
例如 `2026-10-27T10:53:08.820401+08:00`；`run.log` 也使用北京时间并标注 `+08:00`。

## 关闭时间戳目录

在创建 QuantSim 或开始量化任务前，通过公共运行模块配置一次：

```python
from aimet_common.run_context import configure_output

configure_output(timestamp_output=False)
```

后续新建任务直接使用输出根目录，Torch、ONNX 的日志、检查点和导出路径统一生效：

```text
<output_root>/
  run.log
  run.json
  config/
  exports/
  checkpoints/
  _work/
```

- 默认值为 `True`；`configure_output(timestamp_output=True)` 或 `configure_output()` 恢复默认。
- 参数只接受 `bool`。配置是当前进程的默认值，任务创建时保存快照，已开始的任务不变。
- 固定目录模式重写 `run.log`、`run.json`，覆盖同名产物和配置快照；其他文件保留，根目录不会被清空。
- `run.json` 的 `output_config.timestamp_output` 记录实际策略，`outputs` 只记录本次产生或改写的产物。
- `models/`、`datasets/` 和用户其他文件保留。固定模式不保留逐次运行历史，失败也会覆盖同名日志和已写出的产物。
- 同一固定目录同时只允许一个任务写入，冲突时在覆盖日志前报错；不同目录的任务可以并行。

底层调用尚未绑定最终目录时，固定模式的早期日志独立暂存在系统临时目录；
导出时一并迁移，未导出的任务在进程正常结束时归入默认输出根目录。
固定模式的中间文件使用 `_work/run-*/` 独立暂存，成功后仅清理本次工作目录，失败时保留。
时间戳仅从目录结构中移除，执行 ID、日志时间和运行开始结束时间继续记录。
此开关不提供环境变量入口，现有导出接口的参数及返回类型不变。

## 输出文件

| 路径 | 内容 |
| --- | --- |
| `run.log` | 详细运行日志，独立于终端 verbose 设置 |
| `run.json` | 输出策略、状态、开始结束时间、工具与依赖版本、构建来源、配置快照和产物校验值 |
| `config/` | JSON 配置快照；配置内容改变时保留另一份快照 |
| `exports/` | 最终模型、编码及配套产物 metadata |
| `checkpoints/` | KWS 浮点和 QAT 检查点 |
| `_work/` | 本次流程由工具创建的中间文件 |

Torch 与 ONNX PTQ 的最终编译器编码均为 `<prefix>.json`。
`.encodings` 和 ONNX PTQ 的 `.aimet.encodings` 为 AIMET 原始编码，用于恢复量化状态，随最终产物保留。
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
绑定的输出目录不可写则报错。

ONNX 演示准备脚本生成的 `models/` 和 `datasets/` 继续位于输出根目录，供多次任务复用，
不会复制到每个时间戳目录。准备和量化仍按原来的两步命令执行。

## 日志与失败处理

`verbose=False` 保持终端的简略行为，详细诊断仍写入 UTF-8 日志；`verbose=True`
同时在终端显示详情。AIMET 各日志区域也接入同一个文件，终端级别与文件级别独立。
完整流程捕获 Python stdout/stderr、警告和异常堆栈；进度条刷新字符不写入文件。
原生 C++/CUDA 直接写入操作系统文件描述符的文本不经过 Python 日志通路。

已结束的任务不由全局历史索引长期持有；仍存活的模型或显式上下文继续保留其运行记录。
未显式结束的隐式任务在最后一个所属模型和上下文释放后完成记录并关闭日志；
若持久化失败，错误按 Python 对象终结器规则报告，同时释放文件和目录锁。
任务对象释放后，后续导出 metadata 通过 `run.json` 恢复所属记录，不重新打开日志文件。
已结束任务的 metadata 写入先获取目录锁，在锁内重读同一执行 ID 的记录，避免并发写入丢失产物。
目录正在被其他任务或 metadata 写者占用时会报错，调用者可在其完成后重试。
fork 子进程不继承目录锁的所有权，asyncio 子任务也不能复用父任务的 metadata 锁。

任务开始即写入 `status=running`。成功后记录产物路径、大小和 SHA-256，
清理本次 `_work/`。
失败时保留日志、运行记录和中间文件，记录 `status=failed`，继续抛出原来的异常；
默认时间戳模式下，已有成功运行不被失败任务替换，历史任务不会自动删除。
固定模式覆盖同名文件，不保证保留上次成功产物。

仅导入包不会创建运行目录。直接使用独立 GRU/LSTM wheel 而未安装 RX-MET 时，
算子保持原有终端行为，不增加对 AIMET 的强制依赖。
