# 开源仓库的模块目录命名惯例

调研日期：2026-09-17

## 结论

`components/` 是成熟且容易理解的目录名，但不是大多数开源仓库的统一规范。
目录名称通常跟随模块职责、构建系统和发布方式；它本身也不表示外部仓库或 Git
submodule。

## 使用 `components/` 的知名项目

| 项目 | 目录用途 |
| --- | --- |
| [Chromium](https://github.com/chromium/chromium/tree/main/components) | 多处复用的第一方功能和子系统 |
| [ESP-IDF](https://github.com/espressif/esp-idf/tree/master/components) | 可独立编译并链接进应用的模块 |
| [Servo](https://github.com/servo/servo/tree/main/components) | 浏览器引擎内部模块 |
| [Apache Camel](https://github.com/apache/camel/tree/main/components) | 连接器和独立构建模块 |

## 其他常见方式

| 命名方式 | 知名项目示例 | 表达的职责 |
| --- | --- | --- |
| `packages/` | [React](https://github.com/facebook/react/tree/main/packages)、[Angular](https://github.com/angular/angular/tree/main/packages)、[Flutter](https://github.com/flutter/flutter/tree/master/packages) | 可构建或发布的软件包 |
| `crates/` | [rust-analyzer](https://github.com/rust-lang/rust-analyzer/tree/master/crates)、[Bevy](https://github.com/bevyengine/bevy/tree/main/crates) | Rust workspace 成员 |
| `modules/` | [OpenCV](https://github.com/opencv/opencv/tree/4.x/modules) | 功能模块 |
| `extensions/` | [VS Code](https://github.com/microsoft/vscode/tree/main/extensions) | 可选扩展 |
| 按产品平铺 | [LLVM](https://github.com/llvm/llvm-project/tree/main) | `clang/`、`lld/`、`lldb/`、`mlir/` |

rx-met 的 QuantGRU 和未来 QuantLSTM 是量化流程替换算子，因此使用
`operators/` 比 `components/` 更直接：

```text
operators/
├── quant-gru/
└── quant-lstm/
```

以上链接均为项目官方仓库，检索于 2026-09-17。
