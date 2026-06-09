#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${1:?usage: watch_reex_gate_progress.sh <terminal_file> <output_md>}"
OUTPUT_FILE="${2:?usage: watch_reex_gate_progress.sh <terminal_file> <output_md>}"
INTERVAL_SEC="${INTERVAL_SEC:-10}"

while true; do
  done_flag="$(
    python3 - "$INPUT_FILE" "$OUTPUT_FILE" <<'PY'
import os
import json
import re
import subprocess
import sys
from datetime import datetime

def fmt_duration(total_sec: int) -> str:
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"

def extract_operator(line: str) -> str:
    m = re.search(r"\b([A-Z][A-Z0-9_]+)\(", line)
    if m:
        return m.group(1)
    m = re.match(r"(?:OK|FAIL)\s+([A-Za-z0-9_]+)\b", line)
    if m:
        return m.group(1)
    if "reex gemm tests" in line:
        return "reex_gemm"
    if "SUMMARY" in line:
        return "summary"
    return ""

def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"stable": {}, "pending": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("stable", {})
            data.setdefault("pending", {})
            return data
    except Exception:
        pass
    return {"stable": {}, "pending": {}}

def update_stable_field(state: dict, key: str, new_value: str, threshold: int) -> str:
    stable = state.setdefault("stable", {})
    pending = state.setdefault("pending", {})
    current_stable = stable.get(key, "")
    field_pending = pending.get(key, {"value": "", "count": 0})

    if not new_value:
        pending[key] = field_pending
        return current_stable

    if new_value == current_stable:
        pending[key] = {"value": new_value, "count": threshold}
        return current_stable

    if new_value == field_pending.get("value", ""):
        field_pending["count"] = int(field_pending.get("count", 0)) + 1
    else:
        field_pending = {"value": new_value, "count": 1}

    pending[key] = field_pending
    if field_pending["count"] >= threshold:
        stable[key] = new_value
        return new_value
    return current_stable or new_value

def get_live_process_info(container_name: str) -> dict:
    if not container_name:
        return {}
    cmd = [
        "docker", "exec", container_name, "bash", "-lc",
        r"""ps -eo pid=,etime=,pcpu=,args= | awk '
            /llama-perplexity|test-backend-ops|test-reex-|cmake --build| make / {
                if ($0 !~ /awk/ && $0 !~ /watch_reex_gate_progress/) print
            }'"""
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
    except Exception:
        return {}
    if res.returncode != 0:
        return {}
    rows = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    if not rows:
        return {}

    def score(row: str) -> tuple:
        m = re.match(r"(\d+)\s+(\S+)\s+([0-9.]+)\s+(.*)", row)
        if not m:
            return (-1.0, -1)
        cpu = float(m.group(3))
        cmdline = m.group(4)
        priority = 0
        if "llama-perplexity" in cmdline:
            priority = 5
        elif "test-backend-ops" in cmdline:
            priority = 4
        elif "test-reex-" in cmdline:
            priority = 3
        elif "cmake --build" in cmdline or " make " in cmdline:
            priority = 2
        return (priority, cpu)

    rows.sort(key=score, reverse=True)
    best = rows[0]
    m = re.match(r"(\d+)\s+(\S+)\s+([0-9.]+)\s+(.*)", best)
    if not m:
        return {}
    pid, etime, pcpu, cmdline = m.groups()
    mode = ""
    if "llama-perplexity" in cmdline:
        ngl = re.search(r"(?:^|\s)-ngl\s+(-?\d+)", cmdline)
        if ngl:
            mode = "CPU" if ngl.group(1) == "0" else f"GPU(ngl={ngl.group(1)})"
    elif "test-backend-ops" in cmdline:
        mode = "GPU/CUDA" if "cuda" in cmdline.lower() else "backend-ops"
    return {
        "pid": pid,
        "etime": etime,
        "pcpu": pcpu,
        "cmdline": cmdline,
        "mode": mode,
    }

inp, out = sys.argv[1], sys.argv[2]
state_path = out + ".state.json"
stable_threshold = max(2, int(os.environ.get("REEX_PROGRESS_STABLE_COUNT", "3")))
text = ""
if os.path.exists(inp):
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
state = load_state(state_path)

ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
lines = [ansi_re.sub("", line) for line in text.splitlines()]

header = {}
for line in lines[:20]:
    m = re.match(r"([A-Za-z0-9_]+):\s*(.*)", line.strip())
    if m:
        header[m.group(1)] = m.group(2)

step_lines = []
interesting = [
    "Tier 0", "Tier 1", "Project build matrix", "REEX LUT", "REEX GEMM CPU", "FP16 pipeline",
    "Wikitext", "CUDA A/B", "GEMM Q8", "single-block", "full-layer",
    "[失败]", "[跳过]", "[通过]", "[恢复]",
    "=== 汇总 ===", "=== REEX ", "REEX_GATE_PROFILE=", "PROJECT_VARIANTS=",
    "PROJECT_SMOKE_VARIANTS=", "VARIANT_ORDER=", "verify_build_cuda_reex.sh",
]
for line in lines:
    if any(key in line for key in interesting) or re.search(r"\b\d+/\d+\b", line):
        step_lines.append(line)

step_lines = step_lines[-30:]
tail_lines = lines[-40:]

current_step = ""
for line in reversed(step_lines):
    s = line.strip()
    if s:
        current_step = s
        break

major_progress = ""
major_ratio = None
for line in reversed(step_lines):
    m = re.match(r"\s*(\d+)/(\d+)\s+", line)
    if m:
        cur = int(m.group(1))
        total = int(m.group(2))
        if total > 1 and "Backend" not in line:
            major_ratio = cur / total
            major_progress = f"{cur}/{total}"
            break

sub_progress = ""
sub_ratio = None
for line in reversed(lines):
    m = re.search(r"\[(\d+)/(\d+)\]", line)
    if not m:
        m = re.search(r"Backend\s+(\d+)/(\d+)", line)
    if m:
        cur = int(m.group(1))
        total = int(m.group(2))
        if total > 0:
            sub_ratio = cur / total
            sub_progress = f"{cur}/{total}"
            break

gate_started_at = ""
gate_elapsed = ""
for line in lines:
    m = re.search(r"GATE_STARTED_AT=(\S+)", line)
    if m:
        gate_started_at = m.group(1)
        try:
            started_dt = datetime.fromisoformat(gate_started_at.replace("Z", "+00:00"))
            gate_elapsed = fmt_duration(int((datetime.now(started_dt.tzinfo) - started_dt).total_seconds()))
        except Exception:
            gate_elapsed = ""
        break

gate_total_duration = ""
for line in reversed(lines):
    m = re.search(r"GATE_TOTAL_DURATION=(\S+)", line)
    if m:
        gate_total_duration = m.group(1)
        break

profile = ""
for line in lines:
    m = re.search(r"REEX_GATE_PROFILE\s*=\s*([A-Za-z0-9_-]+)", line)
    if m:
        profile = m.group(1)

current_variant = ""
for line in reversed(lines):
    s = line.strip()
    m = re.match(r"-{6,}\s*(.+?)\s*-{6,}$", s)
    if m:
        current_variant = m.group(1)
        break

container_name = ""
for line in lines:
    s = line.strip()
    m = re.search(r"容器:\s*([A-Za-z0-9_.-]+)", s)
    if m:
        container_name = m.group(1)
        break

current_runner = ""
for line in reversed(lines):
    s = line.strip()
    m = re.search(r"\[运行\]\s+(\S+)", s)
    if m:
        current_runner = os.path.basename(m.group(1))
        break

live_process = get_live_process_info(container_name)

current_params = ""
for line in reversed(lines):
    s = line.strip()
    if (
        re.search(r"chunks=\d+.*ctx=\d+.*batch=\d+", s)
        or re.search(r"ngl=\d+.*batch=\d+", s)
        or re.search(r"ctx=\d+ batch=\d+ chunks=\d+ ngl=\d+", s)
    ):
        current_params = s
        break

variant_summary = ""
for line in lines:
    s = line.strip()
    if s.startswith("变体数:"):
        variant_summary = s
        break

key_status = ""
status_patterns = (
    "[通过]", "[失败]", "[跳过]",
    "Built target", "test-backend-ops 通过", "reex gemm tests: all passed",
    "Error:", "FAILED", "failed", "Stop.", "No rule to make target",
)
for line in reversed(lines):
    s = line.strip()
    if s and any(p in s for p in status_patterns):
        key_status = s
        break

current_test_item = ""
current_operator = ""
last_pass_case = ""
last_fail_case = ""
noise_prefixes = (
    "ggml_backend_cuda_graph_compute: CUDA graph warmup",
    "git config --global --add safe.directory",
    "fatal: detected dubious ownership",
    "To add an exception for this directory, call:",
    "CMake Warning",
    "OpenSSL not found",
    "ggml_cuda_init: found",
    "Device 0:",
)
test_case_patterns = (
    r".*:\s+(OK|FAIL|not supported\b).*",
    r"OK vec_dot .*",
    r"OK single_block .*",
    r"OK chain .*",
    r"FAIL chain .*",
)
status_item_patterns = (
    r"\[chain: .*",
    r"=== SUMMARY: .*",
    r"reex gemm tests: all passed",
)
for line in reversed(lines):
    s = line.strip()
    if not s:
        continue
    if s.startswith(noise_prefixes):
        continue
    if any(re.match(p, s) for p in test_case_patterns):
        current_test_item = s
        current_operator = extract_operator(s)
        break

if not current_test_item:
    for line in reversed(lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith(noise_prefixes):
            continue
        if any(re.match(p, s) for p in status_item_patterns):
            current_test_item = s
            current_operator = extract_operator(s)
            break

for line in reversed(lines):
    s = line.strip()
    if not s:
        continue
    if s.startswith(noise_prefixes):
        continue
    if (
        any(re.match(p, s) for p in test_case_patterns)
        and "FAIL" not in s
        and "not supported" not in s
    ):
        last_pass_case = s
        break

for line in reversed(lines):
    s = line.strip()
    if not s:
        continue
    if s.startswith(noise_prefixes):
        continue
    if re.search(r"\bFAIL\b", s):
        last_fail_case = s
        break

display_operator = update_stable_field(state, "current_operator", current_operator, stable_threshold)
display_test_item = update_stable_field(state, "current_test_item", current_test_item, stable_threshold)
display_pass_case = update_stable_field(state, "last_pass_case", last_pass_case, stable_threshold)
display_fail_case = update_stable_field(state, "last_fail_case", last_fail_case, stable_threshold)

failure_summary = ""
failure_patterns = (
    r"Error:.*",
    r".*\[失败\].*",
    r".*No rule to make target.*",
    r".*could not load cache.*",
    r".*FAILED.*",
    r".*Stop\.\s*$",
)
for line in reversed(lines):
    s = line.strip()
    if not s:
        continue
    if s.startswith("CMake Warning") or "warning:" in s.lower() or s.startswith("Remark:"):
        continue
    if any(re.match(p, s) for p in failure_patterns):
        failure_summary = s
        break

running_ms = header.get("running_for_ms", "")
elapsed_ms = header.get("elapsed_ms", "")
exit_code = None
for line in reversed(lines[-20:]):
    m = re.match(r"exit_code:\s*(\S+)", line.strip())
    if m:
        exit_code = m.group(1)
        break

status = "运行中"
if exit_code is not None:
    status = f"已结束（exit_code={exit_code}）"

md = []
md.append("# REEX Gate-All 实时进度")
md.append("")
md.append(f"- 更新时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
md.append(f"- 状态: `{status}`")
if profile:
    md.append(f"- 档位: `{profile}`")
if running_ms:
    md.append(f"- 运行时长(ms): `{running_ms}`")
if gate_started_at:
    md.append(f"- Gate起始时间: `{gate_started_at}`")
if gate_elapsed and exit_code is None:
    md.append(f"- Gate累计耗时: `{gate_elapsed}`")
if elapsed_ms:
    md.append(f"- 总耗时(ms): `{elapsed_ms}`")
if gate_total_duration:
    md.append(f"- Gate总耗时: `{gate_total_duration}`")
cmd = header.get("command")
if cmd:
    md.append(f"- 命令: `{cmd}`")

md.append("")
md.append("## 自动诊断")
md.append("")
md.append(f"- 当前阶段: `{current_step or '未知'}`")
real_stage_parts = []
if current_runner:
    real_stage_parts.append(current_runner)
if current_step and current_step.startswith("Backend "):
    real_stage_parts.append(current_step)
if real_stage_parts:
    md.append(f"- 当前真实执行阶段: `{' / '.join(real_stage_parts)}`")
if container_name:
    md.append(f"- 运行容器: `{container_name}`")
if live_process:
    md.append(
        f"- 容器实时活跃进程: `pid={live_process.get('pid')} etime={live_process.get('etime')} cpu={live_process.get('pcpu')}%`"
    )
    if live_process.get("mode"):
        md.append(f"- 容器实时运行模式: `{live_process.get('mode')}`")
    md.append(f"- 容器实时命令: `{live_process.get('cmdline')}`")
if major_progress:
    major_percent = int(major_ratio * 100)
    major_filled = max(1, min(20, round(major_ratio * 20)))
    major_bar = "█" * major_filled + "░" * (20 - major_filled)
    md.append(f"- 主进度: `{major_progress}` `{major_percent}%`")
    md.append(f"- 主进度条: `{major_bar}`")
if sub_progress:
    sub_percent = int(sub_ratio * 100)
    sub_filled = max(1, min(20, round(sub_ratio * 20)))
    sub_bar = "█" * sub_filled + "░" * (20 - sub_filled)
    md.append(f"- 子进度: `{sub_progress}` `{sub_percent}%`")
    md.append(f"- 子进度条: `{sub_bar}`")
md.append(f"- 最新关键状态: `{key_status or '暂无'}`")
md.append(f"- 字段稳定阈值: `连续{stable_threshold}轮刷新后更新`")
if display_operator:
    md.append(f"- 最近识别到的算子: `{display_operator}`")
if display_test_item:
    md.append(f"- 最近识别到的测试项: `{display_test_item}`")
if display_pass_case:
    md.append(f"- 最近通过用例: `{display_pass_case}`")
if display_fail_case:
    md.append(f"- 最近失败用例: `{display_fail_case}`")
if current_variant:
    md.append(f"- 当前变体: `{current_variant}`")
if variant_summary:
    md.append(f"- 当前变体规模: `{variant_summary}`")
if current_params:
    md.append(f"- 当前运行参数: `{current_params}`")
if exit_code is not None:
    md.append(f"- 最终结论: `{'成功' if exit_code == '0' else '失败'}`")
if failure_summary:
    md.append(f"- 失败摘要: `{failure_summary}`")

md.append("")
md.append("## 最近阶段")
md.append("")
if step_lines:
    for line in step_lines:
        md.append(f"- {line}")
else:
    md.append("- 暂无阶段摘要")

md.append("")
md.append("## 最新输出尾部")
md.append("")
md.append("```text")
if tail_lines:
    md.extend(tail_lines)
else:
    md.append("(暂无输出)")
md.append("```")

with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
with open(state_path, "w", encoding="utf-8") as f:
    json.dump(state, f, ensure_ascii=False)

print("1" if exit_code is not None else "0")
PY
  )"

  if [[ "$done_flag" == "1" ]]; then
    break
  fi
  sleep "$INTERVAL_SEC"
done
