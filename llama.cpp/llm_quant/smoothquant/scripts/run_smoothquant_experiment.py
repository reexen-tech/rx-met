#!/usr/bin/env python3
"""End-to-end SmoothQuant + llama.cpp quantization experiment and report.

Runs the whole chain and records, for every stage, the exact command line, its
exit code, wall time and a full log, so the result can be reproduced and
audited later:

    export_smoothquant_hf.py -> convert_hf_to_gguf.py -> llama-quantize
                             -> llama-perplexity (FP16 / direct quant / smooth quant)

  python llm_quant/smoothquant/scripts/run_smoothquant_experiment.py \\
    --model_path /mnt/data8t/share/models/Qwen/Qwen2.5-7B-Instruct \\
    --output_dir runs/qwen2.5-7b-sq-q4-k-64 \\
    --calib_data /path/to/calib.jsonl \\
    --ppl_dataset /path/to/wiki.test.raw \\
    --alpha 0.85 \\
    --quant-type Q4_K_64
"""
# === REEX_SMOOTHQUANT BEGIN: experiment orchestration and report generation ===
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_quant.smoothquant.runner import _source_model_fingerprint  # noqa: E402

_PPL_FINAL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")

# Every group must share these flags, otherwise the PPL delta measures the flags
# rather than the algorithm.
QUANTIZE_FLAGS = [
    "--token-embedding-type",
    "f16",
    "--output-tensor-type",
    "f16",
    "--leave-output-tensor",
]

BASELINE_PROVENANCE_SCHEMA_VERSION = 1


@dataclass
class Stage:
    name: str
    cmd: list[str] | None = None
    skipped: bool = False
    skip_reason: str = ""
    log_file: Path | None = None
    exit_code: int | None = None
    elapsed_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class SmoothQuantExperiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = Path(args.output_dir).resolve()
        self.logs_dir = self.output_dir / "logs"
        self.stages: list[Stage] = []
        self.manifest: dict[str, Any] = {}
        self._log_index = 0
        self._started_at = ""
        self._finished_at = ""
        self._total_elapsed = 0.0

        self.smooth_hf = self.output_dir / "smoothquant-hf"
        self.act_scales = (
            Path(args.act_scales_cache).expanduser().resolve()
            if args.act_scales_cache
            else self.output_dir / "act_scales.pt"
        )
        self.fp16_gguf = self.output_dir / "model-f16.gguf"
        self.smooth_f16_gguf = self.output_dir / "model-smooth-f16.gguf"
        self.baseline_quant = self.output_dir / f"model-baseline-{args.quant_type}.gguf"
        self.smooth_quant = self.output_dir / f"model-smooth-{args.quant_type}.gguf"
        self.baseline_provenance = self.output_dir / "baseline_provenance.json"
        self.baseline_reuse: dict[str, dict[str, Any]] = {
            "fp16_gguf": {"requested": args.fp16_gguf, "reused": False},
            "baseline_quant": {"requested": args.baseline_quant, "reused": False},
        }
        self._resolve_external_baseline()

    # --- planning -----------------------------------------------------

    def plan(self) -> list[tuple[str, list[str]]]:
        a = self.args
        plan: list[tuple[str, list[str]]] = [
            ("smoothquant_export", self._cmd_export()),
            ("convert_smooth_hf", self._cmd_convert(self.smooth_hf, self.smooth_f16_gguf)),
        ]
        if not a.skip_baseline and not self.baseline_reuse["fp16_gguf"]["reused"]:
            plan.append(("convert_fp16", self._cmd_convert(Path(a.model_path), self.fp16_gguf)))
        plan.append(("quantize_smooth", self._cmd_quantize(self.smooth_f16_gguf, self.smooth_quant)))
        if not a.skip_baseline:
            if not self.baseline_reuse["baseline_quant"]["reused"]:
                plan.append(
                    ("quantize_baseline", self._cmd_quantize(self.fp16_gguf, self.baseline_quant))
                )
            plan += [
                ("ppl_fp16", self._cmd_ppl(self.fp16_gguf)),
                ("ppl_baseline_quant", self._cmd_ppl(self.baseline_quant)),
            ]
        plan.append(("ppl_smooth_quant", self._cmd_ppl(self.smooth_quant)))
        if a.ref_repo:
            plan.append(("ref_compare", self._cmd_ref_compare()))
        return plan

    def _resolve_external_baseline(self) -> None:
        a = self.args
        if not a.fp16_gguf and not a.baseline_quant:
            return
        provenance_path = Path(
            a.baseline_provenance
            or Path(a.baseline_quant or a.fp16_gguf).expanduser().parent / "provenance.json"
        ).expanduser()
        try:
            provenance = json.loads(provenance_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            self._reject_external("fp16_gguf", f"cannot read {provenance_path}: {error}")
            self._reject_external("baseline_quant", f"cannot read {provenance_path}: {error}")
            return

        common_reason = self._validate_provenance_common(provenance)
        if a.fp16_gguf:
            reason = common_reason or self._validate_external_fp16(
                Path(a.fp16_gguf).expanduser(), provenance
            )
            if reason:
                self._reject_external("fp16_gguf", reason)
            else:
                self.fp16_gguf = Path(a.fp16_gguf).expanduser().resolve()
                self.baseline_reuse["fp16_gguf"].update(
                    {"reused": True, "provenance": str(provenance_path.resolve())}
                )
        if a.baseline_quant:
            reason = common_reason or self._validate_external_quant(
                Path(a.baseline_quant).expanduser(), provenance
            )
            if reason:
                self._reject_external("baseline_quant", reason)
            else:
                self.baseline_quant = Path(a.baseline_quant).expanduser().resolve()
                self.baseline_reuse["baseline_quant"].update(
                    {"reused": True, "provenance": str(provenance_path.resolve())}
                )

    def _reject_external(self, artifact: str, reason: str) -> None:
        if not self.baseline_reuse[artifact]["requested"]:
            return
        self.baseline_reuse[artifact]["reason"] = reason
        warnings.warn(
            f"rejecting external {artifact}: {reason}; rebuilding in {self.output_dir}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _validate_provenance_common(self, provenance: dict[str, Any]) -> str | None:
        if provenance.get("schema_version") != BASELINE_PROVENANCE_SCHEMA_VERSION:
            return "baseline provenance schema_version mismatch"
        expected_source = _source_model_fingerprint(self.args.model_path)
        if provenance.get("source_model_fingerprint") != expected_source:
            return "source model/config fingerprint mismatch"
        current_commit = _capture(["git", "-C", str(_ROOT), "rev-parse", "HEAD"])
        if provenance.get("llama_cpp_commit") != current_commit:
            return "llama.cpp commit mismatch"
        return None

    def _validate_external_fp16(
        self, artifact: Path, provenance: dict[str, Any]
    ) -> str | None:
        return _validate_artifact_fingerprint(artifact, provenance.get("fp16_gguf"))

    def _validate_external_quant(
        self, artifact: Path, provenance: dict[str, Any]
    ) -> str | None:
        reason = _validate_artifact_fingerprint(
            artifact, provenance.get("baseline_quant")
        )
        if reason:
            return reason
        quantize = provenance.get("quantize")
        if not isinstance(quantize, dict):
            return "missing quantize provenance"
        recorded_fp16 = provenance.get("fp16_gguf")
        if not isinstance(recorded_fp16, dict) or not recorded_fp16.get("path"):
            return "missing FP16 input provenance"
        reason = _validate_artifact_fingerprint(
            Path(recorded_fp16["path"]), recorded_fp16
        )
        if reason:
            return f"recorded FP16 input {reason}"
        expected_command = self._cmd_quantize(
            Path(recorded_fp16["path"]), artifact.resolve()
        )
        if quantize.get("command") != expected_command:
            return "complete llama-quantize command/flags mismatch"
        if quantize.get("flags") != ([] if self.args.quantize_full else QUANTIZE_FLAGS):
            return "llama-quantize flags mismatch"
        if quantize.get("quant_type") != self.args.quant_type:
            return "quant type mismatch"
        binary_reason = _validate_artifact_fingerprint(
            Path(self.args.llama_quantize), quantize.get("binary")
        )
        if binary_reason:
            return f"quantizer binary {binary_reason}"
        return None

    def _cmd_export(self) -> list[str]:
        a = self.args
        cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "export_smoothquant_hf.py"),
            "--model_path", a.model_path,
            "--output_dir", str(self.smooth_hf),
            "--alpha", str(a.alpha),
            "--n_samples", str(a.n_samples),
            "--seqlen", str(a.seqlen),
            "--calib_data", a.calib_data,
            "--device", a.device,
            "--act_scales_cache", str(self.act_scales),
            "--calibration-mode", a.calibration_mode,
        ]
        if a.min_samples is not None:
            cmd += ["--min_samples", str(a.min_samples)]
        if a.max_samples is not None:
            cmd += ["--max_samples", str(a.max_samples)]
        if a.dtype:
            cmd += ["--dtype", a.dtype]
        if a.reuse_act_scales:
            cmd.append("--reuse_act_scales")
        if a.resume_act_scales:
            cmd.append("--resume_act_scales")
        return cmd

    def _cmd_convert(self, hf_dir: Path, outfile: Path) -> list[str]:
        return [
            sys.executable,
            str(_ROOT / "convert_hf_to_gguf.py"),
            str(hf_dir),
            "--outtype", self.args.outtype,
            "--outfile", str(outfile),
        ]

    def _cmd_quantize(self, src: Path, dst: Path) -> list[str]:
        flags = [] if self.args.quantize_full else QUANTIZE_FLAGS
        return [self.args.llama_quantize, *flags, str(src), str(dst), self.args.quant_type]

    def _cmd_ppl(self, gguf: Path) -> list[str]:
        a = self.args
        cmd = [
            a.llama_perplexity,
            "-m", str(gguf),
            "-f", a.ppl_dataset,
            "-c", str(a.ppl_ctx),
            "-ngl", str(a.ngl),
        ]
        if a.ppl_chunks:
            cmd += ["--chunks", str(a.ppl_chunks)]
        return cmd

    def _cmd_ref_compare(self) -> list[str]:
        a = self.args
        return [
            sys.executable,
            str(_SCRIPT_DIR / "compare_smoothquant_ref.py"),
            "--model_path", a.model_path,
            "--calib_data", a.calib_data,
            "--alpha", str(a.alpha),
            "--n_samples", str(a.ref_compare_samples),
            "--seqlen", str(a.seqlen),
            "--device", a.device,
            "--ref_repo", a.ref_repo,
            "--report", str(self.output_dir / "ref_compare.json"),
        ]

    # --- execution ----------------------------------------------------

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self._write_resolved_config()
        self._build_manifest()

        self._started_at = datetime.datetime.now().isoformat(timespec="seconds")
        t0 = time.time()
        try:
            for name, cmd in self.plan():
                self._run_stage(name, cmd)
        finally:
            self._total_elapsed = time.time() - t0
            self._finished_at = datetime.datetime.now().isoformat(timespec="seconds")
            self._maybe_write_baseline_provenance()
            self._finalize_manifest()
            self._write_manifest()
            self._write_report()
        return self._summary()

    def _maybe_write_baseline_provenance(self) -> None:
        if self.args.skip_baseline:
            return
        generated_fp16 = not self.baseline_reuse["fp16_gguf"]["reused"]
        generated_quant = not self.baseline_reuse["baseline_quant"]["reused"]
        if not (generated_fp16 or generated_quant):
            return
        successful = {stage.name: stage.exit_code == 0 for stage in self.stages}
        if generated_fp16 and not successful.get("convert_fp16"):
            return
        if generated_quant and not successful.get("quantize_baseline"):
            return
        payload = {
            "schema_version": BASELINE_PROVENANCE_SCHEMA_VERSION,
            "source_model_fingerprint": _source_model_fingerprint(self.args.model_path),
            "llama_cpp_commit": _capture(["git", "-C", str(_ROOT), "rev-parse", "HEAD"]),
            "fp16_gguf": _fingerprint_path(str(self.fp16_gguf), hash_files=True),
            "baseline_quant": _fingerprint_path(str(self.baseline_quant), hash_files=True),
            "convert_command": self._cmd_convert(Path(self.args.model_path), self.fp16_gguf),
            "quantize": {
                "command": self._cmd_quantize(self.fp16_gguf, self.baseline_quant),
                "flags": [] if self.args.quantize_full else QUANTIZE_FLAGS,
                "quant_type": self.args.quant_type,
                "binary": _fingerprint_path(self.args.llama_quantize, hash_files=True),
            },
        }
        self.baseline_provenance.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )

    def _run_stage(self, name: str, cmd: list[str]) -> Stage:
        self._log_index += 1
        log_file = self.logs_dir / f"{self._log_index:02d}_{name}.log"
        print(f"\n[sq] === {name} ===")
        print(f"[sq] {shlex.join(cmd)}")
        print(f"[sq] log -> {log_file}", flush=True)

        t0 = time.time()
        with log_file.open("w", encoding="utf-8", errors="replace") as fh:
            fh.write(f"# stage: {name}\n")
            fh.write(f"# cmd: {shlex.join(cmd)}\n")
            fh.write(f"# started_at: {datetime.datetime.now().isoformat()}\n\n")
            fh.flush()
            proc = subprocess.run(
                cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=_ROOT, check=False
            )
        stage = Stage(
            name=name,
            cmd=cmd,
            log_file=log_file,
            exit_code=proc.returncode,
            elapsed_s=time.time() - t0,
        )
        if name.startswith("ppl_"):
            parsed = _parse_perplexity(log_file)
            if parsed:
                stage.extra["perplexity"] = parsed
                print(f"[sq] PPL = {parsed['ppl']} +/- {parsed['stderr']}")
        self.stages.append(stage)
        print(f"[sq] exit={proc.returncode} elapsed={stage.elapsed_s:.1f}s", flush=True)

        if proc.returncode != 0 and not self.args.continue_on_error:
            raise RuntimeError(
                f"stage '{name}' failed with exit code {proc.returncode}; see {log_file}"
            )
        return stage

    # --- manifest and report ------------------------------------------

    def _write_resolved_config(self) -> None:
        path = self.output_dir / "resolved_config.json"
        path.write_text(json.dumps(vars(self.args), indent=2, ensure_ascii=False))

    def _build_manifest(self) -> None:
        self.manifest = {
            "experiment": self.args.name or self.output_dir.name,
            "output_dir": str(self.output_dir),
            "environment": _environment(self.args),
            "config": vars(self.args),
            "inputs": {
                "model": _fingerprint_path(self.args.model_path),
                "calib_data": _fingerprint_path(self.args.calib_data),
                "ppl_dataset": _fingerprint_path(self.args.ppl_dataset),
                "llama_quantize": _fingerprint_path(self.args.llama_quantize),
                "llama_perplexity": _fingerprint_path(self.args.llama_perplexity),
            },
            "baseline_reuse": self.baseline_reuse,
        }

    def _finalize_manifest(self) -> None:
        outputs = {}
        for key, path in {
            "smoothquant_hf": self.smooth_hf,
            "act_scales": self.act_scales,
            "fp16_gguf": self.fp16_gguf,
            "smooth_f16_gguf": self.smooth_f16_gguf,
            "baseline_quant_gguf": self.baseline_quant,
            "smooth_quant_gguf": self.smooth_quant,
        }.items():
            if path.exists():
                outputs[key] = _fingerprint_path(str(path), hash_files=not self.args.no_hash)
        self.manifest["outputs"] = outputs
        self.manifest["started_at"] = self._started_at
        self.manifest["finished_at"] = self._finished_at
        self.manifest["total_elapsed_s"] = round(self._total_elapsed, 3)
        self.manifest["stages"] = [_stage_dict(s) for s in self.stages]
        self.manifest["perplexity"] = self._ppl_table()
        ref_json = self.output_dir / "ref_compare.json"
        if ref_json.is_file():
            self.manifest["ref_compare"] = json.loads(ref_json.read_text())

    def _ppl_table(self) -> dict[str, Any]:
        labels = {
            "ppl_fp16": ("FP16 baseline", self.fp16_gguf),
            "ppl_baseline_quant": (f"no smooth + {self.args.quant_type}", self.baseline_quant),
            "ppl_smooth_quant": (f"smooth + {self.args.quant_type}", self.smooth_quant),
        }
        table = {}
        for stage in self.stages:
            if stage.name in labels:
                label, gguf = labels[stage.name]
                table[stage.name] = {
                    "label": label,
                    "gguf": gguf.name,
                    "log": stage.log_file.name if stage.log_file else None,
                    **(stage.extra.get("perplexity") or {}),
                }
        return table

    def _summary(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "perplexity": self.manifest.get("perplexity", {}),
            "total_elapsed_s": round(self._total_elapsed, 3),
            "failed_stages": [s.name for s in self.stages if s.exit_code not in (0, None)],
        }

    def _write_manifest(self) -> None:
        path = self.output_dir / "manifest.json"
        path.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False))

    def _write_report(self) -> None:
        a = self.args
        env = self.manifest["environment"]
        lines = [
            f"# SmoothQuant + {a.quant_type} experiment: `{self.manifest['experiment']}`",
            "",
            "## 1. 实验元信息",
            "",
            f"- started_at: `{self._started_at}`",
            f"- finished_at: `{self._finished_at}`",
            f"- total_elapsed_s: `{round(self._total_elapsed, 3)}`",
            f"- host: `{env['hostname']}` ({env['uname']})",
            f"- container: `{env['container'] or '(host)'}`",
            f"- python: `{env['python']}`, torch: `{env['torch']}`, cuda: `{env['torch_cuda']}`",
            f"- gpu: `{env['gpu'] or 'n/a'}`",
            f"- git: `{env['git_commit'] or 'n/a'}`",
            f"- workdir: `{env['workdir']}`",
            "",
            "## 2. 配置摘要",
            "",
            f"- model_path: `{a.model_path}`",
            f"- alpha: **{a.alpha}**, n_samples: {a.n_samples}, seqlen: {a.seqlen}",
            f"- calib_data: `{a.calib_data}`",
            f"- dtype: `{a.dtype or 'auto (config.torch_dtype)'}`, gguf outtype: `{a.outtype}`",
            f"- quant type: `{a.quant_type}`",
            f"- quantize flags: `{' '.join(QUANTIZE_FLAGS) if not a.quantize_full else '(none; quantizer defaults)'}`",
            f"- ppl: `-c {a.ppl_ctx} -ngl {a.ngl}"
            + (f" --chunks {a.ppl_chunks}" if a.ppl_chunks else "")
            + f"` on `{a.ppl_dataset}`",
            "",
            "## 3. 输入指纹",
            "",
        ]
        for key, val in self.manifest["inputs"].items():
            lines.append(f"- **{key}**: {_fingerprint_md(val)}")

        lines += ["", "## 4. Stages", ""]
        for stage in self.stages:
            if stage.skipped:
                lines += [f"### {stage.name} — *skipped* ({stage.skip_reason})", ""]
                continue
            lines += [
                f"### {stage.name} — exit={stage.exit_code} elapsed={stage.elapsed_s:.2f}s",
                "",
                "```bash",
                _format_cmd(stage.cmd or []),
                "```",
                "",
                f"log: `logs/{stage.log_file.name}`" if stage.log_file else "",
                "",
            ]

        lines += [
            "## 5. PPL 结果",
            "",
            "| 组别 | GGUF | PPL ± stderr | log |",
            "|---|---|---|---|",
        ]
        for entry in self.manifest["perplexity"].values():
            ppl = (
                f"{entry['ppl']:.4f} ± {entry['stderr']:.4f}"
                if "ppl" in entry
                else "(未解析到 Final estimate)"
            )
            lines.append(
                f"| {entry['label']} | `{entry['gguf']}` | {ppl} | `logs/{entry['log']}` |"
            )

        ref = self.manifest.get("ref_compare")
        if ref:
            lines += [
                "",
                "## 5b. 与 MIT smoothquant 交叉验证",
                "",
                f"- 结论: **{'PASS' if ref['passed'] else 'FAIL'}**"
                f"（reference dispatch: {ref['reference_dispatch']}）",
                f"- act_scales: {ref['n_act_scale_keys']} 个 key，max diff "
                f"{ref['act_scales_max_diff']:.3e}（阈值 {ref['act_scales_atol']:.0e}）",
                f"- 平滑后权重: {ref['n_weight_keys']} 个张量（层 {ref['compared_layers']}），"
                f"max diff {ref['weights_max_diff']:.3e}（阈值 {ref['weights_atol']:.0e}）",
            ]

        lines += ["", "## 6. 产物指纹", ""]
        for key, val in self.manifest["outputs"].items():
            lines.append(f"- **{key}**: {_fingerprint_md(val)}")

        lines += ["", "## 7. 结论", ""] + self._conclusion()
        (self.output_dir / "experiment_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        print(f"\n[sq] report -> {self.output_dir / 'experiment_report.md'}")

    def _conclusion(self) -> list[str]:
        table = self.manifest.get("perplexity", {})
        fp16 = table.get("ppl_fp16", {}).get("ppl")
        base = table.get("ppl_baseline_quant", {}).get("ppl")
        smooth = table.get("ppl_smooth_quant", {}).get("ppl")
        quant_type = self.args.quant_type
        failed = [s.name for s in self.stages if s.exit_code not in (0, None)]
        out = []
        if failed:
            out.append(f"- **失败 stage**：{', '.join(failed)}（见对应 log）")
        if smooth is None:
            out.append(f"- 未取得 smooth {quant_type} 的 PPL，无法判定验收条件。")
            return out
        if base is None:
            out.append(
                f"- smooth + {quant_type} PPL = {smooth:.4f}；本次未跑 baseline"
                f"（--skip_baseline），需与同配置的基线实验对比。"
            )
            return out
        verdict = "满足" if smooth <= base else "**不满足**"
        out.append(
            f"- 验收条件「smooth {quant_type} ≤ 无 smooth {quant_type}」：{verdict} "
            f"（{smooth:.4f} vs {base:.4f}，Δ={smooth - base:+.4f}）"
        )
        if fp16 is not None:
            out.append(
                f"- FP16 sanity 上限：{fp16:.4f}；smooth {quant_type} 相对 FP16 退化 "
                f"{(smooth - fp16) / fp16 * 100:+.2f}%"
            )
        return out


# --- helpers ----------------------------------------------------------


def _parse_perplexity(log_file: Path) -> dict[str, float] | None:
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _PPL_FINAL_RE.search(line)
                if m:
                    return {"ppl": float(m.group(1)), "stderr": float(m.group(2))}
    except OSError:
        pass
    return None


def _sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _fingerprint_path(path: str | None, *, hash_files: bool = True) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    if p.is_dir():
        return {"path": str(p), "kind": "dir", "n_files": sum(1 for _ in p.rglob("*"))}
    info: dict[str, Any] = {
        "path": str(p),
        "kind": "file",
        "size_bytes": p.stat().st_size,
        "mtime": datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
    }
    if hash_files:
        info["sha256"] = _sha256_file(p)
    return info


def _fingerprint_md(val: dict[str, Any] | None) -> str:
    if val is None:
        return "_(not set)_"
    if not val.get("exists", True):
        return f"`{val['path']}` _(missing)_"
    if val.get("kind") == "dir":
        return f"`{val['path']}` (目录, {val['n_files']} files)"
    parts = [f"`{val['path']}`", f"size={val['size_bytes']:,} B"]
    if "sha256" in val:
        parts.append(f"sha256=`{val['sha256']}`")
    return "  \n  ".join(parts)


def _validate_artifact_fingerprint(
    artifact: Path, recorded: dict[str, Any] | None
) -> str | None:
    if not isinstance(recorded, dict):
        return "missing artifact fingerprint"
    resolved = artifact.expanduser().resolve()
    if not resolved.is_file():
        return f"artifact does not exist: {resolved}"
    if recorded.get("path") != str(resolved):
        return "artifact path mismatch"
    if recorded.get("size_bytes") != resolved.stat().st_size:
        return "artifact size mismatch"
    recorded_sha = recorded.get("sha256")
    if not recorded_sha:
        return "artifact SHA-256 missing"
    if recorded_sha != _sha256_file(resolved):
        return "artifact SHA-256 mismatch"
    return None


def _stage_dict(s: Stage) -> dict[str, Any]:
    return {
        "name": s.name,
        "skipped": s.skipped,
        "skip_reason": s.skip_reason,
        "exit_code": s.exit_code,
        "elapsed_s": round(s.elapsed_s, 3),
        "cmd": s.cmd,
        "cmd_str": shlex.join(s.cmd) if s.cmd else None,
        "log_file": f"logs/{s.log_file.name}" if s.log_file else None,
        "extra": s.extra,
    }


_BOOLEAN_FLAGS = frozenset(
    {
        "--leave-output-tensor",
        "--reuse_act_scales",
        "--resume_act_scales",
        "--continue-on-error",
    }
)


def _format_cmd(cmd: list[str]) -> str:
    """One flag (with its value) per line, so the report can be copy-pasted."""
    if not cmd:
        return ""
    lines = [shlex.quote(cmd[0])]
    i = 1
    while i < len(cmd):
        token = cmd[i]
        takes_value = (
            token.startswith("-")
            and token not in _BOOLEAN_FLAGS
            and i + 1 < len(cmd)
            and not cmd[i + 1].startswith("-")
        )
        if takes_value:
            lines.append(f"{shlex.quote(token)} {shlex.quote(cmd[i + 1])}")
            i += 2
        else:
            lines.append(shlex.quote(token))
            i += 1
    return " \\\n  ".join(lines)


def _environment(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
        gpu = ", ".join(
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        )
    except Exception:  # torch is optional for --dry-run
        torch_version = torch_cuda = gpu = None
    return {
        "hostname": socket.gethostname(),
        "uname": " ".join(platform.uname()),
        "container": args.container or os.environ.get("REEX_SQ_CONTAINER"),
        "python": sys.version.split()[0],
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "gpu": gpu,
        "nvidia_smi": _capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                "--format=csv,noheader"]),
        "git_commit": _capture(["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"]),
        "workdir": str(_ROOT),
    }


def _capture(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_quant_type(value: str) -> str:
    quant_type = value.upper()
    if not re.fullmatch(r"[A-Z0-9_]+", quant_type):
        raise argparse.ArgumentTypeError(
            "quant type must contain only letters, digits, and underscores"
        )
    return quant_type


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True, help="Source HF checkpoint")
    p.add_argument("--output_dir", required=True, help="Experiment directory")
    p.add_argument("--name", default=None, help="Experiment name (default: output_dir name)")
    p.add_argument("--calib_data", default="pileval", help="Calibration .jsonl")
    p.add_argument("--ppl_dataset", required=True, help="Raw text file for llama-perplexity")
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--n_samples", type=int, default=512)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument(
        "--calibration-mode", choices=("fixed", "smoke", "formal"), default="fixed"
    )
    p.add_argument("--min_samples", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="auto", help="device_map for calibration")
    p.add_argument(
        "--act-scales-cache",
        default=None,
        help="Shared calibration cache path (may be outside this alpha output directory)",
    )
    cache_mode = p.add_mutually_exclusive_group()
    cache_mode.add_argument("--reuse_act_scales", action="store_true")
    cache_mode.add_argument("--resume_act_scales", action="store_true")
    p.add_argument("--outtype", default="f16", help="convert_hf_to_gguf --outtype")
    p.add_argument(
        "--quant-type",
        type=_parse_quant_type,
        default="Q8_0_64",
        help="llama-quantize target type (default: Q8_0_64)",
    )
    p.add_argument("--llama_quantize", default="./build_cuda_q64/bin/llama-quantize")
    p.add_argument("--llama_perplexity", default="./build_cuda_q64/bin/llama-perplexity")
    p.add_argument(
        "--quantize_full",
        action="store_true",
        help="Use quantizer defaults instead of preserving embedding/output as F16",
    )
    p.add_argument("--ppl_ctx", type=int, default=512)
    p.add_argument("--ppl_chunks", type=int, default=None)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--ref_repo", default=None, help="MIT smoothquant checkout; enables ref_compare")
    p.add_argument("--ref_compare_samples", type=int, default=32)
    p.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Only run the smoothed arm; use when sweeping alpha against a known baseline",
    )
    p.add_argument(
        "--fp16-gguf",
        default=None,
        help="Existing FP16 GGUF; reused only with matching baseline provenance",
    )
    p.add_argument(
        "--baseline-quant",
        default=None,
        help="Existing direct-quant GGUF; reused only with matching provenance",
    )
    p.add_argument(
        "--baseline-provenance",
        default=None,
        help="Provenance JSON for external baseline artifacts (default: sibling provenance.json)",
    )
    p.add_argument("--container", default=None, help="Container name recorded in the report")
    p.add_argument("--continue-on-error", dest="continue_on_error", action="store_true")
    p.add_argument("--no-hash", dest="no_hash", action="store_true",
                   help="Skip sha256 of the produced GGUF files")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Print the commands without running anything")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    experiment = SmoothQuantExperiment(args)
    if args.dry_run:
        for name, cmd in experiment.plan():
            print(f"# {name}\n{_format_cmd(cmd)}\n")
        return 0
    summary = experiment.run()
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["failed_stages"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
# === REEX_SMOOTHQUANT END ===
