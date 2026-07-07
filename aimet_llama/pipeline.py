"""End-to-end LLM quantization pipeline driven by a single JSON config.

Stage order:

    1. Resolve config + write ``resolved_config.json`` to ``output_dir``.
    2. Hash all input artifacts (model, dataset, binaries) for the manifest.
    3. (optional) Run ``llama-imatrix`` to produce the importance matrix.
    4. Run ``llama-quantize`` with translated CLI flags.
    5. (optional) Run ``llama-perplexity`` for PPL evaluation.
    6. Write a Markdown report capturing every command, log file path,
       artifact hash and elapsed time.

The whole stage graph is deterministic given the JSON: re-running it
always produces the same command lines, and the manifest captures the
fingerprints of the inputs, so reproducibility is auditable byte by byte.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cli import (
    build_hw_export_cmd,
    build_imatrix_cmd,
    build_perplexity_cmd,
    build_quantize_cmd,
    format_cmd,
)
from .schema import (
    ConfigError,
    SCHEMA_VERSION,
    hw_export_output_gguf,
    imatrix_path,
    load_config,
    quantized_path,
    resolve_config,
)


@dataclass
class StageResult:
    name: str
    cmd: Optional[List[str]]
    skipped: bool = False
    skip_reason: str = ""
    log_file: Optional[Path] = None
    exit_code: Optional[int] = None
    elapsed_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _safe_hash(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {"path": str(p), "exists": False}
    return {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "sha256": _sha256_file(p),
    }


_PPL_FINAL_RE = re.compile(r"^Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")


def _parse_perplexity(log_file: Path) -> Optional[Dict[str, float]]:
    """Extract the 'Final estimate: PPL = X +/- Y' line from a log file."""
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _PPL_FINAL_RE.search(line)
                if m:
                    return {"ppl": float(m.group(1)), "stderr": float(m.group(2))}
    except OSError:
        pass
    return None


class LLMQuantPipeline:
    """JSON-driven LLM quantization pipeline.

    Construct via :meth:`from_json` (preferred) or directly with a
    pre-resolved config dict.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = resolve_config(config)
        self.output_dir: Path = Path(self.config["experiment"]["output_dir"]).resolve()
        self.results: List[StageResult] = []
        self.manifest: Dict[str, Any] = {}
        self._started_at: Optional[str] = None
        self._finished_at: Optional[str] = None
        self._total_elapsed: float = 0.0

    # --- construction --------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "LLMQuantPipeline":
        return cls(load_config(path))

    # --- main entry ----------------------------------------------------

    def run(self) -> Dict[str, Any]:
        self._setup_output_dir()
        self._save_resolved_config()
        self._build_manifest()

        self._started_at = datetime.datetime.now().isoformat(timespec="seconds")
        t0 = time.time()

        self._run_imatrix()
        self._run_quantize()
        self._run_hw_export()
        self._run_perplexity()

        self._total_elapsed = time.time() - t0
        self._finished_at = datetime.datetime.now().isoformat(timespec="seconds")

        self._finalize_manifest()
        if self.config["report"]["save_manifest"]:
            self._write_manifest()
        if self.config["report"]["write_markdown"]:
            self._write_markdown_report()
        return self._summary()

    # --- planning (no side effects) -----------------------------------

    def plan(self) -> Dict[str, List[str]]:
        """Return the exact CLI commands this run would execute. Side-effect free."""
        plan: Dict[str, List[str]] = {}
        imat = build_imatrix_cmd(self.config)
        if imat is not None:
            plan["imatrix"] = imat
        plan["quantize"] = build_quantize_cmd(self.config)
        hw = build_hw_export_cmd(self.config)
        if hw is not None:
            plan["hw_export"] = hw
        ppl = build_perplexity_cmd(self.config)
        if ppl is not None:
            plan["perplexity"] = ppl
        return plan

    def print_plan(self) -> None:
        for name, cmd in self.plan().items():
            print(f"\n# stage: {name}")
            print(format_cmd(cmd))

    # --- helpers -------------------------------------------------------

    def _setup_output_dir(self) -> None:
        if self.output_dir.exists():
            if not self.output_dir.is_dir():
                raise ConfigError(
                    f"output_dir exists but is not a directory: {self.output_dir}"
                )
            if not self.config["experiment"].get("overwrite", False) and any(
                self.output_dir.iterdir()
            ):
                # Non-empty: allow but warn (we never delete files).
                print(
                    f"[aimet_llama] WARNING: output_dir is non-empty: {self.output_dir}",
                    file=sys.stderr,
                )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_resolved_config(self) -> None:
        if not self.config["report"]["save_resolved_config"]:
            return
        path = self.output_dir / "resolved_config.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.config, fh, indent=2, ensure_ascii=False)

    def _build_manifest(self) -> None:
        cfg = self.config
        b = cfg["binaries"]
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment": cfg["experiment"]["name"],
            "host": {
                "hostname": os.uname().nodename,
                "python": sys.version.split()[0],
                "cwd": os.getcwd(),
            },
            "inputs": {
                "gguf_fp16": _safe_hash(cfg["model"].get("gguf_fp16_path")),
                "calibration_dataset": _safe_hash(
                    cfg["calibration"].get("dataset_file")
                ),
                "ppl_dataset": _safe_hash(
                    cfg["evaluation"]["perplexity"].get("dataset_file")
                ),
                "reuse_imatrix": _safe_hash(cfg["calibration"].get("reuse_imatrix")),
            },
            "binaries": {
                "llama_quantize": _safe_hash(shutil.which(b["llama_quantize"]) or b["llama_quantize"]),
                "llama_imatrix": _safe_hash(shutil.which(b["llama_imatrix"]) or b["llama_imatrix"]),
                "llama_perplexity": _safe_hash(
                    shutil.which(b["llama_perplexity"]) or b["llama_perplexity"]
                ),
            },
            "outputs": {},
            "stages": [],
        }
        if cfg.get("hw_export", {}).get("enabled"):
            hb = cfg["hw_export"]["binary"]
            self.manifest["binaries"]["reex_hw_convert"] = _safe_hash(
                shutil.which(hb) or hb
            )

    def _finalize_manifest(self) -> None:
        out = {}
        imat = imatrix_path(self.config)
        if imat and imat.exists():
            out["imatrix"] = _safe_hash(str(imat))
        qpath = quantized_path(self.config)
        if qpath.exists():
            out["quantized_gguf"] = _safe_hash(str(qpath))
        if self.config.get("hw_export", {}).get("enabled"):
            hw_gguf = hw_export_output_gguf(self.config)
            if hw_gguf.exists():
                out["hw_gguf"] = _safe_hash(str(hw_gguf))
            idx = Path(str(hw_gguf) + ".hw_index.json")
            if idx.exists():
                out["hw_index"] = _safe_hash(str(idx))
        self.manifest["outputs"] = out
        self.manifest["started_at"] = self._started_at
        self.manifest["finished_at"] = self._finished_at
        self.manifest["total_elapsed_s"] = round(self._total_elapsed, 3)
        self.manifest["stages"] = [self._stage_to_dict(r) for r in self.results]

    @staticmethod
    def _stage_to_dict(r: StageResult) -> Dict[str, Any]:
        return {
            "name": r.name,
            "skipped": r.skipped,
            "skip_reason": r.skip_reason,
            "exit_code": r.exit_code,
            "elapsed_s": round(r.elapsed_s, 3),
            "cmd": r.cmd,
            "cmd_str": shlex.join(r.cmd) if r.cmd else None,
            "log_file": str(r.log_file) if r.log_file else None,
            "extra": r.extra,
        }

    def _write_manifest(self) -> None:
        path = self.output_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.manifest, fh, indent=2, ensure_ascii=False)

    # --- subprocess wrapper -------------------------------------------

    def _run_stage(
        self,
        name: str,
        cmd: Optional[List[str]],
        *,
        skip_reason: str = "",
        env_extra: Optional[Dict[str, str]] = None,
        allowed_exit_codes: Optional[set] = None,
    ) -> StageResult:
        if cmd is None:
            res = StageResult(name=name, cmd=None, skipped=True, skip_reason=skip_reason)
            self.results.append(res)
            return res

        log_file = self.output_dir / f"{name}.log"
        env = os.environ.copy()
        ld = self.config["binaries"].get("ld_library_path")
        if ld:
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{ld}:{prev}" if prev else ld
        if env_extra:
            env.update(env_extra)

        print(f"\n[aimet_llama] stage={name} cmd:")
        print(format_cmd(cmd))
        print(f"[aimet_llama] log -> {log_file}\n", flush=True)

        t0 = time.time()
        with log_file.open("w", encoding="utf-8", errors="replace") as fh:
            fh.write(f"# aimet_llama stage: {name}\n")
            fh.write(f"# cmd: {shlex.join(cmd)}\n")
            if env_extra:
                env_str = " ".join(f"{k}={v}" for k, v in env_extra.items())
                fh.write(f"# env: {env_str}\n")
            fh.write(f"# started_at: {datetime.datetime.now().isoformat()}\n\n")
            fh.flush()
            proc = subprocess.run(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        elapsed = time.time() - t0
        res = StageResult(
            name=name,
            cmd=cmd,
            log_file=log_file,
            exit_code=proc.returncode,
            elapsed_s=elapsed,
        )
        if env_extra:
            res.extra["env"] = dict(env_extra)
        self.results.append(res)

        allowed = allowed_exit_codes if allowed_exit_codes is not None else {0}
        if proc.returncode not in allowed:
            raise RuntimeError(
                f"Stage '{name}' failed with exit code {proc.returncode}; "
                f"see {log_file}"
            )
        return res

    # --- stage runners ------------------------------------------------

    def _run_imatrix(self) -> None:
        cmd = build_imatrix_cmd(self.config)
        if cmd is None:
            cal = self.config["calibration"]
            reason = (
                "calibration disabled"
                if not cal["enabled"]
                else f"reusing imatrix={cal['reuse_imatrix']}"
            )
            self._run_stage("imatrix", None, skip_reason=reason)
            return
        self._run_stage("imatrix", cmd)

    def _run_quantize(self) -> None:
        cmd = build_quantize_cmd(self.config)
        self._run_stage("quantize", cmd)

    def _run_hw_export(self) -> None:
        cmd = build_hw_export_cmd(self.config)
        if cmd is None:
            self._run_stage("hw_export", None, skip_reason="hw_export disabled")
            return

        he = self.config["hw_export"]
        ld_parts: List[str] = []
        if he.get("ld_library_path"):
            ld_parts.append(str(he["ld_library_path"]))
        base_ld = self.config["binaries"].get("ld_library_path")
        if base_ld:
            ld_parts.append(str(base_ld))
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        if prev:
            ld_parts.append(prev)
        env_extra = {"LD_LIBRARY_PATH": ":".join(ld_parts)} if ld_parts else None

        # exit 0 = ok, 1 = matched nothing convertible (warning, no GGUF written),
        # 2 = hard error (bad shape / read / parse / write) -> raise.
        res = self._run_stage(
            "hw_export", cmd, env_extra=env_extra, allowed_exit_codes={0, 1}
        )
        out_gguf = hw_export_output_gguf(self.config)
        idx = Path(str(out_gguf) + ".hw_index.json")
        if idx.exists():
            try:
                with idx.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                res.extra["hw_index"] = data.get("summary")
            except (OSError, ValueError):
                pass
        if res.exit_code == 1:
            res.extra["warning"] = "no convertible (Legacy block-64) weight tensors found"

    def _run_perplexity(self) -> None:
        cmd = build_perplexity_cmd(self.config)
        if cmd is None:
            self._run_stage("perplexity", None, skip_reason="evaluation disabled")
            return
        env_extra: Dict[str, str] = {}
        psum_bits = self.config["evaluation"]["perplexity"].get("reex_psum_bits")
        if isinstance(psum_bits, int) and not isinstance(psum_bits, bool) and psum_bits > 0:
            env_extra["REEX_Q64_PSUM_BITS"] = str(psum_bits)
        res = self._run_stage("perplexity", cmd, env_extra=env_extra or None)
        if res.log_file is not None:
            parsed = _parse_perplexity(res.log_file)
            if parsed:
                res.extra["perplexity"] = parsed

    # --- report -------------------------------------------------------

    def _summary(self) -> Dict[str, Any]:
        ppl = None
        for r in self.results:
            if r.name == "perplexity" and "perplexity" in r.extra:
                ppl = r.extra["perplexity"]
        return {
            "output_dir": str(self.output_dir),
            "quantized_gguf": str(quantized_path(self.config)),
            "imatrix": str(imatrix_path(self.config)) if imatrix_path(self.config) else None,
            "perplexity": ppl,
            "total_elapsed_s": round(self._total_elapsed, 3),
            "stages": [self._stage_to_dict(r) for r in self.results],
        }

    def _write_markdown_report(self) -> None:
        cfg = self.config
        exp = cfg["experiment"]
        report = self.output_dir / cfg["report"]["markdown_name"]
        lines: List[str] = []

        lines.append(f"# AIMET-llama experiment report: `{exp['name']}`")
        if exp.get("description"):
            lines.append(f"\n> {exp['description']}\n")
        lines.append("")
        lines.append(f"- schema_version: `{SCHEMA_VERSION}`")
        lines.append(f"- started_at: `{self._started_at}`")
        lines.append(f"- finished_at: `{self._finished_at}`")
        lines.append(f"- total_elapsed_s: `{round(self._total_elapsed, 3)}`")
        lines.append(f"- host: `{self.manifest['host']['hostname']}` "
                     f"(python {self.manifest['host']['python']})")

        lines.append("\n## Configuration summary\n")
        lines.append(f"- default_type: **{cfg['quantization']['default_type']}**")
        if cfg["quantization"].get("output_tensor_type"):
            lines.append(
                f"- output_tensor_type: **{cfg['quantization']['output_tensor_type']}**"
            )
        if cfg["quantization"].get("token_embedding_type"):
            lines.append(
                f"- token_embedding_type: "
                f"**{cfg['quantization']['token_embedding_type']}**"
            )
        overrides = cfg["quantization"]["tensor_type_overrides"]
        if overrides:
            lines.append(f"- tensor_type_overrides ({len(overrides)}):")
            for o in overrides:
                lines.append(f"    - `{o['pattern']}` → **{o['type']}**")
        lines.append(
            f"- imatrix: "
            f"{'used' if cfg['quantization'].get('use_imatrix') else 'not used'}"
        )
        if cfg.get("hw_export", {}).get("enabled"):
            he = cfg["hw_export"]
            pat = ", ".join(f"`{p}`" for p in he.get("patterns", [])) or "(tool defaults)"
            lines.append(
                f"- hw_export: **enabled** → `{hw_export_output_gguf(cfg)}` "
                f"(binary `{he['binary']}`, patterns {pat})"
            )

        lines.append("\n## Inputs (fingerprinted)\n")
        for key, val in self.manifest["inputs"].items():
            if val is None:
                continue
            if val.get("exists") is False:
                lines.append(f"- **{key}**: `{val['path']}` *(missing)*")
            else:
                lines.append(
                    f"- **{key}**: `{val['path']}`  \n"
                    f"  sha256=`{val['sha256']}` size={val['size_bytes']:,} B"
                )

        lines.append("\n## Outputs (fingerprinted)\n")
        if not self.manifest["outputs"]:
            lines.append("_(no outputs generated)_")
        for key, val in self.manifest["outputs"].items():
            if val is None:
                continue
            lines.append(
                f"- **{key}**: `{val['path']}`  \n"
                f"  sha256=`{val['sha256']}` size={val['size_bytes']:,} B"
            )

        lines.append("\n## Stages\n")
        for r in self.results:
            if r.skipped:
                lines.append(
                    f"### {r.name} — *skipped* ({r.skip_reason})\n"
                )
                continue
            lines.append(
                f"### {r.name} — exit={r.exit_code} elapsed={r.elapsed_s:.2f}s\n"
            )
            lines.append("```bash")
            lines.append(format_cmd(r.cmd or []))
            lines.append("```\n")
            if r.log_file:
                lines.append(f"log: `{r.log_file.name}`\n")
            if r.extra:
                lines.append(f"extra: `{json.dumps(r.extra, ensure_ascii=False)}`\n")

        with report.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\n[aimet_llama] report written -> {report}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(
        prog="python -m aimet_llama.pipeline",
        description="JSON-driven LLM quantization pipeline (llama.cpp backend)",
    )
    p.add_argument("config", type=Path, help="Path to JSON config")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved CLI plan and exit (no binaries invoked)",
    )
    args = p.parse_args()

    pipe = LLMQuantPipeline.from_json(args.config)
    if args.dry_run:
        pipe.print_plan()
        return
    summary = pipe.run()
    print("\n[aimet_llama] summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
