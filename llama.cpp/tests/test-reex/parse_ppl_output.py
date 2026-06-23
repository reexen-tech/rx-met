#!/usr/bin/env python3
"""Parse llama-perplexity log and write JSON (lm_evaluator-compatible)."""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reuse lm_evaluator parsers when available.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lm_evaluator"))
    from model_evaluator.backends.perplexity import parse_ppl_output, parse_llama_perf_and_memory
except ImportError:
    def parse_ppl_output(stdout: str, stderr: str):
        text = stdout + "\n" + stderr
        m = re.search(r"Final\s+estimate:\s*PPL\s*=\s*([\d.]+)\s*\+\/\-\s*([\d.]+)", text, re.I)
        if m:
            return {"ppl": float(m.group(1)), "ppl_uncertainty": float(m.group(2)), "metric": "ppl"}
        return {"error": "parse failed"}

    def parse_llama_perf_and_memory(text: str):
        return {}


def main():
    if len(sys.argv) < 4:
        print("usage: parse_ppl_output.py <log> <model.gguf> <out.json>", file=sys.stderr)
        return 2
    log_path = Path(sys.argv[1])
    model_name = Path(sys.argv[2]).name
    out_path = Path(sys.argv[3])
    text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_ppl_output(text, "")
    parsed.update(parse_llama_perf_and_memory(text))
    doc = {
        "model": model_name,
        "dataset": "validation.txt",
        "dataset_type": "ppl",
        "result": parsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    if "error" in parsed:
        print(f"ERROR: {parsed['error']}", file=sys.stderr)
        return 1
    print(f"ppl={parsed.get('ppl')} +/- {parsed.get('ppl_uncertainty')} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
