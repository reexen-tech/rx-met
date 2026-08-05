#!/usr/bin/env python3
"""Build a local calibration .jsonl for SmoothQuant from an offline corpus.

The reference recipe calibrates on ``mit-han-lab/pile-val-backup``. On an
air-gapped machine that download is not available, so this script assembles an
equivalent ``{"text": ...}`` JSONL from a local parquet or plain-text corpus,
keeping only documents long enough to be meaningful at the target calibration
sequence length.

  python llm_quant/smoothquant/scripts/make_smoothquant_calib_jsonl.py \\
    --source /path/to/wikitext-103-raw-v1/train-00000-of-00002.parquet \\
    --output /path/to/calib.jsonl --n_docs 512
"""
# === REEX_SMOOTHQUANT BEGIN: offline calibration corpus builder ===
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_documents(source: Path) -> list[str]:
    """Read a parquet ``text`` column, or split a raw text file on blank lines."""
    if source.suffix == ".parquet":
        import pyarrow.parquet as pq

        column = pq.read_table(source, columns=["text"]).column("text").to_pylist()
        return _join_wikitext_lines(column)
    text = source.read_text(encoding="utf-8", errors="replace")
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def _join_wikitext_lines(lines: list[str]) -> list[str]:
    """WikiText parquet rows are lines; ' = Title = ' starts a new article."""
    documents: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("= ") and stripped.endswith(" =") and not stripped.startswith("= ="):
            if current:
                documents.append("".join(current).strip())
            current = []
        current.append(line)
    if current:
        documents.append("".join(current).strip())
    return [d for d in documents if d]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", nargs="+", required=True, help="Parquet or text corpus files")
    p.add_argument("--output", required=True, help="Destination .jsonl")
    p.add_argument("--n_docs", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=2000, help="Drop shorter documents")
    p.add_argument("--max_chars", type=int, default=20000, help="Truncate longer documents")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    documents: list[str] = []
    for source in args.source:
        documents.extend(read_documents(Path(source)))
    documents = [d for d in documents if len(d) >= args.min_chars]
    if len(documents) < args.n_docs:
        raise SystemExit(
            f"only {len(documents)} documents of >= {args.min_chars} chars, "
            f"need {args.n_docs}; lower --min_chars or add sources"
        )
    random.Random(args.seed).shuffle(documents)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for doc in documents[: args.n_docs]:
            fh.write(json.dumps({"text": doc[: args.max_chars]}, ensure_ascii=False) + "\n")
    print(f"wrote {args.n_docs} documents -> {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# === REEX_SMOOTHQUANT END ===
