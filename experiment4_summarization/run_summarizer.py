"""Summarize an academic report and score the result against its abstract.

Usage:
    python run_summarizer.py
    python run_summarizer.py --document documents/my_paper.pdf
    python run_summarizer.py --models bart distilbart pegasus
    python run_summarizer.py --chunk-tokens 600 --no-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import chunking
import document
import evaluate
import report
import summarize
from config import DEFAULT_DOCUMENT, MODEL_CHOICES, Config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a long academic report with a pre-trained model."
    )
    parser.add_argument("--document", default=str(DEFAULT_DOCUMENT), help="path to a .txt, .md or .pdf report")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bart", "distilbart"],
        choices=sorted(MODEL_CHOICES),
        help="which summarization models to compare",
    )
    parser.add_argument("--output", default="outputs", help="output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--chunk-tokens", type=int, default=900, help="maximum tokens per chunk")
    parser.add_argument("--overlap", type=int, default=2, help="sentences of overlap between chunks")
    parser.add_argument("--baseline-sentences", type=int, default=3)
    parser.add_argument("--no-baseline", action="store_true", help="skip the lead sentence baseline")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(
        document=Path(args.document),
        models=tuple(args.models),
        output_dir=Path(args.output),
        device=args.device,
        seed=args.seed,
        run_baseline=not args.no_baseline,
        baseline_sentences=args.baseline_sentences,
    )
    cfg.chunking.max_tokens = args.chunk_tokens
    cfg.chunking.overlap_sentences = args.overlap
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = cfg.output_dir / "summaries"
    summary_dir.mkdir(exist_ok=True)

    started = time.time()
    device = cfg.resolved_device()

    report_doc = document.load(cfg.document)
    stats = report_doc.stats()

    print("Experiment 4: summarizing an academic report")
    print(f"  document : {cfg.document}")
    print(f"  title    : {report_doc.title[:70]}")
    print(f"  body     : {stats['body_words']} words, {stats['body_sentences']} sentences")
    print(f"  abstract : {stats['abstract_words']} words", end="")
    print("" if report_doc.has_reference else "  (no reference, ROUGE will be skipped)")
    print(f"  device   : {device}")
    print()

    if not report_doc.has_reference:
        print("Warning: no abstract was found, so ROUGE cannot be computed.")
        print()

    reference = report_doc.abstract
    summaries: dict[str, dict[str, Any]] = {}
    chunk_stats: list[dict[str, int]] = []

    for model_key in cfg.models:
        print(f"Summarizing with {model_key} ({cfg.model_id(model_key)})")
        result = summarize.summarize_document(cfg, report_doc.body, model_key)
        metrics = evaluate.score(result["summary"], report_doc.body, reference)

        summaries[model_key] = {
            "label": model_key,
            "model_id": result["model_id"],
            "summary": result["summary"],
            "chunk_summaries": result["chunk_summaries"],
            "reduce_passes": result["reduce_passes"],
            "elapsed_seconds": result["elapsed_seconds"],
            "metrics": metrics,
        }
        (summary_dir / f"summary_{model_key}.txt").write_text(
            result["summary"] + "\n", encoding="utf-8"
        )

        if not chunk_stats:
            chunk_stats = _measure_chunks(result["chunks"], cfg, model_key)

        print(f"  {metrics['summary_words']} words, {metrics['compression_ratio']}x compression, "
              f"{result['elapsed_seconds']} s")
        print()

    if cfg.run_baseline:
        print(f"Extractive baseline: first {cfg.baseline_sentences} sentences")
        lead = summarize.lead_baseline(report_doc.body, cfg.baseline_sentences)
        summaries["lead"] = {
            "label": f"lead-{cfg.baseline_sentences} baseline",
            "model_id": None,
            "summary": lead,
            "chunk_summaries": [],
            "reduce_passes": 0,
            "elapsed_seconds": 0.0,
            "metrics": evaluate.score(lead, report_doc.body, reference),
        }
        (summary_dir / "summary_lead.txt").write_text(lead + "\n", encoding="utf-8")
        print()

    _print_table(summaries, bool(reference))

    results: dict[str, Any] = {
        "document": {
            "path": str(cfg.document),
            "title": report_doc.title,
            "stats": stats,
        },
        "device": device,
        "config": cfg.to_dict(),
        "reference": reference,
        "chunk_stats": chunk_stats,
        "summaries": summaries,
        "elapsed_seconds": round(time.time() - started, 1),
    }

    manifest_path = cfg.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = report.render(results, cfg.output_dir / "report.md")

    print(f"Manifest : {manifest_path}")
    print(f"Report   : {report_path}")
    print(f"Elapsed  : {results['elapsed_seconds']} s")
    return 0


def _measure_chunks(chunks: list[str], cfg: Config, model_key: str) -> list[dict[str, int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id(model_key))
    return chunking.describe(chunks, tokenizer)


def _print_table(summaries: dict[str, dict[str, Any]], has_reference: bool) -> None:
    header = f"{'summary':<22}{'words':>7}{'compress':>10}{'novel':>8}"
    if has_reference:
        header += f"{'R-1':>9}{'R-2':>9}{'R-L':>9}"
    print(header)
    print("-" * len(header))
    for item in summaries.values():
        metrics = item["metrics"]
        row = (
            f"{item['label']:<22}"
            f"{metrics['summary_words']:>7}"
            f"{metrics['compression_ratio']:>9.1f}x"
            f"{metrics['novel_bigram_ratio']:>8.2f}"
        )
        if has_reference:
            row += (
                f"{metrics['rouge1']['f1']:>9.4f}"
                f"{metrics['rouge2']['f1']:>9.4f}"
                f"{metrics['rougeL']['f1']:>9.4f}"
            )
        print(row)
    print()


if __name__ == "__main__":
    sys.exit(main())
