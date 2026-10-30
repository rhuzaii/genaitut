"""Build the vector index from a statute (steps i to iii).

Usage:
    python build_index.py
    python build_index.py --document documents/ipc.pdf
    python build_index.py --strategy fixed --index-dir index_fixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chunking
import ingest
import vectorstore
from config import DEFAULT_DOCUMENT, EMBEDDING_CHOICES, FALLBACK_DOCUMENT, Config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a statute for retrieval.")
    parser.add_argument("--document", default=None, help="path to a .pdf, .txt or .md statute")
    parser.add_argument("--index-dir", default="index")
    parser.add_argument("--strategy", default="section", choices=["section", "fixed"])
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--overlap-tokens", type=int, default=60)
    parser.add_argument("--embedding", default="minilm", choices=sorted(EMBEDDING_CHOICES))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return parser.parse_args(argv)


def resolve_document(given: str | None) -> Path:
    if given:
        return Path(given)
    if DEFAULT_DOCUMENT.exists():
        return DEFAULT_DOCUMENT
    print(
        f"{DEFAULT_DOCUMENT} not found, falling back to {FALLBACK_DOCUMENT}.\n"
        "That file is a fictional test statute, not law. See documents/README.md.\n"
    )
    return FALLBACK_DOCUMENT


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config(
        document=resolve_document(args.document),
        index_dir=Path(args.index_dir),
        device=args.device,
    )
    cfg.chunking.strategy = args.strategy
    cfg.chunking.max_tokens = args.max_tokens
    cfg.chunking.overlap_tokens = args.overlap_tokens
    cfg.retrieval.embedding_key = args.embedding

    print(f"Loading   {cfg.document}")
    statute = ingest.load(cfg.document)
    stats = statute.stats()
    print(f"  title    {statute.title[:70]}")
    print(f"  {stats['words']} words, {stats['sections']} sections, {stats['chapters']} chapters")

    if stats["sections"] == 0:
        print("\nWarning: no numbered sections were detected. The parser expects")
        print("lines like '302. Punishment for murder.' Check the extracted text.")

    print(f"\nChunking  strategy={cfg.chunking.strategy} max_tokens={cfg.chunking.max_tokens}")
    chunks = chunking.build_chunks(
        statute, cfg.chunking.strategy, cfg.chunking.max_tokens, cfg.chunking.overlap_tokens
    )
    described = chunking.describe(chunks)
    print(f"  {described}")

    print(f"\nEmbedding {cfg.embedding_model} on {cfg.resolved_device()}")
    path = vectorstore.build_index(cfg, chunks, stats)
    print(f"\nIndex written to {path.resolve()}")
    print(f"Query it with:  python ask.py --index-dir {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
