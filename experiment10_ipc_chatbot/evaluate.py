"""Measuring the retrieval pipeline rather than eyeballing it.

Three things are measured.

recall@k      the fraction of questions whose governing section appears in the
              top k retrieved passages. This is the ceiling on the whole
              system: a section that is never retrieved can never be cited,
              no matter how capable the language model is.
MRR           mean reciprocal rank, which rewards putting the right section
              first rather than fifth
refusal rate  on deliberately out of domain questions, the fraction correctly
              declined. A legal lookup tool that answers questions about the
              capital of France is not safe to deploy.

Measuring retrieval separately from generation is the point. Most end to end
failures in a RAG system originate in retrieval, and an end to end score
cannot tell you which stage was at fault.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chunking
import ingest
import vectorstore
from config import Config


def load_questions(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "in_domain" not in data:
        raise ValueError("Question file needs an 'in_domain' list.")
    return data


def evaluate_retrieval(
    cfg: Config,
    store,
    questions: dict[str, Any],
    max_k: int = 5,
    section_text: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Recall at every k up to max_k, plus MRR, over the annotated questions.

    A passage counts as a hit when its section label matches, or when it
    actually contains the governing section's heading. The second test matters
    for the fixed size strategy, whose chunks have no reliable section label:
    scoring it on labels alone would measure the labelling rather than the
    retrieval, and understate it unfairly.
    """
    items = questions["in_domain"]
    section_text = section_text or {}
    hits_at_rank: list[int | None] = []

    details = []
    for item in items:
        results = vectorstore.search(store, item["question"], max_k)
        heading = _normalise(section_text.get(str(item["section"]), ""))
        rank = None
        for position, hit in enumerate(results, start=1):
            label_match = str(hit["metadata"].get("section")) == str(item["section"])
            text_match = bool(heading) and heading in _normalise(hit["text"])
            if label_match or text_match:
                rank = position
                break
        hits_at_rank.append(rank)
        details.append(
            {
                "question": item["question"],
                "expected_section": item["section"],
                "rank": rank,
                "top_section": results[0]["metadata"].get("section") if results else None,
                "top_similarity": results[0]["similarity"] if results else 0.0,
            }
        )

    total = len(items)
    recall = {
        f"recall@{k}": round(sum(1 for r in hits_at_rank if r is not None and r <= k) / total, 4)
        for k in range(1, max_k + 1)
    }
    mrr = round(sum(1 / r for r in hits_at_rank if r) / total, 4)

    return {"questions": total, **recall, "mrr": mrr, "details": details}


def evaluate_refusal(cfg: Config, bot, questions: dict[str, Any]) -> dict[str, Any]:
    """How reliably out of domain questions are declined."""
    out_of_domain = questions.get("out_of_domain", [])
    if not out_of_domain:
        return {"questions": 0}

    refused, rows = 0, []
    for question in out_of_domain:
        answer = bot.ask(question)
        refused += int(answer.refused)
        rows.append(
            {
                "question": question,
                "refused": answer.refused,
                "best_similarity": answer.best_similarity,
            }
        )
    return {
        "questions": len(out_of_domain),
        "refused": refused,
        "refusal_rate": round(refused / len(out_of_domain), 4),
        "details": rows,
    }


def compare_chunking(cfg: Config, questions: dict[str, Any], strategies=("section", "fixed")) -> dict[str, Any]:
    """Rebuild the index under each strategy and re-measure retrieval.

    Chunking is a preprocessing choice that costs nothing at query time, so if
    it moves recall it is the cheapest available improvement.
    """
    statute = ingest.load(cfg.document)
    section_text = {s.number: s.heading for s in statute.sections if s.heading}
    comparison: dict[str, Any] = {}

    for strategy in strategies:
        chunks = chunking.build_chunks(
            statute, strategy, cfg.chunking.max_tokens, cfg.chunking.overlap_tokens
        )
        scratch = Config(
            document=cfg.document,
            index_dir=cfg.index_dir.parent / f"index_{strategy}",
            device=cfg.device,
        )
        scratch.chunking.strategy = strategy
        scratch.retrieval = cfg.retrieval
        vectorstore.build_index(scratch, chunks, statute.stats())
        store = vectorstore.load_index(scratch)

        result = evaluate_retrieval(scratch, store, questions, section_text=section_text)
        result.pop("details", None)
        comparison[strategy] = {**chunking.describe(chunks), **result}

    return comparison


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def print_retrieval(result: dict[str, Any]) -> None:
    ks = [key for key in result if key.startswith("recall@")]
    print(f"  questions {result['questions']}")
    for key in ks:
        print(f"  {key:<10} {result[key]:.4f}")
    print(f"  {'mrr':<10} {result['mrr']:.4f}")
    misses = [d for d in result.get("details", []) if d["rank"] is None]
    if misses:
        print(f"  missed {len(misses)}:")
        for miss in misses:
            print(f"    expected Section {miss['expected_section']:<4} "
                  f"got Section {miss['top_section']} ({miss['top_similarity']:.3f})"
                  f"  <- {miss['question']}")
