"""Query the indexed statute from the command line (step iv).

Usage:
    python ask.py --question "What is the punishment for ragging?"
    python ask.py                      # interactive, keeps conversation history
    python ask.py --evaluate           # run the retrieval and refusal metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chatbot
import evaluate
import ingest
import vectorstore
from config import EMBEDDING_CHOICES, Config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions about the indexed statute.")
    parser.add_argument("--question", default=None, help="ask once and exit")
    parser.add_argument("--index-dir", default="index")
    parser.add_argument("--document", default=None, help="only needed for --evaluate")
    parser.add_argument("--backend", default="auto", choices=["auto", "cohere", "local", "extractive"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--embedding", default="minilm", choices=sorted(EMBEDDING_CHOICES))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--evaluate", action="store_true", help="run metrics instead of answering")
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--compare-chunking", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    manifest_path = Path(args.index_dir) / vectorstore.MANIFEST_NAME
    document = args.document
    if document is None and manifest_path.exists():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))["config"]["document"]

    cfg = Config(
        document=document or "documents/sample_statute.txt",
        index_dir=Path(args.index_dir),
        device=args.device,
    )
    cfg.retrieval.top_k = args.top_k
    cfg.retrieval.min_score = args.min_score
    cfg.retrieval.embedding_key = args.embedding
    cfg.generator.backend = args.backend
    return cfg


def show(answer) -> None:
    print()
    if answer.standalone_question != answer.question:
        print(f"  [rewritten as: {answer.standalone_question}]")
    print(answer.text)
    if answer.sources and not answer.refused:
        print("\nSources")
        print(answer.formatted_sources())
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    if args.evaluate or args.compare_chunking:
        return _run_evaluation(cfg, args)

    bot = chatbot.Chatbot(cfg)
    print(f"Index   {cfg.index_dir}")
    print(f"Backend {bot.backend}")
    print(chatbot.DISCLAIMER)

    if args.question:
        show(bot.ask(args.question))
        return 0

    print("\nAsk a question, or type 'exit'.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit", ""}:
            return 0
        answer = bot.ask(question, history)
        show(answer)
        history.append((question, answer.text))


def _run_evaluation(cfg: Config, args: argparse.Namespace) -> int:
    questions = evaluate.load_questions(Path(args.questions))

    if args.compare_chunking:
        print("Chunking strategy comparison")
        for name, result in evaluate.compare_chunking(cfg, questions).items():
            print(
                f"  {name:<8} chunks={result['count']:<5} mean_words={result['mean_words']:<7} "
                f"recall@1={result['recall@1']:.3f}  recall@3={result['recall@3']:.3f}  "
                f"recall@5={result['recall@5']:.3f}  mrr={result['mrr']:.3f}"
            )
        return 0

    statute = ingest.load(cfg.document)
    section_text = {s.number: s.heading for s in statute.sections if s.heading}
    store = vectorstore.load_index(cfg)

    print("Retrieval")
    evaluate.print_retrieval(
        evaluate.evaluate_retrieval(cfg, store, questions, section_text=section_text)
    )

    bot = chatbot.Chatbot(cfg, store=store)
    refusal = evaluate.evaluate_refusal(cfg, bot, questions)
    if refusal.get("questions"):
        print(f"\nRefusal on out of domain questions")
        print(f"  {refusal['refused']}/{refusal['questions']} declined, "
              f"rate {refusal['refusal_rate']:.2f}")
        for row in refusal["details"]:
            state = "declined" if row["refused"] else "ANSWERED"
            print(f"    {state:<9} {row['best_similarity']:+.3f}  {row['question']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
