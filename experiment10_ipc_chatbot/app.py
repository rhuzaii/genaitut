"""Gradio chat interface (step v of the experiment).

Gradio is used rather than Streamlit because it runs inside a Colab cell and
can expose a temporary public link without tunnelling.

Every answer is shown with the passages it was built from and their similarity
scores. That is deliberate. A legal lookup tool whose sources are hidden asks
the reader to trust it, and this one should not be trusted, only checked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chatbot
from config import Config

INTRO = """## Statute question answering

Ask about a provision in plain language. Every answer cites the sections it
came from, and the retrieved passages are shown below it so you can verify
them against the official text.

The assistant declines when nothing in the indexed document is relevant,
rather than answering from general knowledge."""

EXAMPLES = [
    "What is the punishment for ragging?",
    "How long do I have to appeal a decision?",
    "Someone sat an exam in my place. What happens to them?",
    "Is there a time limit on bringing a case?",
]


def make_responder(bot: chatbot.Chatbot):
    """Build the chat callback.

    Kept at module level rather than nested inside the interface so it can be
    tested without constructing or launching a Gradio app.
    """

    def respond(message: str, history: list[dict]) -> str:
        answer = bot.ask(message, to_turns(history))

        text = answer.text
        if answer.standalone_question != answer.question:
            text = f"*Interpreted as: {answer.standalone_question}*\n\n{text}"
        if answer.sources and not answer.refused:
            text += "\n\n---\n**Retrieved passages**\n\n" + answer.formatted_sources()
        return text

    return respond


def to_turns(history: list[dict]) -> list[tuple[str, str]]:
    """Convert Gradio's messages history into (question, answer) pairs."""
    turns: list[tuple[str, str]] = []
    pending: str | None = None
    for message in history or []:
        role, content = message.get("role"), message.get("content", "")
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            turns.append((pending, content))
            pending = None
    return turns


def build_interface(cfg: Config):
    import gradio as gr

    bot = chatbot.Chatbot(cfg)
    respond = make_responder(bot)

    with gr.Blocks(title="Statute question answering", theme=gr.themes.Base()) as demo:
        gr.Markdown(INTRO)
        gr.ChatInterface(
            fn=respond,
            type="messages",
            examples=EXAMPLES,
            chatbot=gr.Chatbot(type="messages", height=440, show_label=False),
        )
        gr.Markdown(
            f"Backend: `{bot.backend}`  |  Index: `{cfg.index_dir}`  |  "
            f"top k: {cfg.retrieval.top_k}  |  relevance threshold: {cfg.retrieval.min_score}"
        )
        gr.Markdown(f"*{chatbot.DISCLAIMER}*")

    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the statute chatbot UI.")
    parser.add_argument("--index-dir", default="index")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "cohere", "local", "extractive"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--share", action="store_true",
                        help="expose a temporary public link, needed in Colab")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config(index_dir=Path(args.index_dir), device=args.device)
    cfg.retrieval.top_k = args.top_k
    cfg.retrieval.min_score = args.min_score
    cfg.generator.backend = args.backend

    demo = build_interface(cfg)
    demo.launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
