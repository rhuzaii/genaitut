"""Retrieval augmented question answering over the statute (step iv).

The design priority here is not fluency, it is grounding. A legal lookup tool
that invents a provision is worse than one that says it does not know, so the
pipeline is built to fail closed:

  retrieve   fetch the top k passages by cosine similarity
  gate       if the best passage scores below a threshold, decline outright
             and never call the language model at all
  generate   answer strictly from the retrieved passages, citing section
             numbers, and say so when the passages do not settle the question
  attach     return the passages alongside the answer so the reader can check

Conversational memory is handled by condensing a follow up into a standalone
question before retrieval. "What is the punishment for it?" retrieves nothing
useful on its own; rewritten against the history it becomes answerable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import vectorstore
from config import Config

DISCLAIMER = (
    "This is an automated lookup over the indexed document. It is not legal "
    "advice. Verify every answer against the official text."
)

ANSWER_PROMPT = """You are a careful legal research assistant answering questions about a statute.

Answer the question using ONLY the numbered passages below. Follow these rules exactly:

1. Cite the section number for every statement you make, in the form (Section 302).
2. If the passages do not contain enough to answer, say so plainly and name what is missing. Do not supply the answer from general knowledge.
3. Quote the operative wording of the provision where it matters.
4. Do not speculate about how a court would apply the provision.
5. Be concise. Three to six sentences unless the question needs more.

PASSAGES
{context}

QUESTION
{question}

ANSWER"""

CONDENSE_PROMPT = """Rewrite the follow up question as a standalone question that can be understood without the conversation.

Keep it short. Preserve any section numbers mentioned. If the question is already standalone, return it unchanged. Return only the rewritten question.

CONVERSATION
{history}

FOLLOW UP
{question}

STANDALONE QUESTION"""

FOLLOW_UP_HINT = re.compile(
    r"\b(it|that|this|they|them|those|there|same|above|previous)\b", re.IGNORECASE
)


@dataclass
class Answer:
    question: str
    standalone_question: str
    text: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    refused: bool = False
    backend: str = ""
    best_similarity: float = 0.0

    def formatted_sources(self) -> str:
        if not self.sources:
            return ""
        lines = []
        for hit in self.sources:
            heading = hit["metadata"].get("heading", "")
            label = f"{hit['citation']}" + (f", {heading}" if heading else "")
            lines.append(f"- {label}  (similarity {hit['similarity']:.3f})")
        return "\n".join(lines)


class Chatbot:
    """Retrieval augmented chatbot over one indexed statute."""

    def __init__(self, cfg: Config, store=None):
        self.cfg = cfg
        self.store = store if store is not None else vectorstore.load_index(cfg)
        self.backend = _resolve_backend(cfg)
        self._generator = None

    # ---------------------------------------------------------------- public

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> Answer:
        history = history or []
        standalone = self.condense(question, history)
        hits = vectorstore.search(self.store, standalone, self.cfg.retrieval.top_k)
        best = hits[0]["similarity"] if hits else 0.0

        if not hits or best < self.cfg.retrieval.min_score:
            return Answer(
                question=question,
                standalone_question=standalone,
                text=(
                    "I could not find anything in the indexed document that "
                    "addresses that. The closest passage scored "
                    f"{best:.3f}, below the {self.cfg.retrieval.min_score} "
                    "relevance threshold, so I am declining rather than "
                    "answering from outside the document."
                ),
                sources=hits,
                refused=True,
                backend=self.backend,
                best_similarity=best,
            )

        context = _format_context(hits)
        text = self._generate(ANSWER_PROMPT.format(context=context, question=standalone))
        return Answer(
            question=question,
            standalone_question=standalone,
            text=text.strip(),
            sources=hits,
            refused=False,
            backend=self.backend,
            best_similarity=best,
        )

    def condense(self, question: str, history: list[tuple[str, str]]) -> str:
        """Rewrite a follow up into a standalone question for retrieval."""
        if not history or not FOLLOW_UP_HINT.search(question):
            return question.strip()

        if self.backend == "extractive":
            # No language model available, so fall back to concatenating the
            # previous question, which is crude but better than nothing.
            previous = history[-1][0]
            return f"{previous} {question}".strip()

        turns = history[-self.cfg.generator.history_turns :]
        rendered = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in turns)
        rewritten = self._generate(
            CONDENSE_PROMPT.format(history=rendered, question=question)
        )
        rewritten = rewritten.strip().strip('"')
        return rewritten or question.strip()

    # --------------------------------------------------------------- private

    def _generate(self, prompt: str) -> str:
        if self.backend == "extractive":
            return _extractive_answer(prompt)
        if self.backend == "cohere":
            return self._generate_cohere(prompt)
        return self._generate_local(prompt)

    def _generate_cohere(self, prompt: str) -> str:
        import cohere

        client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        response = client.chat(
            model=self.cfg.generator.cohere_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.generator.temperature,
            max_tokens=self.cfg.generator.max_tokens,
        )
        return "".join(block.text for block in response.message.content)

    def _generate_local(self, prompt: str) -> str:
        if self._generator is None:
            import torch
            from transformers import pipeline

            device = self.cfg.resolved_device()
            self._generator = pipeline(
                "text-generation",
                model=self.cfg.generator.local_model,
                dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
                device=0 if device == "cuda" else (device if device == "mps" else -1),
            )
        out = self._generator(
            [{"role": "user", "content": prompt}],
            max_new_tokens=self.cfg.generator.max_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return out[0]["generated_text"]


def _resolve_backend(cfg: Config) -> str:
    backend = cfg.generator.backend
    if backend != "auto":
        return backend
    if os.environ.get("COHERE_API_KEY"):
        return "cohere"
    return "local"


def _format_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(f"[{index}] {hit['citation']}\n{hit['text']}")
    return "\n\n".join(blocks)


def _extractive_answer(prompt: str) -> str:
    """Return the retrieved passages verbatim, with no model in the loop.

    This is the zero dependency fallback and also the baseline the generative
    backends are measured against: if a generated answer is no more useful
    than the passages themselves, the language model added nothing.
    """
    match = re.search(r"PASSAGES\n(.*?)\n\nQUESTION", prompt, re.DOTALL)
    passages = match.group(1).strip() if match else ""
    return (
        "Retrieved passages, returned without generation:\n\n"
        f"{passages}"
    )
