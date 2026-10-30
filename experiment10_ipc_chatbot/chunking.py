"""Splitting the statute for retrieval (step ii of the experiment).

Two strategies are provided so the report can compare them.

section  one chunk per numbered provision, split further only when a section
         is too long for the embedding model's input window
fixed    a conventional recursive character splitter, included as the baseline
         that a generic tutorial would use

The section strategy matters because a provision is a self contained unit of
meaning. A fixed splitter will happily cut between an offence and its
punishment, or between a rule and its proviso, and retrieval then returns a
chunk that is confidently wrong because it is missing the qualifying half.
Every section chunk also carries its section number, which is what lets an
answer cite a source the reader can verify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ingest import Statute

SENTENCE_BOUNDARY = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


@dataclass
class Chunk:
    """One retrievable passage, with enough metadata to cite it."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        number = self.metadata.get("section")
        if not number:
            return "Unattributed passage"
        part = self.metadata.get("part", 1)
        total = self.metadata.get("parts", 1)
        suffix = f" (part {part} of {total})" if total > 1 else ""
        return f"Section {number}{suffix}"


def build_chunks(statute: Statute, strategy: str, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    if strategy == "section":
        return _section_chunks(statute, max_tokens)
    if strategy == "fixed":
        return _fixed_chunks(statute, max_tokens, overlap_tokens)
    raise ValueError(f"Unknown chunking strategy {strategy!r}.")


def _section_chunks(statute: Statute, max_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    max_words = _tokens_to_words(max_tokens)

    for section in statute.sections:
        body = section.full_text
        pieces = _split_long(body, max_words)
        for index, piece in enumerate(pieces, start=1):
            chunks.append(
                Chunk(
                    text=piece,
                    metadata={
                        "section": section.number,
                        "heading": section.heading,
                        "chapter": section.chapter,
                        "part": index,
                        "parts": len(pieces),
                        "strategy": "section",
                    },
                )
            )
    return chunks


def _fixed_chunks(statute: Statute, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    """Baseline splitter that ignores the statute's structure entirely."""
    words = statute.raw_text.split()
    size = _tokens_to_words(max_tokens)
    step = max(size - _tokens_to_words(overlap_tokens), 1)

    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(words), step), start=1):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(
            Chunk(
                text=" ".join(window),
                metadata={
                    "section": _guess_section(" ".join(window)),
                    "heading": "",
                    "chapter": "",
                    "part": index,
                    "parts": 1,
                    "strategy": "fixed",
                },
            )
        )
        if start + size >= len(words):
            break
    return chunks


def _split_long(text: str, max_words: int) -> list[str]:
    """Split an over long section on sentence boundaries, keeping the citation."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    header = text.split("\n", 1)[0]
    sentences = SENTENCE_BOUNDARY.split(text)

    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        length = len(sentence.split())
        if current and count + length > max_words:
            pieces.append(" ".join(current))
            # Repeat the header so every piece remains self identifying.
            current, count = [header], len(header.split())
        current.append(sentence)
        count += length
    if current:
        pieces.append(" ".join(current))
    return pieces


def _guess_section(text: str) -> str:
    """Best effort section number for a fixed chunk, which has no structure."""
    match = re.search(r"\b(\d{1,3}[A-Z]{0,2})\.\s+[A-Z]", text)
    return match.group(1) if match else ""


def _tokens_to_words(tokens: int) -> int:
    """English averages near 1.3 subword tokens per whitespace word."""
    return max(int(tokens / 1.3), 20)


def describe(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = [len(chunk.text.split()) for chunk in chunks]
    if not lengths:
        return {"count": 0}
    return {
        "count": len(chunks),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "mean_words": round(sum(lengths) / len(lengths), 1),
        "sections_covered": len({c.metadata.get("section") for c in chunks if c.metadata.get("section")}),
    }
