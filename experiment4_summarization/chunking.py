"""Splitting a long report into pieces that fit the model's input window.

BART accepts 1024 tokens. A report of a few thousand words exceeds that, and
anything past the limit is silently truncated, so the tail of the document
would never reach the model at all. Splitting on sentence boundaries keeps
each chunk grammatical, and a small overlap carries context across the join so
a claim spanning two chunks is not lost.
"""

from __future__ import annotations

from document import split_sentences


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap_sentences: int = 2,
) -> list[str]:
    """Split text into chunks of at most max_tokens, on sentence boundaries."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    lengths = [_token_length(tokenizer, sentence) for sentence in sentences]

    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        total = 0
        end = start
        while end < len(sentences) and total + lengths[end] <= max_tokens:
            total += lengths[end]
            end += 1

        # A single sentence longer than the window still has to go somewhere.
        if end == start:
            end = start + 1

        chunks.append(" ".join(sentences[start:end]))

        if end >= len(sentences):
            break
        start = max(end - overlap_sentences, start + 1)

    return chunks


def _token_length(tokenizer, text: str) -> int:
    if tokenizer is None:
        # Rough fallback used only when no tokenizer is available. English
        # averages near 1.3 subword tokens per whitespace word.
        return int(len(text.split()) * 1.3) + 1
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def describe(chunks: list[str], tokenizer) -> list[dict[str, int]]:
    """Per chunk word and token counts, recorded in the manifest."""
    return [
        {
            "index": index,
            "words": len(chunk.split()),
            "tokens": _token_length(tokenizer, chunk),
        }
        for index, chunk in enumerate(chunks)
    ]
