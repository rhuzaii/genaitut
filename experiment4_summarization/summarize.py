"""Map reduce summarization with a pre-trained Hugging Face model.

The document does not fit in the model's input window, so summarization runs
in two passes:

  map     summarize each chunk independently
  reduce  concatenate the chunk summaries and summarize that

Without the reduce pass the output is simply the chunk summaries joined, which
grows with document length and is not a summary in any useful sense. The reduce
pass is what makes the output short regardless of how long the input was.
"""

from __future__ import annotations

import time

from chunking import chunk_text
from config import Config

MAX_REDUCE_PASSES = 3


def load_pipeline(cfg: Config, model_key: str):
    """Return a summarization pipeline and its tokenizer."""
    import torch
    from transformers import AutoTokenizer, pipeline

    model_id = cfg.model_id(model_key)
    device = cfg.resolved_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    summarizer = pipeline(
        "summarization",
        model=model_id,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    return summarizer, tokenizer


def summarize_document(cfg: Config, text: str, model_key: str) -> dict:
    """Run the full map reduce over one document with one model."""
    import torch

    summarizer, tokenizer = load_pipeline(cfg, model_key)
    torch.manual_seed(cfg.seed)
    started = time.time()

    chunks = chunk_text(
        text,
        tokenizer,
        max_tokens=cfg.chunking.max_tokens,
        overlap_sentences=cfg.chunking.overlap_sentences,
    )
    print(f"  {model_key}: {len(chunks)} chunk(s)")

    chunk_summaries = [
        _generate(
            summarizer,
            chunk,
            max_length=cfg.summary.map_max_tokens,
            min_length=cfg.summary.map_min_tokens,
            cfg=cfg,
        )
        for chunk in chunks
    ]

    final, reduce_passes = _reduce(cfg, summarizer, tokenizer, chunk_summaries)
    elapsed = time.time() - started

    _release(summarizer)
    return {
        "model_key": model_key,
        "model_id": cfg.model_id(model_key),
        "chunks": chunks,
        "chunk_summaries": chunk_summaries,
        "summary": final,
        "reduce_passes": reduce_passes,
        "elapsed_seconds": round(elapsed, 1),
    }


def _reduce(cfg: Config, summarizer, tokenizer, summaries: list[str]) -> tuple[str, int]:
    """Collapse chunk summaries into one, re-chunking while they do not fit."""
    if not summaries:
        return "", 0
    if len(summaries) == 1:
        return summaries[0], 0

    joined = " ".join(summaries)
    for pass_number in range(1, MAX_REDUCE_PASSES + 1):
        pieces = chunk_text(
            joined,
            tokenizer,
            max_tokens=cfg.chunking.max_tokens,
            overlap_sentences=0,
        )
        outputs = [
            _generate(
                summarizer,
                piece,
                max_length=cfg.summary.reduce_max_tokens,
                min_length=cfg.summary.reduce_min_tokens,
                cfg=cfg,
            )
            for piece in pieces
        ]
        if len(outputs) == 1:
            return outputs[0], pass_number
        joined = " ".join(outputs)

    return joined, MAX_REDUCE_PASSES


def _generate(summarizer, text: str, max_length: int, min_length: int, cfg: Config) -> str:
    """One summarization call, with the length bounds clamped to the input.

    Asking for a summary longer than the passage produces a warning and a
    padded, repetitive output, so the bounds are reduced when the input is
    short.
    """
    input_tokens = len(summarizer.tokenizer(text, add_special_tokens=False)["input_ids"])
    upper = max(min(max_length, max(input_tokens // 2, 30)), 30)
    lower = min(min_length, max(upper - 20, 10))

    result = summarizer(
        text,
        max_length=upper,
        min_length=lower,
        do_sample=False,
        num_beams=cfg.summary.num_beams,
        length_penalty=cfg.summary.length_penalty,
        no_repeat_ngram_size=cfg.summary.no_repeat_ngram_size,
        truncation=True,
    )
    return result[0]["summary_text"].strip()


def lead_baseline(text: str, sentences: int) -> str:
    """The first n sentences of the document.

    A strong and often underrated extractive baseline. Academic reports open
    with their thesis, so the opening sentences carry real information. Any
    abstractive model that cannot beat this is not earning its compute.
    """
    from document import split_sentences

    return " ".join(split_sentences(text)[:sentences])


def _release(summarizer) -> None:
    import gc

    import torch

    del summarizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
