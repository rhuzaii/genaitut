"""Metrics for comparing a generated summary against the report's abstract.

Three families of measure are reported.

ROUGE compares the summary to a human written reference, here the paper's own
abstract, by counting overlapping n-grams. It rewards saying the same things
the reference said. It cannot tell whether a fluent summary is factually right,
only whether it used similar wording, which is the standard caveat on this
metric.

Compression says how much shorter the summary is than the source, which is the
whole point of the exercise.

Novel n-gram ratio says how much of the summary is newly worded rather than
copied. It separates genuinely abstractive models from ones that are, in
practice, extracting sentences.

ROUGE is implemented directly rather than pulled from a package, so the
definition being used is visible and the module has no extra dependency. No
stemming is applied, which makes these values slightly conservative compared
with implementations that stem by default.
"""

from __future__ import annotations

import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(prediction: str, reference: str, n: int) -> dict[str, float]:
    """Overlapping n-gram counts, clipped by how often each appears."""
    pred = ngrams(tokenize(prediction), n)
    ref = ngrams(tokenize(reference), n)
    overlap = sum((pred & ref).values())
    return _prf(overlap, sum(pred.values()), sum(ref.values()))


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """Longest common subsequence, which rewards in order overlap."""
    pred = tokenize(prediction)
    ref = tokenize(reference)
    return _prf(_lcs_length(pred, ref), len(pred), len(ref))


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[index], previous[index + 1]))
        previous = current
    return previous[-1]


def _prf(overlap: int, pred_total: int, ref_total: int) -> dict[str, float]:
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def novel_ngram_ratio(summary: str, source: str, n: int = 2) -> float:
    """Fraction of summary n-grams that do not appear in the source.

    Near zero means the model copied. Higher means it rephrased, which is what
    abstractive summarization is supposed to do.
    """
    summary_grams = ngrams(tokenize(summary), n)
    source_grams = set(ngrams(tokenize(source), n))
    total = sum(summary_grams.values())
    if not total:
        return 0.0
    novel = sum(count for gram, count in summary_grams.items() if gram not in source_grams)
    return round(novel / total, 4)


def score(summary: str, source: str, reference: str | None) -> dict:
    """Every metric for one summary."""
    summary_words = len(summary.split())
    source_words = len(source.split())

    result = {
        "summary_words": summary_words,
        "source_words": source_words,
        "compression_ratio": round(source_words / summary_words, 2) if summary_words else 0.0,
        "percent_of_source": round(100 * summary_words / source_words, 1) if source_words else 0.0,
        "novel_bigram_ratio": novel_ngram_ratio(summary, source, 2),
    }

    if reference:
        result["reference_words"] = len(reference.split())
        result["rouge1"] = rouge_n(summary, reference, 1)
        result["rouge2"] = rouge_n(summary, reference, 2)
        result["rougeL"] = rouge_l(summary, reference)

    return result
