"""Loading academic reports and separating the abstract from the body.

An academic report carries its own human written summary in the abstract. That
makes it an unusually good evaluation target: summarize the body only, then
score the machine summary against the abstract as a reference. The abstract
must therefore be held out of the model input, or the experiment would be
scoring the model on text it was given.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ABSTRACT_HEADING = re.compile(r"^\s*abstract\s*:?\s*$", re.IGNORECASE)
BODY_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s+)?(?:introduction|background|1\.|keywords|index terms)\b",
    re.IGNORECASE,
)
TITLE_PREFIX = re.compile(r"^\s*title\s*:\s*", re.IGNORECASE)

# Sentence splitter. Breaks after . ? or ! followed by whitespace and a capital,
# while protecting common abbreviations and decimal numbers.
ABBREVIATIONS = r"(?<!\be\.g)(?<!\bi\.e)(?<!\bet al)(?<!\bFig)(?<!\bNo)(?<!\bvs)(?<!\bDr)(?<!\bProf)"
SENTENCE_BOUNDARY = re.compile(rf"{ABBREVIATIONS}(?<=[.?!])\s+(?=[A-Z\d])")


@dataclass
class Report:
    """One loaded report, split into the parts the experiment needs."""

    path: Path
    title: str
    abstract: str
    body: str

    @property
    def has_reference(self) -> bool:
        return bool(self.abstract.strip())

    def stats(self) -> dict[str, int]:
        return {
            "body_words": count_words(self.body),
            "body_sentences": len(split_sentences(self.body)),
            "abstract_words": count_words(self.abstract),
        }


def load(path: Path) -> Report:
    """Read a report from .txt, .md or .pdf and split off its abstract."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No document at {path}.")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = _read_pdf(path)
    elif suffix in (".txt", ".md"):
        raw = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported document type: {suffix}. Use .txt, .md or .pdf.")

    title, abstract, body = _split(raw)
    return Report(path=path, title=title, abstract=abstract, body=body)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    # PDF extraction hyphenates across line breaks and wraps mid sentence.
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"(?<![.\n])\n(?![\n\d])", " ", text)
    return text


def _split(raw: str) -> tuple[str, str, str]:
    """Return title, abstract and body from the raw text of a report."""
    lines = raw.splitlines()

    title = ""
    for line in lines[:10]:
        if TITLE_PREFIX.match(line):
            title = TITLE_PREFIX.sub("", line).strip()
            break
        if line.strip() and not ABSTRACT_HEADING.match(line):
            title = title or line.strip()

    abstract_start = None
    for index, line in enumerate(lines):
        if ABSTRACT_HEADING.match(line):
            abstract_start = index + 1
            break

    if abstract_start is None:
        return title, "", _clean(raw)

    body_start = len(lines)
    for index in range(abstract_start, len(lines)):
        if BODY_HEADING.match(lines[index]):
            body_start = index
            break

    abstract = _clean("\n".join(lines[abstract_start:body_start]))
    body = _clean("\n".join(lines[body_start:]))
    return title, abstract, body


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, ignoring blank lines and bare headings."""
    sentences: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        for sentence in SENTENCE_BOUNDARY.split(block):
            sentence = sentence.strip()
            if len(sentence.split()) >= 3:
                sentences.append(sentence)
    return sentences


def count_words(text: str) -> int:
    return len(text.split())
