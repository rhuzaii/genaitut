"""Loading and preprocessing the statute (step i of the experiment).

A statute is not prose. It is a numbered hierarchy of chapters and sections,
and that structure is the most useful thing in the document for retrieval. A
generic text loader throws it away. This module keeps it, so every chunk can
later be cited as "Section 302" rather than "chunk 47".

PDF extraction is the messy part. Official statute PDFs hyphenate across line
breaks, wrap sentences mid clause, and repeat page furniture on every page.
All three are repaired here, because each one corrupts chunk boundaries and
therefore retrieval quality downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# "302. Punishment for murder.—Whoever commits murder ..."
# "302A. ..." and "Section 302." are both accepted.
#
# A provision that has been substituted by a later amendment is printed with a
# leading footnote marker, as in "2[375. Rape.—A man is said to commit ...".
# That convention is near universal in Indian bare Acts, and requiring the
# number at the start of the line silently loses every amended section. In the
# Indian Penal Code that is 13 sections, including 34 (common intention) and
# 375 (rape), two of the most frequently cited provisions in the whole Code.
SECTION_START = re.compile(
    r"^[ \t]*(?:\d{1,2}\[)?(?:Section[ \t]+)?(?P<number>\d{1,3}[A-Z]{0,2})\.[ \t]*(?P<rest>\S.*)$"
)
CHAPTER_START = re.compile(
    r"^[ \t]*CHAPTER[ \t]+(?P<numeral>[IVXLC]+|\d+)\b[ \t]*(?P<title>.*)$",
    re.IGNORECASE,
)
# A heading ends at the first em dash, en dash or full stop.
HEADING_SPLIT = re.compile(r"\s*[—–.]\s*")

PAGE_FURNITURE = re.compile(
    r"^\s*(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s*)$", re.IGNORECASE
)


@dataclass
class Section:
    """One numbered provision of the statute."""

    number: str
    heading: str
    text: str
    chapter: str = ""

    @property
    def citation(self) -> str:
        return f"Section {self.number}"

    @property
    def full_text(self) -> str:
        head = f"{self.citation}. {self.heading}".strip().rstrip(".")
        return f"{head}.\n{self.text}".strip()

    def to_dict(self) -> dict[str, str]:
        return {
            "number": self.number,
            "heading": self.heading,
            "chapter": self.chapter,
            "text": self.text,
        }


@dataclass
class Statute:
    path: Path
    title: str
    sections: list[Section] = field(default_factory=list)
    raw_text: str = ""

    def stats(self) -> dict[str, int]:
        return {
            "characters": len(self.raw_text),
            "words": len(self.raw_text.split()),
            "sections": len(self.sections),
            "chapters": len({s.chapter for s in self.sections if s.chapter}),
        }


def load(path: Path) -> Statute:
    """Read a statute from .pdf, .txt or .md and parse it into sections."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No document at {path}. See documents/README.md for where to get "
            "the official IPC text."
        )

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = _read_pdf(path)
    elif suffix in (".txt", ".md"):
        raw = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported document type {suffix}. Use .pdf, .txt or .md.")

    raw = clean(raw)
    sections = parse_sections(raw)
    sections = consolidate(sections)
    title = _guess_title(raw, path)
    return Statute(path=path, title=title, sections=sections, raw_text=raw)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines = [l for l in text.splitlines() if not PAGE_FURNITURE.match(l)]
        pages.append("\n".join(lines))
    return "\n".join(pages)


def clean(text: str) -> str:
    """Repair the damage PDF extraction does to a statute."""
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Words hyphenated across a line break.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Lines wrapped mid sentence, but never before a new numbered section.
    text = re.sub(r"(?<![.:;\n])\n(?![\n\d]|CHAPTER)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_sections(text: str) -> list[Section]:
    """Split the statute into numbered sections, tracking the chapter."""
    lines = text.splitlines()
    sections: list[Section] = []
    chapter = ""
    current: Section | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            current.text = " ".join(buffer).strip()
            if current.text or current.heading:
                sections.append(current)

    for line in lines:
        chapter_match = CHAPTER_START.match(line)
        if chapter_match:
            flush()
            current, buffer = None, []
            numeral = chapter_match.group("numeral")
            title = chapter_match.group("title").strip()
            chapter = f"Chapter {numeral}" + (f": {title}" if title else "")
            continue

        section_match = SECTION_START.match(line)
        if section_match:
            flush()
            rest = section_match.group("rest").strip()
            heading, _, remainder = _split_heading(rest)
            current = Section(
                number=section_match.group("number"),
                heading=heading,
                text="",
                chapter=chapter,
            )
            buffer = [remainder] if remainder else []
            continue

        if current is not None:
            buffer.append(line.strip())

    flush()
    return sections


# Amendment footnotes are printed at the foot of the page as "1. Ins. by Act
# 3 of 1951, s. 3" or "2. Subs. by Act 26 of 1955". After the heading split
# those leave a heading of "Ins", "Subs", "Rep" and so on, and they begin with
# a number and a full stop, so they are indistinguishable from a section
# header by shape alone. Left in, they outweigh the real provision during
# consolidation because a page can carry several, and section 1 of the Indian
# Penal Code resolves to 1500 words of amendment history.
FOOTNOTE_HEADINGS = {"ins", "subs", "rep", "s", "cl", "omitted", "added", "now", "vide"}
FOOTNOTE_PREFIX = re.compile(
    r"^(?:the words|certain words|these words|clause|earlier|before substitution)\b",
    re.IGNORECASE,
)


def _looks_like_footnote(section: "Section") -> bool:
    heading = section.heading.strip().rstrip(".")
    if heading.lower() in FOOTNOTE_HEADINGS:
        return True
    return bool(FOOTNOTE_PREFIX.match(heading))


def consolidate(sections: list[Section], min_body_words: int = 4) -> list[Section]:
    """Drop table of contents entries and keep one entry per section number.

    Two artefacts inflate a naive parse of a statute PDF.

    The document opens with an arrangement of sections that repeats every
    heading with no body, so each heading is seen twice: once empty, once
    real. Those empty entries are dropped.

    Sub clauses and illustrations inside a section are themselves numbered, so
    a line beginning "2." part way through section 300 looks like the start of
    section 2. Keeping, for each number, the entry with the most body text
    selects the operative provision over both the contents line and the stray
    sub clause, because the real provision is almost always the longest.

    Without this the Indian Penal Code parses as about 1400 sections instead
    of roughly 500, and section 302 resolves to an empty contents entry.
    """
    best: dict[str, Section] = {}
    for section in sections:
        words = len(section.text.split())
        if words < min_body_words or _looks_like_footnote(section):
            continue
        current = best.get(section.number)
        if current is None or words > len(current.text.split()):
            best[section.number] = section
    return sorted(best.values(), key=lambda s: _sort_key(s.number))


def _sort_key(number: str) -> tuple[int, str]:
    digits = "".join(ch for ch in number if ch.isdigit())
    suffix = "".join(ch for ch in number if ch.isalpha())
    return (int(digits) if digits else 0, suffix)


def _split_heading(rest: str) -> tuple[str, str, str]:
    """Separate a section heading from the body that follows it."""
    parts = HEADING_SPLIT.split(rest, maxsplit=1)
    heading = parts[0].strip()
    remainder = parts[1].strip() if len(parts) > 1 else ""
    # The separator is often a full stop followed by an em dash, so a dash can
    # survive the split and head the body text.
    remainder = remainder.lstrip("\u2014\u2013- ").strip()
    # A runaway heading means the separator was missing; treat it all as body.
    if len(heading.split()) > 20:
        return "", "", rest
    return heading, "", remainder


def _guess_title(text: str, path: Path) -> str:
    for line in text.splitlines()[:15]:
        line = line.strip()
        if len(line.split()) >= 3 and not SECTION_START.match(line):
            # clean() joins wrapped lines, so the title can absorb the line
            # below it. Keep the first sentence only.
            return re.split(r"(?<=[.!?])\s+", line)[0][:160].strip()
    return path.stem.replace("_", " ").title()
