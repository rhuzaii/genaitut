"""Configuration for Experiment 10: retrieval augmented chatbot over the IPC.

Every tunable value lives here. A build writes the resolved configuration into
the index directory so an index can be traced back to how it was made.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Sentence embedding models. All run locally with no API key.
EMBEDDING_CHOICES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",      # 90 MB, 384 dims
    "mpnet": "sentence-transformers/all-mpnet-base-v2",      # 420 MB, 768 dims
    "bge": "BAAI/bge-small-en-v1.5",                         # 130 MB, 384 dims
}

DEFAULT_DOCUMENT = Path("documents/ipc.pdf")
FALLBACK_DOCUMENT = Path("documents/sample_statute.txt")


@dataclass
class ChunkConfig:
    """How the statute is divided before embedding.

    A statute is already structured: numbered sections, each self contained.
    Splitting on section boundaries keeps a provision together with its
    punishment and its exceptions, which a fixed character splitter would cut
    apart. Long sections are split further, and every piece keeps its section
    number so an answer can cite it.
    """

    strategy: str = "section"  # section or fixed
    max_tokens: int = 400
    overlap_tokens: int = 60


@dataclass
class RetrievalConfig:
    embedding_key: str = "minilm"
    top_k: int = 5
    # Below this cosine similarity a chunk is treated as irrelevant, so the
    # bot declines rather than answering from a weak match.
    min_score: float = 0.25


@dataclass
class GeneratorConfig:
    """Which language model writes the answer.

    auto prefers Cohere when COHERE_API_KEY is set, otherwise a local model.
    """

    backend: str = "auto"  # auto, cohere, local, extractive
    cohere_model: str = "command-r-plus-08-2024"
    # An instruction tuned decoder handles a multi passage prompt far better
    # than a small text to text model, which truncates at 512 tokens.
    local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    temperature: float = 0.1  # low: this is a legal lookup, not a creative task
    max_tokens: int = 400
    history_turns: int = 4


@dataclass
class Config:
    document: Path = DEFAULT_DOCUMENT
    index_dir: Path = Path("index")
    device: str = "auto"
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self) -> None:
        self.document = Path(self.document)
        self.index_dir = Path(self.index_dir)
        if self.retrieval.embedding_key not in EMBEDDING_CHOICES:
            raise ValueError(
                f"Unknown embedding key {self.retrieval.embedding_key!r}. "
                f"Choose from {sorted(EMBEDDING_CHOICES)}."
            )
        if self.chunking.strategy not in ("section", "fixed"):
            raise ValueError("chunking.strategy must be 'section' or 'fixed'.")

    @property
    def embedding_model(self) -> str:
        return EMBEDDING_CHOICES[self.retrieval.embedding_key]

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["document"] = str(self.document)
        data["index_dir"] = str(self.index_dir)
        data["embedding_model"] = self.embedding_model
        return data
