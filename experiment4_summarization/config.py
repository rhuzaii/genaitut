"""Configuration for Experiment 4: abstractive summarization of academic reports.

Every tunable value lives here. A run writes the resolved configuration into
its manifest so a result can be reproduced or defended later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Summarization models, all trained on a summarization objective so the
# pipeline needs no task prefix. Sizes are approximate download sizes.
MODEL_CHOICES = {
    "bart": "facebook/bart-large-cnn",            # 1.6 GB, the usual baseline
    "distilbart": "sshleifer/distilbart-cnn-12-6",  # 1.2 GB, roughly twice as fast
    "pegasus": "google/pegasus-cnn_dailymail",    # 2.3 GB, more aggressive compression
}

DEFAULT_DOCUMENT = Path("documents/sample_report.txt")


@dataclass
class ChunkConfig:
    """How a document longer than the model's input window is divided.

    Encoder decoder summarizers have a fixed input limit, 1024 tokens for BART.
    A long report must therefore be split, summarized piecewise, and the pieces
    combined. Overlap carries a little context across the boundary so a claim
    split between two chunks is not lost.
    """

    max_tokens: int = 900
    overlap_sentences: int = 2


@dataclass
class SummaryConfig:
    """Length bounds for the two summarization passes.

    The map pass summarizes each chunk. The reduce pass summarizes the
    concatenated chunk summaries into the final text, which is what makes the
    output short enough to be worth reading.
    """

    map_max_tokens: int = 180
    map_min_tokens: int = 60
    reduce_max_tokens: int = 200
    reduce_min_tokens: int = 80
    num_beams: int = 4
    length_penalty: float = 2.0
    no_repeat_ngram_size: int = 3


@dataclass
class Config:
    document: Path = DEFAULT_DOCUMENT
    models: tuple[str, ...] = ("bart", "distilbart")
    output_dir: Path = Path("outputs")
    device: str = "auto"
    seed: int = 1729
    run_baseline: bool = True
    baseline_sentences: int = 3
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)

    def __post_init__(self) -> None:
        self.document = Path(self.document)
        self.output_dir = Path(self.output_dir)
        unknown = [name for name in self.models if name not in MODEL_CHOICES]
        if unknown:
            raise ValueError(
                f"Unknown model key(s): {unknown}. "
                f"Choose from {sorted(MODEL_CHOICES)}."
            )

    def model_id(self, key: str) -> str:
        return MODEL_CHOICES[key]

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
        data["output_dir"] = str(self.output_dir)
        data["models"] = list(self.models)
        data["model_ids"] = {key: MODEL_CHOICES[key] for key in self.models}
        return data
