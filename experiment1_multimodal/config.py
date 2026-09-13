"""Configuration for Experiment 1: multimodal generative AI alignment study.

All tunable values live here. Every run writes a copy of the resolved
configuration into its manifest so that a result can be reproduced later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_THEME = (
    "A glass bangle furnace at 4 a.m. in Firozabad, where workers draw molten "
    "glass from a coal fired furnace and shape it into bangles before sunrise"
)

MIN_AUDIO_SECONDS = 15
MAX_AUDIO_SECONDS = 30


@dataclass
class TextConfig:
    """Settings for the language model stage."""

    backend: str = "auto"  # auto, cohere, local or manual
    cohere_model: str = "command-r-plus-08-2024"
    local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 600
    temperature: float = 0.8
    poem_lines: int = 14


@dataclass
class ImageConfig:
    """Settings for the text to image stage.

    Alternatives that also run on a single 16 GB GPU:
      stabilityai/stable-diffusion-2-1-base
      stable-diffusion-v1-5/stable-diffusion-v1-5
      segmind/SSD-1B
    """

    model_id: str = "sd2-community/stable-diffusion-2-1-base"
    height: int = 512
    width: int = 512
    steps: int = 40
    guidance_scale: float = 7.5
    negative_prompt: str = (
        "cartoon, illustration, painting, 3d render, text, watermark, signature, "
        "blurry, deformed hands, extra limbs, oversaturated, clean modern factory"
    )


@dataclass
class AudioConfig:
    """Settings for the text to audio stage.

    MusicGen emits 50 audio tokens per second, so the token budget is derived
    from the requested duration rather than set by hand.
    """

    model_id: str = "facebook/musicgen-small"
    seconds: int = 25
    guidance_scale: float = 3.0
    tokens_per_second: int = 50
    # 16 bit PCM plays everywhere and embeds in Word and PowerPoint, unlike the
    # 32 bit float WAV that scipy writes by default.
    pcm_bit_depth: int = 16
    write_mp3: bool = True
    mp3_bitrate: str = "192k"

    @property
    def max_new_tokens(self) -> int:
        return self.seconds * self.tokens_per_second


@dataclass
class ScoringConfig:
    """Settings for the cross modal similarity metrics."""

    clip_model: str = "openai/clip-vit-base-patch32"
    clap_model: str = "laion/clap-htsat-unfused"
    clap_sample_rate: int = 48000
    clip_chunk_tokens: int = 70


@dataclass
class Config:
    """Top level configuration for one experiment run."""

    theme: str = DEFAULT_THEME
    seed: int = 1729
    device: str = "auto"  # auto, cuda, mps or cpu
    output_dir: Path = Path("outputs")
    run_control: bool = True
    text: TextConfig = field(default_factory=TextConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if not MIN_AUDIO_SECONDS <= self.audio.seconds <= MAX_AUDIO_SECONDS:
            raise ValueError(
                f"The experiment brief requires {MIN_AUDIO_SECONDS} to "
                f"{MAX_AUDIO_SECONDS} seconds of audio, got {self.audio.seconds}."
            )

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
        data["output_dir"] = str(self.output_dir)
        data["audio"]["max_new_tokens"] = self.audio.max_new_tokens
        return data
