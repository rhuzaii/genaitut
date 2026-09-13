"""Quantitative cross modal alignment metrics.

Two similarity scores are reported:

  CLIP  cosine similarity between a text embedding and an image embedding
  CLAP  cosine similarity between a text embedding and an audio embedding

Both encoders were trained contrastively on paired data, so a higher cosine
similarity means the two items sit closer together in a space where paired
examples were pulled together during training.

CLIP truncates text at 77 tokens, which a fourteen line poem exceeds. The poem
is therefore split into chunks that fit the limit, each chunk is embedded, and
the normalised embeddings are averaged. This is stated in the report because it
affects how the numbers should be read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import Config
from runtime import release_model

ACCENT = "#b45309"
NEUTRAL = "#334155"
GRID = "#cbd5e1"


def clip_scores(cfg: Config, text: str, image_paths: dict[str, Path]) -> dict[str, float]:
    """Cosine similarity between the poem and each generated image."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = cfg.resolved_device()
    processor = CLIPProcessor.from_pretrained(cfg.scoring.clip_model)
    model = CLIPModel.from_pretrained(cfg.scoring.clip_model).to(device).eval()

    chunks = _chunk_text(text, processor.tokenizer, cfg.scoring.clip_chunk_tokens)
    with torch.no_grad():
        text_inputs = processor(
            text=chunks, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_vector = _mean_unit_vector(text_features)

        scores: dict[str, float] = {}
        for condition, path in image_paths.items():
            image = Image.open(path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            image_features = model.get_image_features(**image_inputs)
            image_vector = _mean_unit_vector(image_features)
            scores[condition] = float(torch.dot(text_vector, image_vector))

    release_model(model)
    return scores


def clap_scores(cfg: Config, text: str, audio_paths: dict[str, Path]) -> dict[str, float]:
    """Cosine similarity between the poem and each generated audio clip."""
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    device = cfg.resolved_device()
    processor = ClapProcessor.from_pretrained(cfg.scoring.clap_model)
    model = ClapModel.from_pretrained(cfg.scoring.clap_model).to(device).eval()

    target_sr = cfg.scoring.clap_sample_rate
    scores: dict[str, float] = {}

    with torch.no_grad():
        text_inputs = processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(device)
        text_vector = _mean_unit_vector(model.get_text_features(**text_inputs))

        for condition, path in audio_paths.items():
            waveform, _ = librosa.load(path, sr=target_sr, mono=True)
            audio_inputs = _clap_audio_inputs(processor, waveform, target_sr).to(device)
            audio_vector = _mean_unit_vector(model.get_audio_features(**audio_inputs))
            scores[condition] = float(torch.dot(text_vector, audio_vector))

    release_model(model)
    return scores


def _clap_audio_inputs(processor, waveform, sample_rate):
    """Call the CLAP processor in a way that works across transformers versions.

    The keyword was renamed from audios to audio. Older releases accept only
    the old name, newer ones reject it, so try the current name first.
    """
    try:
        return processor(
            audio=waveform, sampling_rate=sample_rate, return_tensors="pt"
        )
    except (TypeError, ValueError):
        return processor(
            audios=waveform, sampling_rate=sample_rate, return_tensors="pt"
        )


def plot_audio(audio_path: Path, output_path: Path) -> Path:
    """Save a waveform and log mel spectrogram figure for one clip.

    The audio cannot be printed in a report, so this figure stands in for it.
    """
    import librosa
    import librosa.display
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=96)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    figure, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    figure.patch.set_facecolor("white")

    times = np.arange(len(waveform)) / sample_rate
    axes[0].plot(times, waveform, color=ACCENT, linewidth=0.6)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(audio_path.stem.replace("_", " "), color=NEUTRAL, loc="left")
    axes[0].grid(True, color=GRID, linewidth=0.5)
    axes[0].set_axisbelow(True)

    image = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_ylabel("Mel frequency")
    figure.colorbar(image, ax=axes[1], format="%+2.0f dB", pad=0.01)

    for axis in axes:
        for spine in axis.spines.values():
            spine.set_color(GRID)
        axis.tick_params(colors=NEUTRAL, labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _chunk_text(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split text into pieces that fit inside the CLIP token limit."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = current + [line]
        token_count = len(tokenizer(" ".join(candidate), add_special_tokens=False)["input_ids"])
        if token_count > max_tokens and current:
            chunks.append(" ".join(current))
            current = [line]
        else:
            current = candidate
    if current:
        chunks.append(" ".join(current))
    return chunks


def _mean_unit_vector(features):
    import torch

    if not torch.is_tensor(features):
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        else:
            raise TypeError(
                f"Expected a tensor or an object with pooler_output, "
                f"got {type(features)}"
            )

    normalised = features / features.norm(dim=-1, keepdim=True)
    pooled = normalised.mean(dim=0)
    return pooled / pooled.norm()
