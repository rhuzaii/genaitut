"""Text to audio stage backed by MusicGen.

MusicGen emits 50 audio tokens per second of output, so the duration required
by the experiment brief is converted into a token budget in AudioConfig.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import Config
from runtime import release_model


def generate_audio(cfg: Config, named_prompts: dict[str, str]) -> dict[str, Path]:
    """Render one clip per prompt and return the paths keyed by condition."""
    import torch
    from scipy.io import wavfile
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = cfg.resolved_device()
    processor = AutoProcessor.from_pretrained(cfg.audio.model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(cfg.audio.model_id)
    model = model.to(device)
    model.eval()

    sample_rate = model.config.audio_encoder.sampling_rate
    audio_dir = cfg.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for condition, prompt in named_prompts.items():
        torch.manual_seed(cfg.seed)
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=cfg.audio.guidance_scale,
                max_new_tokens=cfg.audio.max_new_tokens,
            )

        waveform = tokens[0, 0].detach().cpu().float().numpy()
        waveform = _normalise(waveform)

        path = audio_dir / f"audio_{condition}.wav"
        wavfile.write(path, sample_rate, _to_pcm(waveform, cfg.audio.pcm_bit_depth))
        paths[condition] = path

        duration = len(waveform) / sample_rate
        print(
            f"  wrote {path} ({duration:.1f} s, {sample_rate} Hz, "
            f"{cfg.audio.pcm_bit_depth} bit PCM)"
        )

        if cfg.audio.write_mp3:
            mp3_path = _encode_mp3(path, cfg.audio.mp3_bitrate)
            if mp3_path is not None:
                print(f"  wrote {mp3_path}")

    release_model(model)
    return paths


def _normalise(waveform: np.ndarray, headroom: float = 0.97) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak == 0.0:
        return waveform
    return waveform / peak * headroom


def _to_pcm(waveform: np.ndarray, bit_depth: int) -> np.ndarray:
    """Convert a float waveform in [-1, 1] to integer PCM samples."""
    if bit_depth == 16:
        dtype, ceiling = np.int16, 32767
    elif bit_depth == 32:
        dtype, ceiling = np.int32, 2147483647
    else:
        raise ValueError(f"Unsupported PCM bit depth: {bit_depth}")
    clipped = np.clip(waveform, -1.0, 1.0)
    return (clipped * ceiling).astype(dtype)


def _encode_mp3(wav_path: Path, bitrate: str) -> Path | None:
    """Write an MP3 next to the WAV, or return None when ffmpeg is unavailable.

    MP3 is roughly a tenth the size, which matters when the clips are attached
    to a submitted report.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return None

    mp3_path = wav_path.with_suffix(".mp3")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(wav_path),
            "-codec:a", "libmp3lame", "-b:a", bitrate,
            str(mp3_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  mp3 encoding skipped: {result.stderr.strip()[:120]}")
        return None
    return mp3_path
