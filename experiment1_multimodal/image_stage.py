"""Text to image stage backed by a latent diffusion pipeline.

The pipeline is loaded once, used for every prompt in the run, then released
so that the audio model can claim the same GPU memory.
"""

from __future__ import annotations

from pathlib import Path

from config import Config
from runtime import release_model


def generate_images(cfg: Config, named_prompts: dict[str, str]) -> dict[str, Path]:
    """Render one image per prompt and return the paths keyed by condition."""
    import torch
    from diffusers import StableDiffusionPipeline

    device = cfg.resolved_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.image.model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        pipe.enable_attention_slicing()

    image_dir = cfg.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for condition, prompt in named_prompts.items():
        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=cfg.image.negative_prompt,
            height=cfg.image.height,
            width=cfg.image.width,
            num_inference_steps=cfg.image.steps,
            guidance_scale=cfg.image.guidance_scale,
            generator=generator,
        )
        path = image_dir / f"image_{condition}.png"
        result.images[0].save(path)
        paths[condition] = path
        print(f"  wrote {path}")

    release_model(pipe)
    return paths
