"""Language model stage: generate the poem and extract a structured scene.

Three backends are supported so that the experiment can run whether or not an
API key is available:

  cohere  hosted Cohere chat model, matches the stack used later in the course
  local   a small instruction tuned model pulled from Hugging Face
  manual  read a poem the student wrote or generated elsewhere

The backend is chosen with Config.text.backend. The default, auto, prefers
Cohere when COHERE_API_KEY is set and otherwise falls back to the local model.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import prompts
from config import Config

MANUAL_POEM_FILE = "manual_poem.txt"

_local_pipeline = None


def resolve_backend(cfg: Config) -> str:
    backend = cfg.text.backend
    if backend != "auto":
        return backend
    if os.environ.get("COHERE_API_KEY"):
        return "cohere"
    return "local"


def generate_poem(cfg: Config) -> tuple[str, str, str]:
    """Return the poem, the prompt that produced it and the backend used."""
    backend = resolve_backend(cfg)
    prompt = prompts.poem_prompt(cfg.theme, cfg.text.poem_lines)

    if backend == "manual":
        return _read_manual_poem(cfg), prompt, backend

    raw = _complete(cfg, backend, prompt)
    return _clean_poem(raw, cfg.text.poem_lines), prompt, backend


def extract_scene(cfg: Config, poem: str) -> tuple[dict[str, Any], str]:
    """Extract a structured scene description from the generated poem."""
    backend = resolve_backend(cfg)
    prompt = prompts.scene_extraction_prompt(poem)

    if backend == "manual":
        return _fallback_scene(cfg), prompt

    raw = _complete(cfg, backend, prompt)
    scene = _parse_scene_json(raw)
    if scene is None:
        scene = _fallback_scene(cfg)
    return scene, prompt


def _complete(cfg: Config, backend: str, prompt: str) -> str:
    if backend == "cohere":
        return _complete_cohere(cfg, prompt)
    if backend == "local":
        return _complete_local(cfg, prompt)
    raise ValueError(f"Unknown text backend: {backend}")


def _complete_cohere(cfg: Config, prompt: str) -> str:
    import cohere

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. Export it, or run with "
            "--text-backend local to use a local model instead."
        )

    client = cohere.ClientV2(api_key=api_key)
    response = client.chat(
        model=cfg.text.cohere_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg.text.temperature,
        max_tokens=cfg.text.max_new_tokens,
    )
    return "".join(block.text for block in response.message.content).strip()


def _complete_local(cfg: Config, prompt: str) -> str:
    global _local_pipeline

    import torch
    from transformers import pipeline

    if _local_pipeline is None:
        device = cfg.resolved_device()
        _local_pipeline = pipeline(
            "text-generation",
            model=cfg.text.local_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

    outputs = _local_pipeline(
        [{"role": "user", "content": prompt}],
        max_new_tokens=cfg.text.max_new_tokens,
        temperature=cfg.text.temperature,
        do_sample=True,
        return_full_text=False,
    )
    return outputs[0]["generated_text"].strip()


def _read_manual_poem(cfg: Config) -> str:
    path = cfg.output_dir / MANUAL_POEM_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Manual backend selected but {path} does not exist. "
            "Write the poem into that file and run again."
        )
    return path.read_text(encoding="utf-8").strip()


def _clean_poem(raw: str, expected_lines: int) -> str:
    """Strip preamble, code fences and numbering from a model response."""
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()

    lines = [line.rstrip() for line in text.splitlines()]
    lines = [re.sub(r"^\s*\d+[.)]\s*", "", line) for line in lines]

    # Drop a leading sentence of commentary such as "Here is a poem:".
    while lines and (not lines[0].strip() or lines[0].strip().endswith(":")):
        lines.pop(0)

    lines = [line for line in lines if line.strip()]
    if len(lines) > expected_lines:
        lines = lines[:expected_lines]
    return "\n".join(lines)


def _parse_scene_json(raw: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {key: data.get(key) for key in prompts.SCENE_KEYS}


def _fallback_scene(cfg: Config) -> dict[str, Any]:
    """A hand written scene used when the model returns unusable JSON.

    Recording that the fallback was used matters for the report, so the caller
    stores this dictionary in the manifest exactly as it is returned.
    """
    return {
        "subject": "workers shaping molten glass bangles beside a furnace",
        "setting": "a small glass bangle workshop in Firozabad",
        "time_of_day": "pre dawn, around four in the morning",
        "lighting": "orange furnace glow, one bare bulb, deep shadow",
        "mood": "hot, exhausted, precise",
        "key_visual_details": [
            "molten glass glowing on the end of an iron rod",
            "soot darkened brick walls",
            "stacks of finished bangles in shallow trays",
            "sweat and firelight on a worker's face",
        ],
        "key_sounds": [
            "deep roar of a coal furnace",
            "hiss of hot glass meeting air",
            "rhythmic tapping of iron rods",
            "faint clink of glass bangles",
        ],
        "_fallback": True,
    }
