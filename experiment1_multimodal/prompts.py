"""Prompt construction for the three generative stages.

The experiment compares two ways of prompting the image and audio models:

  direct   the theme string is inserted into a modality specific template
  derived  a structured scene description is first extracted from the
           generated poem, and the image and audio prompts are built from
           that description

Keeping both paths in one module makes the difference between the two
conditions easy to inspect and to quote in the report.
"""

from __future__ import annotations

from typing import Any

SCENE_KEYS = (
    "subject",
    "setting",
    "time_of_day",
    "lighting",
    "mood",
    "key_visual_details",
    "key_sounds",
)

CONTROL_IMAGE_PROMPT = (
    "A bright empty ski slope on a clear winter morning, fresh powder snow, "
    "blue sky, crisp cold light, wide landscape photograph"
)

CONTROL_AUDIO_PROMPT = (
    "Upbeat cheerful ukulele and hand claps, sunny major key, light percussion, "
    "relaxed summer advertisement music, no vocals"
)


def poem_prompt(theme: str, lines: int) -> str:
    return (
        f"Write a {lines} line free verse poem about the following scene.\n\n"
        f"Scene: {theme}\n\n"
        "Requirements:\n"
        f"- Exactly {lines} lines, no title, no numbering.\n"
        "- Free verse. Do not rhyme.\n"
        "- Ground every line in concrete sensory detail: heat, light, sound, "
        "texture, smell.\n"
        "- Contrast the scale and violence of the furnace with the precision of "
        "the human hands working beside it.\n"
        "- Do not explain the poem or add commentary before or after it.\n\n"
        "Poem:"
    )


def scene_extraction_prompt(poem: str) -> str:
    return (
        "Read the poem below and extract a structured description of the scene "
        "it depicts. Return only a single JSON object and nothing else.\n\n"
        "Schema:\n"
        "{\n"
        '  "subject": "the main subject in under 12 words",\n'
        '  "setting": "the physical location in under 12 words",\n'
        '  "time_of_day": "for example pre dawn, midday, dusk",\n'
        '  "lighting": "the dominant light source and quality",\n'
        '  "mood": "three comma separated mood adjectives",\n'
        '  "key_visual_details": ["four short concrete visual details"],\n'
        '  "key_sounds": ["four short concrete sounds present in the scene"]\n'
        "}\n\n"
        "Rules:\n"
        "- Use only what the poem actually contains or clearly implies.\n"
        "- Prefer literal, photographable detail over metaphor.\n"
        "- Keep every string short enough to sit inside an image prompt.\n\n"
        f"Poem:\n{poem}\n\n"
        "JSON:"
    )


def direct_image_prompt(theme: str) -> str:
    return (
        f"{theme}. Documentary photograph, tight interior shot, workers seated "
        "close to the furnace mouth, glowing orange light on faces and walls, "
        "smoke and soot in the air, photorealistic, natural grain, "
        "available light only"
    )


def direct_audio_prompt(theme: str) -> str:
    return (
        f"Ambient field recording of this scene: {theme}. Deep roar of a coal "
        "furnace, hiss of molten glass, rhythmic tapping of iron rods, faint "
        "clink of finished bangles, low voices, night insects outside, "
        "immersive and continuous"
    )


def derived_image_prompt(scene: dict[str, Any]) -> str:
    details = ", ".join(_as_list(scene.get("key_visual_details")))
    parts = [
        _text(scene.get("subject")),
        _text(scene.get("setting")),
        _text(scene.get("time_of_day")),
        _text(scene.get("lighting")),
        details,
        _text(scene.get("mood")),
        "documentary photograph, available light, photorealistic, natural grain, "
        "tight interior composition",
    ]
    return ", ".join(part for part in parts if part)


def derived_audio_prompt(scene: dict[str, Any]) -> str:
    sounds = ", ".join(_as_list(scene.get("key_sounds")))
    mood = _text(scene.get("mood"))
    setting = _text(scene.get("setting"))
    body = sounds or "furnace roar, metal on metal, distant voices"
    return (
        f"Ambient field recording, {setting}. {body}. "
        f"Mood: {mood}. Continuous immersive soundscape, no music, no vocals"
    ).strip()


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(item).strip() for item in value if str(item).strip()]
