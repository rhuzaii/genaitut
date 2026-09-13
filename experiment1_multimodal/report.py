"""Render a Markdown report scaffold from the results of a run.

The numeric results, prompts and file references are filled in automatically.
The qualitative rubric and the discussion sections are left blank on purpose,
since those are the parts of the experiment that have to be written by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONDITION_LABELS = {
    "direct": "Direct (prompt built from the theme string)",
    "derived": "Derived (prompt built from the scene extracted from the poem)",
    "control": "Control (deliberately unrelated prompt)",
}


def render(results: dict[str, Any], output_path: Path) -> Path:
    lines: list[str] = []
    add = lines.append

    add("# Experiment 1: Multimodal Generative AI")
    add("")
    add("## Aim")
    add(
        "To generate text, image and audio outputs from a single theme using "
        "separate generative models, to measure and discuss how well those "
        "outputs align with one another, and to evaluate the strengths, "
        "limitations and ethical challenges of multimodal generative AI."
    )
    add("")

    add("## Theme")
    add("")
    add(f"> {results['theme']}")
    add("")

    add("## Environment and configuration")
    add("")
    add("| Setting | Value |")
    add("| --- | --- |")
    add(f"| Device | {results['device']} |")
    add(f"| Seed | {results['config']['seed']} |")
    add(f"| Text backend | {results['text_backend']} |")
    add(f"| Text model | {results['text_model']} |")
    add(f"| Image model | {results['config']['image']['model_id']} |")
    add(
        f"| Image settings | {results['config']['image']['width']}x"
        f"{results['config']['image']['height']}, "
        f"{results['config']['image']['steps']} steps, guidance "
        f"{results['config']['image']['guidance_scale']} |"
    )
    add(f"| Audio model | {results['config']['audio']['model_id']} |")
    add(
        f"| Audio settings | {results['config']['audio']['seconds']} s, guidance "
        f"{results['config']['audio']['guidance_scale']} |"
    )
    add(f"| CLIP model | {results['config']['scoring']['clip_model']} |")
    add(f"| CLAP model | {results['config']['scoring']['clap_model']} |")
    add("")

    add("## Method")
    add("")
    add(
        "The run compares two ways of prompting the image and audio models "
        "from the same theme."
    )
    add("")
    add(
        "1. **Direct.** The theme string is inserted into a modality specific "
        "template and sent to the image and audio models."
    )
    add(
        "2. **Derived.** The language model first writes the poem, a structured "
        "scene description is then extracted from that poem, and the image and "
        "audio prompts are built from the extracted fields."
    )
    add(
        "3. **Control.** An unrelated prompt is rendered in each modality to "
        "give the similarity scores a baseline to be read against."
    )
    add("")
    add(
        "The hypothesis is that the derived condition scores higher, because "
        "the prompts then carry the concrete detail the language model "
        "actually committed to rather than the more abstract theme string."
    )
    add("")

    add("## Prompts used")
    add("")
    for name, prompt in results["prompts"].items():
        add(f"**{name}**")
        add("")
        add("```")
        add(prompt.strip())
        add("```")
        add("")

    add("## Results")
    add("")
    add("### Generated text")
    add("")
    add("```")
    add(results["poem"].strip())
    add("```")
    add("")

    add("### Extracted scene description")
    add("")
    add("```json")
    add(json.dumps(results["scene"], indent=2, ensure_ascii=False))
    add("```")
    add("")
    if results["scene"].get("_fallback"):
        add(
            "> Note: the model did not return parseable JSON, so the hand "
            "written fallback scene was used. This is recorded here because it "
            "affects how the derived condition should be interpreted."
        )
        add("")

    add("### Generated images")
    add("")
    for condition, path in results["images"].items():
        add(f"**{CONDITION_LABELS.get(condition, condition)}**")
        add("")
        add(f"![{condition}]({_relative(path, output_path)})")
        add("")

    add("### Generated audio")
    add("")
    for condition, path in results["audio"].items():
        add(f"**{CONDITION_LABELS.get(condition, condition)}**")
        add("")
        add(f"Audio file: `{_relative(path, output_path)}`")
        add("")
        figure = results["audio_figures"].get(condition)
        if figure:
            add(f"![{condition} waveform]({_relative(figure, output_path)})")
            add("")

    add("### Quantitative alignment scores")
    add("")
    add(
        "Cosine similarity between the poem and each generated artefact. "
        "Higher means closer in the shared embedding space. CLIP and CLAP are "
        "trained separately, so compare values down a column, not across."
    )
    add("")
    add("| Condition | CLIP (poem vs image) | CLAP (poem vs audio) |")
    add("| --- | --- | --- |")
    conditions = sorted(
        set(results["clip"]) | set(results["clap"]),
        key=lambda name: ("control" in name, name),
    )
    for condition in conditions:
        clip_value = results["clip"].get(condition)
        clap_value = results["clap"].get(condition)
        add(
            f"| {CONDITION_LABELS.get(condition, condition)} "
            f"| {_fmt(clip_value)} | {_fmt(clap_value)} |"
        )
    add("")
    add(
        "The poem was split into "
        f"{results['clip_chunks']} chunk(s) before CLIP text encoding, because "
        "CLIP truncates at 77 tokens. The normalised chunk embeddings were "
        "averaged."
    )
    add("")

    add("## Qualitative alignment analysis")
    add("")
    add("Score each pair from 1 to 5 and justify every score with a specific observation.")
    add("")
    add("| Dimension | Text and image | Text and audio | Image and audio |")
    add("| --- | --- | --- | --- |")
    for dimension in (
        "Subject match",
        "Mood match",
        "Setting match",
        "Detail fidelity",
        "Overall coherence",
    ):
        add(f"| {dimension} | | | |")
    add("")
    add("### Text and image")
    add("")
    add("_Write the discussion here._")
    add("")
    add("### Text and audio")
    add("")
    add("_Write the discussion here._")
    add("")
    add("### Image and audio")
    add("")
    add("_Write the discussion here._")
    add("")
    add("### Direct compared with derived prompting")
    add("")
    add("_Was the hypothesis supported? Quote the scores above._")
    add("")

    for heading in ("Strengths", "Limitations", "Ethical challenges", "Conclusion"):
        add(f"## {heading}")
        add("")
        add("_Write this section here._")
        add("")

    add("## References")
    add("")
    add("_List the models, tools and papers used._")
    add("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _fmt(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.4f}"


def _relative(path: Path, report_path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(report_path.parent.resolve()))
    except ValueError:
        return str(path)
