"""Compare two or more runs of the experiment into a single Markdown document.

Running the same theme and seed through different language model backends turns
the exercise into a controlled comparison: the direct and control artefacts are
built from fixed prompts, so they should come out byte identical across runs,
while the derived artefacts change because they depend on the generated poem.

Usage:
    python compare_runs.py outputs_local outputs_cohere
    python compare_runs.py outputs_local outputs_cohere --labels Qwen Cohere
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

MODALITIES = (("images", "image"), ("audio", "audio"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a comparison report from two or more experiment runs."
    )
    parser.add_argument("run_dirs", nargs="+", help="output directories to compare")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="short name for each run, defaults to the directory name",
    )
    parser.add_argument(
        "--output", default="comparison.md", help="path for the generated document"
    )
    return parser.parse_args(argv)


def load_runs(run_dirs: list[str], labels: list[str] | None) -> list[dict[str, Any]]:
    if labels and len(labels) != len(run_dirs):
        raise ValueError("Give one label per run directory, or no labels at all.")

    runs = []
    for index, directory in enumerate(run_dirs):
        path = Path(directory)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json in {path}.")
        runs.append(
            {
                "label": labels[index] if labels else path.name,
                "dir": path,
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            }
        )
    return runs


def render(runs: list[dict[str, Any]], output_path: Path) -> Path:
    lines: list[str] = []
    add = lines.append

    add("# Experiment 1: comparison across runs")
    add("")
    add(
        "The theme and the seed were held fixed. The language model backend is "
        "the independent variable. Because the direct and control prompts are "
        "built from constants rather than from the poem, their artefacts should "
        "be reproducible across runs, which makes any change in their scores "
        "attributable to the text alone."
    )
    add("")

    add("## Runs")
    add("")
    add("| Run | Text backend | Text model | Seed | Elapsed (s) |")
    add("| --- | --- | --- | --- | --- |")
    for run in runs:
        manifest = run["manifest"]
        add(
            f"| {run['label']} | {manifest.get('text_backend', 'unknown')} "
            f"| `{manifest.get('text_model', 'unknown')}` "
            f"| {manifest['config']['seed']} "
            f"| {manifest.get('elapsed_seconds', 'not recorded')} |"
        )
    add("")

    themes = {run["manifest"]["theme"] for run in runs}
    if len(themes) == 1:
        add(f"Theme held constant across all runs:\n\n> {themes.pop()}")
    else:
        add("Warning: the runs do not share a theme, so they are not comparable.")
        for run in runs:
            add(f"- {run['label']}: {run['manifest']['theme']}")
    add("")

    add("## Alignment scores")
    add("")
    for metric, caption in (("clip", "CLIP, poem against image"), ("clap", "CLAP, poem against audio")):
        add(f"### {caption}")
        add("")
        conditions = _all_conditions(runs, metric)
        add("| Condition | " + " | ".join(run["label"] for run in runs) + " |")
        add("| --- |" + " --- |" * len(runs))
        for condition in conditions:
            cells = [_fmt(run["manifest"][metric].get(condition)) for run in runs]
            add(f"| {condition} | " + " | ".join(cells) + " |")
        add("")

    add("## Artefact reproducibility")
    add("")
    add(
        "SHA256 of each generated file, compared across runs. Direct and control "
        "artefacts are expected to match, since their prompts and seed do not "
        "depend on the poem."
    )
    add("")
    add("| Artefact | " + " | ".join(run["label"] for run in runs) + " | Match |")
    add("| --- |" + " --- |" * (len(runs) + 1))
    for subdir, prefix in MODALITIES:
        for condition in _all_file_conditions(runs, subdir, prefix):
            digests = [_digest(run["dir"] / subdir, prefix, condition) for run in runs]
            present = [d for d in digests if d]
            match = "identical" if len(set(present)) == 1 and len(present) == len(runs) else "different"
            cells = [d[:12] if d else "missing" for d in digests]
            add(f"| `{prefix}_{condition}` | " + " | ".join(cells) + f" | {match} |")
    add("")
    add(
        "Where an artefact is identical but its score moved, the change is "
        "caused entirely by the poem it was scored against, not by the artefact. "
        "This is direct evidence that CLIP and CLAP measure text to artefact "
        "alignment rather than artefact quality."
    )
    add("")

    add("## Generated text")
    add("")
    for run in runs:
        add(f"### {run['label']}")
        add("")
        add("```")
        add(run["manifest"]["poem"].strip())
        add("```")
        add("")
        add(f"Lines: {len(run['manifest']['poem'].strip().splitlines())}")
        add("")

    add("## Extracted scene descriptions")
    add("")
    for run in runs:
        add(f"### {run['label']}")
        add("")
        add("```json")
        add(json.dumps(run["manifest"]["scene"], indent=2, ensure_ascii=False))
        add("```")
        add("")

    add("## Interpretation")
    add("")
    for prompt in (
        "Which backend produced the higher scores, and in which modality?",
        "Did the direct against derived pattern hold across both runs?",
        "Which artefacts were reproducible, and what does that prove?",
        "Where did the scene extraction lose information from the theme?",
        "What does any change in the control score tell you?",
    ):
        add(f"**{prompt}**")
        add("")
        add("_Write the answer here._")
        add("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _all_conditions(runs: list[dict[str, Any]], metric: str) -> list[str]:
    names: set[str] = set()
    for run in runs:
        names.update(run["manifest"].get(metric, {}))
    return sorted(names, key=lambda name: (name == "control", name))


def _all_file_conditions(runs: list[dict[str, Any]], subdir: str, prefix: str) -> list[str]:
    names: set[str] = set()
    for run in runs:
        directory = run["dir"] / subdir
        if directory.exists():
            for path in directory.glob(f"{prefix}_*"):
                if path.suffix in (".png", ".wav"):
                    names.add(path.stem[len(prefix) + 1:])
    return sorted(names, key=lambda name: (name == "control", name))


def _digest(directory: Path, prefix: str, condition: str) -> str | None:
    for suffix in (".png", ".wav"):
        path = directory / f"{prefix}_{condition}{suffix}"
        if path.exists():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def _fmt(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = load_runs(args.run_dirs, args.labels)
    path = render(runs, Path(args.output))
    print(f"Compared {len(runs)} runs")
    for run in runs:
        print(f"  {run['label']:<12} {run['dir']}")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
