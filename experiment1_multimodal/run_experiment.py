"""Run the full multimodal generation and alignment experiment.

Usage:
    python run_experiment.py
    python run_experiment.py --theme "a salt pan at noon" --seed 7
    python run_experiment.py --text-backend local --no-control
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import alignment
import audio_stage
import image_stage
import prompts
import report
import text_stage
from config import DEFAULT_THEME, Config
from runtime import describe_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aligned text, image and audio from one theme."
    )
    parser.add_argument("--theme", default=DEFAULT_THEME, help="theme to generate from")
    parser.add_argument("--seed", type=int, default=1729, help="random seed")
    parser.add_argument("--output", default="outputs", help="output directory")
    parser.add_argument(
        "--text-backend",
        default="auto",
        choices=["auto", "cohere", "local", "manual"],
        help="which language model to use for the poem and scene extraction",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument(
        "--audio-seconds", type=int, default=25, help="clip length, 15 to 30"
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="skip the deliberately unrelated baseline condition",
    )
    parser.add_argument(
        "--no-derived",
        action="store_true",
        help="only run the direct condition, useful for a quick check",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help=(
            "reuse the text, images and audio already in the output directory "
            "and run only the scoring and report stages"
        ),
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(
        theme=args.theme,
        seed=args.seed,
        device=args.device,
        output_dir=Path(args.output),
        run_control=not args.no_control,
    )
    cfg.text.backend = args.text_backend
    cfg.audio.seconds = args.audio_seconds
    cfg.__post_init__()
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.resolved_device()
    started = time.time()

    print("Experiment 1: multimodal generative AI")
    print(f"  theme  : {cfg.theme}")
    print(f"  device : {describe_device(device)}")
    print(f"  output : {cfg.output_dir.resolve()}")
    print()

    if args.score_only:
        poem, scene, backend = _load_existing_text(cfg)
        print("Stages 1 to 3 skipped, reusing the existing outputs")
        print(f"  poem  : {len(poem.splitlines())} lines")
    else:
        print("Stage 1 of 4: text")
        poem, _, backend = text_stage.generate_poem(cfg)
        (cfg.output_dir / "poem.txt").write_text(poem + "\n", encoding="utf-8")
        print(f"  backend {backend}, {len(poem.splitlines())} lines")

        scene, _ = text_stage.extract_scene(cfg, poem)
        (cfg.output_dir / "scene.json").write_text(
            json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("  scene description extracted")
    print()

    poem_prompt_text = prompts.poem_prompt(cfg.theme, cfg.text.poem_lines)
    scene_prompt_text = prompts.scene_extraction_prompt(poem)

    image_prompts = {"direct": prompts.direct_image_prompt(cfg.theme)}
    audio_prompts = {"direct": prompts.direct_audio_prompt(cfg.theme)}
    if not args.no_derived:
        image_prompts["derived"] = prompts.derived_image_prompt(scene)
        audio_prompts["derived"] = prompts.derived_audio_prompt(scene)
    if cfg.run_control:
        image_prompts["control"] = prompts.CONTROL_IMAGE_PROMPT
        audio_prompts["control"] = prompts.CONTROL_AUDIO_PROMPT

    if args.score_only:
        image_paths = _find_existing(cfg.output_dir / "images", "image_", ".png")
        audio_paths = _find_existing(cfg.output_dir / "audio", "audio_", ".wav")
        image_prompts = {k: v for k, v in image_prompts.items() if k in image_paths}
        audio_prompts = {k: v for k, v in audio_prompts.items() if k in audio_paths}
        print(f"Reusing {len(image_paths)} image(s) and {len(audio_paths)} clip(s)")
        print()
    else:
        print(f"Stage 2 of 4: image ({len(image_prompts)} renders)")
        image_paths = image_stage.generate_images(cfg, image_prompts)
        print()

        print(f"Stage 3 of 4: audio ({len(audio_prompts)} clips)")
        audio_paths = audio_stage.generate_audio(cfg, audio_prompts)
        print()

    print("Stage 4 of 4: alignment scoring")
    clip = alignment.clip_scores(cfg, poem, image_paths)
    clap = alignment.clap_scores(cfg, poem, audio_paths)
    for condition, value in clip.items():
        print(f"  CLIP {condition:<8} {value:.4f}")
    for condition, value in clap.items():
        print(f"  CLAP {condition:<8} {value:.4f}")

    figure_dir = cfg.output_dir / "figures"
    audio_figures = {
        condition: alignment.plot_audio(path, figure_dir / f"{path.stem}.png")
        for condition, path in audio_paths.items()
    }
    print()

    prompt_record = {
        "text: poem": poem_prompt_text,
        "text: scene extraction": scene_prompt_text,
        **{f"image: {name}": value for name, value in image_prompts.items()},
        **{f"audio: {name}": value for name, value in audio_prompts.items()},
    }

    results: dict[str, Any] = {
        "theme": cfg.theme,
        "device": describe_device(device),
        "text_backend": backend,
        "text_model": (
            cfg.text.cohere_model if backend == "cohere" else cfg.text.local_model
        ),
        "config": cfg.to_dict(),
        "prompts": prompt_record,
        "poem": poem,
        "scene": scene,
        "images": {name: str(path) for name, path in image_paths.items()},
        "audio": {name: str(path) for name, path in audio_paths.items()},
        "audio_figures": {name: str(path) for name, path in audio_figures.items()},
        "clip": clip,
        "clap": clap,
        "clip_chunks": _clip_chunk_count(cfg, poem),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    manifest_path = cfg.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results["images"] = {name: Path(path) for name, path in results["images"].items()}
    results["audio"] = {name: Path(path) for name, path in results["audio"].items()}
    results["audio_figures"] = {
        name: Path(path) for name, path in results["audio_figures"].items()
    }
    report_path = report.render(results, cfg.output_dir / "report.md")

    print(f"Manifest : {manifest_path}")
    print(f"Report   : {report_path}")
    print(f"Elapsed  : {results['elapsed_seconds']} s")
    return 0


def _load_existing_text(cfg: Config) -> tuple[str, dict[str, Any], str]:
    """Read the poem and scene written by an earlier run of this experiment."""
    poem_path = cfg.output_dir / "poem.txt"
    scene_path = cfg.output_dir / "scene.json"
    if not poem_path.exists() or not scene_path.exists():
        raise FileNotFoundError(
            f"--score-only needs {poem_path} and {scene_path} from an earlier "
            "run. Run without the flag first."
        )

    manifest_path = cfg.output_dir / "manifest.json"
    backend = "unknown"
    if manifest_path.exists():
        try:
            backend = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "text_backend", "unknown"
            )
        except json.JSONDecodeError:
            pass

    poem = poem_path.read_text(encoding="utf-8").strip()
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    return poem, scene, backend


def _find_existing(directory: Path, prefix: str, suffix: str) -> dict[str, Path]:
    """Map condition name to file for artefacts an earlier run left behind."""
    if not directory.exists():
        raise FileNotFoundError(
            f"--score-only expected generated files in {directory}."
        )
    found = {
        path.stem[len(prefix):]: path
        for path in sorted(directory.glob(f"{prefix}*{suffix}"))
    }
    if not found:
        raise FileNotFoundError(f"No {prefix}*{suffix} files found in {directory}.")
    return found


def _clip_chunk_count(cfg: Config, poem: str) -> int:
    try:
        from transformers import CLIPProcessor

        processor = CLIPProcessor.from_pretrained(cfg.scoring.clip_model)
        return len(
            alignment._chunk_text(poem, processor.tokenizer, cfg.scoring.clip_chunk_tokens)
        )
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
