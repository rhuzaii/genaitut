# Experiment 1: Multimodal Generative AI

CSL75 Skill Enhancement Laboratory, Generative AI.

Generate a poem, an image and a 15 to 30 second audio clip from a single theme
using three separate generative models, then measure and discuss how well the
three outputs align with one another.

Theme used for the submitted run:

> A glass bangle furnace at 4 a.m. in Firozabad, where workers draw molten
> glass from a coal fired furnace and shape it into bangles before sunrise.

## What the run actually tests

The three models share no state. The only thing connecting them is the prompt,
so the experiment treats prompting strategy as the independent variable and
compares two conditions plus a baseline.

| Condition | How the image and audio prompts are built |
| --- | --- |
| `direct` | The theme string is inserted into a modality specific template. |
| `derived` | The poem is generated first, a structured scene description is extracted from it, and the prompts are built from those fields. |
| `control` | An unrelated prompt, used to show what a low similarity score looks like. |

Hypothesis: the derived condition scores higher, because those prompts carry the
concrete detail the language model actually committed to rather than the more
abstract theme string.

## Pipeline

```
theme
  |
  +-- language model ------> poem
  |                            |
  |                            +-- scene extraction --> scene.json
  |                                                        |
  +-- direct prompts ------------------+                   |
                                       |   derived prompts -+
                                       v                    v
                              latent diffusion         MusicGen
                                       |                    |
                                    image.png            audio.wav
                                       |                    |
                                  CLIP score           CLAP score
                                       \                   /
                                        +--- report.md ---+
```

## Models

| Stage | Model | Notes |
| --- | --- | --- |
| Text | Cohere `command-r-plus-08-2024`, or `Qwen/Qwen2.5-1.5B-Instruct` locally | Autoregressive decoder only transformer. |
| Image | `sd2-community/stable-diffusion-2-1-base` | Latent diffusion: VAE, U-Net denoiser, CLIP text encoder. |
| Audio | `facebook/musicgen-small` | Transformer over discrete audio tokens, 50 tokens per second. |
| Text to image scoring | `openai/clip-vit-base-patch32` | Contrastive image and text encoder. |
| Text to audio scoring | `laion/clap-htsat-unfused` | Contrastive audio and text encoder. |

## Running it

### Google Colab, recommended

Open `experiment1_colab.ipynb`, set the runtime to a T4 GPU, upload this folder
to `/content/experiment1_multimodal`, and run the cells in order. A full run
takes roughly eight to twelve minutes.

### Locally

Needs Python 3.10 to 3.12 and, realistically, a CUDA GPU with 8 GB or more.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export COHERE_API_KEY=your_key_here
python run_experiment.py
```

Useful flags:

```
python run_experiment.py --theme "a salt pan at noon in the Rann of Kutch"
python run_experiment.py --text-backend local --no-control
python run_experiment.py --seed 42 --audio-seconds 20
python run_experiment.py --score-only
```

`--score-only` reuses the poem, images and audio already sitting in the output
directory and runs just the scoring and report stages. Useful when scoring
fails after a long generation run, since it avoids regenerating everything.

## Comparing two runs

Running the same theme and seed through two different language model backends
turns the exercise into a controlled comparison. Send each run to its own
output directory, then build the comparison document:

```
python run_experiment.py --text-backend local  --output outputs_local
python run_experiment.py --text-backend cohere --output outputs_cohere
python compare_runs.py outputs_local outputs_cohere --labels Qwen Cohere
```

`compare_runs.py` writes `comparison.md` containing the score tables side by
side, the poems and scene descriptions, and a SHA256 table showing which
artefacts were reproducible.

The direct and control prompts are built from constants, so with a fixed seed
their images and clips come out byte identical across runs. Only the derived
artefacts change, because only they depend on the poem. Where a score moves
even though the file is identical, the change is caused entirely by the text it
was scored against, which is direct evidence that CLIP and CLAP measure
alignment rather than artefact quality.

The theme is an input, never hardcoded into the prompts, so the same code can be
demonstrated live on any theme an examiner asks for.

## Outputs

```
outputs/
  poem.txt                    generated text
  scene.json                  structured scene extracted from the poem
  images/image_direct.png     one render per condition
  audio/audio_direct.wav      one clip per condition, 25 seconds, 16 bit PCM
  audio/audio_direct.mp3      same clip, written when ffmpeg is available
  figures/audio_direct.png    waveform and log mel spectrogram
  manifest.json               every prompt, setting, path and score
  report.md                   report scaffold with results filled in
```

`manifest.json` records the resolved configuration and the seed, so any run can
be reproduced or defended in a viva.

## Reading the scores

CLIP and CLAP both return a cosine similarity between two embeddings that were
trained to sit close together for matching pairs. Higher means better aligned.

The two encoders were trained separately on different data, so compare values
within a column, never across. The `control` row exists to make the comparison
meaningful: a score is only high or low relative to something.

CLIP truncates text at 77 tokens and a fourteen line poem exceeds that, so the
poem is split into chunks that fit, each chunk is embedded, and the normalised
embeddings are averaged. This is noted in the generated report because it
affects interpretation.

## Files

| File | Responsibility |
| --- | --- |
| `config.py` | Every tunable value, and validation of the 15 to 30 second rule. |
| `prompts.py` | Prompt templates for both conditions and the control. |
| `text_stage.py` | Poem generation and scene extraction, with three backends. |
| `image_stage.py` | Diffusion pipeline, one render per condition. |
| `audio_stage.py` | MusicGen, duration to token budget conversion. |
| `alignment.py` | CLIP and CLAP scoring, waveform and spectrogram figures. |
| `report.py` | Markdown report scaffold rendered from the run results. |
| `runtime.py` | Device description and GPU memory release between stages. |
| `run_experiment.py` | Command line entry point that sequences the stages. |
| `compare_runs.py` | Builds a comparison document from two or more runs. |

## What still has to be written by hand

The generated report leaves these blank on purpose:

- the 1 to 5 qualitative rubric for each modality pair
- the pairwise discussion of what matched and what did not
- whether the direct against derived hypothesis was supported
- strengths, limitations and ethical challenges
- the conclusion and references
