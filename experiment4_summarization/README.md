# Experiment 4: Summarizing Academic Reports

CSL75 Skill Enhancement Laboratory, Generative AI.

Use a pre-trained Hugging Face summarization model to turn a long academic
report into a concise summary, then measure how good that summary actually is.

## The evaluation idea

Most summarization exercises stop at "it produced a summary", which cannot be
graded objectively. This one uses a property of academic writing: **a paper
already contains a human written summary of itself, its abstract.**

So the pipeline holds the abstract out, summarizes only the body, and scores
the machine summary against the abstract with ROUGE. The model never sees the
text it is judged on, which makes the comparison honest and the result a
number rather than an opinion.

An extractive **lead-3 baseline**, the first three sentences of the body, is
scored the same way. Academic reports state their thesis early, so this is a
genuinely strong baseline. Any abstractive model that cannot beat it is not
earning its compute.

## Why map reduce

BART accepts 1024 tokens, roughly 750 words. A real report is longer, and
anything past the limit is silently truncated, so the end of the document would
never reach the model.

```
report body
    |
    +-- chunk on sentence boundaries, 900 tokens each, 2 sentence overlap
            |
            +-- MAP: summarize each chunk independently
                    |
                    +-- REDUCE: summarize the joined chunk summaries
                            |
                            +-- final summary
```

Without the reduce pass the output is just the chunk summaries concatenated,
which grows with document length. The reduce pass is what makes the output
short no matter how long the input was.

## Metrics

| Metric | What it measures | Reading it |
| --- | --- | --- |
| ROUGE-1 | Unigram overlap with the abstract | Content words in common |
| ROUGE-2 | Bigram overlap | Phrasing in common, much stricter |
| ROUGE-L | Longest common subsequence | In order overlap, rewards structure |
| Compression | Source words divided by summary words | How much shorter it got |
| Novel bigram ratio | Fraction of summary bigrams absent from the source | Near 0 means copying, higher means real rewriting |

ROUGE is implemented directly in `evaluate.py` rather than imported, so the
definition in use is visible and there is one less dependency. No stemming is
applied, which makes the values slightly conservative against implementations
that stem by default.

The novel bigram ratio matters because a model can score well on ROUGE by
copying sentences. Reporting both separates a genuinely abstractive model from
one that is quietly extracting.

## Models compared

| Key | Model | Notes |
| --- | --- | --- |
| `bart` | `facebook/bart-large-cnn` | The standard baseline, 1.6 GB |
| `distilbart` | `sshleifer/distilbart-cnn-12-6` | Distilled, roughly twice as fast |
| `pegasus` | `google/pegasus-cnn_dailymail` | More aggressive compression, 2.3 GB |

All three were trained on news, not academic prose. That domain mismatch is
itself worth reporting in the observations.

## Running it

### Google Colab

Open `experiment4_colab.ipynb`, set the runtime to a T4 GPU, and run the cells
in order. A full run with two models takes two to four minutes.

### Locally

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_summarizer.py
```

Useful flags:

```
python run_summarizer.py --document documents/my_paper.pdf
python run_summarizer.py --models bart distilbart pegasus
python run_summarizer.py --chunk-tokens 600 --overlap 1
python run_summarizer.py --no-baseline
```

`.txt`, `.md` and `.pdf` inputs are supported. PDF extraction repairs hyphens
broken across line endings and rejoins wrapped lines before chunking.

## Outputs

```
outputs/
  summaries/summary_bart.txt        one file per model
  summaries/summary_lead.txt        the extractive baseline
  manifest.json                     every setting, summary and metric
  report.md                         report scaffold with results filled in
```

## Files

| File | Responsibility |
| --- | --- |
| `config.py` | Model registry, chunk sizes, generation length bounds |
| `document.py` | Loading .txt, .md and .pdf, splitting abstract from body, sentence splitting |
| `chunking.py` | Token aware splitting on sentence boundaries with overlap |
| `summarize.py` | Pipeline loading, map reduce, lead sentence baseline |
| `evaluate.py` | ROUGE-1, ROUGE-2, ROUGE-L, compression, novel n-gram ratio |
| `report.py` | Markdown report scaffold rendered from the run |
| `run_summarizer.py` | Command line entry point |

## What still has to be written by hand

The generated report leaves the observations blank on purpose:

- which model won, and whether anything beat the lead-3 baseline
- whether the compression is actually usable
- what the novel bigram ratio says about each model
- anything the summary omitted or distorted
- the limits of ROUGE as the only measure
