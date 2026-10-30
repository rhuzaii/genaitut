# Experiment 10: Retrieval Augmented Chatbot over the Indian Penal Code

CSL75 Skill Enhancement Laboratory, Generative AI.

A chatbot that answers questions about a statute by retrieving the governing
provisions and answering only from them, citing section numbers so every
answer can be checked.

The five steps the exercise asks for map onto the modules directly:

| Step | Requirement | Module |
| --- | --- | --- |
| i | Load and preprocess the document, extracting text from PDF | `ingest.py` |
| ii | Split into smaller chunks for efficient retrieval | `chunking.py` |
| iii | Create an embedding based vector store for semantic search | `vectorstore.py` |
| iv | Integrate a retrieval based QA pipeline with a language model | `chatbot.py` |
| v | Deploy a chatbot interface | `app.py` |

## Getting the document

The IPC is not in this repository. Download it and save it as
`documents/ipc.pdf`. See `documents/README.md` for the official source.

Until you do, everything runs against `documents/sample_statute.txt`, a short
fictional statute written to test the pipeline. **It is not law.** It exists so
the code can be verified end to end before the real document is available.

## The design decision that matters

A statute is not prose. It is a numbered hierarchy, and that structure is the
most useful thing in the document for retrieval.

A generic character splitter throws it away, and will cheerfully cut between
an offence and its punishment, or between a rule and its proviso. Retrieval
then returns a passage that is confidently wrong because the qualifying half
is missing.

This project splits on section boundaries instead, and keeps the section
number on every chunk, which is what allows an answer to cite a source the
reader can verify. Both strategies are implemented so the report can measure
the difference rather than assert it.

Measured on the 18 annotated questions in `eval_questions.json`:

| Strategy | Chunks | Mean words | recall@1 | recall@3 | MRR |
| --- | --- | --- | --- | --- | --- |
| section | 20 | 43.2 | 0.944 | 1.000 | 0.972 |
| fixed | 4 | 252.2 | 0.611 | 0.944 | 0.764 |

A passage counts as a hit when its section label matches **or** when it
contains the governing section's heading. The second test matters: fixed
chunks have no reliable section label, so scoring on labels alone would
measure the labelling rather than the retrieval. Without that correction the
fixed strategy appeared to score 0.333, which was an artifact.

## What the real IPC broke

The parser was written against a clean statute and then run on the official
Indian Penal Code PDF, 119 pages and 84,000 words. Three things went wrong,
all of them specific to how Indian bare Acts are typeset, and all of them
silent failures that produce a working but wrong index.

**The arrangement of sections.** The document opens with a contents list that
repeats every heading with no body. Each section therefore parsed twice, once
empty and once real, and the Code came out as 1435 sections instead of roughly
500. Section 302 resolved to the empty contents entry.

**Amended provisions carry a footnote marker.** A substituted section is
printed as `2[375. Rape.-A man is said to commit ...`, so the number is not at
the start of the line. Requiring it there silently dropped 13 sections,
including 34 (common intention) and 375 (rape), two of the most cited
provisions in the Code.

**Amendment footnotes look exactly like section headers.** A page foot carries
`1. Ins. by Act 3 of 1951, s. 3`, which has the same shape as a section header
and wins consolidation because a page can carry several. Section 1 resolved to
1500 words of amendment history.

After fixing all three: 581 sections, 24 chapters, and of sections 1 to 511
only four are absent, two of which (16 and 17) are genuinely repealed.

The general lesson is worth stating in the report. None of these raised an
error. Each produced an index that looked fine and answered confidently from
the wrong text, which is exactly the failure mode retrieval augmentation is
supposed to prevent. Verifying that a known section resolves to its known text
is a necessary step, not a nicety.

## Failing closed

A legal lookup tool that invents a provision is worse than one that says it
does not know, so the pipeline declines rather than guesses:

1. Retrieve the top k passages by cosine similarity.
2. If the best passage scores below `min_score`, decline and **never call the
   language model at all**.
3. Otherwise answer strictly from the passages, citing section numbers.
4. Return the passages alongside the answer so the reader can check them.

Cosine similarity is computed explicitly rather than taken from the
framework's relevance mapping. Embeddings are L2 normalised at encode time, so
for unit vectors `||a - b||^2 = 2 - 2(a . b)`, therefore
`cosine = 1 - distance / 2`.

On the five out of domain questions in the evaluation set, four are declined.
The one that gets through is "What is the punishment for murder?", which
scores 0.361 against the fictional test statute because it is semantically
close to that document's punishment provisions without being in it. The
threshold cannot separate "topically similar but absent" from "present". With
the real IPC indexed this particular question would be correctly answerable
from Section 302, so re-run the refusal evaluation after indexing the IPC and
report what you actually observe.

## Conversational memory

A follow up like "Does consent matter?" retrieves nothing useful on its own.
Before retrieval, the question is rewritten against recent history into a
standalone question. With a language model available this is an explicit
rewrite step; without one it falls back to concatenating the previous
question, which is crude but better than nothing.

## Running it

### Locally

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python build_index.py --document documents/ipc.pdf
export COHERE_API_KEY=your_key_here
python ask.py --question "What is the punishment for criminal breach of trust?"
python app.py
```

### Google Colab

Open `experiment10_colab.ipynb`, set the runtime to a T4 GPU, and run the
cells. The Gradio cell gives a public link you can open in another tab.

### Useful flags

```
python build_index.py --strategy fixed --index-dir index_fixed
python ask.py                              # interactive, keeps history
python ask.py --backend extractive         # no language model at all
python ask.py --evaluate                   # retrieval and refusal metrics
python ask.py --compare-chunking           # section against fixed
python app.py --share                      # public link, needed in Colab
```

## Generator backends

| Backend | Model | Notes |
| --- | --- | --- |
| `cohere` | `command-r-plus-08-2024` | Best answers. Needs `COHERE_API_KEY`. |
| `local` | `Qwen/Qwen2.5-1.5B-Instruct` | No key needed, about 3 GB |
| `extractive` | none | Returns the retrieved passages verbatim |

`extractive` is not a toy. It is the baseline the generative backends are
measured against: if a generated answer is no more useful than the passages
themselves, the language model added nothing but latency and risk.

## Files

| File | Responsibility |
| --- | --- |
| `config.py` | Model registry, chunk sizes, retrieval threshold |
| `ingest.py` | PDF and text loading, chapter and section parsing |
| `chunking.py` | Section aware and fixed size splitting |
| `vectorstore.py` | Embeddings, FAISS index, cosine similarity search |
| `chatbot.py` | Grounded prompt, refusal gate, question condensing |
| `evaluate.py` | recall@k, MRR, refusal rate, chunking comparison |
| `build_index.py` | CLI for steps i to iii |
| `ask.py` | CLI for step iv, plus the evaluation modes |
| `app.py` | Gradio interface, step v |

## A note on currency

The Bharatiya Nyaya Sanhita, 2023 replaced the IPC for offences committed on
or after 1 July 2024. The exercise names the IPC, so index the IPC, but state
in your report which document you indexed. The chatbot can only ever be as
current as the text it was given, which is itself one of the arguments for
retrieval augmentation over a model's parametric memory.

## What still has to be written by hand

- your own annotated question set for the IPC, in `eval_questions.json`
- the observations: what recall you measured, where retrieval failed, whether
  the refusal threshold held, and what the generated answers got wrong
