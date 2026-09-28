"""Render a Markdown report scaffold from the results of a summarization run.

Numbers, summaries and settings are filled in automatically. The discussion
sections are left blank, since those are the parts that have to be written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render(results: dict[str, Any], output_path: Path) -> Path:
    lines: list[str] = []
    add = lines.append

    add("# Experiment 4: Summarizing Academic Reports with a Pre-trained Model")
    add("")
    add("## Aim")
    add(
        "To summarize a long academic report into a concise summary using a "
        "pre-trained Hugging Face summarization model, and to evaluate the "
        "result against the report's own abstract as a human written reference."
    )
    add("")

    add("## Document")
    add("")
    add("| Property | Value |")
    add("| --- | --- |")
    add(f"| File | `{results['document']['path']}` |")
    add(f"| Title | {results['document']['title']} |")
    add(f"| Body words | {results['document']['stats']['body_words']} |")
    add(f"| Body sentences | {results['document']['stats']['body_sentences']} |")
    add(f"| Abstract words | {results['document']['stats']['abstract_words']} |")
    add("")
    add(
        "The abstract is held out of the model input and used only as the "
        "reference summary, so the model is never shown the text it is scored "
        "against."
    )
    add("")

    add("## Configuration")
    add("")
    add("| Setting | Value |")
    add("| --- | --- |")
    add(f"| Device | {results['device']} |")
    add(f"| Seed | {results['config']['seed']} |")
    add(f"| Chunk size | {results['config']['chunking']['max_tokens']} tokens |")
    add(f"| Chunk overlap | {results['config']['chunking']['overlap_sentences']} sentences |")
    add(f"| Map pass length | {results['config']['summary']['map_min_tokens']} to {results['config']['summary']['map_max_tokens']} tokens |")
    add(f"| Reduce pass length | {results['config']['summary']['reduce_min_tokens']} to {results['config']['summary']['reduce_max_tokens']} tokens |")
    add(f"| Beams | {results['config']['summary']['num_beams']} |")
    add("")

    add("## Method")
    add("")
    add(
        "The report is longer than the model's 1024 token input window, so "
        "summarization runs as a map reduce."
    )
    add("")
    add("1. **Load and split.** Read the report, separate the abstract from the body.")
    add("2. **Chunk.** Divide the body on sentence boundaries into pieces that fit the window, with a small overlap.")
    add("3. **Map.** Summarize each chunk independently.")
    add("4. **Reduce.** Concatenate the chunk summaries and summarize again, so the output length does not grow with document length.")
    add("5. **Baseline.** Take the first few sentences of the body as an extractive comparison.")
    add("6. **Score.** Compare every summary against the abstract with ROUGE, and measure compression and novel n-gram ratio.")
    add("")

    add("## Results")
    add("")
    add("### Chunking")
    add("")
    add("| Chunk | Words | Tokens |")
    add("| --- | --- | --- |")
    for entry in results["chunk_stats"]:
        add(f"| {entry['index']} | {entry['words']} | {entry['tokens']} |")
    add("")

    add("### Reference abstract")
    add("")
    add("```")
    add(results["reference"].strip() or "(no abstract found in the document)")
    add("```")
    add("")

    for name, item in results["summaries"].items():
        add(f"### Summary: {item['label']}")
        add("")
        if item.get("model_id"):
            add(f"Model: `{item['model_id']}`")
            add("")
        add("```")
        add(item["summary"].strip())
        add("```")
        add("")
        if item.get("chunk_summaries") and len(item["chunk_summaries"]) > 1:
            add(f"Intermediate chunk summaries before the reduce pass: {len(item['chunk_summaries'])}")
            add("")

    add("### Metrics")
    add("")
    add(
        "ROUGE F1 against the abstract. Compression is source words divided by "
        "summary words. Novel bigram ratio is the fraction of summary bigrams "
        "absent from the source, so higher means more rewriting and less copying."
    )
    add("")
    add("| Summary | ROUGE-1 | ROUGE-2 | ROUGE-L | Words | Compression | Novel bigrams | Time (s) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, item in results["summaries"].items():
        metrics = item["metrics"]
        add(
            f"| {item['label']} "
            f"| {_f1(metrics, 'rouge1')} | {_f1(metrics, 'rouge2')} | {_f1(metrics, 'rougeL')} "
            f"| {metrics['summary_words']} "
            f"| {metrics['compression_ratio']}x "
            f"| {metrics['novel_bigram_ratio']} "
            f"| {item.get('elapsed_seconds', 'n/a')} |"
        )
    add("")

    add("## Observations")
    add("")
    for prompt in (
        "Which model scored highest, and did any model beat the lead sentence baseline?",
        "How much was the document compressed, and is the summary actually usable?",
        "What does the novel bigram ratio say about how abstractive each model is?",
        "Did the summary omit or distort anything important from the report?",
        "What are the limits of ROUGE as the only measure here?",
    ):
        add(f"**{prompt}**")
        add("")
        add("_Write the answer here._")
        add("")

    add("## Result")
    add("")
    add("_One or two sentences confirming the aim was met._")
    add("")

    add("## References")
    add("")
    add("_List the models and libraries used._")
    add("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _f1(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    return f"{value['f1']:.4f}" if value else "not measured"
