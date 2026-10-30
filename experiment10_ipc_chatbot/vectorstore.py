"""Embedding based vector store (step iii of the experiment).

Chunks are embedded with a sentence transformer and indexed with FAISS, which
runs in process and needs no server. Embeddings are L2 normalised at encode
time, so the squared L2 distance FAISS returns converts exactly to cosine
similarity:

    for unit vectors,  ||a - b||^2 = 2 - 2(a . b)
    therefore          cosine      = 1 - distance / 2

That conversion is done here rather than relying on the framework's own
relevance mapping, so the number the chatbot thresholds on has a definition
that can be stated in the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chunking import Chunk
from config import Config

INDEX_NAME = "statute"
MANIFEST_NAME = "index_manifest.json"


def build_embeddings(cfg: Config):
    from langchain_huggingface import HuggingFaceEmbeddings

    device = cfg.resolved_device()
    return HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(cfg: Config, chunks: list[Chunk], statute_stats: dict[str, Any]) -> Path:
    """Embed every chunk and write a FAISS index to disk."""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    if not chunks:
        raise ValueError("No chunks to index. Check that the document parsed correctly.")

    documents = [Document(page_content=c.text, metadata=c.metadata) for c in chunks]
    embeddings = build_embeddings(cfg)
    store = FAISS.from_documents(documents, embeddings)

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(cfg.index_dir), index_name=INDEX_NAME)

    manifest = {
        "config": cfg.to_dict(),
        "statute": statute_stats,
        "chunks": len(chunks),
        "embedding_model": cfg.embedding_model,
    }
    (cfg.index_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return cfg.index_dir


def load_index(cfg: Config):
    """Load a previously built index."""
    from langchain_community.vectorstores import FAISS

    if not (cfg.index_dir / f"{INDEX_NAME}.faiss").exists():
        raise FileNotFoundError(
            f"No index in {cfg.index_dir}. Run build_index.py first."
        )
    embeddings = build_embeddings(cfg)
    # The index is produced by this project and read back from local disk, so
    # deserialising it is safe. Never enable this for an index from elsewhere.
    return FAISS.load_local(
        str(cfg.index_dir),
        embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )


def read_manifest(cfg: Config) -> dict[str, Any]:
    path = cfg.index_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def search(store, query: str, top_k: int) -> list[dict[str, Any]]:
    """Return the top_k passages with a cosine similarity in [-1, 1]."""
    hits = store.similarity_search_with_score(query, k=top_k)
    results = []
    for document, distance in hits:
        results.append(
            {
                "text": document.page_content,
                "metadata": document.metadata,
                "similarity": round(1.0 - float(distance) / 2.0, 4),
                "citation": _citation(document.metadata),
            }
        )
    return results


def _citation(metadata: dict[str, Any]) -> str:
    number = metadata.get("section")
    if not number:
        return "Unattributed passage"
    parts = metadata.get("parts", 1)
    part = metadata.get("part", 1)
    suffix = f" (part {part} of {parts})" if parts and parts > 1 else ""
    return f"Section {number}{suffix}"
