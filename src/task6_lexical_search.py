"""
Task 6 - Lexical Search Module.

BM25 keyword retrieval over the same chunks used by Task 5. The corpus is loaded
from Task 4's local vector-store fallback when available; otherwise it is built
from standardized Markdown documents.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from .task4_chunking_indexing import LOCAL_INDEX_PATH, chunk_documents, load_documents


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _load_corpus() -> list[dict[str, Any]]:
    if LOCAL_INDEX_PATH.exists():
        index = json.loads(LOCAL_INDEX_PATH.read_text(encoding="utf-8"))
        return [
            {
                "content": chunk.get("content", ""),
                "metadata": dict(chunk.get("metadata") or {}),
            }
            for chunk in index.get("chunks", [])
            if chunk.get("content")
        ]

    return chunk_documents(load_documents())


@lru_cache(maxsize=1)
def _get_bm25_resources():
    from rank_bm25 import BM25Okapi

    corpus = _load_corpus()
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    return corpus, bm25


def build_bm25_index(corpus: list[dict]):
    """
    Build a BM25 index from a corpus of {'content': str, 'metadata': dict}.
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Return BM25 retrieval results sorted by score descending.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    if top_k <= 0 or not query.strip():
        return []

    corpus, bm25 = _get_bm25_resources()
    if not corpus or bm25 is None:
        return []

    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    results: list[dict] = []
    for index in ranked_indices[:top_k]:
        score = float(scores[index])
        if score <= 0:
            continue

        doc = corpus[index]
        results.append(
            {
                "content": doc["content"],
                "score": score,
                "metadata": dict(doc.get("metadata") or {}),
            }
        )

    return results


if __name__ == "__main__":
    for result in lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
