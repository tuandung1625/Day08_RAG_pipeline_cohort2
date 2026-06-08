"""
Task 5 - Semantic Search Module.

Dense retrieval over the vector store created in Task 4.

The Task 4 pipeline tries Weaviate first, then writes a complete local JSON
index when Weaviate is unavailable. This module supports that classroom-friendly
fallback and keeps the same result shape for both stores.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

from .task4_chunking_indexing import (
    EMBEDDING_MODEL,
    LOCAL_INDEX_PATH,
    WEAVIATE_COLLECTION,
    _hash_embedding,
)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


@lru_cache(maxsize=1)
def _load_local_index() -> dict[str, Any]:
    if not LOCAL_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Local vector index not found: {LOCAL_INDEX_PATH}. Run Task 4 first."
        )
    return json.loads(LOCAL_INDEX_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _embed_query(query: str, backend: str | None = None) -> list[float]:
    if backend == "hashing-fallback":
        return _hash_embedding(query)

    try:
        model = _load_sentence_transformer()
        embedding = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
    except Exception:
        return _hash_embedding(query)


def _semantic_search_local(query: str, top_k: int) -> list[dict]:
    index = _load_local_index()
    chunks = index.get("chunks", [])
    backend = index.get("embedding_backend")
    query_embedding = _embed_query(query, backend=backend)

    results: list[dict] = []
    for chunk in chunks:
        chunk_embedding = chunk.get("embedding") or []
        score = _cosine_similarity(query_embedding, chunk_embedding)
        results.append(
            {
                "content": chunk.get("content", ""),
                "score": float(score),
                "metadata": dict(chunk.get("metadata") or {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _semantic_search_weaviate(query: str, top_k: int) -> list[dict]:
    import weaviate
    from weaviate.classes.query import MetadataQuery

    query_embedding = _embed_query(query)
    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get(WEAVIATE_COLLECTION)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results: list[dict] = []
        for obj in response.objects:
            properties = obj.properties or {}
            distance = obj.metadata.distance
            score = 1.0 - float(distance) if distance is not None else 0.0
            results.append(
                {
                    "content": properties.get("content", ""),
                    "score": score,
                    "metadata": {
                        "source": properties.get("source"),
                        "path": properties.get("path"),
                        "type": properties.get("doc_type"),
                        "chunk_index": properties.get("chunk_index"),
                    },
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results
    finally:
        client.close()


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Return dense retrieval results sorted by score descending.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    if top_k <= 0 or not query.strip():
        return []

    if LOCAL_INDEX_PATH.exists():
        return _semantic_search_local(query, top_k)

    try:
        return _semantic_search_weaviate(query, top_k)
    except Exception:
        return []


if __name__ == "__main__":
    for result in semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
