"""
Task 9 - Complete Retrieval Pipeline.

Pipeline:
    1. Run semantic search and lexical search.
    2. Merge both ranked lists with Reciprocal Rank Fusion (RRF).
    3. Rerank merged results with the local Task 7 reranker.
    4. If hybrid confidence is too low, try PageIndex fallback.
    5. Return normalized results.
"""

from __future__ import annotations

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _normalize_results(results: list[dict], source: str) -> list[dict]:
    normalized: list[dict] = []
    for item in results:
        normalized.append(
            {
                "content": item.get("content", ""),
                "score": float(item.get("score") or 0.0),
                "metadata": dict(item.get("metadata") or {}),
                "source": item.get("source", source),
            }
        )
    return normalized


def _fallback_pageindex(query: str, top_k: int) -> list[dict]:
    try:
        return _normalize_results(pageindex_search(query, top_k=top_k), "pageindex")
    except Exception:
        # PageIndex requires external account/API setup. Returning an empty list
        # keeps the pipeline callable until Task 8 is configured.
        return []


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Complete retrieval pipeline with PageIndex fallback.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': str}
    """
    if top_k <= 0 or not query.strip():
        return []

    search_k = max(top_k * 3, top_k)
    dense_results = semantic_search(query, top_k=search_k)
    sparse_results = lexical_search(query, top_k=search_k)

    merged = rerank_rrf([dense_results, sparse_results], top_k=search_k)
    merged = _normalize_results(merged, "hybrid")

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = _normalize_results(final_results, "hybrid")
    else:
        final_results = merged[:top_k]

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        fallback_results = _fallback_pageindex(query, top_k)
        if fallback_results:
            return fallback_results[:top_k]
        return final_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    for question in [
        "Hinh phat cho toi tang tru trai phep chat ma tuy",
        "Luat phong chong ma tuy quy dinh gi ve cai nghien",
    ]:
        print(f"\nQuery: {question}")
        for index, result in enumerate(retrieve(question, top_k=3), start=1):
            print(f"{index}. [{result['score']:.3f}] [{result['source']}] {result['content'][:90]}...")
