"""
Task 7 - Reranking Module.

Chosen approach for the default `rerank()`:
    Lightweight local relevance reranker.

Why:
    It runs offline, does not require a Jina/Qwen API key, and is enough for the
    automated tests. It re-scores each candidate by combining token overlap with
    the original retrieval score. The module also implements MMR and RRF because
    Task 9 uses RRF to merge dense and lexical retrieval results.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _lexical_relevance(query: str, document: str) -> float:
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(document)
    if not query_tokens or not doc_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    doc_counts = Counter(doc_tokens)
    overlap = sum(min(count, doc_counts[token]) for token, count in query_counts.items())
    recall = overlap / max(sum(query_counts.values()), 1)
    precision = overlap / max(sum(doc_counts.values()), 1)
    phrase_bonus = 0.15 if query.lower() in document.lower() else 0.0
    return min(1.0, (0.75 * recall) + (0.25 * precision) + phrase_bonus)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Re-score candidates with a local cross-encoder-style relevance function.

    This is not a neural cross-encoder; it is the offline fallback selected for
    this assignment. It preserves candidate metadata and replaces `score` with a
    normalized rerank score sorted descending.
    """
    if top_k <= 0 or not candidates:
        return []

    scored: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        original_score = float(candidate.get("score") or 0.0)
        lexical_score = _lexical_relevance(query, str(candidate.get("content", "")))

        # Keep retrieval signal as a tie-breaker while letting query relevance lead.
        rerank_score = (0.85 * lexical_score) + (0.15 * max(original_score, 0.0))
        item["score"] = float(rerank_score)
        item["metadata"] = dict(candidate.get("metadata") or {})
        item["metadata"]["rerank_method"] = "local_lexical"
        item["metadata"]["original_rank"] = rank
        item["metadata"]["original_score"] = original_score
        scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance: select relevant but less-duplicative candidates.

    MMR = lambda * sim(query, doc) - (1 - lambda) * max(sim(doc, selected_docs))
    """
    if top_k <= 0 or not candidates:
        return []

    selected: list[int] = []
    remaining = set(range(len(candidates)))
    limit = min(top_k, len(candidates))

    for _ in range(limit):
        best_idx: int | None = None
        best_score = float("-inf")

        for idx in remaining:
            candidate_embedding = candidates[idx].get("embedding") or []
            relevance = _cosine_similarity(query_embedding, candidate_embedding)

            max_similarity_to_selected = 0.0
            for selected_idx in selected:
                selected_embedding = candidates[selected_idx].get("embedding") or []
                similarity = _cosine_similarity(candidate_embedding, selected_embedding)
                max_similarity_to_selected = max(max_similarity_to_selected, similarity)

            mmr_score = (
                lambda_param * relevance
                - (1.0 - lambda_param) * max_similarity_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    results: list[dict] = []
    for idx in selected:
        item = dict(candidates[idx])
        item["score"] = float(item.get("score") or 0.0)
        item["metadata"] = dict(item.get("metadata") or {})
        item["metadata"]["rerank_method"] = "mmr"
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion for merging results from multiple rankers.

    RRF(d) = sum(1 / (k + rank_r(d)))
    """
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            content = str(item.get("content", ""))
            if not content:
                continue

            scores[content] = scores.get(content, 0.0) + (1.0 / (k + rank))
            content_map.setdefault(content, item)

    fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    results: list[dict] = []
    for content, score in fused[:top_k]:
        item = dict(content_map[content])
        item["score"] = float(score)
        item["metadata"] = dict(item.get("metadata") or {})
        item["metadata"]["rerank_method"] = "rrf"
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Re-score and re-order candidates based on relevance to query.
    """
    if method in {"cross_encoder", "local", "lexical"}:
        return rerank_cross_encoder(query, candidates, top_k)

    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)

    if method == "mmr":
        raise ValueError("Use rerank_mmr(query_embedding, candidates, ...) for MMR.")

    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Dieu 248: Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si bi bat vi su dung ma tuy", "score": 0.7, "metadata": {}},
        {"content": "Python programming", "score": 0.4, "metadata": {}},
    ]
    for result in rerank("hinh phat ma tuy", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
