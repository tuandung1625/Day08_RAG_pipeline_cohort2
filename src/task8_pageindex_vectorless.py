"""
Task 8 - PageIndex Vectorless RAG.

PageIndex builds a tree index over uploaded PDF documents and retrieves relevant
sections through reasoning-based tree search. The SDK currently exposes
`PageIndexClient`, `submit_document`, `get_document`, `is_retrieval_ready`,
`submit_query`, and `get_retrieval`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PROJECT_DIR = Path(__file__).resolve().parent.parent
LEGAL_LANDING_DIR = PROJECT_DIR / "data" / "landing" / "legal"
PAGEINDEX_CACHE_PATH = PROJECT_DIR / "data" / "pageindex_documents.json"

DEFAULT_POLL_SECONDS = 5
DEFAULT_WAIT_SECONDS = int(os.getenv("PAGEINDEX_WAIT_SECONDS", "0"))


def _client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not set in .env")

    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_cache() -> dict[str, Any]:
    if not PAGEINDEX_CACHE_PATH.exists():
        return {"documents": []}
    return json.loads(PAGEINDEX_CACHE_PATH.read_text(encoding="utf-8"))


def _save_cache(cache: dict[str, Any]) -> None:
    PAGEINDEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _pdf_files() -> list[Path]:
    return sorted(
        path
        for path in LEGAL_LANDING_DIR.glob("*.pdf")
        if path.is_file() and path.stat().st_size > 0
    )


def _cached_by_path(cache: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["path"]: item
        for item in cache.get("documents", [])
        if item.get("path") and item.get("doc_id")
    }


def _refresh_status(client, document: dict[str, Any]) -> dict[str, Any]:
    try:
        metadata = client.get_document(document["doc_id"])
        document["status"] = metadata.get("status", document.get("status"))
        document["page_count"] = metadata.get("pageNum", document.get("page_count"))
        document["retrieval_ready"] = client.is_retrieval_ready(document["doc_id"])
        document.pop("last_error", None)
    except Exception as exc:
        document["last_error"] = str(exc)
    return document


def upload_documents(wait: bool = False, max_wait_seconds: int = 300) -> list[dict]:
    """
    Upload legal PDFs to PageIndex and cache their doc_ids.

    Args:
        wait: If True, poll until documents are retrieval-ready or timeout.
        max_wait_seconds: Maximum total wait time when wait=True.
    """
    client = _client()
    cache = _load_cache()
    by_path = _cached_by_path(cache)
    documents = list(by_path.values())

    for pdf_path in _pdf_files():
        cache_key = str(pdf_path.relative_to(PROJECT_DIR)).replace("\\", "/")
        if cache_key in by_path:
            continue

        result = client.submit_document(str(pdf_path))
        document = {
            "doc_id": result["doc_id"],
            "name": pdf_path.name,
            "path": cache_key,
            "status": "submitted",
            "retrieval_ready": False,
        }
        documents.append(document)
        by_path[cache_key] = document

    cache["documents"] = documents
    _save_cache(cache)

    deadline = time.time() + max_wait_seconds
    while wait and time.time() < deadline:
        cache["documents"] = [_refresh_status(client, doc) for doc in cache["documents"]]
        _save_cache(cache)
        if cache["documents"] and all(doc.get("retrieval_ready") for doc in cache["documents"]):
            break
        time.sleep(DEFAULT_POLL_SECONDS)

    if not wait:
        cache["documents"] = [_refresh_status(client, doc) for doc in cache["documents"]]
        _save_cache(cache)

    return cache["documents"]


def _ready_documents() -> list[dict[str, Any]]:
    cache = _load_cache()
    documents = cache.get("documents", [])
    if not documents:
        documents = upload_documents(wait=False)
    else:
        client = _client()
        documents = [_refresh_status(client, doc) for doc in documents]
        cache["documents"] = documents
        _save_cache(cache)

    return [doc for doc in documents if doc.get("retrieval_ready")]


def _poll_retrieval(client, retrieval_id: str, max_wait_seconds: int) -> dict[str, Any]:
    deadline = time.time() + max_wait_seconds
    last_result: dict[str, Any] = {}

    while time.time() < deadline:
        last_result = client.get_retrieval(retrieval_id)
        if last_result.get("status") == "completed":
            return last_result
        if last_result.get("status") == "failed":
            return last_result
        time.sleep(DEFAULT_POLL_SECONDS)

    return last_result


def _extract_results(
    retrieval_result: dict[str, Any],
    document: dict[str, Any],
    top_k: int,
) -> list[dict]:
    extracted: list[dict] = []
    nodes = retrieval_result.get("retrieved_nodes") or []

    def iter_content_items(value: Any):
        if isinstance(value, dict):
            yield value
        elif isinstance(value, list):
            for child in value:
                yield from iter_content_items(child)

    for node_rank, node in enumerate(nodes, start=1):
        title = node.get("title") or node.get("node_id") or f"Node {node_rank}"
        contents = list(iter_content_items(node.get("relevant_contents") or []))
        for content_rank, content_item in enumerate(contents, start=1):
            text = (
                content_item.get("relevant_content")
                or content_item.get("content")
                or content_item.get("text")
                or ""
            ).strip()
            if not text:
                continue

            extracted.append(
                {
                    "content": text,
                    "score": 1.0 / (node_rank + content_rank - 1),
                    "metadata": {
                        "source": document.get("name"),
                        "path": document.get("path"),
                        "doc_id": document.get("doc_id"),
                        "title": title,
                        "page_index": content_item.get("page_index"),
                        "type": "pageindex",
                    },
                    "source": "pageindex",
                }
            )

    return extracted[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    if top_k <= 0 or not query.strip():
        return []

    ready_documents = _ready_documents()
    if not ready_documents:
        return []

    client = _client()
    results: list[dict] = []
    per_doc_k = max(top_k, 3)

    for document in ready_documents:
        retrieval = client.submit_query(
            doc_id=document["doc_id"],
            query=query,
            thinking=False,
        )
        retrieval_result = _poll_retrieval(
            client,
            retrieval["retrieval_id"],
            max_wait_seconds=max(DEFAULT_WAIT_SECONDS, 30),
        )
        results.extend(_extract_results(retrieval_result, document, per_doc_k))

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("Set PAGEINDEX_API_KEY in .env first.")
    else:
        docs = upload_documents(wait=True, max_wait_seconds=300)
        print(f"Tracked {len(docs)} PageIndex documents")
        for doc in docs:
            print(f"- {doc.get('name')}: {doc.get('status')} ready={doc.get('retrieval_ready')}")

        for result in pageindex_search("hinh phat su dung ma tuy", top_k=3):
            print(f"[{result['score']:.3f}] {result['content'][:120]}...")
