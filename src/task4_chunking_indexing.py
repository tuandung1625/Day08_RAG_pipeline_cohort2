"""
Task 4 - Chunk Markdown documents, embed chunks, and index them.

Chosen chunking strategy:
    RecursiveCharacterTextSplitter from langchain-text-splitters.
    It is the safest default for mixed legal PDFs and crawled news because the
    documents do not all have reliable Markdown headings. It tries paragraph,
    line, sentence, then word boundaries before falling back to characters.

Chunk settings:
    CHUNK_SIZE = 500 characters keeps chunks small enough for precise retrieval.
    CHUNK_OVERLAP = 80 characters preserves context across boundaries, useful
    for legal clauses and Vietnamese sentences that span multiple lines.

Chosen embedding model:
    sentence-transformers/all-MiniLM-L6-v2, dimension 384.
    It is lightweight and fast for local classroom/demo indexing. For a stronger
    multilingual production setup, BAAI/bge-m3 (1024 dim) is a good upgrade.

Vector store:
    Weaviate is attempted first, using custom vectors so hybrid search is
    available when a local Weaviate server is running. If Weaviate is not
    reachable, the pipeline writes the complete embedded index to
    data/vector_store/drug_law_docs.json so indexing still succeeds locally.
"""

from __future__ import annotations

import json
import math
import re
from hashlib import blake2b
from pathlib import Path
from typing import Any

STANDARDIZED_DIR = Path(__file__).resolve().parent.parent / "data" / "standardized"
VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store"
LOCAL_INDEX_PATH = VECTOR_STORE_DIR / "drug_law_docs.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

VECTOR_STORE = "weaviate"
WEAVIATE_COLLECTION = "DrugLawDocs"
_EMBEDDING_BACKEND = "sentence-transformers"


def load_documents() -> list[dict[str, Any]]:
    """
    Read all Markdown files from data/standardized/.

    Returns:
        A list of {"content": str, "metadata": dict}.
    """
    documents: list[dict[str, Any]] = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(relative_path).replace("\\", "/"),
                    "type": doc_type,
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Split documents with RecursiveCharacterTextSplitter.

    Returns:
        A list of {"content": str, "metadata": dict}, one item per chunk.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[dict[str, Any]] = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for chunk_index, chunk_text in enumerate(splits):
            text = chunk_text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "content": text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": chunk_index,
                        "chunking_method": CHUNKING_METHOD,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                }
            )

    return chunks


def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Offline fallback embedding with the same configured dimension."""
    vector = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    for token in tokens:
        digest = blake2b(token.encode("utf-8"), digest_size=4).digest()
        bucket = int.from_bytes(digest, byteorder="big") % dim
        vector[bucket] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add an embedding vector to each chunk using the configured SentenceTransformer.
    """
    if not chunks:
        return chunks

    global _EMBEDDING_BACKEND

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        texts = [chunk["content"] for chunk in chunks]
        embeddings = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
            chunk["metadata"]["embedding_backend"] = "sentence-transformers"
        _EMBEDDING_BACKEND = "sentence-transformers"
    except Exception as exc:
        # Keeps the local indexing demo runnable when Hugging Face is blocked by
        # SSL/network policy. The dimension remains the documented 384.
        print(f"Embedding model unavailable; using local hashing fallback: {exc}")
        for chunk in chunks:
            chunk["embedding"] = _hash_embedding(chunk["content"])
            chunk["metadata"]["embedding_backend"] = "hashing-fallback"
        _EMBEDDING_BACKEND = "hashing-fallback"

    return chunks


def _index_to_local_json(chunks: list[dict[str, Any]], reason: str | None = None) -> Path:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "vector_store": "local_json",
        "fallback_reason": reason,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "embedding_backend": _EMBEDDING_BACKEND,
        "chunking": {
            "method": CHUNKING_METHOD,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "documents_indexed": len({chunk["metadata"]["path"] for chunk in chunks}),
        "chunks": chunks,
    }
    LOCAL_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed locally: {LOCAL_INDEX_PATH}")
    return LOCAL_INDEX_PATH


def _index_to_weaviate(chunks: list[dict[str, Any]]) -> None:
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property

    client = weaviate.connect_to_local()
    try:
        if client.collections.exists(WEAVIATE_COLLECTION):
            client.collections.delete(WEAVIATE_COLLECTION)

        collection = client.collections.create(
            name=WEAVIATE_COLLECTION,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="path", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )

        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk["metadata"]
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": metadata["source"],
                        "path": metadata["path"],
                        "doc_type": metadata["type"],
                        "chunk_index": metadata["chunk_index"],
                    },
                    vector=chunk["embedding"],
                )
    finally:
        client.close()


def index_to_vectorstore(chunks: list[dict[str, Any]]) -> Path | str:
    """
    Index chunks to Weaviate when available, otherwise persist a local JSON index.
    """
    if not chunks:
        raise ValueError("No chunks to index")

    try:
        _index_to_weaviate(chunks)
        print(f"Indexed to Weaviate collection: {WEAVIATE_COLLECTION}")
        return WEAVIATE_COLLECTION
    except Exception as exc:
        reason = f"Weaviate unavailable, using local JSON fallback: {exc}"
        print(reason)
        return _index_to_local_json(chunks, reason=reason)


def run_pipeline() -> Path | str:
    """Run load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"Vector Store: {VECTOR_STORE} with local JSON fallback")
    print("=" * 50)

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_target = index_to_vectorstore(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents")
    return index_target


if __name__ == "__main__":
    run_pipeline()
