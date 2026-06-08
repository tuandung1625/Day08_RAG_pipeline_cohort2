"""
Task 10 - Generation with Citation.

This module supports two execution modes:
    1. OpenAI generation when OPENAI_API_KEY is configured.
    2. Offline extractive fallback that still returns cited evidence.
"""

from __future__ import annotations

import os
import re
from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()


# top_k = 5 keeps enough evidence for legal/news questions while staying short
# enough to reduce "lost in the middle" risk in a classroom demo.
TOP_K = 5

# top_p = 0.9 allows natural Vietnamese wording without making factual RAG too
# random. Temperature stays low because citations should be evidence-bound.
TOP_P = 0.9
TEMPERATURE = 0.3
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


SYSTEM_PROMPT = """Answer the following question comprehensively.
For every statement of fact or claim, immediately insert a citation
in brackets linking to the specific source
(e.g., [Author/Platform Name, Year]).
If the information is not explicitly stated in the provided context
or knowledge base, state 'I cannot verify this information'
rather than guessing."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to place strong evidence at the beginning and end.

    Input sorted by score: [1, 2, 3, 4, 5]
    Output pattern:        [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = [chunks[index] for index in range(0, len(chunks), 2)]
    back = [chunks[index] for index in range(1, len(chunks), 2)]
    back.reverse()
    return front + back


def _source_label(chunk: dict, fallback_index: int) -> str:
    metadata = chunk.get("metadata") or {}
    source = (
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("path")
        or f"Source {fallback_index}"
    )
    return str(source)


def _year_from_text(*values: str) -> str:
    for value in values:
        match = re.search(r"(20\d{2}|19\d{2})", value or "")
        if match:
            return match.group(1)
    return "Khong ro nam"


def _citation(chunk: dict, index: int) -> str:
    source = _source_label(chunk, index)
    metadata = chunk.get("metadata") or {}
    year = str(metadata.get("year") or _year_from_text(source, chunk.get("content", "")))
    return f"[{source}, {year}]"


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks into a prompt context with citation-ready source labels.
    """
    context_parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        source = _source_label(chunk, index)
        doc_type = metadata.get("type") or metadata.get("doc_type") or "unknown"
        citation = _citation(chunk, index)
        context_parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type} | Citation: {citation}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(context_parts)


def _call_openai(query: str, context: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        user_message = f"Context:\n{context}\n\nQuestion: {query}"
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return None


def _call_gemini(query: str, context: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import requests

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": TEMPERATURE,
                    "topP": TOP_P,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip() or None
    except Exception:
        return None


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I cannot verify this information"

    sentences: list[str] = []
    query_terms = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
    for index, chunk in enumerate(chunks, start=1):
        content = chunk.get("content", "").replace("\n", " ")
        parts = [part.strip() for part in re.split(r"(?<=[.!?。])\s+", content) if part.strip()]
        if not parts:
            parts = [content[:350].strip()]

        best_sentence = max(
            parts[:8],
            key=lambda sentence: len(query_terms & set(re.findall(r"\w+", sentence.lower(), flags=re.UNICODE))),
        )
        if best_sentence:
            sentences.append(f"{best_sentence} {_citation(chunk, index)}")

        if len(sentences) >= 2:
            break

    if not sentences:
        return "I cannot verify this information"
    return " ".join(sentences)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    Retrieve context, reorder it, and generate an answer with citations.
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = _call_openai(query, context) or _call_gemini(query, context)
    if not answer:
        answer = _extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": reordered[0].get("source", "none") if reordered else "none",
        "context": context,
    }


if __name__ == "__main__":
    result = generate_with_citation("Hinh phat tang tru ma tuy?")
    print(result["answer"])
    print(f"Sources: {len(result['sources'])} via {result['retrieval_source']}")
