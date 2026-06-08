"""
Task 1 - Collect legal documents about drug prevention and controlled substances.

The actual PDF files are stored in data/landing/legal/. This module keeps a
small inventory helper plus an optional download function for direct links.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "legal"
VALID_EXTENSIONS = {".pdf", ".docx", ".doc"}


class LegalDocument(TypedDict):
    title: str
    filename: str
    source_note: str


LEGAL_DOCUMENTS: list[LegalDocument] = [
    {
        "title": "Luat Phong, chong ma tuy 2021",
        "filename": "73luat.pdf",
        "source_note": "Luat so 73/2021/QH15, stored as a local PDF.",
    },
    {
        "title": "Hoi dap Luat Phong, chong ma tuy",
        "filename": "HOI DAP LUAT PHONG CHONG MA TUY.signed.pdf",
        "source_note": "Question-answer legal guidance document, stored as a local PDF.",
    },
    {
        "title": "Nghi dinh 28/2026/ND-CP",
        "filename": "Nghị-định-28-2026-NĐ-CP.pdf",
        "source_note": "Government decree PDF collected for the legal corpus.",
    },
]


def setup_directory() -> Path:
    """Create data/landing/legal/ when it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Legal landing directory ready: {DATA_DIR}")
    return DATA_DIR


def download_file(url: str, filename: str, timeout: int = 30) -> Path:
    """
    Download a PDF/DOC/DOCX from a direct URL into data/landing/legal/.

    This is optional because the current repository already contains the
    required collected files. It is useful when replacing or adding documents.
    """
    setup_directory()
    filepath = DATA_DIR / filename

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    filepath.write_bytes(response.content)
    print(f"Downloaded: {filepath}")
    return filepath


def list_collected_documents() -> list[Path]:
    """Return collected legal files that match the required extensions."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        filepath
        for filepath in DATA_DIR.iterdir()
        if filepath.is_file() and filepath.suffix.lower() in VALID_EXTENSIONS
    )


def validate_collection(min_files: int = 3) -> bool:
    """Check that at least min_files legal documents have been collected."""
    files = list_collected_documents()
    valid_files = [filepath for filepath in files if filepath.stat().st_size > 1024]
    return len(valid_files) >= min_files


def main() -> None:
    setup_directory()
    files = list_collected_documents()
    print(f"Collected {len(files)} legal files:")
    for filepath in files:
        print(f"- {filepath.name} ({filepath.stat().st_size} bytes)")

    if not validate_collection():
        raise RuntimeError("Need at least 3 non-empty PDF/DOCX legal documents.")


if __name__ == "__main__":
    main()
