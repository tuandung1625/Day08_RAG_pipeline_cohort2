"""
Task 3 - Convert every supported file in data/landing/ to Markdown.

Output is written to data/standardized/ while preserving subfolders such as
legal/ and news/. PDF/DOCX conversion uses Microsoft MarkItDown. Crawled news
JSON files are normalized into Markdown with a small metadata header.
"""

import json
import sys
from pathlib import Path
from typing import Iterable

from markitdown import MarkItDown

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardized"

MARKITDOWN_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".txt", ".md"}
IGNORED_FILENAMES = {".gitkeep"}


def _output_path_for(input_path: Path) -> Path:
    relative_path = input_path.relative_to(LANDING_DIR)
    return (OUTPUT_DIR / relative_path).with_suffix(".md")


def _iter_landing_files() -> Iterable[Path]:
    for filepath in sorted(LANDING_DIR.rglob("*")):
        if filepath.is_file() and filepath.name not in IGNORED_FILENAMES:
            yield filepath


def _json_to_markdown(filepath: Path) -> str:
    data = json.loads(filepath.read_text(encoding="utf-8"))

    title = data.get("title") or filepath.stem
    url = data.get("url") or "N/A"
    crawl_date = data.get("crawl_date") or data.get("date_crawled") or "N/A"
    content = (
        data.get("content_markdown")
        or data.get("markdown")
        or data.get("content")
        or data.get("text")
        or ""
    )

    metadata = [
        f"# {title}",
        "",
        f"**Source:** {url}",
        f"**Crawled:** {crawl_date}",
        "",
        "---",
        "",
    ]
    return "\n".join(metadata) + str(content).strip() + "\n"


def _pdfplumber_to_markdown(filepath: Path) -> str:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(str(filepath)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"## Page {page_number}\n\n{text.strip()}")

    return "\n\n".join(pages).strip()


def _empty_conversion_note(filepath: Path) -> str:
    return (
        f"# {filepath.stem}\n\n"
        f"**Source file:** {filepath.relative_to(LANDING_DIR)}\n\n"
        "MarkItDown completed but did not extract selectable text from this file. "
        "The document is likely image-based or scanned and needs OCR for full text "
        "extraction. This placeholder keeps the standardized corpus complete for "
        "Task 4 indexing while recording the conversion limitation.\n"
    )


def convert_file(filepath: Path, md_converter: MarkItDown | None = None) -> Path | None:
    """Convert one landing file and return the output path, or None if skipped."""
    suffix = filepath.suffix.lower()
    output_path = _output_path_for(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".json":
        markdown = _json_to_markdown(filepath)
    elif suffix in MARKITDOWN_EXTENSIONS:
        converter = md_converter or MarkItDown()
        result = converter.convert(str(filepath))
        markdown = result.text_content
        if suffix == ".pdf" and not markdown.strip():
            markdown = _pdfplumber_to_markdown(filepath)
        if not markdown.strip():
            markdown = _empty_conversion_note(filepath)
    else:
        print(f"Skipping unsupported file: {filepath.relative_to(LANDING_DIR)}")
        return None

    output_path.write_text(markdown, encoding="utf-8")
    print(f"Saved: {output_path.relative_to(PROJECT_DIR)}")
    return output_path


def convert_all() -> list[Path]:
    """Convert all files under data/landing/ to data/standardized/."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    md_converter = MarkItDown()
    outputs: list[Path] = []

    for filepath in _iter_landing_files():
        print(f"Converting: {filepath.relative_to(LANDING_DIR)}")
        output_path = convert_file(filepath, md_converter)
        if output_path is not None:
            outputs.append(output_path)

    print(f"\nDone. Converted {len(outputs)} files into {OUTPUT_DIR}")
    return outputs


if __name__ == "__main__":
    convert_all()
