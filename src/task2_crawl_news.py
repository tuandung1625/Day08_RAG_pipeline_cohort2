"""
Task 2 - Crawl news articles about Vietnamese artists related to drug cases.

This module contains the crawling logic that was previously in crawl_news.py.
It uses Crawl4AI, stores one JSON file per article, and records the metadata
required by the assignment: url, title, crawl date, and Markdown content.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://tuoitre.vn/ca-si-long-nhat-thua-nhan-da-nhieu-lan-dat-mua-ma-tuy-ve-su-dung-20260520161117184.htm",
    "https://tuoitre.vn/khoi-to-3-bi-can-trong-vu-ca-si-miu-le-su-dung-ma-tuy-o-cat-ba-20260514230349573.htm",
    "https://thanhnien.vn/chi-dan-huu-tin-va-loat-sao-viet-gay-on-ao-vi-dinh-toi-ma-tuy-185241110141122628.htm",
    "https://thanhnien.vn/ca-si-long-nhat-bi-bat-showbiz-viet-lien-tiep-chan-dong-vi-ma-tuy-18526052013032001.htm",
]


def setup_directory() -> Path:
    """Create data/landing/news/ when it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def save_article(data: dict[str, Any], filename: str) -> Path:
    """Save one crawled article as UTF-8 JSON."""
    setup_directory()
    filepath = DATA_DIR / filename
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    return filepath


def _article_from_result(url: str, result: Any) -> dict[str, Any]:
    metadata = getattr(result, "metadata", None) or {}
    return {
        "url": url,
        "crawl_date": datetime.now().isoformat(),
        "title": metadata.get("title", ""),
        "content_markdown": getattr(result, "markdown", "") or "",
    }


async def crawl_article(url: str, crawler: Any | None = None) -> dict[str, Any]:
    """
    Crawl a single article and return metadata plus Markdown content.

    Passing an existing crawler lets crawl_all reuse one browser session.
    """
    if crawler is not None:
        result = await crawler.arun(url=url)
        return _article_from_result(url, result)

    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as local_crawler:
        result = await local_crawler.arun(url=url)
        return _article_from_result(url, result)


async def crawl_and_save_article(crawler: Any, url: str, index: int) -> Path | None:
    """Crawl one URL and save it as article_{index}.json."""
    try:
        article = await crawl_article(url, crawler=crawler)
        filename = f"article_{index}.json"
        filepath = save_article(article, filename)
        print(f"Saved: {filepath}")
        return filepath
    except Exception as exc:
        print(f"Error crawling {url}: {exc}")
        return None


async def crawl_all(urls: list[str] | None = None) -> list[Path]:
    """Crawl all configured article URLs and save them under data/landing/news/."""
    from crawl4ai import AsyncWebCrawler

    setup_directory()
    target_urls = urls or ARTICLE_URLS
    saved_files: list[Path] = []

    async with AsyncWebCrawler() as crawler:
        tasks = [
            crawl_and_save_article(crawler, url, index)
            for index, url in enumerate(target_urls, start=1)
        ]
        results = await asyncio.gather(*tasks)

    for filepath in results:
        if filepath is not None:
            saved_files.append(filepath)
    return saved_files


def main() -> None:
    if not ARTICLE_URLS:
        raise RuntimeError("ARTICLE_URLS is empty. Add at least 5 article URLs.")
    asyncio.run(crawl_all())


if __name__ == "__main__":
    main()
