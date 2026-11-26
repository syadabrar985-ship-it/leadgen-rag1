import re
from pathlib import Path
from typing import Any
from pypdf import PdfReader


def extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract cleaned text and page numbers from a PDF file."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = re.sub(r"[ \t]+", " ", raw)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned:
            pages.append(
                {
                    "page": page_num,
                    "source": pdf_path.name,
                    "text": cleaned,
                }
            )
    return pages


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[dict[str, Any]]:
    """Split extracted pages into overlapping chunks with page metadata."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[dict[str, Any]] = []
    step = chunk_size - chunk_overlap

    for page_item in pages:
        text = page_item["text"]
        page_num = page_item["page"]
        source = page_item["source"]

        start = 0
        idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            piece = text[start:end].strip()
            if piece:
                chunk_id = f"p{page_num}_c{idx}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": piece,
                        "metadata": {
                            "chunk_id": chunk_id,
                            "page": page_num,
                            "source": source,
                        },
                    }
                )
                idx += 1
            if end >= text_len:
                break
            start += step

    return chunks
