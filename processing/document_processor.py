"""PDF extraction and text chunking."""

import logging
from typing import List

import pymupdf

logger = logging.getLogger(__name__)


def extract_pdf_text(file_path: str) -> str:
    """Extracts all text from a PDF file using PyMuPDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        The full extracted text from all pages.
    """
    text = ""
    doc = pymupdf.open(file_path)
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text


def create_chunks(text: str, max_size: int = 1200, overlap: int = 200) -> List[str]:
    """Splits text into overlapping chunks using a sliding window.

    Args:
        text: The full document text.
        max_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + max_size

        # Try to break at a sentence boundary (period, newline) for cleaner chunks
        if end < text_len:
            # Look for the last sentence-ending punctuation within the chunk
            last_period = text.rfind('. ', start, end)
            last_newline = text.rfind('\n', start, end)
            break_point = max(last_period, last_newline)

            if break_point > start + max_size // 2:
                end = break_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= text_len:
            break

    logger.info(f"Created {len(chunks)} chunks from text ({text_len} chars)")
    return chunks
