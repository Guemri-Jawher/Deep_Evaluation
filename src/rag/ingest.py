"""PDF loading, boilerplate cleaning and chunking."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from utils.config import DOCUMENTS_DIR, get_settings
from rag.models import Chunk

DOC_ALIASES = {
    "baswe_evaluation_field_guide": "eval-guide",
    "baswe-agentic-design-patterns-guide": "agentic-patterns",
}

"""Boilerplate patterns to remove from PDF pages."""
_BOILERPLATE = [
    re.compile(r"^baswe\.ai engineer accelerator™.*$", re.I),
    re.compile(r"^agentic design patterns\s*\|\s*skool\.com/baswe-ai\s*$", re.I),
    re.compile(r"^baswe\.ai\s*$", re.I),
    re.compile(r"^\d{1,2}\s*$"),  # bare page numbers
    re.compile(r"^•\s*$"),
]
MIN_CHUNK_CHARS = 80


def doc_alias(path: Path) -> str:
    """Return the canonical alias for a document path.

    If the document has a known alias in DOC_ALIASES, return that.
    Otherwise, generate a slug from the file stem.
    """
    stem = path.stem
    if stem in DOC_ALIASES:
        return DOC_ALIASES[stem]
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def clean_page_text(text: str) -> str:
    """Remove boilerplate lines and excessive newlines from a page's text."""
    kept = [
        ln.rstrip()
        for ln in text.splitlines()
        if not any(p.match(ln.strip()) for p in _BOILERPLATE)
    ]
    cleaned = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def load_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Load and clean text from each page of a PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_page_text(page.extract_text() or "")
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(
    source: str,
    pages: list[tuple[int, str]],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split pages into smaller text chunks suitable for embedding."""
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or s.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks: list[Chunk] = []
    for page_no, text in pages:
        for idx, piece in enumerate(
            p for p in splitter.split_text(text) if len(p) >= MIN_CHUNK_CHARS
        ):
            chunks.append(Chunk(f"{source}:p{page_no}:c{idx}", piece.strip(), source, page_no, idx))
    return chunks


def load_corpus(documents_dir: Path | None = None) -> list[Chunk]:
    """Chunk every PDF in the documents directory."""
    directory = documents_dir or DOCUMENTS_DIR
    chunks: list[Chunk] = []
    for pdf in sorted(directory.glob("*.pdf")):
        chunks.extend(chunk_pages(doc_alias(pdf), load_pages(pdf)))
    if not chunks:
        raise FileNotFoundError(f"no PDF content found in {directory}")
    return chunks
