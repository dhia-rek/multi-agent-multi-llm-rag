"""Load PDF documents from the corpus directory."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RawDocument:
    """One parsed PDF: provenance plus per-page text."""

    source: str            # absolute path
    framework: str         # subfolder name, e.g. "dt_framework" or "bikeshop_roadmap"
    title: str             # filename without extension
    pages: list[str]       # text per page


def _extract_pdf(path: Path) -> list[str]:
    """Return stripped text for each page of a PDF ('' when a page has no extractable text)."""
    reader = PdfReader(str(path))
    return [(p.extract_text() or "").strip() for p in reader.pages]


def load_corpus(corpus_dir: Path) -> list[RawDocument]:
    """Walk every subfolder of corpus_dir and parse PDFs found there.

    Args:
        corpus_dir: Root directory searched recursively for ``*.pdf``.

    Returns:
        One RawDocument per PDF, in sorted path order; ``framework`` is the PDF's
        immediate parent folder name.

    Raises:
        FileNotFoundError: If ``corpus_dir`` does not exist.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    docs: list[RawDocument] = []
    for pdf_path in sorted(corpus_dir.rglob("*.pdf")):
        framework = pdf_path.parent.name
        log.info("Loading %s/%s", framework, pdf_path.name)
        pages = _extract_pdf(pdf_path)
        docs.append(
            RawDocument(
                source=str(pdf_path),
                framework=framework,
                title=pdf_path.stem,
                pages=pages,
            )
        )
    log.info("Loaded %d PDF document(s)", len(docs))
    return docs


def iter_pages(docs: Iterable[RawDocument]) -> Iterator[tuple[RawDocument, int, str]]:
    """Yield (doc, page_number, page_text) for non-empty pages; page numbers are 1-based."""
    for doc in docs:
        for i, text in enumerate(doc.pages, start=1):
            if text:
                yield doc, i, text
