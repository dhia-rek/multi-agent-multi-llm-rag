"""Load PDF documents from the corpus directory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RawDocument:
    source: str            # absolute path
    framework: str         # subfolder name, e.g. "dt_framework" or "bikeshop_roadmap"
    title: str             # filename without extension
    pages: list[str]       # text per page


def _extract_pdf(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    return [(p.extract_text() or "").strip() for p in reader.pages]


def load_corpus(corpus_dir: Path) -> list[RawDocument]:
    """Walk every subfolder of corpus_dir and parse PDFs found there."""
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


def iter_pages(docs: Iterable[RawDocument]):
    """Yield (doc, page_number, page_text) for non-empty pages."""
    for doc in docs:
        for i, text in enumerate(doc.pages, start=1):
            if text:
                yield doc, i, text
