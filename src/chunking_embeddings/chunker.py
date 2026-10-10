"""Sliding-window chunker on character offsets, paragraph-aware."""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from src.data_ingestion.loader import RawDocument, iter_pages


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    framework: str
    title: str
    page: int


_PARA = re.compile(r"\n\s*\n")


def _split_paragraphs(text: str, max_len: int) -> list[str]:
    """Split on blank lines; break anything longer than max_len into lines, then hard-wrap.

    pypdf rarely emits blank lines, so without the fallback a whole page becomes one
    "paragraph" and one oversized chunk that the embedder silently truncates.
    """
    out: list[str] = []
    for para in (p.strip() for p in _PARA.split(text)):
        if not para:
            continue
        if len(para) <= max_len:
            out.append(para)
            continue
        for line in (ln.strip() for ln in para.split("\n")):
            while len(line) > max_len:
                out.append(line[:max_len])
                line = line[max_len:]
            if line:
                out.append(line)
    return out


def chunk_documents(
    docs: Iterable[RawDocument], chunk_size: int = 900, overlap: int = 150
) -> list[Chunk]:
    """Group paragraphs into ~chunk_size character windows with overlap."""
    chunks: list[Chunk] = []
    counter = 0

    for doc, page_num, page_text in iter_pages(docs):
        paragraphs = _split_paragraphs(page_text, chunk_size)
        buffer = ""
        for para in paragraphs:
            if not buffer:
                buffer = para
                continue
            if len(buffer) + len(para) + 1 <= chunk_size:
                buffer = f"{buffer}\n{para}"
            else:
                chunks.append(_make_chunk(counter, buffer, doc, page_num))
                counter += 1
                # carry tail as overlap (buffer[-0:] would be the whole buffer)
                tail = buffer[-overlap:] if overlap > 0 else ""
                buffer = (tail + "\n" + para).strip()
        if buffer:
            chunks.append(_make_chunk(counter, buffer, doc, page_num))
            counter += 1

    return chunks


def _make_chunk(idx: int, text: str, doc: RawDocument, page: int) -> Chunk:
    return Chunk(
        chunk_id=f"c{idx:05d}",
        text=text,
        source=doc.source,
        framework=doc.framework,
        title=doc.title,
        page=page,
    )
