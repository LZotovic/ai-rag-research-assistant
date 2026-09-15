from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Chunk:
    """A searchable passage and the metadata needed to cite its origin."""

    text: str
    source: str
    index: int
    page: int | None = None
    document_id: str | None = None


def chunk_words(
    text: str,
    *,
    source: str,
    chunk_size: int = 60,
    overlap: int = 10,
) -> list[Chunk]:
    """Split text into overlapping word windows.

    Overlap prevents information near a chunk boundary from losing its context.
    This simple strategy will later be compared with paragraph-aware chunking.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    words = text.split()
    step = chunk_size - overlap
    chunks: list[Chunk] = []

    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(Chunk(text=" ".join(window), source=source, index=len(chunks)))
        if start + chunk_size >= len(words):
            break

    return chunks


def chunk_paragraphs(text: str, *, source: str) -> list[Chunk]:
    """Create one chunk per non-empty paragraph.

    Unlike fixed word windows, this preserves boundaries chosen by the author.
    It is a useful baseline for papers whose paragraphs are reasonably sized.
    """
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n\s*\n", text.strip())
        if paragraph.strip()
    ]
    return [
        Chunk(text=paragraph, source=source, index=index)
        for index, paragraph in enumerate(paragraphs)
    ]


def chunk_pages(
    pages: list[str],
    *,
    source: str,
    document_id: str,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    """Chunk extracted PDF pages without losing citation metadata."""
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        # Chunk each page independently: a citation must never span two pages.
        page_chunks = chunk_words(
            page_text,
            source=source,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(
            Chunk(
                text=chunk.text,
                source=source,
                index=len(chunks),
                page=page_number,
                document_id=document_id,
            )
            for chunk in page_chunks
        )
    return chunks
