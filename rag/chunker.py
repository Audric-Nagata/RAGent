"""Document chunker — splits raw document text into overlapping chunks.

Chunking strategy
-----------------
The default strategy is **recursive sentence-aware splitting**:
  1. Try to split on double newlines (paragraph boundaries).
  2. Fall back to single newlines, then sentences (`. `), then words.
  3. Enforce ``chunk_size`` with ``chunk_overlap`` tokens of context carry-over.

This mirrors LlamaIndex's ``SentenceSplitter`` behaviour but implemented
in plain Python so there is no heavy dependency for the core splitting logic.
LlamaIndex is still available for more advanced strategies if needed.

All public functions are ``async`` for consistency with the rest of the
pipeline, even though the splitting itself is CPU-bound (it's fast enough
to run inline without an executor for typical document sizes).
"""

from __future__ import annotations

import re
from typing import Sequence

from models.documents import ParsedChunk, RawDocument


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE: int = 512          # characters (not tokens)
DEFAULT_CHUNK_OVERLAP: int = 64        # characters carried over between chunks
MIN_CHUNK_SIZE: int = 50               # discard chunks shorter than this


# ── Splitting helpers ──────────────────────────────────────────────────────────

_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Recursively split ``text`` into chunks of at most ``chunk_size`` chars.

    The algorithm tries each separator in ``_SEPARATORS`` in order, collecting
    candidate splits and merging them into chunks that respect the size limit
    with ``chunk_overlap`` chars of preceding context.

    Args:
        text:          The full text to split.
        chunk_size:    Maximum character length per chunk.
        chunk_overlap: Characters from the end of one chunk to prepend to the next.

    Returns:
        A list of non-empty text chunks.
    """
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks: list[str] = []
    current: str = ""

    # Choose the first separator that actually splits the text
    chosen_sep = ""
    for sep in _SEPARATORS:
        if sep and sep in text:
            chosen_sep = sep
            break

    parts = text.split(chosen_sep) if chosen_sep else list(text)

    for part in parts:
        candidate = (current + chosen_sep + part).strip() if current else part.strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Flush current chunk
            if current and len(current.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(current.strip())

            # Carry-over overlap
            overlap_text = current[-chunk_overlap:] if chunk_overlap and current else ""
            overlap_text = overlap_text.strip()

            # If the part itself is too big, recurse
            if len(part) > chunk_size:
                sub_chunks = _split_text(part, chunk_size, chunk_overlap)
                if sub_chunks:
                    if overlap_text:
                        sub_chunks[0] = (overlap_text + " " + sub_chunks[0]).strip()
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1]
                else:
                    current = overlap_text
            else:
                current = (overlap_text + " " + part).strip() if overlap_text else part.strip()

    if current and len(current.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current.strip())

    return chunks


# ── Page-aware splitting ───────────────────────────────────────────────────────

_PAGE_BREAK_RE = re.compile(r"\f|--- ?[Pp]age \d+ ?---")


def _detect_pages(text: str) -> list[tuple[int, str]]:
    """Split text on form-feed or explicit page markers.

    Returns:
        List of ``(page_number, page_text)`` tuples (1-indexed).
    """
    raw_pages = _PAGE_BREAK_RE.split(text)
    return [(i + 1, p) for i, p in enumerate(raw_pages) if p.strip()]


# ── Public API ────────────────────────────────────────────────────────────────

async def chunk_document(
    doc: RawDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ParsedChunk]:
    """Chunk a :class:`~models.documents.RawDocument` into ``ParsedChunk`` objects.

    Detects page boundaries (form-feed ``\\f`` or ``--- Page N ---`` markers)
    and records ``page_number`` on each chunk.  Falls back to treating the
    entire document as page 1 if no markers are found.

    Args:
        doc:           The raw document to split.
        chunk_size:    Maximum character length per chunk.
        chunk_overlap: Overlap in characters between consecutive chunks.

    Returns:
        An ordered list of :class:`~models.documents.ParsedChunk` objects
        ready for embedding.
    """
    pages = _detect_pages(doc.content)
    # If no page markers found, _detect_pages returns [(1, full_text)]

    parsed: list[ParsedChunk] = []
    global_index: int = 0

    for page_num, page_text in pages:
        raw_chunks = _split_text(page_text, chunk_size, chunk_overlap)
        for text in raw_chunks:
            parsed.append(
                ParsedChunk(
                    document_id=doc.id,
                    text=text,
                    chunk_index=global_index,
                    page_number=page_num,
                    metadata={
                        "source": doc.source,
                        "document_type": doc.document_type.value,
                    },
                )
            )
            global_index += 1

    return parsed


async def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    start_index: int = 0,
) -> list[ParsedChunk]:
    """Chunk arbitrary plain text without a full RawDocument object.

    Useful for chunking content fetched from web search or other ad-hoc
    sources that don't go through the document ingestion pipeline.

    Args:
        text:          Plain text to split.
        document_id:   ID to associate each chunk with.
        chunk_size:    Maximum character length per chunk.
        chunk_overlap: Overlap in characters between consecutive chunks.
        start_index:   Starting ``chunk_index`` value (for appending to an
                       existing chunk list).

    Returns:
        A list of :class:`~models.documents.ParsedChunk` objects.
    """
    raw_chunks = _split_text(text, chunk_size, chunk_overlap)
    return [
        ParsedChunk(
            document_id=document_id,
            text=chunk,
            chunk_index=start_index + i,
        )
        for i, chunk in enumerate(raw_chunks)
    ]
