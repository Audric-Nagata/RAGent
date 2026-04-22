"""Document embedder — generates embedding vectors via Qwen/Qwen3-Embedding-0.6B.

Uses the HuggingFace ``Qwen/Qwen3-Embedding-0.6B`` model (1024 dimensions),
loaded locally once at import time via ``sentence-transformers``.

Async safety
------------
The ``SentenceTransformer.encode()`` call is synchronous (PyTorch CPU/GPU).
To avoid blocking the event loop it is offloaded with ``asyncio.to_thread``.

Batch embedding
---------------
Batch operations run chunks concurrently (up to ``DEFAULT_CONCURRENCY``
workers) via ``asyncio.Semaphore`` + ``asyncio.gather``, exactly as before.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from sentence_transformers import SentenceTransformer

from config import Settings
from models.documents import EmbeddedChunk, ParsedChunk

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EMBED_MODEL_NAME: str = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_CONCURRENCY: int = 8        # max simultaneous thread-pool tasks
EMBEDDING_DIMENSION: int = 1024     # Qwen3-Embedding-0.6B default output dim

# ── Module-level singleton (loaded once) ──────────────────────────────────────

_embed_model: SentenceTransformer = SentenceTransformer(
    EMBED_MODEL_NAME,
    tokenizer_kwargs={"padding_side": "left"},
)


# ── Core embedding function ───────────────────────────────────────────────────

async def embed_text(
    text: str,
    *,
    prompt_name: str | None = None,
) -> list[float]:
    """Embed a single text string using Qwen/Qwen3-Embedding-0.6B.

    The synchronous ``SentenceTransformer.encode()`` call is offloaded to the
    default thread-pool executor via ``asyncio.to_thread`` so the event loop
    is never blocked.

    Args:
        text:        The text to embed.
        prompt_name: Optional prompt name (e.g. ``"query"`` for retrieval
                     queries as recommended by the Qwen3-Embedding docs).

    Returns:
        A list of 1024 floats representing the embedding vector.
    """
    def _encode() -> list[float]:
        vec = _embed_model.encode(
            text,
            prompt_name=prompt_name,
            normalize_embeddings=True,
        )
        return vec.tolist()

    return await asyncio.to_thread(_encode)


# ── Batch embedding ───────────────────────────────────────────────────────────

async def embed_chunks(
    chunks: Sequence[ParsedChunk],
    concurrency: int = DEFAULT_CONCURRENCY,
    settings: Settings | None = None,
) -> list[EmbeddedChunk]:
    """Embed a sequence of :class:`~models.documents.ParsedChunk` objects.

    Runs embeddings concurrently (up to ``concurrency`` thread-pool tasks)
    using an ``asyncio.Semaphore``.

    Args:
        chunks:      The chunks to embed. Order is preserved in the output.
        concurrency: Max simultaneous ``to_thread`` tasks (default 8).
        settings:    Unused — kept for API compatibility.

    Returns:
        A list of :class:`~models.documents.EmbeddedChunk` objects in the
        same order as the input.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _embed_one(chunk: ParsedChunk) -> EmbeddedChunk:
        async with sem:
            vector = await embed_text(chunk.text)
        return EmbeddedChunk(
            document_id=chunk.document_id,
            chunk_id=chunk.id,
            text=chunk.text,
            embedding=vector,
            chunk_index=chunk.chunk_index,
            metadata=chunk.metadata,
        )

    tasks = [_embed_one(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)

    return list(results)


async def embed_single(
    text: str,
    settings: Settings | None = None,
) -> list[float]:
    """Embed a single arbitrary string — convenience wrapper for query embedding.

    This is the function used by :mod:`rag.retriever` to embed a search query
    before performing vector search.

    Args:
        text:     The text to embed.
        settings: Unused — kept for API compatibility.

    Returns:
        A 1024-dimensional embedding vector.
    """
    return await embed_text(text, prompt_name="query")


# ── Document-level convenience ────────────────────────────────────────────────

async def embed_document_chunks(
    chunks: Sequence[ParsedChunk],
    concurrency: int = DEFAULT_CONCURRENCY,
    settings: Settings | None = None,
) -> list[EmbeddedChunk]:
    """Alias for :func:`embed_chunks` — used for clarity at call sites.

    Call this after :func:`rag.chunker.chunk_document` to produce
    ``EmbeddedChunk`` objects ready to be stored in the vector DB via the
    ``vector-db`` MCP server.
    """
    return await embed_chunks(chunks, concurrency=concurrency, settings=settings)
