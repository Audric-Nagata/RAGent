"""Document embedder — generates embedding vectors via Gemini text-embedding-004.

Uses Google AI Studio's free ``text-embedding-004`` model (768 dimensions).
All requests are made with ``httpx.AsyncClient`` so the embedder fits
naturally into the async pipeline without blocking the event loop.

Rate-limit awareness
--------------------
Gemini's free tier allows 1 500 RPM for embeddings.  For batch operations
the embedder processes chunks **concurrently** using ``asyncio.gather`` with
a configurable semaphore to avoid hammering the API.  Single-item calls are
fire-and-forget.

Retry logic
-----------
Transient 429 / 5xx errors are retried up to ``MAX_RETRIES`` times with
exponential backoff (starting at 1 s, doubling each attempt).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Sequence

import httpx

from config import Settings
from models.documents import EmbeddedChunk, ParsedChunk

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/text-embedding-004:embedContent"
)

MAX_RETRIES: int = 3
BASE_RETRY_DELAY: float = 1.0       # seconds; doubles each retry
DEFAULT_CONCURRENCY: int = 8        # max simultaneous embed requests
EMBEDDING_DIMENSION: int = 768


# ── Core embedding function ───────────────────────────────────────────────────

async def embed_text(
    text: str,
    client: httpx.AsyncClient,
    api_key: str,
) -> list[float]:
    """Embed a single text string using Gemini text-embedding-004.

    Retries on rate-limit (429) and server errors (5xx) with exponential
    backoff.

    Args:
        text:    The text to embed (max ~2 048 tokens).
        client:  A shared ``httpx.AsyncClient`` instance.
        api_key: Gemini API key from settings.

    Returns:
        A list of 768 floats representing the embedding vector.

    Raises:
        httpx.HTTPStatusError: If all retries are exhausted.
        ValueError:            If the API response is malformed.
    """
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]},
    }

    delay = BASE_RETRY_DELAY
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.post(
                EMBED_URL,
                params={"key": api_key},
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            values: list[float] = data["embedding"]["values"]
            return values

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                logger.warning(
                    "Embed request failed (status=%d), retry %d/%d in %.1fs",
                    status, attempt, MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
                last_exc = exc
                continue
            raise

        except httpx.RequestError as exc:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(delay)
                delay *= 2
                last_exc = exc
                continue
            raise

    # Should never reach here, but satisfies mypy
    raise RuntimeError("embed_text: all retries exhausted") from last_exc


# ── Batch embedding ───────────────────────────────────────────────────────────

async def embed_chunks(
    chunks: Sequence[ParsedChunk],
    concurrency: int = DEFAULT_CONCURRENCY,
    settings: Settings | None = None,
) -> list[EmbeddedChunk]:
    """Embed a sequence of :class:`~models.documents.ParsedChunk` objects.

    Runs embeddings concurrently (up to ``concurrency`` requests at a time)
    using a shared ``httpx.AsyncClient`` and an ``asyncio.Semaphore``.

    Args:
        chunks:      The chunks to embed. Order is preserved in the output.
        concurrency: Max simultaneous requests (default 8, safe for free tier).
        settings:    Optional pre-loaded settings; loaded from env if ``None``.

    Returns:
        A list of :class:`~models.documents.EmbeddedChunk` objects in the
        same order as the input.
    """
    cfg = settings or Settings()
    api_key = cfg.gemini_api_key
    sem = asyncio.Semaphore(concurrency)

    async def _embed_one(chunk: ParsedChunk, client: httpx.AsyncClient) -> EmbeddedChunk:
        async with sem:
            vector = await embed_text(chunk.text, client, api_key)
        return EmbeddedChunk(
            document_id=chunk.document_id,
            chunk_id=chunk.id,
            text=chunk.text,
            embedding=vector,
            chunk_index=chunk.chunk_index,
            metadata=chunk.metadata,
        )

    async with httpx.AsyncClient() as client:
        tasks = [_embed_one(chunk, client) for chunk in chunks]
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
        settings: Optional pre-loaded settings; loaded from env if ``None``.

    Returns:
        A 768-dimensional embedding vector.
    """
    cfg = settings or Settings()
    async with httpx.AsyncClient() as client:
        return await embed_text(text, client, cfg.gemini_api_key)


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
