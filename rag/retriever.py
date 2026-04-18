"""Async vector retriever — queries Qdrant and re-ranks results.

The retriever is the read-path of the RAG layer.  It:
  1. Embeds the query string via :func:`rag.embedder.embed_single`.
  2. Sends a cosine-similarity search to Qdrant via the MCP ``vector-db``
     server (to respect the MCP architecture).
  3. Optionally applies a lightweight re-ranking step using
     cross-encoder scores (reciprocal rank fusion if no cross-encoder is
     available — pure Python, no model download required).

Design note
-----------
The retriever deliberately goes through the **MCP client** rather than
talking to Qdrant directly.  This keeps the MCP server as the single
authoritative interface for vector storage so that agents and non-agent
code (e.g. a CLI indexing script) use the same transport.

If you need to bypass MCP (e.g. in a standalone test), use
:func:`retrieve_direct` which connects to Qdrant without the MCP layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from mcp_server.client import MCPClient
from models.documents import ParsedChunk
from mcp_server.schemas import StoredChunk
from rag.embedder import embed_single
from config import Settings

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(slots=True)
class RetrievedChunk:
    """A retrieved chunk with its relevance score.

    Attributes:
        chunk_id:    Qdrant point ID (hex string).
        document_id: Parent document's ID.
        text:        The chunk text.
        chunk_index: Position of this chunk in the original document.
        score:       Cosine-similarity score in [0.0, 1.0] (higher is better).
        metadata:    Extra payload stored alongside the vector.
    """

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    score: float
    metadata: dict


# ── MCP-based retriever (primary path) ────────────────────────────────────────

async def retrieve(
    query: str,
    mcp_client: MCPClient,
    top_k: int = 5,
    document_id: str | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top-k most relevant chunks for a query via MCP.

    This is the **primary retrieval function** used by :mod:`agents.rag_agent`.
    It calls the ``vector-db`` MCP server's ``retrieve_chunks`` tool, which
    handles both embedding the query and the Qdrant search internally.

    Args:
        query:       Natural-language retrieval query.
        mcp_client:  The shared :class:`~mcp.client.MCPClient` instance.
        top_k:       Number of chunks to return (1-50).
        document_id: Optional — restrict results to a single document.
        settings:    Unused here (MCP server owns its own config); reserved
                     for future direct-mode parity.

    Returns:
        A list of :class:`RetrievedChunk` objects ordered by score descending.
    """
    arguments: dict[str, object] = {
        "query": query,
        "top_k": max(1, min(top_k, 50)),
    }
    if document_id:
        arguments["document_id"] = document_id

    raw = await mcp_client.call_tool(
        server="vector-db",
        tool="retrieve_chunks",
        arguments=arguments,
    )

    chunks: list[RetrievedChunk] = []
    for item in raw:
        try:
            data = json.loads(getattr(item, "text", "{}"))
            stored = StoredChunk.model_validate(data)
            chunks.append(
                RetrievedChunk(
                    chunk_id=stored.chunk_id,
                    document_id=stored.document_id,
                    text=stored.text,
                    chunk_index=stored.chunk_index,
                    score=stored.score or 0.0,
                    metadata=stored.metadata,
                )
            )
        except Exception as exc:
            logger.warning("Failed to parse retrieved chunk: %s", exc)

    return chunks


# ── Re-ranking ────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF is a parameter-light fusion method that works well without a
    trained cross-encoder.  Useful when combining results from two different
    queries (e.g. the original query + a reformulated one).

    Args:
        lists: Two or more ranked lists of :class:`RetrievedChunk`.
        k:     RRF constant — higher values dampen rank differences.

    Returns:
        A single merged and re-ranked list.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}

    for ranked in lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk.chunk_id] = chunk

    return sorted(chunks.values(), key=lambda c: scores[c.chunk_id], reverse=True)


async def retrieve_and_rerank(
    query: str,
    mcp_client: MCPClient,
    top_k: int = 5,
    document_id: str | None = None,
    rerank_queries: list[str] | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Retrieve chunks and optionally re-rank using multiple query formulations.

    When ``rerank_queries`` is provided, the retriever runs each query in
    parallel and fuses the results via RRF.  This increases recall when a
    single query might miss semantically valid chunks.

    Args:
        query:           Primary retrieval query.
        mcp_client:      Shared MCP client.
        top_k:           Number of final chunks to return.
        document_id:     Optional document scope.
        rerank_queries:  Additional query variants for multi-query retrieval.
        settings:        Optional settings override.

    Returns:
        Re-ranked list of at most ``top_k`` :class:`RetrievedChunk` objects.
    """
    all_queries = [query] + (rerank_queries or [])
    # Fetch top_k * 2 per query so fusion has enough material to work with
    fetch_k = min(top_k * 2, 50)

    results = await asyncio.gather(
        *[
            retrieve(q, mcp_client, top_k=fetch_k, document_id=document_id)
            for q in all_queries
        ]
    )

    if len(results) == 1:
        return results[0][:top_k]

    fused = _reciprocal_rank_fusion(list(results))
    return fused[:top_k]


# ── Direct Qdrant retriever (bypass MCP — for CLI / indexing scripts) ─────────

async def retrieve_direct(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Retrieve chunks by talking directly to Qdrant (no MCP hop).

    Use this in offline indexing scripts or unit tests where the MCP server
    is not running.  The ``agents/`` code always uses :func:`retrieve` instead.

    Args:
        query:       Retrieval query.
        top_k:       Number of results.
        document_id: Optional document scope.
        settings:    Settings (loaded from env if ``None``).

    Returns:
        A list of :class:`RetrievedChunk` objects.
    """
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import FieldCondition, MatchValue, Filter

    cfg = settings or Settings()
    vector = await embed_single(query, cfg)

    client = AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)

    qdrant_filter = None
    if document_id:
        qdrant_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )

    results = await client.query_points(
        collection_name=cfg.qdrant_collection,
        query=vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    chunks: list[RetrievedChunk] = []
    for scored in results.points:
        payload = scored.payload or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=str(scored.id),
                document_id=payload.get("document_id", "unknown"),
                text=payload.get("text", ""),
                chunk_index=payload.get("chunk_index", 0),
                score=round(scored.score, 4) if scored.score else 0.0,
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k not in ("document_id", "text", "chunk_index")
                },
            )
        )

    return chunks
