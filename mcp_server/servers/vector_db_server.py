"""MCP server: vector database — embed and retrieve chunks via Qdrant."""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, EmbeddedResource
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

import httpx

from config import Settings
from mcp_server.schemas import (
    RetrieveInput,
    EmbedInput,
    EmbedBatchInput,
    StoredChunk,
    EmbedResult,
)

# Use stderr for logging — stdout is reserved for the MCP JSON-RPC wire protocol
logging.basicConfig(stream=sys.stderr, level=logging.INFO)


app = Server("vector-db")

# ── Globals (initialised on first use) ────────────────────────────

_client: AsyncQdrantClient | None = None
_collection: str = "ragent-chunks"
_dimension: int = 1024


def _get_client() -> AsyncQdrantClient:
    """Lazy-initialise the Qdrant client and ensure the collection exists."""
    global _client, _collection, _dimension

    if _client is None:
        settings = Settings()
        _collection = settings.qdrant_collection
        _dimension = settings.embedding_dimension

        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    return _client


async def _ensure_collection() -> None:
    """Create the collection if it doesn't already exist."""
    client = _get_client()
    exists = await client.collection_exists(_collection)
    if not exists:
        await client.create_collection(
            collection_name=_collection,
            vectors_config=VectorParams(
                size=_dimension,
                distance=Distance.COSINE,
            ),
        )


# ── MCP Tool Handlers ─────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="retrieve_chunks",
            description="Retrieve top-k relevant chunks for a query",
            inputSchema=RetrieveInput.model_json_schema(),
        ),
        Tool(
            name="embed_chunk",
            description="Store a single text chunk with its embedding vector",
            inputSchema=EmbedInput.model_json_schema(),
        ),
        Tool(
            name="embed_batch",
            description="Store multiple text chunks with their embedding vectors",
            inputSchema=EmbedBatchInput.model_json_schema(),
        ),
    ]


@app.call_tool()
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> list[TextContent | EmbeddedResource]:
    await _ensure_collection()

    match name:
        case "retrieve_chunks":
            inp = RetrieveInput.model_validate(arguments)
            query_embedding = await _get_embedding(inp.query)

            filter_payload = None
            if inp.document_id:
                from qdrant_client.models import FieldCondition, MatchValue, Filter
                filter_payload = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=inp.document_id),
                        )
                    ]
                )

            results = await _get_client().query_points(
                collection_name=_collection,
                query=query_embedding,
                query_filter=filter_payload,
                limit=inp.top_k,
                with_payload=True,
            )

            chunks: list[StoredChunk] = []
            for scored in results.points:
                payload = scored.payload or {}
                chunks.append(
                    StoredChunk(
                        chunk_id=scored.id,
                        document_id=payload.get("document_id", "unknown"),
                        text=payload.get("text", ""),
                        chunk_index=payload.get("chunk_index", 0),
                        score=round(scored.score, 4) if scored.score else None,
                        metadata={
                            k: v
                            for k, v in payload.items()
                            if k not in ("document_id", "text", "chunk_index")
                        },
                    )
                )
            return [_text(c.model_dump_json()) for c in chunks]

        case "embed_chunk":
            inp = EmbedInput.model_validate(arguments)
            chunk_id = uuid.uuid4().hex
            embedding = await _get_embedding(inp.text)

            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "document_id": "unknown",
                    "text": inp.text,
                    "chunk_index": 0,
                },
            )
            await _get_client().upsert(
                collection_name=_collection,
                points=[point],
            )
            result = EmbedResult(
                chunk_id=chunk_id,
                embedding_dimension=len(embedding),
            )
            return [_text(result.model_dump_json())]

        case "embed_batch":
            inp = EmbedBatchInput.model_validate(arguments)
            points: list[PointStruct] = []
            results: list[EmbedResult] = []

            for i, text in enumerate(inp.texts):
                chunk_id = uuid.uuid4().hex
                embedding = await _get_embedding(text)
                points.append(
                    PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload={
                            "document_id": "unknown",
                            "text": text,
                            "chunk_index": i,
                        },
                    )
                )
                results.append(
                    EmbedResult(
                        chunk_id=chunk_id,
                        embedding_dimension=len(embedding),
                    )
                )

            await _get_client().upsert(
                collection_name=_collection,
                points=points,
            )
            return [_text(r.model_dump_json()) for r in results]

        case _:
            raise ValueError(f"Unknown tool: {name}")


# ── Embedding helper (uses Qwen/Qwen3-Embedding-0.6B via sentence-transformers) ─
async def _get_embedding(text: str) -> list[float]:
    """Embed text using the local Qwen3-Embedding-0.6B model."""
    return await embed_single(text)


def _get_client_dimension() -> int:
    return _dimension


def _hash_embedding(text: str, dimension: int = 768) -> list[float]:
    """Create a deterministic pseudo-random embedding from text (testing only)."""
    import hashlib
    import math

    h = hashlib.sha256(text.encode()).digest()
    while len(h) < dimension * 4:
        h += hashlib.sha256(h).digest()
    values = []
    for i in range(dimension):
        val = int.from_bytes(h[i * 4 : (i + 1) * 4], "big") % 10000 / 10000.0 - 0.5
        values.append(val)
    norm = math.sqrt(sum(v * v for v in values))
    if norm > 0:
        values = [v / norm for v in values]
    return values


def _text(content: str) -> TextContent:
    return TextContent(type="text", text=content)


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
