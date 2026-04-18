"""Typed MCP tool input/output schemas shared across servers and clients."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Document Store Schemas ─────────────────────────────────────────────

class StoreDocumentInput(BaseModel):
    """Input schema for storing a parsed document."""

    document_id: str
    content: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GetDocumentInput(BaseModel):
    """Input schema for retrieving a document by ID."""

    document_id: str


class ListDocumentsInput(BaseModel):
    """Input schema for listing documents with optional filtering."""

    status: str | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class StoredDocument(BaseModel):
    """Output schema for a stored document."""

    document_id: str
    content: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    stored_at: str  # ISO timestamp


# ── Vector DB Schemas ──────────────────────────────────────────────────

class RetrieveInput(BaseModel):
    """Input schema for vector retrieval."""

    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: str | None = None  # Optional scope to a single document


class EmbedInput(BaseModel):
    """Input schema for embedding a text chunk."""

    text: str


class EmbedBatchInput(BaseModel):
    """Input schema for batch embedding."""

    texts: list[str]


class StoredChunk(BaseModel):
    """Output schema for a stored/retrieved chunk."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    score: float | None = None  # Relevance score for retrieval
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbedResult(BaseModel):
    """Output schema for embedding operation."""

    chunk_id: str
    embedding_dimension: int


# ── Web Search Schemas ─────────────────────────────────────────────────

class SearchInput(BaseModel):
    """Input schema for web search."""

    query: str
    num_results: int = Field(default=5, ge=1, le=20)
    language: str = Field(default="en")


class SearchResult(BaseModel):
    """Output schema for a single search result."""

    title: str
    url: str
    snippet: str
    source: str | None = None


class SearchResults(BaseModel):
    """Output schema for web search."""

    query: str
    results: list[SearchResult]
    total_available: int | None = None
