"""Pydantic models for document representation throughout the pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Supported document types for classification and routing."""

    INVOICE = "invoice"
    CONTRACT = "contract"
    REPORT = "report"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


class DocumentStatus(str, Enum):
    """Lifecycle status of a document in the pipeline."""

    PENDING = "pending"
    INGESTED = "ingested"
    CLASSIFIED = "classified"
    PROCESSED = "processed"
    FAILED = "failed"


class RawDocument(BaseModel):
    """Represents a raw document as it enters the pipeline."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str
    source: str | None = None
    document_type: DocumentType = DocumentType.UNKNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: DocumentStatus = DocumentStatus.PENDING

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Document content cannot be empty or whitespace-only")
        return v


class ParsedChunk(BaseModel):
    """A chunk of text extracted from a document with metadata."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    document_id: str
    text: str
    chunk_index: int
    page_number: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text cannot be empty or whitespace-only")
        return v


class EmbeddedChunk(BaseModel):
    """A chunk with its embedding vector for vector storage."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    document_id: str
    chunk_id: str
    text: str
    embedding: list[float]
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("embedding")
    @classmethod
    def embedding_must_not_be_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Embedding vector cannot be empty")
        return v
