"""Pydantic models for SSE streaming events and chunks."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Generic, TypeVar, Literal

from pydantic import BaseModel, Field

from models.extractions import ExtractionResult
from models.documents import DocumentStatus

T = TypeVar("T", bound=BaseModel)


class StreamEventType(StrEnum):
    """Types of events that can be streamed."""

    PARTIAL = "partial"
    COMPLETE = "complete"
    ERROR = "error"
    PROGRESS = "progress"


class StreamChunk(BaseModel, Generic[T]):
    """A single chunk in the SSE stream, parameterized by payload type."""

    event: Literal["partial", "complete", "error", "progress"]
    payload: T | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sequence: int | None = None  # Optional ordering index

    def unwrap(self) -> T:
        """Unwrap the payload, raising on error or missing payload."""
        if self.event == "error":
            from models.results import PipelineError
            raise PipelineError(self.error or "Unknown error")
        if self.payload is None:
            raise ValueError("Empty payload on non-error chunk")
        return self.payload


class ProgressData(BaseModel):
    """Payload for 'progress' stream events."""

    stage: str  # e.g., "classifying", "retrieving", "extracting"
    message: str
    percentage: float | None = Field(default=None, ge=0.0, le=100.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PartialExtraction(BaseModel):
    """Payload for 'partial' stream events during extraction."""

    document_id: str
    partial_text: str | None = None
    fields_extracted_so_far: dict[str, Any] = Field(default_factory=dict)


# Convenience type for the most common stream usage
ExtractionStreamChunk = StreamChunk[ExtractionResult]
