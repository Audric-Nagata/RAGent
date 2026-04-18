"""Pydantic models for pipeline results and validation reports."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from models.extractions import (
    ExtractionResult,
    ExtractionType,
    InvoiceData,
    ContractData,
    ReportData,
    ReceiptData,
    GenericExtraction,
)
from models.documents import RawDocument

EXTRACTION_MAP: dict[ExtractionType, type[ExtractionResult]] = {
    ExtractionType.INVOICE: InvoiceData,
    ExtractionType.CONTRACT: ContractData,
    ExtractionType.REPORT: ReportData,
    ExtractionType.RECEIPT: ReceiptData,
    ExtractionType.GENERIC: GenericExtraction,
}


def get_extraction_model(extraction_type: ExtractionType) -> type[ExtractionResult]:
    """Return the correct Pydantic model for a given document type.

    Raises:
        ValueError: If the extraction type is not mapped.
    """
    model = EXTRACTION_MAP.get(extraction_type)
    if model is None:
        raise ValueError(f"No extraction model mapped for {extraction_type!r}")
    return model

class PipelineError(Exception):
    """Custom exception for pipeline failures."""

    def __init__(self, message: str | None = None, cause: Exception | None = None):
        self.message = message or "Pipeline execution failed"
        self.cause = cause
        super().__init__(self.message)


class ValidationIssue(BaseModel):
    """A single validation issue found in extracted data."""

    field: str
    message: str
    severity: Literal["warning", "error"] = "error"
    value: Any | None = None


class ValidationReport(BaseModel):
    """Report on the validity of extracted data."""

    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")


class IntakeResult(BaseModel):
    """Result from the intake agent (classification + routing)."""

    document_id: str
    extraction_type: ExtractionType  # e.g., "invoice", "contract"
    confidence: float = Field(ge=0.0, le=1.0)
    query: str | None = None  # Suggested query for RAG
    routing_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResult(BaseModel):
    """Result from the RAG agent (retrieval + synthesis)."""

    query: str
    context: str  # Synthesized context from retrieved chunks
    retrieved_chunks: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Final result of a full pipeline run."""

    document_id: str
    original_document: RawDocument | None = None
    extraction: ExtractionResult
    validation: ValidationReport | None = None
    processing_time_ms: float | None = None
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.validation.is_valid if self.validation else True
