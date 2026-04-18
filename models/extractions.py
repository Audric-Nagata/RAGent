"""Pydantic models for structured extraction results by document type."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ExtractionType(str, Enum):
    """The type of extraction performed."""

    INVOICE = "invoice"
    CONTRACT = "contract"
    REPORT = "report"
    RECEIPT = "receipt"
    GENERIC = "generic"


class BaseExtraction(BaseModel):
    """Base class for all extraction results."""

    extraction_type: ExtractionType
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    raw_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class LineItem(BaseModel):
    """A single line item, e.g., on an invoice or receipt."""

    description: str
    quantity: float | None = None
    unit_price: float | None = None
    total: float | None = None
    tax: float | None = None


class InvoiceData(BaseExtraction):
    """Structured data extracted from an invoice."""

    extraction_type: Literal[ExtractionType.INVOICE] = ExtractionType.INVOICE
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    vendor_name: str | None = None
    vendor_address: str | None = None
    customer_name: str | None = None
    customer_address: str | None = None
    subtotal: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None
    currency: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)


class ContractClause(BaseModel):
    """A single clause extracted from a contract."""

    clause_type: str  # e.g., "termination", "liability", "confidentiality"
    text: str
    summary: str | None = None
    effective_date: date | None = None
    expiration_date: date | None = None


class ContractData(BaseExtraction):
    """Structured data extracted from a contract."""

    extraction_type: Literal[ExtractionType.CONTRACT] = ExtractionType.CONTRACT
    contract_title: str | None = None
    contract_date: date | None = None
    effective_date: date | None = None
    expiration_date: date | None = None
    parties: list[str] = Field(default_factory=list)
    governing_law: str | None = None
    clauses: list[ContractClause] = Field(default_factory=list)
    key_obligations: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    """A section extracted from a report."""

    title: str
    content: str
    page_number: int | None = None


class ReportData(BaseExtraction):
    """Structured data extracted from a report."""

    extraction_type: Literal[ExtractionType.REPORT] = ExtractionType.REPORT
    report_title: str | None = None
    report_date: date | None = None
    author: str | None = None
    executive_summary: str | None = None
    sections: list[ReportSection] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)


class ReceiptData(BaseExtraction):
    """Structured data extracted from a receipt."""

    extraction_type: Literal[ExtractionType.RECEIPT] = ExtractionType.RECEIPT
    merchant_name: str | None = None
    merchant_address: str | None = None
    transaction_date: date | None = None
    transaction_time: datetime | None = None
    total_amount: float | None = None
    tax_amount: float | None = None
    payment_method: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)


class GenericExtraction(BaseExtraction):
    """Fallback extraction for unknown/unclassified document types."""

    extraction_type: Literal[ExtractionType.GENERIC] = ExtractionType.GENERIC
    title: str | None = None
    summary: str | None = None
    key_entities: list[str] = Field(default_factory=list)
    dates_found: list[date] = Field(default_factory=list)
    amounts_found: list[float] = Field(default_factory=list)


# Union type for all possible extraction outputs
ExtractionResult = (
    InvoiceData | ContractData | ReportData | ReceiptData | GenericExtraction
)
