"""Extractor agent — structured field extraction from documents.

Model : HuggingFace  ``mistralai/Ministral-3-3B-Reasoning-2512-GGUF``
Output: :class:`~models.extractions.ExtractionResult`  (discriminated union)

The extractor agent is the *third and final* stage of the pipeline.  It
receives:
  - the original document content
  - the RAG context synthesised by the RAG agent
  - the document type determined by the intake agent

It then produces a fully-typed, validated extraction matching the correct
Pydantic model for the detected document type (InvoiceData, ContractData,
ReportData, ReceiptData, or GenericExtraction).
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.huggingface import HuggingFaceModel

from agents.base import AgentDeps
from models.documents import DocumentType
from models.extractions import (
    ExtractionResult,
    ExtractionType,
    InvoiceData,
    ContractData,
    ReportData,
    ReceiptData,
    GenericExtraction,
)
from models.results import IntakeResult, RAGResult

# ── Model ─────────────────────────────────────────────────────────────────────

_model = HuggingFaceModel("mistralai/Ministral-3-3B-Reasoning-2512-GGUF")

# ── Per-type agents (all share the same model, different system prompts) ───────

_INVOICE_SYSTEM = """You are a financial document specialist extracting invoice data.
Extract every field you can find. Leave fields as null if not present.
Always populate: invoice_number, invoice_date, vendor_name, total_amount.
Output MUST be a valid InvoiceData object."""

_CONTRACT_SYSTEM = """You are a legal document specialist extracting contract data.
Extract parties, dates, governing law, and key clauses.
Output MUST be a valid ContractData object."""

_REPORT_SYSTEM = """You are a business analyst extracting structured data from reports.
Extract the title, author, executive summary, sections, and key findings.
Output MUST be a valid ReportData object."""

_RECEIPT_SYSTEM = """You are a receipt extraction specialist.
Extract merchant, date, total, payment method, and line items.
Output MUST be a valid ReceiptData object."""

_GENERIC_SYSTEM = """You are a general document analyst.
Extract a title, summary, key entities, dates, and monetary amounts.
Output MUST be a valid GenericExtraction object."""

_invoice_agent: Agent[AgentDeps, InvoiceData] = Agent(
    model=_model, output_type=InvoiceData, deps_type=AgentDeps,
    system_prompt=_INVOICE_SYSTEM,
)
_contract_agent: Agent[AgentDeps, ContractData] = Agent(
    model=_model, output_type=ContractData, deps_type=AgentDeps,
    system_prompt=_CONTRACT_SYSTEM,
)
_report_agent: Agent[AgentDeps, ReportData] = Agent(
    model=_model, output_type=ReportData, deps_type=AgentDeps,
    system_prompt=_REPORT_SYSTEM,
)
_receipt_agent: Agent[AgentDeps, ReceiptData] = Agent(
    model=_model, output_type=ReceiptData, deps_type=AgentDeps,
    system_prompt=_RECEIPT_SYSTEM,
)
_generic_agent: Agent[AgentDeps, GenericExtraction] = Agent(
    model=_model, output_type=GenericExtraction, deps_type=AgentDeps,
    system_prompt=_GENERIC_SYSTEM,
)

# Map from DocumentType → concrete agent
_AGENT_MAP = {
    DocumentType.INVOICE:  _invoice_agent,
    DocumentType.CONTRACT: _contract_agent,
    DocumentType.REPORT:   _report_agent,
    DocumentType.RECEIPT:  _receipt_agent,
    DocumentType.UNKNOWN:  _generic_agent,
}


# ── Shared tool (all sub-agents can call it if needed) ────────────────────────

def _register_lookup_tool(agent: Agent) -> None:  # type: ignore[type-arg]
    """Register a doc-store lookup tool on a sub-agent."""

    @agent.tool  # type: ignore[attr-defined]
    async def lookup_document(ctx: RunContext[AgentDeps], document_id: str) -> str:
        """Retrieve the full stored document text from the doc-store MCP server.

        Args:
            ctx:         Agent run context.
            document_id: The ID of the document to retrieve.

        Returns:
            A JSON string of the stored document.
        """
        with logfire.span("extractor_agent.lookup_document", document_id=document_id):
            result = await ctx.deps.mcp_client.call_tool(
                server="doc-store",
                tool="get_document",
                arguments={"document_id": document_id},
            )
        return result[0].text if result else "{}"


for _a in _AGENT_MAP.values():
    _register_lookup_tool(_a)


# ── Convenience runner ─────────────────────────────────────────────────────────

async def run_extractor(
    doc_content: str,
    intake: IntakeResult,
    rag_result: RAGResult,
    deps: AgentDeps,
) -> ExtractionResult:
    """Run the appropriate extractor agent for the document type.

    Routes to the correct typed sub-agent based on the intake classification,
    then validates and returns the structured extraction result.

    Args:
        doc_content: The original raw document text.
        intake:      The :class:`~models.results.IntakeResult` from stage 1.
        rag_result:  The :class:`~models.results.RAGResult` from stage 2.
        deps:        Shared agent dependencies.

    Returns:
        A validated :class:`~models.extractions.ExtractionResult` (typed
        union: InvoiceData | ContractData | ReportData | ReceiptData |
        GenericExtraction).
    """
    # Map ExtractionType → DocumentType for agent lookup
    try:
        doc_type = DocumentType(intake.extraction_type.value)
    except ValueError:
        doc_type = DocumentType.UNKNOWN

    agent = _AGENT_MAP.get(doc_type, _generic_agent)

    prompt = (
        f"Document ID : {intake.document_id}\n"
        f"Document type: {intake.extraction_type.value}\n\n"
        f"--- RAG Context ---\n{rag_result.context}\n--- End Context ---\n\n"
        f"--- Original Document ---\n{doc_content}\n--- End Document ---\n\n"
        "Extract all structured fields from this document."
    )

    with logfire.span(
        "extractor_agent.run",
        document_id=intake.document_id,
        doc_type=doc_type.value,
    ):
        run_result = await agent.run(prompt, deps=deps)

    return run_result.output  # type: ignore[return-value]
