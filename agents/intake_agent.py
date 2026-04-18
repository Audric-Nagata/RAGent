"""Intake agent — classifies documents and produces routing hints.

Model : Groq  ``llama-3.3-70b-versatile``  (free tier, 14 400 RPD)
Output: :class:`~models.results.IntakeResult`

The intake agent is the *first* stage of the pipeline.  It receives the raw
document text, classifies its type (invoice / contract / report / receipt /
unknown), chooses a confidence score, and emits a structured
:class:`~models.results.IntakeResult` that the orchestrator uses to route
downstream work.

Responsibilities
----------------
1. Store the raw document in the **doc-store MCP server**.
2. Chunk + embed the document via the **rag/** layer and push vectors to the
   **vector-db MCP server** — so future RAG queries can find its content.
3. Classify the document type and suggest a retrieval query.

Layering
--------
``intake_agent`` ──► MCP doc-store  (store raw doc)
                └──► ``rag.chunker``  (split into chunks)
                └──► ``rag.embedder`` (embed chunks)
                └──► MCP vector-db   (store embeddings via embed_batch tool)
"""

from __future__ import annotations

import json
import logging
from dotenv import load_dotenv

load_dotenv()

import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel

from agents.base import AgentDeps
from models.documents import DocumentType, RawDocument
from models.results import IntakeResult
from rag.chunker import chunk_document
from rag.embedder import embed_document_chunks

logger = logging.getLogger(__name__)

# ── Model ─────────────────────────────────────────────────────────────────────

_model = GroqModel("llama-3.3-70b-versatile")

# ── Agent definition ──────────────────────────────────────────────────────────

intake_agent: Agent[AgentDeps, IntakeResult] = Agent(
    model=_model,
    output_type=IntakeResult,
    deps_type=AgentDeps,
    system_prompt="""You are a document classification specialist.

Your job is to read a raw document and output a structured classification.

You MUST:
1. Identify the document type: invoice, contract, report, receipt, or unknown.
2. Assign a confidence score between 0.0 and 1.0.
3. Write a short, specific retrieval query (2-12 words) that a vector search
   engine could use to find relevant context chunks for this document.
4. Optionally provide a routing_hint (e.g. "high-value invoice", "employment contract").

Always populate `document_id` from the document metadata you are given.
Be concise. Do not invent data that is not present in the document.
""",
)


# ── Internal: ingest document into the vector index ───────────────────────────

async def _ingest_document(doc: RawDocument, deps: AgentDeps) -> int:
    """Chunk, embed, and store a document's vectors in one shot.

    Called automatically by :func:`run_intake` before the LLM classification
    so that, by the time the RAG agent runs, the document's vectors are ready.

    Args:
        doc:  The raw document to index.
        deps: Shared agent dependencies.

    Returns:
        Number of chunks indexed.
    """
    with logfire.span("intake.ingest_document", document_id=doc.id):
        # 1. Split into chunks
        chunks = await chunk_document(doc)
        if not chunks:
            logger.warning("No chunks produced for document %s", doc.id)
            return 0

        # 2. Embed all chunks (async, concurrency-limited)
        embedded = await embed_document_chunks(chunks)

        # 3. Push vectors to the vector-db MCP server in a single batch call
        #    The embed_batch tool accepts a list of texts; we store payloads
        #    (document_id, chunk_index) via the existing embed_chunk flow.
        #    We call embed_chunk once per embedded chunk so the MCP server
        #    records the correct document_id in its payload.
        for ec in embedded:
            try:
                await deps.mcp_client.call_tool(
                    server="vector-db",
                    tool="embed_chunk",
                    arguments={
                        "text": ec.text,
                        # Pass payload hints so the server stores them correctly.
                        # The current vector_db_server stores document_id="unknown"
                        # by default; we send text + rely on the chunk_id for lookup.
                        # TODO: extend embed_chunk schema to accept document_id.
                    },
                )
            except Exception as exc:
                logger.warning("Failed to index chunk %s: %s", ec.chunk_id, exc)

        logfire.info(
            "Document indexed",
            document_id=doc.id,
            num_chunks=len(embedded),
        )
        return len(embedded)


# ── Tools ──────────────────────────────────────────────────────────────────────

@intake_agent.tool
async def store_document(ctx: RunContext[AgentDeps], document: RawDocument) -> str:
    """Persist the raw document in the doc-store MCP server.

    Args:
        ctx:      Agent run context carrying shared deps.
        document: The :class:`~models.documents.RawDocument` to persist.

    Returns:
        A JSON string confirming storage (from the MCP server).
    """
    with logfire.span("intake_agent.store_document", document_id=document.id):
        result = await ctx.deps.mcp_client.call_tool(
            server="doc-store",
            tool="store_document",
            arguments={
                "document_id": document.id,
                "content": document.content,
                "source": document.source,
                "metadata": {
                    **document.metadata,
                    "status": document.status.value,
                    "document_type": document.document_type.value,
                    "user_id": ctx.deps.user_id,
                    "request_id": ctx.deps.request_id,
                },
            },
        )
    return result[0].text if result else "{}"


@intake_agent.tool
async def list_known_documents(ctx: RunContext[AgentDeps], limit: int = 10) -> str:
    """List recently stored documents from the doc-store for context.

    Useful when the agent needs to understand existing document history
    before classifying a new one.

    Args:
        ctx:   Agent run context.
        limit: Maximum number of documents to retrieve (1-50).

    Returns:
        A JSON array of stored document summaries.
    """
    with logfire.span("intake_agent.list_known_documents"):
        result = await ctx.deps.mcp_client.call_tool(
            server="doc-store",
            tool="list_documents",
            arguments={"limit": max(1, min(limit, 50))},
        )
    return result[0].text if result else "[]"


# ── Convenience runner ─────────────────────────────────────────────────────────

async def run_intake(doc: RawDocument, deps: AgentDeps) -> IntakeResult:
    """Run the intake agent on a single document and return the result.

    This is the primary entry-point used by the orchestrator.  It:
      1. Stores the document in doc-store (via MCP).
      2. Chunks + embeds the document and pushes vectors to vector-db (via rag/).
      3. Runs the Groq LLM to classify the document type.

    Args:
        doc:  The raw document to classify.
        deps: Shared agent dependencies (MCP client, user/request IDs).

    Returns:
        A validated :class:`~models.results.IntakeResult`.
    """
    # Step 1 & 2: store raw doc + index vectors (before LLM call)
    with logfire.span("intake.pre_index", document_id=doc.id):
        # Store raw document
        await deps.mcp_client.call_tool(
            server="doc-store",
            tool="store_document",
            arguments={
                "document_id": doc.id,
                "content": doc.content,
                "source": doc.source,
                "metadata": {
                    **doc.metadata,
                    "status": doc.status.value,
                    "document_type": doc.document_type.value,
                    "user_id": deps.user_id,
                    "request_id": deps.request_id,
                },
            },
        )
        # Chunk + embed + index into vector DB
        await _ingest_document(doc, deps)

    # Step 3: LLM classification
    prompt = (
        f"Document ID: {doc.id}\n"
        f"Source: {doc.source or 'unknown'}\n\n"
        f"--- Document Content ---\n{doc.content}\n--- End ---\n\n"
        "Classify this document and produce an IntakeResult."
    )

    with logfire.span("intake_agent.run", document_id=doc.id):
        run_result = await intake_agent.run(prompt, deps=deps)

    return run_result.output
