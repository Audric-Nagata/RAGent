"""Orchestrator — TaskGroup-based multi-agent pipeline runner.

This module is the *engine* of RAGent.  It coordinates the three agents
(intake → RAG → extractor) using ``asyncio.TaskGroup`` for structured
concurrency and an ``asyncio.Queue`` for backpressure-safe streaming.

Pipeline stages
---------------
1. **Intake agent** (Groq)   — classifies document, gates further processing.
2. **RAG agent**   (Mistral)  — retrieves context from vector DB (concurrent
                               with step 1 output available).
3. **Extractor agent** (Mistral) — structured field extraction using RAG context.

The queue accepts :class:`~models.stream.StreamChunk` payloads and is
consumed by the SSE emitter in ``api/sse.py``.  A bounded ``maxsize=10``
prevents unbounded memory use if the consumer is slow.
"""

from __future__ import annotations

import asyncio
import time

import logfire

from agents.base import AgentDeps
from agents.intake_agent import run_intake
from agents.rag_agent import run_rag
from agents.extractor_agent import run_extractor
from models.documents import RawDocument
from models.results import IntakeResult, RAGResult, PipelineResult, ValidationReport, ValidationIssue
from models.stream import StreamChunk, ProgressData



# ── Internal helpers ──────────────────────────────────────────────────────────

async def _emit_progress(
    queue: asyncio.Queue[StreamChunk],  # type: ignore[type-arg]
    stage: str,
    message: str,
    percentage: float | None = None,
) -> None:
    """Put a ``progress`` event onto the queue (non-blocking if room)."""
    chunk: StreamChunk[ProgressData] = StreamChunk(
        event="progress",
        payload=ProgressData(stage=stage, message=message, percentage=percentage),
    )
    try:
        queue.put_nowait(chunk)
    except asyncio.QueueFull:
        pass  # Drop progress event rather than stall the pipeline


def _validate_extraction(result: PipelineResult) -> ValidationReport:
    """Run basic sanity checks on the extraction result.

    Returns a :class:`~models.results.ValidationReport` with any issues found.
    """
    issues: list[ValidationIssue] = []
    extraction = result.extraction

    # Confidence sanity check
    if extraction.confidence < 0.5:
        issues.append(
            ValidationIssue(
                field="confidence",
                message=f"Low extraction confidence: {extraction.confidence:.2f}",
                severity="warning",
                value=extraction.confidence,
            )
        )

    return ValidationReport(
        is_valid=all(i.severity != "error" for i in issues),
        issues=issues,
    )


# ── Public pipeline entry-point ───────────────────────────────────────────────

async def run_pipeline(
    file_bytes: bytes,
    filename: str,
    doc_type: str,
    metadata: dict,
    queue: asyncio.Queue[StreamChunk],  # type: ignore[type-arg]
    deps: AgentDeps,
) -> None:
    """Execute the full three-stage document intelligence pipeline.

    This coroutine is designed to run as an ``asyncio`` task.  It pushes
    ``StreamChunk`` events onto *queue* as processing progresses and puts a
    final ``complete`` or ``error`` chunk when done.

    Args:
        file_bytes: The raw document bytes.
        filename: Original file name.
        doc_type: Document type hint.
        metadata: External metadata map.
        queue: Bounded asyncio Queue consumed by the SSE emitter.
        deps:  Shared agent dependencies (MCP client, user/request IDs).

    Queue events emitted (in order):
        - ``progress`` — extracting_text
        - ``progress`` — classifying
        - ``progress`` — retrieving
        - ``progress`` — extracting
        - ``complete``  — final :class:`~models.results.PipelineResult`
        - ``error``     — on any unhandled exception
    """
    start_ms = time.monotonic() * 1000
    
    from utils.document_parser import parse_document
    from models.documents import DocumentType
    
    try:
        document_enum = DocumentType(doc_type)
    except ValueError:
        document_enum = DocumentType.UNKNOWN

    with logfire.span(
        "orchestrator.run_pipeline",
        filename=filename,
        user_id=deps.user_id,
        request_id=deps.request_id,
    ):
        try:
            # ── Stage 0: Text Extraction ──────────────────────────────────────
            await _emit_progress(
                queue, stage="extracting_text",
                message="Extracting document text...", percentage=5.0,
            )
            
            content = await parse_document(file_bytes, filename)
            
            doc = RawDocument(
                content=content,
                source=filename,
                document_type=document_enum,
                metadata=metadata or {},
            )
            
            logfire.info("Successfully extracted document text", document_id=doc.id)

            # ── Stage 1: Intake (serial — gates everything else) ──────────────
            await _emit_progress(
                queue, stage="classifying",
                message="Classifying document type…", percentage=10.0,
            )

            async with asyncio.TaskGroup() as tg:
                intake_task = tg.create_task(
                    run_intake(doc, deps),
                    name="intake",
                )

            intake: IntakeResult = intake_task.result()
            logfire.info(
                "Intake complete",
                document_id=doc.id,
                extraction_type=intake.extraction_type.value,
                confidence=intake.confidence,
            )

            # ── Stage 2: RAG (can run as soon as intake is done) ──────────────
            await _emit_progress(
                queue, stage="retrieving",
                message=f"Retrieving context for \"{intake.query}\"…",
                percentage=40.0,
            )

            async with asyncio.TaskGroup() as tg:
                rag_task = tg.create_task(
                    run_rag(
                        query=intake.query or doc.content[:200],
                        deps=deps,
                        document_id=intake.document_id,
                    ),
                    name="rag",
                )

            rag_result: RAGResult = rag_task.result()
            logfire.info(
                "RAG complete",
                document_id=doc.id,
                num_chunks=len(rag_result.retrieved_chunks),
                rag_confidence=rag_result.confidence,
            )

            # ── Stage 3: Extraction ───────────────────────────────────────────
            await _emit_progress(
                queue, stage="extracting",
                message="Extracting structured fields…", percentage=70.0,
            )

            extraction = await run_extractor(
                doc_content=doc.content,
                intake=intake,
                rag_result=rag_result,
                deps=deps,
            )
            logfire.info(
                "Extraction complete",
                document_id=doc.id,
                extraction_type=extraction.extraction_type.value,
            )

            # ── Assemble final result ─────────────────────────────────────────
            elapsed_ms = time.monotonic() * 1000 - start_ms

            pipeline_result = PipelineResult(
                document_id=doc.id,
                original_document=doc,
                extraction=extraction,
                processing_time_ms=round(elapsed_ms, 2),
                metadata={
                    "user_id": deps.user_id,
                    "request_id": deps.request_id,
                    "rag_confidence": rag_result.confidence,
                    "intake_confidence": intake.confidence,
                },
            )
            pipeline_result = pipeline_result.model_copy(
                update={"validation": _validate_extraction(pipeline_result)}
            )

            await _emit_progress(
                queue, stage="done",
                message="Pipeline complete.", percentage=100.0,
            )

            complete_chunk: StreamChunk[PipelineResult] = StreamChunk(
                event="complete",
                payload=pipeline_result,
            )
            await queue.put(complete_chunk)

        except* Exception as eg:
            # ``except*`` handles ExceptionGroups from TaskGroup
            first = eg.exceptions[0]
            logfire.error(
                "Pipeline failed",
                document_id=doc.id,
                error=str(first),
                exc_info=first,
            )
            error_chunk: StreamChunk[PipelineResult] = StreamChunk(
                event="error",
                error=str(first),
            )
            await queue.put(error_chunk)
