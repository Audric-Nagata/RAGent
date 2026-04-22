"""SSE emitter — converts an asyncio.Queue into a Server-Sent Events stream.

This module owns the stream lifecycle:
  1. A background ``asyncio`` task runs :func:`~agents.orchestrator.run_pipeline`.
  2. The pipeline pushes :class:`~models.stream.StreamChunk` objects onto a
     bounded ``asyncio.Queue``.
  3. :func:`stream_pipeline` drains the queue and yields SSE-formatted lines
     to the FastAPI ``StreamingResponse``.

SSE format
----------
Each event is serialised as::

    data: <StreamChunk JSON>\\n\\n

The stream terminates when a chunk with ``event="complete"`` or
``event="error"`` is dequeued.

Backpressure
------------
The queue ``maxsize=10`` prevents unbounded memory growth when the HTTP
consumer is slow.  Progress events are dropped (non-blocking put) inside the
orchestrator, so the pipeline is never stalled by a slow SSE client.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi.responses import StreamingResponse

from agents.orchestrator import run_pipeline
from api.dependencies import AppDeps
from models.results import PipelineResult
from models.stream import StreamChunk

logger = logging.getLogger(__name__)

_QUEUE_MAX: int = 10


# ── Core SSE generator ────────────────────────────────────────────────────────

async def _event_generator(
    file_bytes: bytes,
    filename: str,
    doc_type: str,
    metadata: dict,
    deps: AppDeps,
) -> AsyncGenerator[str, None]:
    """Async generator that drives the pipeline and yields SSE event strings.

    Args:
        file_bytes: The raw document bytes.
        filename: Original filename.
        doc_type: The document type hint.
        metadata: Metadata map.
        deps: Populated ``AppDeps`` carrying the MCP client and IDs.

    Yields:
        SSE-formatted strings: ``"data: <JSON>\\n\\n"``
    """
    queue: asyncio.Queue[StreamChunk[Any]] = asyncio.Queue(maxsize=_QUEUE_MAX)
    sequence: int = 0

    # Fire off the pipeline as a background task
    pipeline_task = asyncio.create_task(
        run_pipeline(file_bytes, filename, doc_type, metadata, queue, deps),
        name=f"pipeline-{deps.request_id}",
    )

    try:
        while True:
            try:
                # Wait up to 30 s for the next chunk before giving up
                chunk = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "SSE stream timed out waiting for chunk (request_id=%s)",
                    deps.request_id,
                )
                timeout_chunk: StreamChunk[PipelineResult] = StreamChunk(
                    event="error",
                    error="Pipeline timed out — no response within 30 seconds.",
                    sequence=sequence,
                )
                yield _format_sse(timeout_chunk)
                break

            # Stamp the sequence number so the client can detect gaps
            stamped = chunk.model_copy(update={"sequence": sequence})
            sequence += 1

            yield _format_sse(stamped)
            queue.task_done()

            if stamped.event in ("complete", "error"):
                break

    finally:
        # Cancel the pipeline task if the client disconnects early
        if not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except (asyncio.CancelledError, Exception):
                pass


def _format_sse(chunk: StreamChunk[Any]) -> str:
    """Serialise a StreamChunk to an SSE data line.

    Args:
        chunk: The :class:`~models.stream.StreamChunk` to serialise.

    Returns:
        A string in the form ``"data: <JSON>\\n\\n"``.
    """
    # model_dump_json handles Generic[T] correctly in Pydantic v2
    return f"data: {chunk.model_dump_json()}\n\n"


# ── Public streaming response factory ────────────────────────────────────────

def make_stream_response(
    file_bytes: bytes,
    filename: str,
    doc_type: str,
    metadata: dict,
    deps: AppDeps,
) -> StreamingResponse:
    """Create a :class:`fastapi.responses.StreamingResponse` for a pipeline run.

    This is the single function called by route handlers.  It wires the
    async generator to the FastAPI streaming machinery.

    Args:
        file_bytes: Original file bytes.
        filename: Name of the uploaded file.
        doc_type: Expected document type.
        metadata: Meta dictionary.
        deps: Populated ``AppDeps`` for this request.

    Returns:
        A ``StreamingResponse`` with ``Content-Type: text/event-stream``.
    """
    return StreamingResponse(
        _event_generator(file_bytes, filename, doc_type, metadata, deps),
        media_type="text/event-stream",
        headers={
            # Prevent proxies/browsers from buffering the stream
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": deps.request_id,
        },
    )
