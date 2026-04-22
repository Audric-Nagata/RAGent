"""FastAPI route handlers — REST + SSE endpoints for the RAGent pipeline.

Endpoints
---------

POST /process
    Submit a document for processing.  The pipeline runs immediately and
    streams SSE events back on the same connection.  This is the primary
    endpoint for real-time document intelligence.

POST /process/async
    Submit a document and receive a ``request_id`` immediately.  The pipeline
    runs in the background; results can be polled via GET /results/{request_id}.

GET /stream/{request_id}
    Re-attach to a running pipeline's SSE stream by request ID.
    (Only works while the pipeline task is still alive in the current process.)

GET /results/{request_id}
    Poll for the final :class:`~models.results.PipelineResult` of a completed
    pipeline run.  Returns 202 while still in-progress, 200 when done.

GET /health
    Liveness probe — returns 200 with basic status info.

GET /docs-list
    List documents currently stored in the doc-store MCP server.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.dependencies import AppDeps, DepsDep
from api.sse import make_stream_response
from models.documents import DocumentType
from models.results import PipelineResult

logger = logging.getLogger(__name__)

router = APIRouter()

# ── In-memory result store (replace with a real DB in production) ─────────────
# Maps request_id → PipelineResult (or None if still in progress)
_results: dict[str, PipelineResult | None] = {}
_stream_queues: dict[str, asyncio.Queue[Any]] = {}


# ── Request / response schemas ────────────────────────────────────────────────

class ProcessAccepted(BaseModel):
    """Response body for POST /process/async."""

    request_id: str
    document_id: str
    message: str = "Document accepted. Use GET /results/{request_id} to poll."


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    mcp_connected: bool


# ── Health probe ──────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["admin"],
)
async def health(deps: DepsDep) -> HealthResponse:
    """Return 200 if the server is alive and the MCP client is connected."""
    try:
        # Try listing tools on doc-store as a lightweight MCP ping
        await deps.mcp_client.list_tools("doc-store")
        mcp_ok = True
    except Exception:
        mcp_ok = False

    return HealthResponse(status="ok", mcp_connected=mcp_ok)


# ── Synchronous streaming process (primary endpoint) ─────────────────────────

@router.post(
    "/process",
    summary="Submit a document and stream SSE results",
    response_class=StreamingResponse,
    tags=["pipeline"],
)
async def process_document(
    deps: DepsDep,
    file: UploadFile = File(...),
    document_type: str = Form("unknown"),
    metadata: str = Form("{}"),
) -> StreamingResponse:
    """Process a document and stream :class:`~models.stream.StreamChunk` events.

    The response is a Server-Sent Events stream.  Connect with::

        const es = new EventSource('/process');  // or fetch with ReadableStream

    Each event is a JSON-serialised :class:`~models.stream.StreamChunk`.
    The stream ends with ``event=complete`` (success) or ``event=error``.

    Events in order:

    1. ``progress`` — extracting_text (5%)
    2. ``progress`` — classifying (10%)
    3. ``progress`` — retrieving (40%)
    4. ``progress`` — extracting (70%)
    5. ``progress`` — done (100%)
    6. ``complete`` — :class:`~models.results.PipelineResult` payload

    Args:
        deps: Injected :class:`~api.dependencies.AppDeps`.
        file: Uploaded PDF or DOCX file.
        document_type: Hint for the document type.
        metadata: JSON string for arbitrary metadata.

    Returns:
        A ``text/event-stream`` ``StreamingResponse``.
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"
    try:
        meta_dict = json.loads(metadata)
    except json.JSONDecodeError:
        meta_dict = {}

    logger.info(
        "Streaming pipeline started",
        extra={"filename": filename, "request_id": deps.request_id},
    )
    return make_stream_response(file_bytes, filename, document_type, meta_dict, deps)


# ── Async (fire-and-forget) process ──────────────────────────────────────────

@router.post(
    "/process/async",
    response_model=ProcessAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a document for background processing",
    tags=["pipeline"],
)
async def process_document_async(
    background_tasks: BackgroundTasks,
    deps: DepsDep,
    file: UploadFile = File(...),
    document_type: str = Form("unknown"),
    metadata: str = Form("{}"),
) -> ProcessAccepted:
    """Submit a document for background processing and return immediately.

    The pipeline runs as an ``asyncio`` background task.  Poll the result via
    ``GET /results/{request_id}``.

    Args:
        background_tasks: FastAPI ``BackgroundTasks`` for fire-and-forget.
        deps:             Injected :class:`~api.dependencies.AppDeps`.
        file: Uploaded PDF or DOCX file.
        document_type: Hint for the document type.
        metadata: JSON string for arbitrary metadata.

    Returns:
        A :class:`ProcessAccepted` with ``request_id`` and ``document_id``.
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"
    try:
        meta_dict = json.loads(metadata)
    except json.JSONDecodeError:
        meta_dict = {}

    request_id = deps.request_id
    _results[request_id] = None  # Mark as in-progress

    async def _run() -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=10)
        _stream_queues[request_id] = queue
        from agents.orchestrator import run_pipeline

        await run_pipeline(file_bytes, filename, document_type, meta_dict, queue, deps)

        # Drain the queue to find the final result
        while not queue.empty():
            chunk = queue.get_nowait()
            if chunk.event == "complete" and chunk.payload is not None:
                _results[request_id] = chunk.payload
                break
            elif chunk.event == "error":
                logger.error(
                    "Background pipeline error (request_id=%s): %s",
                    request_id, chunk.error,
                )
                break

        _stream_queues.pop(request_id, None)

    background_tasks.add_task(_run)

    logger.info(
        "Async pipeline accepted",
        extra={"filename": filename, "request_id": request_id},
    )
    return ProcessAccepted(
        request_id=request_id,
        document_id="generated-post-extraction",
    )


# ── Result polling ────────────────────────────────────────────────────────────

@router.get(
    "/results/{request_id}",
    summary="Poll for a pipeline result",
    tags=["pipeline"],
)
async def get_result(request_id: str) -> PipelineResult | dict[str, str]:
    """Return the :class:`~models.results.PipelineResult` for a completed run.

    - **200** — pipeline finished; returns the full ``PipelineResult``.
    - **202** — pipeline still in progress.
    - **404** — unknown ``request_id``.

    Args:
        request_id: The ``request_id`` returned by ``POST /process/async``.
    """
    if request_id not in _results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pipeline run found for request_id={request_id!r}.",
        )

    result = _results[request_id]
    if result is None:
        # Still in progress
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"status": "in_progress", "request_id": request_id},
        )

    return result


# ── Document listing ──────────────────────────────────────────────────────────

@router.get(
    "/docs-list",
    summary="List stored documents",
    tags=["admin"],
)
async def list_documents(
    deps: DepsDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
) -> dict[str, Any]:
    """List documents stored in the doc-store MCP server.

    Args:
        deps:          Injected deps (provides the MCP client).
        limit:         Max documents to return (1-200).
        offset:        Pagination offset.
        status_filter: Optional filter by document status string.

    Returns:
        A dict with ``items`` (list of stored documents) and ``count``.
    """
    arguments: dict[str, Any] = {"limit": limit, "offset": offset}
    if status_filter:
        arguments["status"] = status_filter

    raw = await deps.mcp_client.call_tool(
        server="doc-store",
        tool="list_documents",
        arguments=arguments,
    )

    import json
    items: list[Any] = []
    if raw:
        try:
            items = json.loads(raw[0].text)
        except (json.JSONDecodeError, AttributeError):
            pass

    return {"items": items, "count": len(items)}
