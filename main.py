"""RAGent application entrypoint.

Startup sequence
----------------
1. Validate settings (Pydantic-settings raises on missing required env vars).
2. Configure Logfire tracing (no-op if token is empty).
3. Connect to all three MCP servers concurrently via :class:`~mcp.client.MCPClient`.
4. Mount the FastAPI router.
5. Expose the ASGI app for Uvicorn.

Run locally::

    uvicorn main:app --reload --port 8000

Or with the module path::

    python -m uvicorn main:app --reload
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os
from dotenv import load_dotenv
load_dotenv()

import logfire
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import set_mcp_client
from api.routes import router
from config import Settings
from mcp_server.client import MCPClient

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── MCP server configuration ──────────────────────────────────────────────────

def _build_server_configs() -> dict[str, dict[str, object]]:
    """Return the MCP server subprocess configurations.

    Each entry maps a server name to the ``command`` and ``args`` needed to
    launch it as a stdio subprocess.  Adjust ``args`` if your project root
    differs from the CWD.
    """
    return {
        "doc-store": {
            "command": "python",
            "args": ["-m", "mcp_server.servers.doc_store_server"],
        },
        "vector-db": {
            "command": "python",
            "args": ["-m", "mcp_server.servers.vector_db_server"],
        },
        "web-search": {
            "command": "python",
            "args": ["-m", "mcp_server.servers.web_search_server"],
        },
    }


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager.

    Runs startup logic before the first request and teardown after the last.
    """
    settings = Settings()
    
    # ── Logfire ──────────────────────────────────────────────────────────────
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if logfire_token:
        logfire.configure(token=logfire_token.strip())
        logfire.instrument_fastapi(app)
        logger.info("Logfire tracing enabled.")
    else:
        logger.warning(
            "LOGFIRE_TOKEN not set — tracing disabled. "
            "Set it in .env to enable full observability."
        )

    # ── MCP client startup ────────────────────────────────────────────────────
    server_configs = _build_server_configs()
    client = MCPClient(server_configs=server_configs)

    logger.info("Connecting to MCP servers: %s", list(server_configs.keys()))
    try:
        await client.connect_all()
        set_mcp_client(client)
        logger.info("All MCP servers connected.")
    except Exception as exc:
        logger.exception("Failed to connect to MCP servers: %s", exc)
        raise RuntimeError("MCP startup failed — aborting.") from exc

    # ── Hand control to the application ──────────────────────────────────────
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down — disconnecting MCP servers…")
    await client.disconnect_all()
    logger.info("MCP servers disconnected. Goodbye.")


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title="RAGent",
    description=(
        "Multi-Agent Document Intelligence Pipeline with MCP and Typed RAG. "
        "Streams structured JSON extraction results via SSE."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins in development; lock this down for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes under /api/v1
app.include_router(router, prefix="/api/v1")


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
