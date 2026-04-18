"""FastAPI dependency providers — inject shared resources into route handlers.

This module implements the FastAPI dependency-injection layer.  Every
injectable here is an ``async`` generator or callable that FastAPI resolves
per-request (or per-lifespan for singletons).

Resources managed here
----------------------
- :class:`AppDeps` — the concrete implementation of ``AgentDeps`` that wires
  the shared :class:`~mcp.client.MCPClient` into every route.
- :func:`get_deps` — per-request FastAPI dependency that extracts a user ID
  from the request header and assembles an ``AppDeps`` ready for the pipeline.
- :data:`mcp_client_state` — module-level singleton set during lifespan startup
  so it can be safely shared across concurrent requests.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Request, status

from config import Settings
from mcp_server.client import MCPClient


# ── Singleton state (injected by lifespan in main.py) ───────────────────────

# Set during app startup in main.py lifespan; read by get_deps per-request.
_mcp_client: MCPClient | None = None


def set_mcp_client(client: MCPClient) -> None:
    """Register the shared MCPClient instance (called from lifespan)."""
    global _mcp_client
    _mcp_client = client


def get_mcp_client() -> MCPClient:
    """Return the shared MCPClient; raises 503 if not yet initialised."""
    if _mcp_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP client not initialised — server is still starting up.",
        )
    return _mcp_client


# ── AppDeps (concrete AgentDeps for FastAPI) ─────────────────────────────────

class AppDeps:
    """Concrete implementation of the ``AgentDeps`` protocol for FastAPI.

    Satisfies the ``AgentDeps`` protocol (``mcp_client``, ``user_id``,
    ``request_id``) so it can be passed directly to any agent's ``deps``
    parameter without the agents importing anything from this module.

    Attributes:
        mcp_client:  Shared :class:`~mcp.client.MCPClient` singleton.
        user_id:     Value of the ``X-User-ID`` request header (or ``"anon"``).
        request_id:  Fresh UUID generated per request for tracing.
        settings:    Loaded application settings.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        user_id: str,
        request_id: str,
        settings: Settings,
    ) -> None:
        self.mcp_client = mcp_client
        self.user_id = user_id
        self.request_id = request_id
        self.settings = settings


# ── FastAPI dependency functions ─────────────────────────────────────────────

def _get_settings() -> Settings:
    """Load application settings (cached by FastAPI's DI if used with Depends)."""
    return Settings()


async def get_deps(
    request: Request,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    settings: Settings = Depends(_get_settings),
) -> AppDeps:
    """Per-request dependency that assembles :class:`AppDeps`.

    Reads the optional ``X-User-ID`` header (falls back to ``"anon"``),
    generates a fresh ``request_id`` UUID, and returns a populated
    :class:`AppDeps` ready for injection into route handlers.

    Args:
        request:    FastAPI request object (used for future auth middleware).
        x_user_id: Optional ``X-User-ID`` header value.
        settings:  Loaded :class:`~config.Settings`.

    Returns:
        A fully populated :class:`AppDeps` instance.
    """
    client = get_mcp_client()
    return AppDeps(
        mcp_client=client,
        user_id=x_user_id or "anon",
        request_id=uuid.uuid4().hex,
        settings=settings,
    )


# ── Type alias for route annotations ─────────────────────────────────────────

DepsDep = Annotated[AppDeps, Depends(get_deps)]
