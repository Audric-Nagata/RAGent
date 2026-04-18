"""Shared agent dependencies protocol and typed context for all agents."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mcp_server.client import MCPClient


@runtime_checkable
class AgentDeps(Protocol):
    """Protocol defining the shared dependency context injected into every agent.

    Concrete implementations (e.g. AppDeps in api/dependencies.py) must
    satisfy this interface.  Using a Protocol — not a base class — keeps
    agents fully decoupled from the FastAPI layer.

    Attributes:
        mcp_client: Shared MCP client connected to all three servers
                    (doc-store, vector-db, web-search).
        user_id:    Identifier for the user/session running the pipeline.
        request_id: Unique ID for the current pipeline run (for tracing).
    """

    mcp_client: MCPClient
    user_id: str
    request_id: str


class ConcreteAgentDeps:
    """Minimal concrete implementation of AgentDeps for direct use.

    Use this in tests, CLI scripts, or whenever you need to construct deps
    without the full FastAPI dependency-injection machinery.

    Example::

        async with MCPClient(server_configs) as client:
            deps = ConcreteAgentDeps(
                mcp_client=client,
                user_id="local",
                request_id="run-001",
            )
            result = await intake_agent.run(doc.content, deps=deps)
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        user_id: str = "default",
        request_id: str = "default",
    ) -> None:
        self.mcp_client = mcp_client
        self.user_id = user_id
        self.request_id = request_id
