"""Shared MCP client — manages connections to multiple MCP servers."""

from __future__ import annotations

import asyncio
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """Unified client for interacting with multiple MCP servers via stdio.

    Each server is registered with a name and its command/args.
    Sessions are created on demand and cleaned up via context management.
    """

    def __init__(self, server_configs: dict[str, dict[str, Any]] | None = None):
        """Initialize the MCP client.

        Args:
            server_configs: Mapping of server_name → {
                "command": str,       # e.g., "python"
                "args": list[str],    # e.g., ["-m", "mcp.servers.doc_store_server"]
            }
        """
        self._server_configs: dict[str, dict[str, Any]] = server_configs or {}
        self._sessions: dict[str, ClientSession] = {}
        self._contexts: dict[str, Any] = {}  # stdio_client context managers
        self._lock = asyncio.Lock()

    async def connect_server(self, server_name: str) -> None:
        """Connect to a single MCP server by name.

        Args:
            server_name: The registered name of the server.

        Raises:
            ValueError: If the server is not configured.
        """
        config = self._server_configs.get(server_name)
        if config is None:
            raise ValueError(f"Server '{server_name}' is not configured")

        params = StdioServerParameters(
            command=config["command"],
            args=config["args"],
        )

        ctx = stdio_client(params)
        read, write = await ctx.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()

        async with self._lock:
            self._sessions[server_name] = session
            self._contexts[server_name] = ctx

    async def connect_all(self) -> None:
        """Connect to all configured servers sequentially.

        Note: servers must be connected sequentially (not with asyncio.gather)
        because mcp's stdio_client uses anyio cancel scopes internally, which
        must be entered and exited within the same task. Concurrent tasks
        (via gather) cross task boundaries and cause a RuntimeError.
        """
        for name in self._server_configs:
            await self.connect_server(name)

    async def disconnect_server(self, server_name: str) -> None:
        """Disconnect from a single MCP server."""
        async with self._lock:
            session = self._sessions.pop(server_name, None)
            ctx = self._contexts.pop(server_name, None)

        if session is not None:
            await session.__aexit__(None, None, None)
        if ctx is not None:
            await ctx.__aexit__(None, None, None)

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        server_names = list(self._sessions.keys())
        for name in server_names:
            await self.disconnect_server(name)

    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any],
    ) -> list[Any]:
        """Call a tool on a specific MCP server.

        Args:
            server: Target server name.
            tool: Tool name to invoke.
            arguments: Tool input arguments.

        Returns:
            List of content objects returned by the tool.

        Raises:
            ValueError: If the server is not connected.
        """
        session = self._sessions.get(server)
        if session is None:
            raise ValueError(f"Server '{server}' is not connected")

        result = await session.call_tool(tool, arguments)
        return result.content

    async def list_tools(self, server: str) -> list[Any]:
        """List available tools on a specific server.

        Args:
            server: Target server name.

        Returns:
            List of tool descriptors.
        """
        session = self._sessions.get(server)
        if session is None:
            raise ValueError(f"Server '{server}' is not connected")

        result = await session.list_tools()
        return result.tools

    async def __aenter__(self) -> MCPClient:
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect_all()
