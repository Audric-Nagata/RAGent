"""Bridge: re-exports mcp.server.Server from the installed MCP SDK.

With the local ``mcp/`` package taking precedence, ``from mcp.server import Server``
would look for ``mcp/server.py`` locally (this file) rather than in the installed
SDK.  We load the installed SDK's server module via the ``_mcp_sdk`` alias
registered in ``mcp/__init__.py`` and re-export ``Server``.
"""

from __future__ import annotations

import importlib

# _mcp_sdk is registered in sys.modules by mcp/__init__.py at import time
_server_mod = importlib.import_module("_mcp_sdk.server")

Server = _server_mod.Server

__all__ = ["Server"]
