"""Bridge: re-exports MCP type primitives from the installed MCP SDK.

With local ``mcp/`` shadowing the installed package, ``from mcp.types import X``
looks here (``mcp/types.py``) first.  We delegate to the installed SDK's
``types`` module via the ``_mcp_sdk`` alias.
"""

from __future__ import annotations

import importlib

_types_mod = importlib.import_module("_mcp_sdk.types")

Tool = _types_mod.Tool
TextContent = _types_mod.TextContent
EmbeddedResource = _types_mod.EmbeddedResource

__all__ = ["Tool", "TextContent", "EmbeddedResource"]
