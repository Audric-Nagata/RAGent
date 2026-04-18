"""MCP server: document store — read/write parsed documents."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, EmbeddedResource

# Use stderr for logging — stdout is reserved for the MCP JSON-RPC wire protocol
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

from mcp_server.schemas import (
    StoreDocumentInput,
    GetDocumentInput,
    ListDocumentsInput,
    StoredDocument,
)

app = Server("doc-store")

# In-memory store — replace with a real DB in production
_store: dict[str, StoredDocument] = {}


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="store_document",
            description="Store a parsed document for later retrieval",
            inputSchema=StoreDocumentInput.model_json_schema(),
        ),
        Tool(
            name="get_document",
            description="Retrieve a document by its ID",
            inputSchema=GetDocumentInput.model_json_schema(),
        ),
        Tool(
            name="list_documents",
            description="List stored documents with optional filtering",
            inputSchema=ListDocumentsInput.model_json_schema(),
        ),
    ]


@app.call_tool()
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> list[TextContent | EmbeddedResource]:
    match name:
        case "store_document":
            inp = StoreDocumentInput.model_validate(arguments)
            doc = StoredDocument(
                document_id=inp.document_id,
                content=inp.content,
                source=inp.source,
                metadata=inp.metadata,
                stored_at=datetime.utcnow().isoformat(),
            )
            _store[doc.document_id] = doc
            return [_text(doc.model_dump_json())]

        case "get_document":
            inp = GetDocumentInput.model_validate(arguments)
            doc = _store.get(inp.document_id)
            if doc is None:
                return [_text(f'{{"error": "Document {inp.document_id} not found"}}')]
            return [_text(doc.model_dump_json())]

        case "list_documents":
            inp = ListDocumentsInput.model_validate(arguments)
            docs = list(_store.values())
            if inp.status:
                docs = [d for d in docs if d.metadata.get("status") == inp.status]
            docs = docs[inp.offset : inp.offset + inp.limit]
            payload = [d.model_dump_json() for d in docs]
            return [_text(f"[{','.join(payload)}]")]

        case _:
            raise ValueError(f"Unknown tool: {name}")


def _text(content: str) -> TextContent:
    return TextContent(type="text", text=content)


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
