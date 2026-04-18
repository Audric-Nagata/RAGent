"""MCP server: web search — wraps Tavily Search API."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, EmbeddedResource

# Use stderr for logging — stdout is reserved for the MCP JSON-RPC wire protocol
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

from mcp_server.schemas import SearchInput, SearchResult, SearchResults

app = Server("web-search")

TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description="Search the web using Tavily for supplementary information",
            inputSchema=SearchInput.model_json_schema(),
        ),
    ]


@app.call_tool()
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> list[TextContent | EmbeddedResource]:
    match name:
        case "web_search":
            inp = SearchInput.model_validate(arguments)
            results = await _perform_search(inp.query, inp.num_results, inp.language)
            return [_text(results.model_dump_json())]

        case _:
            raise ValueError(f"Unknown tool: {name}")


async def _perform_search(
    query: str, num_results: int, language: str
) -> SearchResults:
    """Execute a Tavily web search and return structured results."""
    if not TAVILY_API_KEY:
        return SearchResults(
            query=query,
            results=[],
            total_available=0,
        )

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            TAVILY_API_URL,
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": num_results,
                "include_answer": False,
                "lang": language,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    search_results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
            source=item.get("source"),
        )
        for item in data.get("results", [])
    ]

    return SearchResults(
        query=query,
        results=search_results,
        total_available=len(search_results),
    )


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
