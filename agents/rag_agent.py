"""RAG agent — retrieves relevant chunks and synthesises context.

Model : HuggingFace  ``mistralai/Ministral-3-3B-Reasoning-2512-GGUF``
Output: :class:`~models.results.RAGResult`

The RAG agent is the *second* stage of the pipeline.  It receives the
retrieval query produced by the intake agent, delegates semantic search to
:func:`rag.retriever.retrieve` (which owns the MCP ↔ Qdrant path), optionally
supplements with a Tavily web search via the ``web-search`` MCP server, and
then synthesises a concise context paragraph for the extractor agent.

Layering
--------
``agents/rag_agent.py`` ──► ``rag/retriever.py`` ──► MCP vector-db server
                        └──► MCP web-search server  (direct, lightweight)

The agent tools are thin wrappers: they call the ``rag/`` layer rather than
raw MCP calls.  This keeps business logic in the ``rag/`` layer and keeps
the agent focused on prompting and output validation.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.huggingface import HuggingFaceModel

from agents.base import AgentDeps
from models.results import RAGResult
from rag.retriever import retrieve, retrieve_and_rerank

# ── Model ─────────────────────────────────────────────────────────────────────

_model = HuggingFaceModel("mistralai/Ministral-3-3B-Reasoning-2512-GGUF")

# ── Agent definition ──────────────────────────────────────────────────────────

rag_agent: Agent[AgentDeps, RAGResult] = Agent(
    model=_model,
    output_type=RAGResult,
    deps_type=AgentDeps,
    system_prompt="""You are a retrieval-augmented generation specialist.

You have two tools available:
- `retrieve_chunks`: semantic vector search against stored document chunks.
- `web_search`: fallback web search for additional context (Tavily).

Your job:
1. Call `retrieve_chunks` with the provided query (and optionally reformulated variants).
2. If the returned chunks are insufficient or empty, call `web_search`.
3. Synthesise a clear, factual context paragraph (200-400 words) from the
   retrieved information. Cite which chunks you used.
4. Populate `retrieved_chunks` with the raw chunk texts you used.
5. Report a confidence score (0.0-1.0) based on how relevant the retrieved
   material is to the query. Use 0.1 if nothing useful was found.

Do NOT invent information. If nothing useful was retrieved, say so in the
context field and set confidence to 0.1.
""",
)


# ── Tools ──────────────────────────────────────────────────────────────────────

@rag_agent.tool
async def retrieve_chunks(
    ctx: RunContext[AgentDeps],
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    rerank_queries: list[str] | None = None,
) -> list[str]:
    """Retrieve the most relevant text chunks from the vector DB.

    Delegates to :func:`rag.retriever.retrieve_and_rerank` when
    ``rerank_queries`` are provided, otherwise to :func:`rag.retriever.retrieve`.

    Args:
        ctx:            Agent run context.
        query:          Primary natural-language search query.
        top_k:          Number of chunks to retrieve (1-20).
        document_id:    Optional — restrict retrieval to a single document.
        rerank_queries: Optional additional query variants for RRF re-ranking.

    Returns:
        A list of retrieved text strings ordered by relevance (best first).
    """
    with logfire.span(
        "rag_agent.retrieve_chunks",
        query=query,
        top_k=top_k,
        has_rerank=bool(rerank_queries),
    ):
        if rerank_queries:
            chunks = await retrieve_and_rerank(
                query=query,
                mcp_client=ctx.deps.mcp_client,
                top_k=max(1, min(top_k, 20)),
                document_id=document_id,
                rerank_queries=rerank_queries,
            )
        else:
            chunks = await retrieve(
                query=query,
                mcp_client=ctx.deps.mcp_client,
                top_k=max(1, min(top_k, 20)),
                document_id=document_id,
            )

    return [c.text for c in chunks]


@rag_agent.tool
async def web_search(
    ctx: RunContext[AgentDeps],
    query: str,
    num_results: int = 5,
) -> str:
    """Search the web for supplementary context using Tavily.

    Use this tool only when vector retrieval returns empty or low-quality
    results.

    Args:
        ctx:         Agent run context.
        query:       Search query string.
        num_results: Number of web results to request (1-10).

    Returns:
        A JSON string containing search results (title, url, snippet).
    """
    with logfire.span("rag_agent.web_search", query=query):
        result = await ctx.deps.mcp_client.call_tool(
            server="web-search",
            tool="web_search",
            arguments={
                "query": query,
                "num_results": max(1, min(num_results, 10)),
            },
        )
    return result[0].text if result else '{"results": []}'


# ── Convenience runner ─────────────────────────────────────────────────────────

async def run_rag(
    query: str,
    deps: AgentDeps,
    document_id: str | None = None,
) -> RAGResult:
    """Run the RAG agent for the given retrieval query.

    This is the primary entry-point used by the orchestrator.

    Args:
        query:       The retrieval query (produced by the intake agent).
        deps:        Shared agent dependencies.
        document_id: Optional document scope for vector retrieval.

    Returns:
        A validated :class:`~models.results.RAGResult`.
    """
    scope_note = f" (scoped to document {document_id})" if document_id else ""
    prompt = (
        f"Retrieval query{scope_note}: {query}\n\n"
        "Use your tools to retrieve relevant chunks and synthesise context."
    )

    with logfire.span("rag_agent.run", query=query):
        run_result = await rag_agent.run(prompt, deps=deps)

    return run_result.output
