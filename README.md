# RAGent 🤖

**Multi-Agent Document Intelligence Pipeline with MCP and Typed RAG**

> Python 3.12+ · Pydantic AI · Model Context Protocol · Strict Typing · Real Asyncio · SSE Streaming

---

## What is RAGent?

RAGent ingests unstructured documents (invoices, contracts, reports, receipts), runs a three-stage multi-agent pipeline backed by open-source models on HuggingFace, and streams fully-typed, validated JSON extractions to any frontend via Server-Sent Events.

Every external tool — the vector database, document store, and web search — is exposed through **MCP (Model Context Protocol) servers**, keeping agents cleanly decoupled from infrastructure.

**Monthly cloud API cost: $0.** All hosted models use free-tier APIs; embeddings run 100% locally.

---

## Model Stack

| Provider | Model | Agent Role | Notes |
|---|---|---|---|
| **HuggingFace Serverless** | `mistralai/Ministral-3-3B-Reasoning-2512-GGUF` | RAG synthesis + structured extraction | Free serverless inference |
| **Groq** | `llama-3.3-70b-versatile` | Document classification & routing | 14,400 RPD free tier |
| **Local (sentence-transformers)** | `Qwen/Qwen3-Embedding-0.6B` | Vector embedding (1024-dim) | Runs fully offline, no API key needed |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI  (SSE + REST)                       │
│   POST /api/v1/process  →  GET /api/v1/stream/{id}          │
│         streams StreamChunk[PipelineResult] → frontend       │
└──────────────────────────┬──────────────────────────────────┘
                           │  asyncio.Queue (backpressure, maxsize=10)
              ┌────────────▼────────────┐
              │       Orchestrator       │
              │  (asyncio.TaskGroup per  │
              │       request)           │
              └──┬──────────┬───────────┘
                 │          │
        ┌────────▼───┐  ┌───▼──────────────┐
        │   Intake   │  │  RAG + Extractor  │
        │   Agent    │  │    Agents (HF)    │
        │   (Groq)   │  └───────┬───────────┘
        └────────┬───┘          │
                 │              │
  ───────────────▼──────────────▼──────────────────
                    MCP CLIENT (shared singleton)
  ────────────┬───────────────┬──────────────────┬──
              │               │                  │
   ┌──────────▼──┐  ┌─────────▼──┐  ┌────────────▼──┐
   │  MCP Server  │  │ MCP Server  │  │  MCP Server   │
   │  doc-store   │  │  vector-db  │  │  web-search   │
   │ (in-memory)  │  │  (Qdrant)   │  │   (Tavily)    │
   └─────────────┘  └────────────┘  └───────────────┘
```

### Pipeline Stages

```
RawDocument
    │
    ▼ Stage 1 — Intake Agent (Groq)
    │   • Stores document in doc-store MCP
    │   • Chunks text  →  rag/chunker.py
    │   • Embeds chunks →  rag/embedder.py  →  vector-db MCP
    │   • Classifies document type (invoice/contract/report/receipt/unknown)
    │   • Outputs: IntakeResult (type, confidence, RAG query)
    │
    ▼ Stage 2 — RAG Agent (HuggingFace Ministral)
    │   • Embeds the query (local Qwen3-Embedding-0.6B)
    │   • Retrieves top-k chunks  →  rag/retriever.py  →  vector-db MCP
    │   • Optionally calls web-search MCP (Tavily) as fallback
    │   • Synthesises a grounding context paragraph
    │   • Outputs: RAGResult (context, retrieved_chunks, confidence)
    │
    ▼ Stage 3 — Extractor Agent (HuggingFace Ministral)
        • Routes to the correct typed sub-agent for the document type
        • Extracts all structured fields using RAG context as grounding
        • Outputs: InvoiceData | ContractData | ReportData | ReceiptData | GenericExtraction
```

---

## Project Structure

```
RAGent/
│
├── agents/                        # Pydantic AI agents
│   ├── base.py                    # AgentDeps protocol + ConcreteAgentDeps
│   ├── intake_agent.py            # Groq — classifies, chunks, indexes, routes
│   ├── rag_agent.py               # Gemini — retrieves context via rag/retriever
│   ├── extractor_agent.py         # Gemini — 5 typed sub-agents (one per doc type)
│   └── orchestrator.py            # asyncio.TaskGroup pipeline + SSE queue
│
├── mcp/                           # MCP layer
│   ├── __init__.py                # Makes local mcp/ a proper package; bridges SDK
│   ├── server.py                  # Bridge → installed mcp SDK's Server class
│   ├── types.py                   # Bridge → installed mcp SDK's Tool, TextContent
│   ├── client.py                  # Shared MCPClient wrapper (stdio sessions)
│   ├── schemas.py                 # Typed Pydantic I/O schemas for all MCP tools
│   └── servers/
│       ├── doc_store_server.py    # MCP server: in-memory document store
│       ├── vector_db_server.py    # MCP server: Qdrant embed + retrieve
│       └── web_search_server.py   # MCP server: Tavily search API
│
├── rag/                           # RAG pipeline layer
│   ├── chunker.py                 # Recursive sentence-aware text splitter
│   ├── embedder.py                # Qwen3-Embedding-0.6B (local, async via asyncio.to_thread)
│   └── retriever.py               # Qdrant retrieval + RRF multi-query reranking
│
├── models/                        # All Pydantic models (strict)
│   ├── documents.py               # RawDocument, ParsedChunk, EmbeddedChunk
│   ├── extractions.py             # InvoiceData, ContractData, ReportData, ReceiptData, GenericExtraction
│   ├── results.py                 # IntakeResult, RAGResult, PipelineResult, ValidationReport
│   └── stream.py                  # StreamChunk[T], ProgressData — for SSE
│
├── api/                           # FastAPI layer
│   ├── dependencies.py            # AppDeps, get_deps, MCPClient singleton injection
│   ├── sse.py                     # SSE emitter — asyncio.Queue → EventStream
│   └── routes.py                  # POST /process, POST /process/async, GET /results/{id}, GET /health
│
├── config.py                      # pydantic-settings — strict env validation
├── main.py                        # FastAPI app + asynccontextmanager lifespan (MCP startup)
├── pyproject.toml                 # Project metadata, mypy strict config
└── requirements.txt               # All dependencies (sentence-transformers, torch, pydantic-ai[huggingface], ...)
```

---

## Prerequisites

### 1. Python 3.12+
```bash
python --version   # Must be 3.12 or higher
```

### 2. Qdrant (local vector database)
```bash
# Via Docker (recommended)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Or download the binary from https://qdrant.tech/documentation/quick-start/
```

### 3. API Keys

Get **free** keys from:

| Key | Where to get it |
|---|---|
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token (read access) |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) → API (optional — web search degrades gracefully without it) |
| `LOGFIRE_TOKEN` | [logfire.pydantic.dev](https://logfire.pydantic.dev) → (optional — tracing) |

---

## Setup

### 1. Clone and create a virtual environment
```bash
git clone <repo-url>
cd RAGent

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env`
```env
# Required — HuggingFace token (for Ministral inference)
HF_TOKEN=hf_...

# Required — Groq (intake classification)
GROQ_API_KEY=gsk_...

# Optional — web search (Tavily)
TAVILY_API_KEY=tvly-...

# Optional — observability (Logfire)
LOGFIRE_TOKEN=...

# Optional — Qdrant (defaults shown)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=ragent-chunks
```

### 4. Start Qdrant
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### 5. Run the server
```bash
uvicorn main:app --reload --port 8000
```

The server starts 3 MCP servers as subprocesses automatically (doc-store, vector-db, web-search) and connects to them before accepting requests.

Interactive API docs: **http://localhost:8000/docs**

---

## API Usage

### POST `/api/v1/process` — Stream results in real-time

Submit a document and receive a live SSE stream of processing events.

**Request**
```bash
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user-123" \
  -d '{
    "content": "INVOICE #INV-2024-001\nVendor: Acme Corp\nDate: 2024-01-15\nTotal: $1,250.00",
    "source": "invoice_jan.txt",
    "document_type": "invoice"
  }'
```

**SSE Response stream**
```
data: {"event":"progress","payload":{"stage":"classifying","message":"Classifying document type…","percentage":10.0},"timestamp":"...","sequence":0}

data: {"event":"progress","payload":{"stage":"retrieving","message":"Retrieving context for \"invoice vendor payment\"…","percentage":40.0},"timestamp":"...","sequence":1}

data: {"event":"progress","payload":{"stage":"extracting","message":"Extracting structured fields…","percentage":70.0},"timestamp":"...","sequence":2}

data: {"event":"progress","payload":{"stage":"done","message":"Pipeline complete.","percentage":100.0},"timestamp":"...","sequence":3}

data: {"event":"complete","payload":{...PipelineResult...},"timestamp":"...","sequence":4}
```

---

### POST `/api/v1/process/async` — Fire-and-forget

```bash
curl -X POST http://localhost:8000/api/v1/process/async \
  -H "Content-Type: application/json" \
  -d '{"content": "CONTRACT between Party A and Party B..."}'
```

**Response (202 Accepted)**
```json
{
  "request_id": "a3f1e2d0c4b5...",
  "document_id": "9b8c7d6e5f4a...",
  "message": "Document accepted. Use GET /results/{request_id} to poll."
}
```

### GET `/api/v1/results/{request_id}` — Poll for result

```bash
curl http://localhost:8000/api/v1/results/a3f1e2d0c4b5...
```
- **202** while processing
- **200** with full `PipelineResult` when done

### GET `/api/v1/health` — Liveness probe
```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","mcp_connected":true}
```

### GET `/api/v1/docs-list` — List stored documents
```bash
curl "http://localhost:8000/api/v1/docs-list?limit=20"
```

---

## Extraction Results by Document Type

All results are wrapped in a `PipelineResult`:

```json
{
  "document_id": "9b8c...",
  "extraction": { ...type-specific fields... },
  "validation": { "is_valid": true, "issues": [] },
  "processing_time_ms": 4821.3,
  "completed_at": "2024-01-15T10:23:45Z"
}
```

---

### 📄 Invoice (`document_type: "invoice"`)

Extracts financial fields from supplier invoices, purchase orders, and billing statements.

**Expected `extraction` object:**
```json
{
  "extraction_type": "invoice",
  "confidence": 0.94,
  "invoice_number": "INV-2024-001",
  "invoice_date": "2024-01-15",
  "due_date": "2024-02-15",
  "vendor_name": "Acme Corp",
  "vendor_address": "123 Main St, Springfield, IL 62701",
  "customer_name": "Globex Corp",
  "customer_address": "742 Evergreen Terrace, Springfield, IL 62702",
  "subtotal": 1150.00,
  "tax_amount": 100.00,
  "total_amount": 1250.00,
  "currency": "USD",
  "line_items": [
    {
      "description": "Widget Model X (x10)",
      "quantity": 10,
      "unit_price": 115.00,
      "total": 1150.00,
      "tax": null
    }
  ]
}
```

---

### 📋 Contract (`document_type: "contract"`)

Extracts parties, key dates, governing law, and clause summaries from legal agreements.

**Expected `extraction` object:**
```json
{
  "extraction_type": "contract",
  "confidence": 0.91,
  "contract_title": "Software Development Services Agreement",
  "contract_date": "2024-01-10",
  "effective_date": "2024-02-01",
  "expiration_date": "2025-01-31",
  "parties": ["TechCorp Inc.", "DevStudio LLC"],
  "governing_law": "State of California",
  "key_obligations": [
    "TechCorp shall pay monthly retainer of $15,000",
    "DevStudio shall deliver sprint demos every two weeks"
  ],
  "clauses": [
    {
      "clause_type": "confidentiality",
      "text": "Each party agrees to keep all proprietary information confidential...",
      "summary": "Standard mutual NDA clause, 3-year term post-contract.",
      "effective_date": "2024-02-01",
      "expiration_date": "2028-02-01"
    },
    {
      "clause_type": "termination",
      "text": "Either party may terminate with 30 days written notice...",
      "summary": "30-day termination notice required from either party.",
      "effective_date": null,
      "expiration_date": null
    }
  ]
}
```

---

### 📊 Report (`document_type: "report"`)

Extracts structured sections, key findings, and executive summary from business or research reports.

**Expected `extraction` object:**
```json
{
  "extraction_type": "report",
  "confidence": 0.88,
  "report_title": "Q4 2023 Sales Performance Analysis",
  "report_date": "2024-01-05",
  "author": "Jane Smith, Head of Analytics",
  "executive_summary": "Q4 2023 saw a 23% YoY revenue increase driven primarily by APAC expansion...",
  "key_findings": [
    "Revenue grew 23% YoY to $12.4M",
    "APAC region contributed 41% of total revenue",
    "Customer churn decreased from 8.2% to 5.7%"
  ],
  "sections": [
    {
      "title": "Revenue Overview",
      "content": "Total Q4 revenue reached $12.4M, up from $10.1M in Q4 2022...",
      "page_number": 2
    },
    {
      "title": "Regional Breakdown",
      "content": "APAC: $5.1M (+65%), EMEA: $4.2M (+12%), Americas: $3.1M (+8%)...",
      "page_number": 4
    }
  ]
}
```

---

### 🧾 Receipt (`document_type: "receipt"`)

Extracts transaction details from retail receipts and point-of-sale records.

**Expected `extraction` object:**
```json
{
  "extraction_type": "receipt",
  "confidence": 0.96,
  "merchant_name": "Whole Foods Market",
  "merchant_address": "1440 P St NW, Washington, DC 20005",
  "transaction_date": "2024-01-15",
  "transaction_time": "2024-01-15T14:32:00",
  "total_amount": 47.83,
  "tax_amount": 3.21,
  "payment_method": "Visa ending 4242",
  "line_items": [
    { "description": "Organic Apples 2lb", "quantity": 1, "unit_price": 5.99, "total": 5.99, "tax": null },
    { "description": "Almond Milk 64oz", "quantity": 2, "unit_price": 4.49, "total": 8.98, "tax": null },
    { "description": "Sourdough Bread", "quantity": 1, "unit_price": 7.99, "total": 7.99, "tax": null }
  ]
}
```

---

### 🗂️ Unknown / Generic (`document_type: "unknown"`)

Fallback for unclassified documents. Extracts a title, summary, key entities, dates, and any monetary amounts found.

**Expected `extraction` object:**
```json
{
  "extraction_type": "generic",
  "confidence": 0.72,
  "title": "Meeting Notes — Product Roadmap Review",
  "summary": "Internal meeting discussing Q2 feature priorities, timeline adjustments, and resource allocation...",
  "key_entities": ["Alice Chen", "Bob Torres", "mobile app redesign", "API v3"],
  "dates_found": ["2024-01-12", "2024-03-01", "2024-06-30"],
  "amounts_found": [50000.00, 120000.00]
}
```

---

## JavaScript Client Example

```javascript
const response = await fetch('http://localhost:8000/api/v1/process', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'X-User-ID': 'user-001' },
  body: JSON.stringify({ content: documentText, source: 'upload.pdf' }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const lines = decoder.decode(value).split('\n\n').filter(Boolean);
  for (const line of lines) {
    if (!line.startsWith('data: ')) continue;
    const chunk = JSON.parse(line.slice(6));

    if (chunk.event === 'progress') {
      console.log(`[${chunk.payload.percentage}%] ${chunk.payload.message}`);
    } else if (chunk.event === 'complete') {
      console.log('Result:', chunk.payload.extraction);
    } else if (chunk.event === 'error') {
      console.error('Pipeline error:', chunk.error);
    }
  }
}
```

---

## Validation

Every extraction goes through a `ValidationReport`:

```json
{
  "is_valid": true,
  "issues": [],
  "checked_at": "2024-01-15T10:23:46Z"
}
```

If confidence is low (`< 0.5`):
```json
{
  "is_valid": true,
  "issues": [
    {
      "field": "confidence",
      "message": "Low extraction confidence: 0.42",
      "severity": "warning",
      "value": 0.42
    }
  ]
}
```

Severity levels: `"warning"` (non-blocking) and `"error"` (marks `is_valid: false`).

---

## Configuration Reference

All settings are loaded from `.env` via `pydantic-settings`.

| Variable | Default | Required | Description |
|---|---|---|---|
| `HF_TOKEN` | `""` | ✅ Yes | HuggingFace token for Ministral inference via Serverless API |
| `GROQ_API_KEY` | `""` | ✅ Yes | Groq key for intake classification |
| `TAVILY_API_KEY` | `""` | ❌ No | Web search fallback (returns empty if unset) |
| `LOGFIRE_TOKEN` | `""` | ❌ No | Pydantic Logfire tracing token |
| `QDRANT_URL` | `http://localhost:6333` | ❌ No | Qdrant instance URL |
| `QDRANT_API_KEY` | `null` | ❌ No | Qdrant API key (for cloud instances) |
| `QDRANT_COLLECTION` | `ragent-chunks` | ❌ No | Qdrant collection name |
| `EMBEDDING_DIMENSION` | `1024` | ❌ No | Must match the model (1024 for Qwen3-Embedding-0.6B) |

---

## Observability

When `LOGFIRE_TOKEN` is set, every pipeline run traces:

- `orchestrator.run_pipeline` — full end-to-end span
- `intake_agent.run` — Groq classification span
- `intake.ingest_document` — chunking + embedding span (Qwen3-Embedding-0.6B)
- `rag_agent.run` — Ministral retrieval span
- `rag_agent.retrieve_chunks` — MCP vector-db call
- `rag_agent.web_search` — Tavily fallback call
- `extractor_agent.run` — Gemini extraction span

View traces at **https://logfire.pydantic.dev**

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| Async | `asyncio.TaskGroup`, `asyncio.Queue`, `asyncio.Semaphore` |
| Agents | Pydantic AI (multi-agent, typed `output_type`, `RunContext`) |
| Integration | Model Context Protocol (MCP) — stdio subprocess servers |
| Validation | Pydantic v2 — all LLM outputs, all MCP I/O |
| RAG | Custom chunker + Qwen3-Embedding-0.6B (local) + Qdrant + RRF reranking |
| LLM (agents) | Groq `llama-3.3-70b-versatile` (intake) · HuggingFace `Ministral-3B` (RAG + extraction) |
| API | FastAPI + SSE (`StreamingResponse`) |
| Observability | Logfire (auto-instrumented) |
| Type checking | mypy `--strict` |

---

## Running mypy

```bash
pip install mypy
mypy . --strict
```

Target: **0 errors**.

---

## License

MIT
