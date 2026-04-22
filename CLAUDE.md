# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SuperBizAgent** is a Python-based intelligent on-call/AIOps agent system built on LangChain + LangGraph. It provides two primary agent workflows:
1. **RAG Chat** – conversational Q&A backed by a Milvus vector store knowledge base
2. **AIOps Diagnostics** – autonomous Plan-Execute-Replan loop for automated incident investigation

The backend is a FastAPI server (port 9900) that exposes SSE-streamed responses. External tools are provided by two MCP (Model Context Protocol) servers: a CLS log server (port 8003) and a monitoring server (port 8004).

## Prerequisites

- Python 3.11–3.13, Docker (for Milvus vector DB)
- Copy `.env` from template and set `DASHSCOPE_API_KEY` and `DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1`
- The project uses a `.venv` virtualenv; Makefile targets call `.venv/bin/python` directly

## Common Commands

```bash
# First-time setup: start Docker, all services, and ingest docs
make init

# Start/stop all services (Milvus must be running first)
make up          # start Milvus container
make start       # start CLS MCP + Monitor MCP + FastAPI
make stop        # stop all services

# Development (foreground, hot-reload)
make dev         # uvicorn --reload on port 9900

# Upload knowledge base docs to Milvus
make upload      # uploads all aiops-docs/*.md

# Code quality
make format      # ruff format + isort
make lint        # ruff check
make fix         # ruff --fix
make type-check  # mypy

# Tests
make test        # pytest with coverage (targets app/)
make test-quick  # pytest without coverage
python3 -m pytest tests/path/to/test_file.py::test_name -v  # single test

# Service health
make check       # curl /health
make status-mcp  # check CLS and Monitor MCP status
make logs        # tail -f server.log
```

## Architecture

### Request Flow

```
HTTP/SSE Client
    → FastAPI (app/main.py, port 9900)
        → API routers (app/api/)
            → Services (app/services/)
                → LangGraph agents (app/agent/)
                    → MCP tools (mcp_servers/, via MCPClient)
                    → RAG tools (app/tools/, Milvus)
                    → LLM (Alibaba DashScope / Qwen models)
```

### Key Modules

| Path | Role |
|---|---|
| `app/main.py` | FastAPI app entry point; registers routers, Milvus lifecycle |
| `app/config.py` | Pydantic Settings; all config loaded from `.env` |
| `app/api/` | Route handlers: `chat`, `aiops`, `file`, `health` |
| `app/services/rag_agent_service.py` | LangGraph ReAct agent for chat; manages message history trimming (keeps first system message + last 6 messages) |
| `app/services/aiops_service.py` | LangGraph Plan-Execute-Replan workflow for AIOps diagnostics |
| `app/agent/mcp_client.py` | MCPClient connecting to external MCP tool servers |
| `app/agent/aiops.py` | Planner / executor / replanner node logic for AIOps graph |
| `app/core/milvus_client.py` | Milvus connection manager (singleton) |
| `app/tools/` | Internal LangChain tools: `retrieve_knowledge` (RAG), `get_current_time` |
| `app/services/vector_*` | Vector embedding, indexing, search, and store management |
| `mcp_servers/cls_server.py` | FastMCP server exposing CLS log query tools (port 8003) |
| `mcp_servers/monitor_server.py` | FastMCP server exposing monitoring/metrics tools (port 8004) |
| `aiops-docs/` | Markdown files ingested as the default RAG knowledge base |

### Two Agent Patterns

**RAG Agent** (`RagAgentService`): Standard ReAct loop using `ChatQwen` + LangGraph with `MemorySaver` for multi-turn conversation. Tools: `retrieve_knowledge`, `get_current_time`, plus all MCP tools loaded dynamically at init.

**AIOps Agent** (`AIOpsService`): Plan-Execute-Replan loop — a `planner` node generates a step list, the `executor` node runs one step using MCP tools, and the `replanner` node either updates the plan or generates a final response. The graph exits when `state["response"]` is set.

### MCP Tool Integration

Both agent types use `get_mcp_client_with_retry()` from `app/agent/mcp_client.py` to load tools from the CLS and Monitor MCP servers at service initialization time. MCP server URLs are configured via `MCP_CLS_URL` and `MCP_MONITOR_URL` in `.env`.

### Data Flow for Document Ingestion

`POST /api/upload` → `file.py` router → `DocumentSplitterService` (chunking, 800 tokens / 100 overlap) → `VectorEmbeddingService` (DashScope `text-embedding-v4`, 1024-dim) → `VectorStoreManager` → Milvus

## Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DASHSCOPE_API_KEY` | *(required)* | Aliyun DashScope API key |
| `DASHSCOPE_API_BASE` | *(required)* | Set to `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `DASHSCOPE_MODEL` | `qwen-max` | Chat model for RAG |
| `DASHSCOPE_EMBEDDING_MODEL` | `text-embedding-v4` | Embedding model |
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus port |
| `MCP_CLS_URL` | `http://localhost:8003/mcp` | CLS MCP endpoint |
| `MCP_MONITOR_URL` | `http://localhost:8004/mcp` | Monitor MCP endpoint |
| `DEBUG` | `false` | Enable hot-reload and debug mode |

## Testing

Tests live in `tests/` mirroring `app/` structure. Uses `pytest-asyncio` in auto mode. `pyproject.toml` configures coverage for the `app/` package by default, so `make test` always reports coverage. Use `make test-quick` to skip the coverage overhead during development.

## RAG Optimization Roadmap (plans.md / eval.md)

Three-phase plan documented in `plans.md`; difficulty/framework evaluation in `eval.md`.

### Current RAG Limitations
- Single dense vector field (`vector`, 1024-dim, **L2**, IVF_FLAT, nlist=128)
- Collection name: `biz`; no sparse vector, no reranking, no query preprocessing
- `retrieve_knowledge` in `app/tools/knowledge_tool.py` is **synchronous** — must stay sync or all callers updated together
- `pymilvus` currently pinned at `>=2.3.5`; enhanced RAG requires `>=2.4.6`

### Phase 1 — Pluggable Interface (P0, low effort)
New module `app/retriever/` with `base.py` (ABC), `basic.py` (wraps current logic), `factory.py` (returns impl by config).  
New config field: `rag_mode: Literal["basic", "enhanced"] = "basic"` / `.env`: `RAG_MODE=basic`.  
`retrieve_knowledge` tool signature (`@tool(response_format="content_and_artifact")`) must not change.

### Phase 2 — Enhanced RAG (P1–P2, medium effort)
- **Dual-vector schema**: new collection `biz_enhanced` (keep `biz` for basic mode); add `dense_vector` (FLOAT_VECTOR, 1024-dim, COSINE) + `sparse_vector` (SPARSE_FLOAT_VECTOR)
- **BM25 sparse encoding**: `pymilvus.model.sparse.BM25EmbeddingFunction`; fit on full corpus after ingestion; serialize to `data/bm25_model.pkl`. Short-term: full rebuild from Milvus `content` field on each update (corpus ~20–50 chunks, <1s). Long-term: migrate to Milvus 2.5 built-in BM25 (`FunctionType.BM25` + Chinese analyzer).
- **Hybrid search**: `AnnSearchRequest × 2 + RRFRanker(k=60)` via `pymilvus` native API
- **Query preprocessing** (`app/retriever/preprocessing/`): `none` / `rewrite` / `hyde` / `multi_query`; HyDE uses hypothetical doc for Dense but original query for Sparse
- **Reranker**: `BAAI/bge-reranker-v2-m3` via `FlagEmbedding` (560MB, CPU ~200–800ms); always scores against original query
- New config fields: `query_preprocessor_type`, `reranker_type`, `reranker_top_k`, `rerank_coarse_top_k`, `bm25_refit_strategy`
- **pymilvus monkey-patch** (`_patch_pymilvus_milvus_client_orm_alias` in `milvus_client.py`) must be re-validated after upgrading pymilvus

### Phase 3 — RAGAs Evaluation (P2, medium effort)
`tests/evaluation/` with `rag_testset.py` (15–25 Q&A pairs from `aiops-docs/`), `evaluate_rag.py`, `compare_reports.py`.  
LLM judge reuses `ChatQwen` via `LangchainLLMWrapper`. Target: `context_precision` and `context_recall` ≥ 0.7 (basic baseline); enhanced mode +0.10 on both.  
New deps: `ragas>=0.1.0`, `datasets>=2.0.0`, `FlagEmbedding>=1.2.0`, `rank_bm25>=0.2.2`.
