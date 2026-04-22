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
