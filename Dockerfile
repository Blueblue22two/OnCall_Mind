# ============================================================
# OnCall Mind — Docker Image (精简版 / Lite)
#
# 不含 FlagEmbedding/PyTorch/ragas/datasets
# 仅支持 RAG_MODE=basic
# 如需 Enhanced RAG（Cross-Encoder 精排），请参考完整版镜像
# ============================================================

# ----------------------------------------------------------
# Stage 1: Builder — 安装依赖（排除重型包）
# ----------------------------------------------------------
FROM python:3.11-slim AS builder

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 复制依赖清单（利用 Docker 层缓存）
COPY pyproject.toml ./

# 安装生产依赖，排除重型包
# 使用 uv pip install 直接读取 pyproject.toml（无需 lockfile）
# --no-install-package: 跳过指定包及其所有依赖（FlagEmbedding→torch ~2GB）
# 不加 build-essential，所有依赖均有预编译 arm64/amd64 wheel
RUN uv venv && \
    uv pip install --no-cache \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    "sse-starlette>=2.1.0" \
    "langchain>=0.1.0" \
    "langchain-community>=0.0.20" \
    "langchain-core>=0.1.0" \
    "langchain-openai>=1.0.0" \
    "langgraph>=0.0.40" \
    "dashscope>=1.14.0" \
    "openai>=1.10.0" \
    "pymilvus>=2.4.6" \
    "pydantic>=2.5.0,<3.0.0" \
    "pydantic-settings>=2.1.0" \
    "httpx>=0.26.0" \
    "aiohttp>=3.9.0" \
    "aiofiles>=23.2.0" \
    "python-multipart>=0.0.6" \
    "loguru>=0.7.2" \
    "python-dotenv>=1.0.0" \
    "langchain-milvus>=0.3.3" \
    "langchain-text-splitters>=1.1.0" \
    "langchain-mcp-adapters>=0.2.1" \
    "fastmcp>=2.14.0" \
    "langchain-qwq>=0.3.4" \
    "pypdf>=3.0.0" \
    "prometheus_client>=0.17.0" \
    "mcp>=1.0.0" \
    "typing_extensions>=4.0.0" \
    "redis>=5.0.0"

# ----------------------------------------------------------
# Stage 2: Runtime — 精简运行镜像
# ----------------------------------------------------------
FROM python:3.11-slim AS runtime

# 运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 复制整个 venv（比复制 site-packages 更可靠）
COPY --from=builder /app/.venv /app/.venv

# 复制应用源码
COPY app/           ./app/
COPY mcp_servers/   ./mcp_servers/
COPY static/        ./static/
COPY aiops-docs/    ./aiops-docs/
COPY pyproject.toml ./

# 复制启动脚本
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# 将 venv 加入 PATH，使 python/pip 指向 venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV MCP_BIND_HOST=0.0.0.0

EXPOSE 9900 8003 8004

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9900/health || exit 1

# tini 作为 PID 1 处理信号转发
ENTRYPOINT ["tini", "--"]
CMD ["/app/docker/entrypoint.sh"]
