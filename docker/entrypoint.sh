#!/usr/bin/env bash
# ============================================================
# OnCall Mind — Container Entrypoint
# Manages 3 processes: CLS MCP, Monitor MCP, FastAPI
# ============================================================
set -euo pipefail

CLS_PID=0
MONITOR_PID=0

cleanup() {
    echo "=== Shutting down MCP servers ==="
    kill $CLS_PID $MONITOR_PID 2>/dev/null || true
    wait $CLS_PID $MONITOR_PID 2>/dev/null || true
    echo "=== All processes stopped ==="
}
trap cleanup EXIT

echo "============================================================"
echo " OnCall Mind — Starting Services"
echo " MCP_BIND_HOST=${MCP_BIND_HOST:-0.0.0.0}"
echo "============================================================"

# ----------------------------------------------------------
# 1. CLS MCP Server (background)
# ----------------------------------------------------------
echo "[1/3] Starting CLS MCP Server on ${MCP_BIND_HOST:-0.0.0.0}:8003 ..."
python mcp_servers/cls_server.py &
CLS_PID=$!

# ----------------------------------------------------------
# 2. Monitor MCP Server (background)
# ----------------------------------------------------------
echo "[2/3] Starting Monitor MCP Server on ${MCP_BIND_HOST:-0.0.0.0}:8004 ..."
python mcp_servers/monitor_server.py &
MONITOR_PID=$!

# ----------------------------------------------------------
# 3. Brief pause for MCP servers to bind
# ----------------------------------------------------------
sleep 2

# ----------------------------------------------------------
# 4. FastAPI (foreground — tini tracks this PID)
# ----------------------------------------------------------
echo "[3/3] Starting FastAPI on 0.0.0.0:9900 ..."
echo "============================================================"
echo " All services started. Access at http://localhost:9900"
echo "============================================================"
exec uvicorn app.main:app --host 0.0.0.0 --port 9900
