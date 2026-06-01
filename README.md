# OnCall Mind

> 面向智能运维场景的 On-Call 诊断助手，当前支持 RAG 知识库问答、日志/监控工具调用、AIOps 故障诊断和基础评估能力。

OnCall Mind 是一个基于 FastAPI、LangChain、LangGraph 和 Milvus 的智能运维原型系统。项目当前重点用于帮助运维人员围绕常见故障进行知识检索、告警排查、日志与监控数据分析，并生成结构化诊断报告。

## 当前用途

- **RAG 知识库问答**：上传运维 SOP 文档后，可基于知识库回答 CPU、内存、磁盘、网络、服务不可用等常见故障问题。
- **AIOps 故障诊断**：使用 Plan-Execute-Replan 流程自动制定诊断计划，调用工具收集信息，并输出 Markdown 诊断报告。
- **日志与监控工具接入**：通过 MCP Server 接入模拟 CLS 日志查询和 Monitor 监控查询工具。
- **流式交互界面**：提供 Web 页面和 SSE 接口，展示诊断计划、步骤执行和最终报告。
- **评估实验**：提供 RAG Eval、Agent Eval、AIOps Eval 脚本，用于评估检索质量、工具调用准确率和诊断流程表现。

## 技术栈

| 模块 | 技术 |
|---|---|
| Web/API | FastAPI、Uvicorn、SSE |
| Agent 编排 | LangChain、LangGraph |
| LLM | 阿里云 DashScope / 通义千问，默认 `qwen-max` |
| Embedding | DashScope `text-embedding-v4` |
| 向量数据库 | Milvus |
| RAG | Dense 向量检索、Hybrid Search、BM25、RRF、Cross-Encoder Rerank |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| 工具协议 | MCP (Model Context Protocol) |
| 会话与 Trace | MemorySaver / RedisSaver、TraceStore |
| 评估 | RAGAs、LLM Judge、自定义工具调用指标 |
| 前端 | 原生 HTML / CSS / JavaScript |

## 功能概览

### 1. RAG 知识问答

系统支持两种检索模式：

- `basic`：基于 Dense Embedding 的向量检索。
- `enhanced`：查询预处理 + Dense / BM25 混合检索 + RRF 融合 + Cross-Encoder 精排。

知识库文档默认放在 `aiops-docs/` 目录，包含常见运维故障排查 SOP。

### 2. AIOps 诊断

AIOps 诊断基于 Plan-Execute-Replan 流程：

```text
用户诊断请求
  -> Planner 制定诊断计划
  -> Executor 调用知识库、日志、监控等工具
  -> Replanner 判断继续执行、重新规划或生成报告
  -> 输出结构化诊断报告
```

当前主要能力是辅助分析和建议生成，不默认执行真实生产变更操作。

### 3. MCP 工具服务

项目内置两个 MCP 服务：

- `mcp_servers/cls_server.py`：日志查询服务，默认端口 `8003`。
- `mcp_servers/monitor_server.py`：监控指标服务，默认端口 `8004`。

这些服务用于本地演示和评估，可替换为真实日志、监控或运维平台工具。

### 4. 评估能力

评估脚本位于 `tests/evaluation/`：

- `evaluate_rag.py`：RAG 检索与生成质量评估。
- `evaluate_agent.py`：RAG Agent 工具调用与目标达成评估。
- `evaluate_aiops_agent.py`：AIOps 诊断流程评估。
- `run_ablation.py`：RAG 消融实验。

## 项目结构

```text
.
├── app/
│   ├── api/                    # FastAPI 路由
│   ├── agent/                  # Agent 与 MCP 客户端
│   ├── core/                   # LLM、Milvus、Metrics 等核心组件
│   ├── retriever/              # Basic / Enhanced RAG 检索模块
│   ├── services/               # RAG、AIOps、索引、Trace 等服务
│   └── tools/                  # 本地工具，如知识检索和时间工具
├── aiops-docs/                 # 示例运维知识库文档
├── mcp_servers/                # 本地 MCP 日志/监控工具服务
├── static/                     # Web 前端
├── tests/evaluation/           # RAG / Agent / AIOps 评估脚本
├── vector-database.yml         # Milvus / Redis 等基础服务
├── Makefile                    # macOS/Linux 常用命令
├── start-windows.bat           # Windows 启动脚本
└── pyproject.toml              # Python 项目配置
```

## 环境要求

- Python `>=3.11,<3.14`
- Docker / Docker Compose
- 阿里云 DashScope API Key
- 推荐使用 `uv` 管理 Python 虚拟环境和依赖

## 配置说明

创建 `.env` 文件并配置必要参数：

```bash
# DashScope 配置，必填
DASHSCOPE_API_KEY=your-dashscope-api-key
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-max
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# RAG 模式：basic 或 enhanced
RAG_MODE=basic
RAG_TOP_K=3

# Enhanced RAG 配置
QUERY_PREPROCESSOR_TYPE=none
RERANKER_TYPE=cross_encoder
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_TOP_K=3
RERANK_COARSE_TOP_K=10

# Redis 可选；不配置时使用进程内 MemorySaver
REDIS_URL=redis://localhost:6379

# MCP 服务地址
MCP_CLS_URL=http://localhost:8003/mcp
MCP_MONITOR_URL=http://localhost:8004/mcp

# 评估 Judge 配置，可选
EVAL_JUDGE_MODEL=qwen3.5-plus
EVAL_JUDGE_TEMPERATURE=0.0
EVAL_JUDGE_API_BASE=
EVAL_JUDGE_API_KEY=
```

> **注意** !!! 如果各位用户部署该项目时，不要把真实 API Key 提交到代码仓库避免API KEY泄露。 

## 部署与启动

### macOS / Linux

1. 安装依赖

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .
```

2. 配置环境变量

```bash
cp .env.example .env  # 如果仓库中存在模板文件
vim .env
```

至少需要填写 `DASHSCOPE_API_KEY`。

3. 启动基础服务

```bash
docker compose -f vector-database.yml up -d
```

4. 启动应用服务

推荐使用 Makefile：

```bash
make start
```

该命令会启动：

- CLS MCP 服务：`http://localhost:8003/mcp`
- Monitor MCP 服务：`http://localhost:8004/mcp`
- FastAPI 主服务：`http://localhost:9900`

5. 初始化知识库

```bash
make upload
```

或者一键完成基础服务启动、应用启动和文档上传：

```bash
make init
```

### Windows

1. 创建虚拟环境并安装依赖

```powershell
pip install uv
uv venv
.venv\Scripts\activate
uv pip install -e .
```

2. 配置 `.env`

```powershell
notepad .env
```

3. 启动 Docker Desktop，然后启动基础服务

```powershell
docker compose -f vector-database.yml up -d
```

4. 使用脚本启动服务

```powershell
.\start-windows.bat
```

停止服务：

```powershell
.\stop-windows.bat
```

## 访问地址

- Web 界面：`http://localhost:9900`
- API 文档：`http://localhost:9900/docs`
- Milvus Attu：`http://localhost:8000`
- MinIO 控制台：`http://localhost:9001`

## 常用 API

| 功能 | 方法 | 路径 |
|---|---|---|
| 普通对话 | POST | `/api/chat` |
| 流式对话 | POST | `/api/chat_stream` |
| 清空会话 | POST | `/api/chat/clear` |
| 会话历史 | GET | `/api/chat/session/{session_id}` |
| AIOps 诊断 | POST | `/api/aiops` |
| 诊断历史 | GET | `/api/aiops/diagnosis/{session_id}` |
| 文件上传 | POST | `/api/upload` |
| 目录索引 | POST | `/api/index_directory` |
| 健康检查 | GET | `/api/health` |
| Prometheus 指标 | GET | `/metrics` |

示例：

```bash
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-001","Question":"CPU 使用率过高怎么排查？"}'
```

```bash
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-001"}' \
  --no-buffer
```

## 评估命令

```bash
# RAG 检索评估
python -m tests.evaluation.evaluate_rag

# RAG 检索 + 生成评估
python -m tests.evaluation.evaluate_rag --with-generation

# Agent 工具调用评估
python -m tests.evaluation.evaluate_agent --skip-goal

```

## 常用开发命令

```bash
make help          # 查看可用命令
make start         # 启动 MCP + FastAPI 服务
make stop          # 停止服务
make restart       # 重启服务
make dev           # 开发模式启动 FastAPI
make upload        # 上传 aiops-docs 下的文档
make check         # 健康检查
make status-mcp    # 查看 MCP 服务状态
make test          # 运行测试
make lint          # 代码检查
make format        # 代码格式化
```

