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

| 模块        | 技术                                                     |
| --------- | ------------------------------------------------------ |
| Web/API   | FastAPI、Uvicorn、SSE                                    |
| Agent 编排  | LangChain、LangGraph                                    |
| LLM       | 阿里云 DashScope / 通义千问，默认 `qwen-max`                     |
| Embedding | DashScope `text-embedding-v4`                          |
| 向量数据库     | Milvus                                                 |
| RAG       | Dense 向量检索、Hybrid Search、BM25、RRF、Cross-Encoder Rerank |
| Reranker  | `BAAI/bge-reranker-v2-m3`                              |
| 工具协议      | MCP (Model Context Protocol)                           |
| 会话与 Trace | MemorySaver / RedisSaver、TraceStore                    |
| 评估        | RAGAs、LLM Judge、自定义工具调用指标                              |
| 前端        | 原生 HTML / CSS / JavaScript                             |

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
├── docker/
│   └── entrypoint.sh           # Docker 容器启动脚本
├── mcp_servers/                # 本地 MCP 日志/监控工具服务
├── static/                     # Web 前端
├── tests/evaluation/           # RAG / Agent / AIOps 评估脚本
├── Dockerfile                  # Docker 镜像构建文件（精简版）
├── docker-compose.yml          # 统一编排（基础设施 + 应用）
├── vector-database.yml         # Milvus / Redis 等基础服务（独立）
├── .env.docker                 # Docker 环境配置模板
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

# v4 RAG 实验开关（默认关闭；先通过 make eval-rag-v4-ablation 验证）
RAG_QUERY_ROUTING=false
RAG_CHUNK_STRATEGY=legacy
RAG_INCLUDE_SECTION_PREFIX=false
RAG_PARENT_CONTEXT=false
ENHANCED_COLLECTION_NAME=biz_enhanced

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
EVAL_JUDGE_CACHE_PATH=reports/judge_cache.sqlite
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

1. 配置环境变量

```bash
cp .env.example .env  # 如果仓库中存在模板文件
vim .env
```

至少需要填写 `DASHSCOPE_API_KEY`。

1. 启动基础服务

```bash
docker compose -f vector-database.yml up -d
```

1. 启动应用服务

推荐使用 Makefile：

```bash
make start
```

该命令会启动：

- CLS MCP 服务：`http://localhost:8003/mcp`
- Monitor MCP 服务：`http://localhost:8004/mcp`
- FastAPI 主服务：`http://localhost:9900`

1. 初始化知识库

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

1. 配置 `.env`

```powershell
notepad .env
```

1. 启动 Docker Desktop，然后启动基础服务

```powershell
docker compose -f vector-database.yml up -d
```

1. 使用脚本启动服务

```powershell
.\start-windows.bat
```

停止服务：

```powershell
.\stop-windows.bat
```

### Docker 一键部署（推荐）

项目提供完整的 Docker Compose 编排，**一条命令启动所有服务**（基础设施 + 应用）。

> **镜像说明**：当前发布的 Docker 镜像为**精简版（Lite）**，不含 FlagEmbedding / PyTorch / RAGAs 等重型依赖，仅支持 `RAG_MODE=basic`。如需 Enhanced RAG（Cross-Encoder 精排），请参考下方[完整版镜像](#完整版镜像)。

**前置条件**：Docker 和 Docker Compose V2

**步骤**：

```bash
# 1. 克隆项目
git clone <repo-url>
cd <project-dir>

# 2. 配置环境变量
cp .env.docker .env
# 编辑 .env，填入你的 DashScope API Key
vim .env   # 或 nano .env

# 3. 一键启动（基础设施 + 应用，共 7 个容器）
docker compose up -d

# 4. 上传知识库文档（首次部署需要）
make upload
```

**包含的容器**：

| 容器                | 镜像                              | 端口        | 用途                |
| ----------------- | ------------------------------- | --------- | ----------------- |
| oncall-mind       | `blueblue22/oncall-mind:latest` | 9900      | FastAPI + MCP 服务器 |
| milvus-standalone | `milvusdb/milvus:v2.5.10`       | 19530     | 向量数据库             |
| oncallmind-redis  | `redis:7-alpine`                | 6379      | 会话/Trace 持久化      |
| milvus-etcd       | `quay.io/coreos/etcd:v3.5.18`   | —         | Milvus 协调         |
| milvus-minio      | `minio/minio`                   | 9000/9001 | Milvus 对象存储       |
| milvus-attu       | `zilliz/attu:v2.5`              | 8000      | Milvus Web UI     |

**常用 Docker 命令**：

```bash
make docker-up        # 启动所有服务
make docker-down      # 停止所有服务
make docker-logs      # 查看应用日志
make docker-status    # 查看服务状态
make docker-build     # 从源码构建镜像
make docker-push      # 推送到 Docker Hub
```

**直接从 Docker Hub 拉取**：

```bash
docker pull blueblue22/oncall-mind:latest
docker pull blueblue22/oncall-mind:v1.2.1-lite
```

#### 完整版镜像

精简版镜像（约 629 MB）不含以下依赖，仅支持 `RAG_MODE=basic`：

- `FlagEmbedding`（含 PyTorch，约 1.5 GB）— 用于 Cross-Encoder 精排
- `ragas` + `datasets` — 用于 RAG 评估

如需使用 `RAG_MODE=enhanced`（查询改写 + Dense/BM25 混合检索 + RRF 融合 + Cross-Encoder 精排），需要构建完整版镜像：

```bash
# 修改 Dockerfile，移除以下两行：
#   --no-install-package FlagEmbedding
#   --no-install-package ragas

# 然后重新构建
docker build -t oncall-mind:full .
```

或在本地 Python 环境中安装：

```bash
uv pip install FlagEmbedding ragas datasets
```

> 完整版镜像预计约 2.2 GB。

### 下载 Reranker 模型（Enhanced RAG 必需）

当使用 `RAG_MODE=enhanced` + `RERANKER_TYPE=cross_encoder` 时，系统需要加载 `BAAI/bge-reranker-v2-m3` 模型（约 560 MB）。模型首次使用时会自动从 HuggingFace 下载，但由于网络原因国内可能较慢或失败。推荐使用 ModelScope 预下载：

**方法一：ModelScope CLI 下载（推荐）**

```bash
# 安装 ModelScope（如已安装可跳过）
pip install modelscope

# 下载模型到项目 models/ 目录
modelscope download BAAI/bge-reranker-v2-m3 --local_dir ./models/BAAI/bge-reranker-v2-m3
```

下载完成后，在 `.env` 中将模型路径指向本地目录：

```bash
RERANKER_MODEL=./models/BAAI/bge-reranker-v2-m3
```

**方法二：Python 脚本下载**

```python
from modelscope import snapshot_download

model_dir = snapshot_download("BAAI/bge-reranker-v2-m3", cache_dir="./models")
print(f"模型已下载到: {model_dir}")
```

**方法三：sentence-transformers 自动下载**

```bash
# 设置 HuggingFace 镜像（可选，加速下载）
export HF_ENDPOINT=https://hf-mirror.com

# 首次运行时会自动下载到 ~/.cache/huggingface/hub/
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

> `models/` 目录已加入 `.gitignore`，模型文件不会被提交到代码仓库。


---

## 访问地址

- Web 界面：`http://localhost:9900`
- API 文档：`http://localhost:9900/docs`
- Milvus Attu：`http://localhost:8000`
- MinIO 控制台：`http://localhost:9001`

## 常用 API

| 功能            | 方法   | 路径                                  |
| ------------- | ---- | ----------------------------------- |
| 普通对话          | POST | `/api/chat`                         |
| 流式对话          | POST | `/api/chat_stream`                  |
| 清空会话          | POST | `/api/chat/clear`                   |
| 会话历史          | GET  | `/api/chat/session/{session_id}`    |
| AIOps 诊断      | POST | `/api/aiops`                        |
| 诊断历史          | GET  | `/api/aiops/diagnosis/{session_id}` |
| 文件上传          | POST | `/api/upload`                       |
| 目录索引          | POST | `/api/index_directory`              |
| 健康检查          | GET  | `/api/health`                       |
| Prometheus 指标 | GET  | `/metrics`                          |

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

# RAG 消融实验
python -m tests.evaluation.run_ablation

# Agent 工具调用评估
python -m tests.evaluation.evaluate_agent --skip-goal

# AIOps 诊断流程评估
python -m tests.evaluation.evaluate_aiops_agent
```

## 常用开发命令

```bash
make help          # 查看可用命令
make start         # 启动 MCP + FastAPI 服务（本地开发）
make stop          # 停止服务
make restart       # 重启服务
make dev           # 开发模式启动 FastAPI
make upload        # 上传 aiops-docs 下的文档
make check         # 健康检查
make status-mcp    # 查看 MCP 服务状态
make test          # 运行测试
make lint          # 代码检查
make format        # 代码格式化

# Docker 相关
make docker-build  # 构建 Docker 镜像（精简版）
make docker-up     # 一键启动所有服务（基础设施 + 应用）
make docker-down   # 停止所有 Docker 服务
make docker-logs   # 查看应用容器日志
make docker-push   # 推送镜像到 Docker Hub
```
