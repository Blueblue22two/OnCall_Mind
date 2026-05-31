# SuperBizAgent 架构文档

## 1. 系统概述

### 1.1 项目背景

SuperBizAgent 是一个基于 LangChain + LangGraph 的智能运维（AIOps）系统，旨在通过 RAG（检索增强生成）和 Agent 技术实现自动化的故障诊断和运维辅助决策。

### 1.2 技术栈

| 层级 | 技术选型 |
|------|----------|
| **LLM** | 阿里千问（ChatQwen via DashScope API） |
| **向量数据库** | Milvus（支持 Dense Vector + Sparse BM25） |
| **Agent 框架** | LangGraph（Plan-Execute-Replan 模式） |
| **工具协议** | MCP（Model Context Protocol） |
| **Web 框架** | FastAPI + SSE 流式响应 |
| **嵌入模型** | DashScope text-embedding-v4（1024 维） |
| **精排模型** | BAAI/bge-reranker-v2-m3 |

### 1.3 核心能力

- **智能对话**：基于 RAG 的知识问答，支持多轮对话
- **故障诊断**：自动获取告警、分析根因、生成诊断报告
- **知识管理**：文档上传、分块、向量化、检索
- **混合检索**：Dense ANN + Sparse BM25 + RRF 融合
- **精排优化**：Cross-Encoder 重排序提升检索质量
- **流式输出**：SSE 实时推送诊断过程和结果

---

## 2. 架构设计

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                        │
│         Static Web UI (HTML/CSS/JS) + FastAPI Routes            │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                          API Layer                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ /chat    │ │ /file    │ │ /aiops   │ │ /health  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                        Service Layer                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ RagAgentService │ │ AIOpsService    │ │ VectorServices  │   │
│  │ (对话代理)       │ │ (诊断服务)       │ │ (向量存储服务)   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Layer                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Plan-Execute-Replan Workflow                 │   │
│  │  ┌────────┐    ┌──────────┐    ┌───────────┐            │   │
│  │  │Planner │ -> │ Executor │ -> │ Replanner │ <-> 循环    │   │
│  │  └────────┘    └──────────┘    └───────────┘            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                       Retriever Layer                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Enhanced RAG Pipeline                  │   │
│  │  Query Preprocessing -> Hybrid Search -> Reranking       │   │
│  │  (rewrite/none)      (Dense+BM25+RRF)  (Cross-Encoder)   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                     Infrastructure Layer                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ Milvus      │ │ MCP Client  │ │ LLM Factory │               │
│  │ (向量存储)   │ │ (工具协议)   │ │ (模型调用)   │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      External Services                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ DashScope   │ │ MCP Servers │ │ Milvus      │               │
│  │ (LLM/Embed) │ │ (CLS/Monitor)│ │ (Database)  │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
app/
├── main.py                 # FastAPI 应用入口
├── config.py               # 配置管理（Pydantic Settings）
├── api/                    # API 路由层
│   ├── chat.py             # 对话接口
│   ├── file.py             # 文件上传接口
│   ├── aiops.py            # AIOps 诊断接口
│   └── health.py           # 健康检查接口
├── services/               # 业务服务层
│   ├── rag_agent_service.py      # RAG Agent 服务
│   ├── aiops_service.py          # AIOps 诊断服务
│   ├── vector_store_manager.py   # 基础向量存储管理
│   ├── enhanced_vector_store_manager.py  # 增强向量存储管理
│   ├── vector_embedding_service.py       # 嵌入服务
│   ├── vector_index_service.py           # 索引服务
│   ├── vector_search_service.py          # 搜索服务
│   └── document_splitter_service.py      # 文档分块服务
├── agent/                  # Agent 模块
│   ├── mcp_client.py       # MCP 客户端管理
│   └── aiops/              # AIOps Agent
│       ├── state.py        # 状态定义
│       ├── planner.py      # 规划节点
│       ├── executor.py     # 执行节点
│       ├── replanner.py    # 重规划节点
│       └── utils.py        # 工具函数
├── retriever/              # 检索模块
│   ├── base.py             # 基类
│   ├── basic.py            # 基础检索器
│   ├── enhanced.py         # 增强检索器
│   ├── factory.py          # 工厂方法
│   ├── preprocessing/      # 查询预处理
│   │   ├── rewrite.py      # LLM 改写
│   │   └── passthrough.py  # 直通
│   └── reranker/           # 精排器
│       ├── cross_encoder.py  # Cross-Encoder 精排
│       └── passthrough.py    # 直通
├── tools/                  # 工具模块
│   ├── knowledge_tool.py   # 知识检索工具
│   └── time_tool.py        # 时间工具
├── core/                   # 核心组件
│   ├── milvus_client.py    # Milvus 客户端
│   └── llm_factory.py      # LLM 工厂
├── models/                 # 数据模型
│   ├── request.py          # 请求模型
│   ├── response.py         # 响应模型
│   ├── document.py         # 文档模型
│   └── aiops.py            # AIOps 模型
└── utils/                  # 工具函数
    └── logger.py           # 日志配置
```

---

## 3. 核心组件详解

### 3.1 Agent 系统

#### 3.1.1 RAG Agent（对话代理）

**文件**: `app/services/rag_agent_service.py`

**职责**: 处理用户对话请求，结合知识检索生成回答

**核心流程**:
```
用户问题 -> System Prompt + Tools -> LLM 推理 -> 工具调用 -> 生成回答
```

**关键特性**:
- 使用 `ChatQwen` 原生集成，支持流式输出
- 工具包括：`retrieve_knowledge`（知识检索）、`get_current_time`（时间）
- MCP 工具动态加载（CLS 日志查询、监控指标查询）
- 会话管理：`MemorySaver` 持久化对话历史
- 消息修剪：保留首条系统消息 + 最近 6 条消息

#### 3.1.2 AIOps Agent（诊断代理）

**文件**: `app/services/aiops_service.py`, `app/agent/aiops/`

**职责**: 自动化故障诊断，生成诊断报告

**核心流程**: Plan-Execute-Replan 模式

```
┌─────────┐     ┌──────────┐     ┌───────────┐
│ Planner │ --> │ Executor │ --> │ Replanner │
└─────────┘     └──────────┘     └───────────┘
     │               │                  │
     │ 制定计划       │ 执行步骤          │ 评估结果
     │               │                  │
     │               └──────────────────┘
     │                      是否完成？
     │                     /        \
     │                   否          是
     │                    \          /
     │                     \        /
     │                    继续执行  END
     └──────────────────────────────
```

**节点职责**:

| 节点 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **Planner** | 制定执行计划 | 用户任务 | 步骤列表 |
| **Executor** | 执行单个步骤 | 当前步骤 | 执行结果 |
| **Replanner** | 评估结果，决定下一步 | 执行结果 | 新计划/最终响应 |

**状态定义** (`PlanExecuteState`):
```python
{
    "input": str,           # 用户输入
    "plan": List[str],      # 待执行步骤
    "past_steps": List,     # 已执行步骤及结果
    "response": str         # 最终响应
}
```

### 3.2 Retriever 系统

#### 3.2.1 Basic RAG Retriever

**文件**: `app/retriever/basic.py`

**流程**: 简单的向量相似度检索

```
Query -> Embedding -> Milvus ANN Search -> Top-K Documents
```

#### 3.2.2 Enhanced RAG Retriever

**文件**: `app/retriever/enhanced.py`

**流程**: 三阶段增强检索

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Query Preprocessing                                │
│   - none: 直接使用原始查询                                   │
│   - rewrite: LLM 语义改写                                    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Hybrid Search (粗排)                               │
│   - Dense ANN: DashScope text-embedding-v4 (COSINE)         │
│   - Sparse BM25: Milvus 内置 BM25 (Jieba 中文分词)          │
│   - RRF 融合 (k=60)                                         │
│   - 候选数: rerank_coarse_top_k (默认 20)                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Reranking (精排)                                   │
│   - none: 直接截断到 top_k                                  │
│   - cross_encoder: BGE bge-reranker-v2-m3                   │
│   - 使用原始查询打分（非改写后查询）                          │
└─────────────────────────────────────────────────────────────┘
```

**关键设计决策**:
- 预处理后的查询用于混合检索，提升召回
- 精排始终使用**原始查询**打分，确保分数反映用户真实意图

### 3.3 向量存储系统

#### 3.3.1 Milvus Collection 设计

**基础 Collection (`biz`)**:
```python
fields = [
    FieldSchema(name="id", dtype=VARCHAR, max_length=100, is_primary=True),
    FieldSchema(name="vector", dtype=FLOAT_VECTOR, dim=1024),
    FieldSchema(name="content", dtype=VARCHAR, max_length=8000),
    FieldSchema(name="metadata", dtype=JSON),
]
# 索引: IVF_FLAT, metric=L2
```

**增强 Collection (`biz_enhanced`)**:
```python
fields = [
    FieldSchema(name="id", dtype=VARCHAR, max_length=100, is_primary=True),
    FieldSchema(name="dense_vector", dtype=FLOAT_VECTOR, dim=1024),
    FieldSchema(name="content_text", dtype=VARCHAR, max_length=8000, 
                enable_analyzer=True, analyzer_params={"type": "chinese"}),
    FieldSchema(name="sparse_vector", dtype=SPARSE_FLOAT_VECTOR),  # BM25 自动生成
    FieldSchema(name="metadata", dtype=JSON),
]
# 索引: 
#   - dense_vector: IVF_FLAT, metric=COSINE
#   - sparse_vector: SPARSE_INVERTED_INDEX, metric=BM25
```

### 3.4 MCP 工具系统

#### 3.4.1 MCP 客户端

**文件**: `app/agent/mcp_client.py`

**职责**: 管理 MCP 服务器连接，提供工具调用能力

**关键特性**:
- 全局单例模式，避免重复初始化
- 重试拦截器：指数退避策略（最多 3 次重试）
- 支持多服务器配置

**配置示例**:
```python
mcp_servers = {
    "cls": {
        "transport": "streamable-http",
        "url": "http://localhost:8003/mcp"
    },
    "monitor": {
        "transport": "streamable-http",
        "url": "http://localhost:8004/mcp"
    }
}
```

#### 3.4.2 MCP 服务器

**文件**: `mcp_servers/`

| 服务器 | 端口 | 功能 |
|--------|------|------|
| `cls_server.py` | 8003 | 云日志服务查询 |
| `monitor_server.py` | 8004 | 监控指标查询 |

---

## 4. API 接口规范

### 4.1 对话接口

**POST** `/api/chat`

请求:
```json
{
    "question": "如何处理 CPU 高使用率问题？",
    "session_id": "session-123"
}
```

响应（SSE 流式）:
```
event: message
data: {"type": "content", "data": "根据知识库..."}

event: message
data: {"type": "complete"}
```

### 4.2 文件上传接口

**POST** `/api/files/upload`

请求: `multipart/form-data`
- `file`: 文件内容
- `collection_name`: 集合名称（可选）

响应:
```json
{
    "message": "文件上传成功",
    "filename": "document.pdf",
    "chunks_count": 15
}
```

### 4.3 AIOps 诊断接口

**POST** `/api/aiops`

请求:
```json
{
    "session_id": "session-123"
}
```

响应（SSE 流式事件类型）:

| 事件类型 | 说明 |
|----------|------|
| `status` | 状态更新 |
| `plan` | 诊断计划制定完成 |
| `step_complete` | 步骤执行完成 |
| `report` | 最终诊断报告 |
| `complete` | 诊断完成 |
| `error` | 错误信息 |

### 4.4 健康检查接口

**GET** `/api/health`

响应:
```json
{
    "status": "healthy",
    "milvus": "connected"
}
```

---

## 5. 配置说明

### 5.1 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `APP_NAME` | SuperBizAgent | 应用名称 |
| `APP_VERSION` | 1.0.0 | 应用版本 |
| `DEBUG` | False | 调试模式 |
| `HOST` | 0.0.0.0 | 监听地址 |
| `PORT` | 9900 | 监听端口 |
| `DASHSCOPE_API_KEY` | - | DashScope API 密钥 |
| `DASHSCOPE_MODEL` | qwen-max | LLM 模型 |
| `DASHSCOPE_EMBEDDING_MODEL` | text-embedding-v4 | 嵌入模型 |
| `MILVUS_HOST` | localhost | Milvus 地址 |
| `MILVUS_PORT` | 19530 | Milvus 端口 |
| `RAG_MODE` | basic | RAG 模式（basic/enhanced） |
| `RAG_TOP_K` | 3 | 检索文档数 |
| `QUERY_PREPROCESSOR_TYPE` | none | 查询预处理方式 |
| `RERANKER_TYPE` | cross_encoder | 精排器类型 |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | 精排模型 |

### 5.2 RAG 模式配置

**Basic 模式**:
```env
RAG_MODE=basic
RAG_TOP_K=3
```

**Enhanced 模式**:
```env
RAG_MODE=enhanced
QUERY_PREPROCESSOR_TYPE=rewrite
RERANKER_TYPE=cross_encoder
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_TOP_K=3
RERANK_COARSE_TOP_K=20
```

---

## 6. 部署架构

### 6.1 依赖服务

```
┌─────────────────────────────────────────────────────────────┐
│                    SuperBizAgent                            │
│                    (FastAPI :9900)                          │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Milvus      │ │ DashScope   │ │ MCP CLS     │ │ MCP Monitor │
│ :19530      │ │ API         │ │ :8003       │ │ :8004       │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

### 6.2 启动命令

```bash
# 安装依赖
uv sync

# 启动 Milvus
docker-compose -f vector-database.yml up -d

# 启动 MCP 服务器
python mcp_servers/cls_server.py &
python mcp_servers/monitor_server.py &

# 启动应用
uv run python -m app.main
```

---

## 7. 架构优化建议

### 7.1 当前问题

| 问题 | 影响 | 优先级 |
|------|------|--------|
| **服务层耦合** | RagAgentService 职责过重，直接依赖 MCP 客户端 | 高 |
| **向量存储分散** | 存在两套向量存储管理器，代码重复 | 高 |
| **错误恢复不足** | AIOps 流程中步骤失败后缺乏重试/回退 | 中 |
| **配置热更新缺失** | 无法运行时调整参数 | 中 |
| **测试覆盖不足** | 仅有 RAG 评估测试，缺少单元测试 | 中 |
| **会话持久化** | MemorySaver 重启后丢失 | 低 |

### 7.2 优化方案

#### 7.2.1 抽取工具注册中心

```python
# 建议新增: app/tools/registry.py
class ToolRegistry:
    def __init__(self):
        self._tools = {}
    
    def register(self, name: str, tool):
        self._tools[name] = tool
    
    def get_all_tools(self) -> List:
        return list(self._tools.values())
```

#### 7.2.2 统一向量存储管理器

```python
# 建议合并为: app/services/vector_store_manager.py
class VectorStoreManager:
    def __init__(self, mode: Literal["basic", "enhanced"]):
        self.mode = mode
        # 统一接口
```

#### 7.2.3 添加错误处理节点

```python
# 在 StateGraph 中添加
workflow.add_node("error_handler", error_handler_node)
workflow.add_edge(NODE_EXECUTOR, "error_handler")
```

#### 7.2.4 引入 Redis 会话持久化

```python
# 替换 MemorySaver
from langgraph.checkpoint.redis import RedisSaver
checkpointer = RedisSaver.from_conn_string("redis://localhost:6379")
```

---

## 8. 需求实现清单

### 8.1 已实现功能

| 功能模块 | 功能点 | 状态 | 说明 |
|----------|--------|------|------|
| **RAG Agent** | 对话式问答 | ✅ 完整 | 支持多轮对话 |
| | 知识检索 | ✅ 完整 | Basic/Enhanced 两种模式 |
| | 流式输出 | ✅ 完整 | SSE 实时推送 |
| | 会话管理 | ✅ 完整 | MemorySaver |
| **AIOps Agent** | Plan-Execute-Replan | ✅ 完整 | LangGraph 实现 |
| | 自动诊断 | ✅ 完整 | 告警获取+分析+报告 |
| | 流式诊断过程 | ✅ 完整 | SSE 事件流 |
| **向量存储** | 文档上传 | ✅ 完整 | 支持多种格式 |
| | 文档分块 | ✅ 完整 | 递归字符分块 |
| | 向量嵌入 | ✅ 完整 | DashScope API |
| | 混合检索 | ✅ 完整 | Dense + BM25 + RRF |
| | 精排优化 | ✅ 完整 | Cross-Encoder |
| **MCP 工具** | CLS 日志查询 | ✅ 完整 | MCP 服务器 |
| | 监控指标查询 | ✅ 完整 | MCP 服务器 |
| | 重试机制 | ✅ 完整 | 指数退避 |
| **API** | 对话接口 | ✅ 完整 | /api/chat |
| | 文件上传 | ✅ 完整 | /api/files/upload |
| | AIOps 诊断 | ✅ 完整 | /api/aiops |
| | 健康检查 | ✅ 完整 | /api/health |

### 8.2 未实现/待完善功能

| 功能模块 | 功能点 | 优先级 | 说明 |
|----------|--------|--------|------|
| **RAG 评估** | 自动化评估流程 | 高 | `eval.md` 提到但未完全实现 |
| | 评估指标计算 | 高 | RAGAS 指标 |
| | 评估报告生成 | 中 | 对比报告 |
| **前端 UI** | 诊断结果可视化 | 中 | 当前仅有基础 UI |
| | 实时诊断进度 | 中 | WebSocket/SSE 展示 |
| **告警系统** | 主动推送机制 | 中 | 当前仅被动查询 |
| | 告警订阅管理 | 低 | 多租户告警隔离 |
| **持久化** | 会话历史持久化 | 低 | 当前使用内存 |
| | 诊断报告存储 | 低 | 历史报告查询 |
| **多租户** | 租户隔离 | 低 | 当前无隔离 |
| | 权限控制 | 低 | RBAC |
| **可观测性** | 链路追踪 | 低 | OpenTelemetry |
| | 指标监控 | 低 | Prometheus |

### 8.3 接口实现状态

| 接口路径 | 方法 | 状态 | 备注 |
|----------|------|------|------|
| `/api/chat` | POST | ✅ 已实现 | 流式对话 |
| `/api/chat/history` | GET | ✅ 已实现 | 获取会话历史 |
| `/api/chat/clear` | DELETE | ✅ 已实现 | 清空会话 |
| `/api/files/upload` | POST | ✅ 已实现 | 文件上传 |
| `/api/files/collections` | GET | ✅ 已实现 | 获取集合列表 |
| `/api/files/collections/{name}` | DELETE | ✅ 已实现 | 删除集合 |
| `/api/aiops` | POST | ✅ 已实现 | AIOps 诊断 |
| `/api/health` | GET | ✅ 已实现 | 健康检查 |

---

