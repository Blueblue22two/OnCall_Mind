# OnCall Mind Architecture

> 文档日期：2026-06-01  
> 项目名称：OnCall Mind  
> 项目定位：面向智能运维场景的 On-Call 诊断助手  

## 1. 系统概述

OnCall Mind 是一个基于 FastAPI、LangChain、LangGraph、Milvus 和 MCP 的智能运维辅助系统。系统当前主要用于 RAG 知识库问答、日志/监控工具调用、AIOps 故障诊断、诊断报告生成和基础评估实验。

当前系统定位是“智能运维辅助系统”，重点帮助运维人员完成信息检索、告警排查、根因分析和处理建议生成。系统暂不默认执行真实生产变更操作，自动化处置、审批、回滚、工单集成和复盘沉淀属于后续演进方向。

## 2. 核心能力

| 能力 | 当前实现 |
|---|---|
| RAG 知识问答 | 支持 Basic / Enhanced 两种检索模式 |
| AIOps 诊断 | 基于 Plan-Execute-Replan 的诊断工作流 |
| 工具调用 | 通过 MCP 接入日志查询和监控指标查询 |
| Web 交互 | FastAPI + 静态前端 + SSE 流式输出 |
| 文档索引 | 支持 Markdown / TXT / PDF 上传、分块、向量化 |
| 会话管理 | 支持 MemorySaver / RedisSaver |
| 诊断持久化 | 支持诊断报告和 trace 存储 |
| 评估体系 | 支持 RAG Eval、Agent Eval、AIOps Eval |
| 可观测性 | 支持日志、TraceStore、Prometheus Metrics |

## 3. 总体架构

```text
┌─────────────────────────────────────────────────────────────┐
│                        Web / API Layer                       │
│  Static UI + FastAPI Routes                                  │
│  /api/chat /api/chat_stream /api/aiops /api/upload /metrics  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                       Service Layer                          │
│  RagAgentService  AIOpsService  VectorIndexService           │
│  DiagnosisStore   TraceStore    VectorStoreManager           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                         Agent Layer                          │
│  RAG Agent: ReAct-style tool calling                         │
│  AIOps Agent: Planner -> Executor -> Replanner               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                       Retriever Layer                        │
│  Basic Retriever: Dense Vector Search                        │
│  Enhanced Retriever: Query Rewrite + Hybrid Search + Rerank  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                         Tool Layer                           │
│  Local Tools: retrieve_knowledge, get_current_time            │
│  MCP Tools: CLS log tools, Monitor metric tools               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                      │
│  DashScope LLM / Embedding  Milvus  Redis  MCP Servers        │
└─────────────────────────────────────────────────────────────┘
```

## 4. 代码结构

```text
app/
├── api/                    # FastAPI 路由
├── agent/                  # Agent 工作流与 MCP 客户端
│   └── aiops/              # Planner / Executor / Replanner
├── core/                   # LLM 工厂、Milvus 客户端、Metrics
├── models/                 # Pydantic 请求和响应模型
├── retriever/              # Basic / Enhanced RAG 检索器
│   ├── preprocessing/      # 查询预处理
│   └── reranker/           # 精排器
├── services/               # RAG、AIOps、索引、存储、Trace 服务
├── tools/                  # 本地工具
└── utils/                  # 日志等通用能力

mcp_servers/
├── cls_server.py           # 日志查询 MCP 服务
└── monitor_server.py       # 监控指标 MCP 服务

tests/evaluation/
├── evaluate_rag.py         # RAG 评估
├── evaluate_agent.py       # Agent 工具调用评估
├── evaluate_aiops_agent.py # AIOps 诊断流程评估
└── metrics/                # 自定义评估指标
```

## 5. API 层设计

API 层基于 FastAPI 实现，主要负责请求接入、参数校验、流式响应和服务调用。

| 路由 | 职责 |
|---|---|
| `app/api/chat.py` | 普通对话、流式对话、会话管理 |
| `app/api/aiops.py` | AIOps 诊断和诊断历史查询 |
| `app/api/file.py` | 文件上传和目录索引 |
| `app/api/health.py` | 健康检查和 Prometheus 指标 |

核心 API：

| 功能 | 方法 | 路径 |
|---|---|---|
| 普通对话 | POST | `/api/chat` |
| 流式对话 | POST | `/api/chat_stream` |
| AIOps 诊断 | POST | `/api/aiops` |
| 文件上传 | POST | `/api/upload` |
| 目录索引 | POST | `/api/index_directory` |
| 健康检查 | GET | `/api/health` |
| Metrics | GET | `/metrics` |

## 6. Service 层设计

Service 层承载系统主要业务逻辑。

| 服务 | 职责 |
|---|---|
| `RagAgentService` | 构建 RAG Agent，处理对话和工具调用 |
| `AIOpsService` | 构建 Plan-Execute-Replan 诊断工作流 |
| `VectorIndexService` | 读取文件、分块、写入向量库 |
| `DocumentSplitterService` | Markdown / 文本分块 |
| `VectorStoreManager` | Basic 向量集合管理 |
| `EnhancedVectorStoreManager` | Enhanced 混合检索集合管理 |
| `TraceStore` | 保存 Agent 和 RAG 运行轨迹 |
| `DiagnosisStore` | 保存诊断报告 |

Service 层向上为 API 提供业务接口，向下调用 Agent、Retriever、Tool、Milvus、Redis 等组件。

## 7. RAG 架构

### 7.1 Basic RAG

Basic 模式是单阶段 Dense 向量检索：

```text
Query
  -> Embedding
  -> Milvus Dense Vector Search
  -> Top-K Documents
  -> LLM Answer
```

特点：

- 实现简单。
- 延迟较低。
- 适合语义明确、知识库规模较小的场景。

### 7.2 Enhanced RAG

Enhanced 模式是三阶段检索流水线：

```text
Query
  -> Query Preprocessing
  -> Dense + Sparse Hybrid Search
  -> RRF Fusion
  -> Cross-Encoder Reranking
  -> Top-K Documents
  -> LLM Answer
```

阶段说明：

| 阶段 | 说明 |
|---|---|
| Query Preprocessing | 可选 LLM 查询改写，提升召回 |
| Hybrid Search | Dense 语义向量 + BM25 稀疏关键词检索 |
| RRF Fusion | 融合 Dense 与 Sparse 排名 |
| Reranking | 使用 Cross-Encoder 对候选文档精排 |

降级策略：

- 查询改写失败：回退到原始 query。
- 精排失败：回退到粗排 Top-K。
- 混合检索失败：抛出异常，不静默降级。

### 7.3 文档索引流程

```text
Upload File / Index Directory
  -> Read Content
  -> Split Document
  -> Generate Embeddings
  -> Write Basic Collection
  -> Write Enhanced Collection
```

系统会将文档同时写入 Basic 和 Enhanced 两套集合，便于通过配置切换检索模式。

## 8. Agent 架构

### 8.1 RAG Agent

RAG Agent 使用 LangChain / LangGraph Agent 能力，绑定本地工具和 MCP 工具。

可用工具：

- `retrieve_knowledge`：从 RAG 知识库检索相关内容。
- `get_current_time`：获取当前时间。
- MCP 日志工具：查询日志主题、搜索日志等。
- MCP 监控工具：查询 CPU、内存等指标。

处理流程：

```text
User Question
  -> System Prompt
  -> LLM Tool Decision
  -> Tool Calls
  -> Tool Results
  -> Final Answer
```

### 8.2 AIOps Agent

AIOps Agent 使用 Plan-Execute-Replan 工作流。

```text
Input
  -> Planner
  -> Executor
  -> Replanner
  -> Executor / Final Response
```

节点职责：

| 节点 | 职责 |
|---|---|
| Planner | 结合用户任务、知识库和可用工具制定诊断计划 |
| Executor | 执行当前步骤，调用知识库、日志、监控等工具 |
| Replanner | 判断是否继续执行、重新规划或生成最终报告 |
| Error Handler | 处理错误次数超限、工具失败等异常路径 |

核心状态：

```python
{
    "input": str,
    "plan": list[str],
    "past_steps": list[tuple[str, str]],
    "response": str,
    "trace_id": str,
    "error_count": int,
    "max_errors": int,
    "last_error": str,
}
```

当前 AIOps Agent 主要用于辅助诊断和生成建议，不负责默认执行真实变更操作。

## 9. MCP 工具架构

系统通过 MCP 将外部工具抽象为 LLM 可调用工具。

```text
Agent
  -> MultiServerMCPClient
  -> CLS MCP Server
  -> Monitor MCP Server
```

当前 MCP 服务：

| 服务 | 默认端口 | 作用 |
|---|---:|---|
| CLS MCP Server | 8003 | 模拟日志主题查询、日志搜索 |
| Monitor MCP Server | 8004 | 模拟 CPU、内存等监控指标查询 |

MCP Client 具备指数退避重试能力。当工具失败时，会重试并记录错误结果，避免单次失败直接中断整个诊断流程。

## 10. 数据存储架构

| 数据 | 存储 |
|---|---|
| 文档向量 | Milvus |
| 稀疏检索数据 | Milvus BM25 Function / Enhanced Collection |
| 会话状态 | RedisSaver 或 MemorySaver |
| 诊断报告 | Redis 或本地 JSON 文件 |
| Agent Trace | TraceStore，支持文件或 Redis 后端 |
| 日志 | Loguru 本地日志文件 |
| 评估结果 | `reports/` 目录下 JSON / CSV |

## 11. 主要数据流

### 11.1 RAG 问答数据流

```text
User Question
  -> /api/chat or /api/chat_stream
  -> RagAgentService
  -> LLM decides retrieve_knowledge
  -> Retriever searches Milvus
  -> Documents formatted as context
  -> LLM generates answer
  -> API returns answer / SSE tokens
```

### 11.2 AIOps 诊断数据流

```text
Diagnosis Request
  -> /api/aiops
  -> AIOpsService
  -> Planner creates plan
  -> Executor calls tools
  -> MCP Client calls CLS / Monitor servers
  -> Replanner evaluates evidence
  -> Final report generated
  -> DiagnosisStore saves record
  -> TraceStore saves trace
```

### 11.3 文档索引数据流

```text
Upload / Index Directory
  -> VectorIndexService
  -> DocumentSplitterService
  -> Embedding Service
  -> Milvus Basic Collection
  -> Milvus Enhanced Collection
```

## 12. 配置架构

系统配置集中在 `app/config.py`，通过 `.env` 注入。

主要配置域：

| 配置域 | 示例 |
|---|---|
| LLM | `DASHSCOPE_MODEL`、`LLM_TIMEOUT`、`LLM_MAX_RETRIES` |
| Embedding | `DASHSCOPE_EMBEDDING_MODEL` |
| Milvus | `MILVUS_HOST`、`MILVUS_PORT`、`MILVUS_NPROBE` |
| RAG | `RAG_MODE`、`RAG_TOP_K` |
| Enhanced RAG | `QUERY_PREPROCESSOR_TYPE`、`RERANKER_TYPE`、`RERANK_COARSE_TOP_K` |
| Redis | `REDIS_URL` |
| MCP | `MCP_CLS_URL`、`MCP_MONITOR_URL` |
| Eval | `EVAL_JUDGE_MODEL`、`EVAL_JUDGE_API_BASE` |

## 13. 部署架构

本地部署由三类进程组成：

```text
Docker Compose
  -> Milvus
  -> Etcd
  -> MinIO
  -> Attu
  -> Redis

Python Processes
  -> FastAPI App :9900
  -> CLS MCP Server :8003
  -> Monitor MCP Server :8004

External Services
  -> DashScope LLM API
  -> DashScope Embedding API
```

访问入口：

| 服务 | 地址 |
|---|---|
| Web UI | `http://localhost:9900` |
| API Docs | `http://localhost:9900/docs` |
| CLS MCP | `http://localhost:8003/mcp` |
| Monitor MCP | `http://localhost:8004/mcp` |
| Milvus | `localhost:19530` |
| Attu | `http://localhost:8000` |

## 14. 可观测性

当前可观测性能力：

- Loguru 本地日志。
- Agent 节点 trace。
- RAG trace。
- 工具调用 trace。
- token usage 记录。
- Prometheus `/metrics` 端点。
- SSE 诊断过程事件。

建议继续增强：

- 统一 trace_id 贯穿 API、Agent、Tool、Report。
- 工具调用参数和返回值标准化。
- 增加 trajectory quality 评估。
- 增加错误分类和失败原因聚合。
- 增加线上延迟、成本、成功率看板。

## 15. 安全边界

当前系统主要用于诊断辅助和本地演示，默认不执行真实生产变更操作。

后续若接入执行类工具，必须补充：

- 用户身份认证。
- RBAC 权限控制。
- 操作分级。
- 人工审批。
- dry-run。
- 回滚机制。
- 执行后验证。
- 审计日志。

建议操作分级：

| 等级 | 类型 | 策略 |
|---|---|---|
| L0 | 只读查询 | 可自动执行 |
| L1 | 低风险诊断操作 | 自动执行并记录 |
| L2 | 可逆变更 | 需要人工审批 |
| L3 | 高风险变更 | 多人审批和变更窗口 |
| L4 | 破坏性操作 | 默认禁止 |

## 16. 评估架构

系统提供三类评估：

| 评估 | 脚本 | 目标 |
|---|---|---|
| RAG Eval | `tests/evaluation/evaluate_rag.py` | 评估检索质量和生成质量 |
| Agent Eval | `tests/evaluation/evaluate_agent.py` | 评估工具调用准确率和目标达成 |
| AIOps Eval | `tests/evaluation/evaluate_aiops_agent.py` | 评估诊断计划、工具调用、结论命中 |

已有指标：

- context precision。
- context recall。
- hit rate。
- MRR。
- faithfulness。
- answer relevancy。
- tool exact match。
- tool precision。
- tool recall。
- goal accuracy。

建议补充指标：

- Tool Argument Accuracy。
- RCA Correctness。
- Evidence Coverage。
- Citation Accuracy。
- Hallucination Rate。
- Trajectory Quality。
- Error Recovery Rate。
- Latency P95。
- Cost per Diagnosis。

## 17. 当前架构限制

1. 当前不是完整多 Agent 架构，而是单诊断 Agent 的多节点工作流。
2. 系统主要支持诊断和建议生成，不支持生产级自动处置闭环。
3. RAG 回答尚未强制绑定引用证据。
4. 工具调用参数校验仍需增强。
5. 知识库缺少版本、过期和审核机制。
6. 评估数据集规模仍需扩充。
7. 权限、审批、回滚和审计能力尚未形成。

## 18. 后续演进方向

短期：

- 强化工具参数 schema 校验。
- 增加 RAG 引用和证据链输出。
- 扩展评估集并纳入 CI。
- 优化 MCP 工具异常处理。

中期：

- 接入真实告警事件。
- 增加 Ticket / 工单集成。
- 增加人工审批和操作分级。
- 增加处置后 Verification Agent。
- 增加知识版本和 freshness 治理。

长期：

- 演进为 Supervisor 多 Agent 架构。
- 建立 Evidence Graph。
- 引入 Runbook DSL。
- 支持半自动或自动化处置。
- 建立线上持续评估和 Shadow Mode。

## 19. 总结

OnCall Mind 当前已经具备智能运维诊断助手的核心技术底座：RAG、Agent 编排、MCP 工具调用、诊断报告、Trace 和评估体系。现阶段架构适合作为智能运维辅助分析平台或 AIOps 原型系统。

下一阶段的架构重点应从“能回答、能诊断”转向“有证据、可追溯、可评估、可审批、可验证”。在完成工具参数可靠性、证据链、权限审批和评估闭环后，系统才能进一步演进为生产级 Multi-Agent AIOps 平台。
