# 智能运维 Agent System

> 企业级智能对话和运维助手，支持可插拔 RAG 知识库问答和 AIOps 智能诊断

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-purple.svg)](https://langchain-ai.github.io/langgraph/)

## 核心特性

- **智能对话** — LangGraph ReAct Agent 多轮对话 + SSE 流式输出，支持工具调用过程可视化
- **可插拔 RAG** — 两种检索模式：Basic（Dense 向量检索）和 Enhanced（查询改写 + 双向量混合检索 + Cross-Encoder 精排），通过配置切换
- **AIOps 诊断** — Plan-Execute-Replan 自动故障诊断，流式输出诊断过程和结构化报告
- **RAG 评估体系** — 基于 RAGAs 的两阶段评估（检索质量 + 生成质量），支持消融实验和 Basic vs Enhanced 对比报告
- **Agent 评估体系** — 工具调用准确率 + LLM Judge 目标达成率，覆盖 6 类诊断场景
- **Memory 管理** — RedisSaver 持久化会话（可选），tiktoken 上下文自动裁剪，诊断报告持久化存储
- **MCP 集成** — 日志查询和监控数据工具接入，指数退避重试
- **Web 界面** — 现代化 UI，支持 RAG 快速问答 / 流式对话 / AIOps 智能诊断三种模式

## 技术栈

- **框架**: FastAPI + LangChain + LangGraph
- **LLM**: 阿里云 DashScope（通义千问 qwen-max）
- **Embedding**: DashScope text-embedding-v4（1024 维）
- **向量库**: Milvus（双集合：biz + biz_enhanced）
- **精排模型**: BAAI/bge-reranker-v2-m3（Cross-Encoder）
- **评估**: RAGAs + LLM Judge
- **会话持久化**: Redis（可选，不配置则使用 MemorySaver）
- **工具协议**: MCP (Model Context Protocol)

## 快速开始

### 环境要求

- Python 3.11–3.13
- Docker（用于 Milvus 向量数据库）
- 阿里云 DashScope API Key（[获取地址](https://dashscope.aliyun.com/)）

### 安装和启动

#### Linux/macOS 环境

```bash
# 1. 克隆项目
git clone <repository_url>
cd super_biz_agent_py

# 2. 安装依赖（推荐使用 uv）
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 3. 编辑配置文件
cp .env.example .env   # 如有模板文件
vim .env                # 填入 DASHSCOPE_API_KEY

# 4. 一键初始化（启动 Docker + 服务 + 上传文档）
make init

# 5. 一键启动
make start
```

#### Windows 环境（PowerShell/CMD）

如果 Windows 不支持 `make` 命令，可以手动执行以下步骤：

```powershell
# 1. 克隆项目
git clone <repository_url>
cd super_biz_agent_py

# 2. 创建虚拟环境并安装依赖
pip install uv
uv venv
.venv\Scripts\activate
uv pip install -e .

# 3. 编辑 .env 文件，填入 DASHSCOPE_API_KEY
notepad .env

# 4. 启动 Docker Desktop，确保正在运行

# 5. 启动 Milvus 向量数据库
docker compose -f vector-database.yml up -d

# 6. 等待 Milvus 启动（约 5-10 秒）
timeout /t 10

# 7. 启动各服务（分别在新 PowerShell 窗口中执行）
python mcp_servers/cls_server.py        # CLS 日志服务（端口 8003）
python mcp_servers/monitor_server.py    # Monitor 监控服务（端口 8004）
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900  # 主服务

# 8. 上传知识库文档
python -c "import requests, os, time; [requests.post('http://localhost:9900/api/upload', files={'file': open(f'aiops-docs/{f}', 'rb')}) or time.sleep(1) for f in os.listdir('aiops-docs') if f.endswith('.md')]"
```

**Windows 一键启动脚本**（推荐）：

```powershell
.\start-windows.bat    # 启动所有服务
.\stop-windows.bat     # 停止所有服务
```

### 访问服务

- **Web 界面**: http://localhost:9900
- **API 文档**: http://localhost:9900/docs

## API 接口

### 核心接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 普通对话 | POST | `/api/chat` | 一次性返回 |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式输出，含工具调用事件 |
| 清空会话 | POST | `/api/chat/clear` | 清除会话历史 |
| 会话历史 | GET | `/api/chat/session/{session_id}` | 查询会话消息历史 |
| AIOps 诊断 | POST | `/api/aiops` | 自动故障诊断（SSE 流式） |
| 诊断历史 | GET | `/api/aiops/diagnosis/{session_id}` | 查询历史诊断报告 |
| 文件上传 | POST | `/api/upload` | 上传并索引文档到向量库 |
| 目录索引 | POST | `/api/index_directory` | 批量索引指定目录下的文档 |
| 健康检查 | GET | `/api/health` | 服务状态 + Milvus 连接检查 |

### 使用示例

```bash
# 普通对话
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}'

# 流式对话
curl -X POST "http://localhost:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"CPU 使用率过高怎么排查？"}' \
  --no-buffer

# AIOps 诊断
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-123"}' \
  --no-buffer

# 查询诊断历史
curl "http://localhost:9900/api/aiops/diagnosis/session-123?limit=10"

# 清空会话
curl -X POST "http://localhost:9900/api/chat/clear" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-123"}'
```

## 项目结构

```
super_biz_agent_py/
├── app/                                    # 应用核心
│   ├── main.py                             # FastAPI 应用入口，生命周期管理
│   ├── config.py                           # Pydantic Settings 配置管理
│   │
│   ├── api/                                # API 路由层
│   │   ├── chat.py                         # 对话接口（RAG 聊天 + 会话管理）
│   │   ├── aiops.py                        # AIOps 接口（诊断 + 历史查询）
│   │   ├── file.py                         # 文件管理（上传 + 目录索引）
│   │   └── health.py                       # 健康检查
│   │
│   ├── models/                             # Pydantic 数据模型
│   │   ├── request.py                      # 请求模型（ChatRequest, ClearRequest）
│   │   ├── response.py                     # 响应模型（ChatResponse, SessionInfoResponse）
│   │   ├── aiops.py                        # AIOps 模型（AlertInfo, DiagnosisResponse）
│   │   └── document.py                     # 文档模型（DocumentChunk）
│   │
│   ├── services/                           # 业务服务层
│   │   ├── rag_agent_service.py            # RAG ReAct Agent（LangGraph 状态图）
│   │   ├── aiops_service.py                # AIOps Plan-Execute-Replan 工作流
│   │   ├── vector_store_manager.py         # 基础向量存储管理（biz 集合，Dense/L2）
│   │   ├── enhanced_vector_store_manager.py # 增强向量存储管理（biz_enhanced 集合，Dense+BM25/COSINE+RRF）
│   │   ├── vector_embedding_service.py     # DashScope 向量 Embedding 服务
│   │   ├── vector_search_service.py        # 向量检索服务
│   │   ├── vector_index_service.py         # 文件索引服务（双集合写入）
│   │   ├── document_splitter_service.py    # 文档分割（Markdown + 递归分割）
│   │   └── diagnosis_store.py              # 诊断记录持久化（Redis + 文件双后端）
│   │
│   ├── agent/                              # Agent 模块
│   │   ├── mcp_client.py                   # MCP 客户端（重试拦截器、全局单例）
│   │   └── aiops/                          # AIOps 核心逻辑
│   │       ├── state.py                    # PlanExecuteState 状态定义
│   │       ├── planner.py                  # 计划制定器（LLM 生成诊断步骤）
│   │       ├── executor.py                 # 步骤执行器（ToolNode 调用 MCP 工具）
│   │       ├── replanner.py                # 重规划器（continue/replan/respond 决策）
│   │       └── utils.py                    # 工具函数（格式化工具描述）
│   │
│   ├── retriever/                          # 可插拔 RAG 检索系统 ★
│   │   ├── base.py                         # BaseRAGRetriever 抽象基类
│   │   ├── factory.py                      # get_rag_retriever() 工厂（basic/enhanced）
│   │   ├── basic.py                        # BasicRAGRetriever：Dense 向量检索
│   │   ├── enhanced.py                     # EnhancedRAGRetriever：三阶段流水线
│   │   ├── preprocessing/                  # 查询预处理插件
│   │   │   ├── base.py                     # BaseQueryPreprocessor 抽象基类
│   │   │   ├── factory.py                  # get_query_preprocessor() 工厂
│   │   │   ├── passthrough.py              # PassthroughPreprocessor（透传）
│   │   │   └── rewrite.py                  # QueryRewritePreprocessor（LLM 改写）
│   │   └── reranker/                       # 结果精排插件
│   │       ├── base.py                     # BaseReranker 抽象基类
│   │       ├── factory.py                  # get_reranker() 工厂
│   │       ├── passthrough.py              # PassthroughReranker（直接截断）
│   │       └── cross_encoder.py            # CrossEncoderReranker（BGE-Reranker）
│   │
│   ├── tools/                              # Agent 工具集
│   │   ├── knowledge_tool.py               # retrieve_knowledge（从工厂获取检索器）
│   │   └── time_tool.py                    # get_current_time
│   │
│   ├── core/                               # 核心组件
│   │   ├── llm_factory.py                  # LLM 工厂（DashScope 兼容模式）
│   │   └── milvus_client.py                # Milvus 客户端（双集合 + BM25 Function）
│   │
│   └── utils/                              # 工具类
│       └── logger.py                       # Loguru 日志配置（控制台 + 按日轮转）
│
├── tests/                                  # 测试与评估
│   └── evaluation/                         # 评估框架 ★
│       ├── rag_testset.py                  # RAG 评估数据集（78 条，v1.1.2）
│       ├── agent_testset.py                # Agent 评估数据集（12 条，v1.0.0）
│       ├── evaluate_rag.py                 # RAGAs 两阶段评估（检索 + 生成）
│       ├── evaluate_agent.py               # Agent 评估（工具准确率 + 目标达成率）
│       ├── run_ablation.py                 # 消融实验（12 组参数组合）
│       ├── compare_reports.py              # Basic vs Enhanced 对比报告
│       ├── generate_questions.py           # LLM 辅助评估问题生成
│       ├── generate_docs.py                # LLM 辅助知识库文档生成
│       ├── import_questions.py             # 候选问题导入评估数据集
│       ├── validate_dataset.py             # 数据集质量检查
│       └── metrics/                        # 评估指标
│           ├── hit_rate.py                 # Hit Rate@k
│           ├── mrr.py                      # MRR (Mean Reciprocal Rank)
│           ├── tool_call_accuracy.py       # 工具调用精确率/召回率
│           └── goal_accuracy.py            # LLM Judge 目标达成率（0/1/2 评分）
│
├── mcp_servers/                            # MCP 工具服务器
│   ├── cls_server.py                       # CLS 日志查询服务（端口 8003，5 个工具）
│   └── monitor_server.py                   # 监控数据服务（端口 8004，2 个工具）
│
├── aiops-docs/                             # 运维知识库（12 个故障排查 SOP）
│   ├── cpu_high_usage.md                   # CPU 使用率过高
│   ├── memory_high_usage.md                # 内存使用率过高
│   ├── disk_high_usage.md                  # 磁盘使用率过高
│   ├── service_unavailable.md              # 服务不可用
│   ├── slow_response.md                    # 服务响应慢
│   ├── network_high_latency.md             # 网络延迟高
│   ├── api_error_rate_spike.md             # API 错误率飙升
│   ├── cache_avalanche.md                  # 缓存雪崩
│   ├── certificate_expiry.md               # 证书过期
│   ├── container_oom_killed.md             # 容器 OOM
│   ├── database_connection_pool_exhaustion.md # 数据库连接池耗尽
│   └── message_queue_backlog.md            # 消息队列积压
│
├── static/                                 # Web 前端（纯静态）
│   ├── index.html                          # 主页面（三种对话模式）
│   ├── app.js                              # 前端逻辑
│   └── styles.css                          # 样式表
│
├── Project Docs/                           # 架构设计与方案文档
│   ├── Architecture.md                     # 系统架构总览
│   ├── model_config.md                     # 模型配置说明
│   └── plans/                              # 12 份渐进式增强方案
│
├── logs/                                   # 日志目录（Loguru 自动创建）
├── uploads/                                # 上传文件临时目录
├── diagnosis_reports/                      # 诊断报告持久化目录（文件后端）
├── reports/                                # 评估报告输出目录
├── volumes/                                # Milvus 数据持久化目录
│
├── .env                                    # 环境变量配置
├── Makefile                                # 项目管理命令（Linux/macOS）
├── start-windows.bat                       # Windows 启动脚本
├── stop-windows.bat                        # Windows 停止脚本
├── vector-database.yml                     # Milvus Docker Compose 配置
├── pyproject.toml                          # 项目配置（依赖、版本、工具链）
└── README.md
```

## 配置说明

通过 `.env` 文件配置：

```bash
# ===== 阿里云 DashScope 配置（必填）=====
# 秘钥管理：https://bailian.console.aliyun.com/
DASHSCOPE_API_KEY=your-api-key
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-max
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4

# ===== Milvus 配置 =====
MILVUS_HOST=localhost
MILVUS_PORT=19530

# ===== RAG 检索模式 =====
# basic: Dense 向量检索（L2 距离）
# enhanced: 查询改写 + 双向量混合检索 + Cross-Encoder 精排
RAG_MODE=basic
RAG_TOP_K=3

# ===== Enhanced RAG 配置（RAG_MODE=enhanced 时生效）=====
# 查询预处理: none（透传） / rewrite（LLM 改写）
QUERY_PREPROCESSOR_TYPE=none
# 精排器: none（直接截断） / cross_encoder（BGE-Reranker）
RERANKER_TYPE=cross_encoder
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_TOP_K=3             # 精排后最终返回数
RERANK_COARSE_TOP_K=20        # 混合检索粗排候选数

# ===== 评估 Judge 配置（独立于线上模型）=====
EVAL_JUDGE_MODEL=qwen3.5-plus
EVAL_JUDGE_TEMPERATURE=0.0
EVAL_JUDGE_API_BASE=https://api.vveai.com/v1
EVAL_JUDGE_API_KEY=sk-your-key

# ===== 文档分块配置 =====
CHUNK_MAX_SIZE=800
CHUNK_OVERLAP=100

# ===== Redis 配置（可选，不配置则使用内存会话）=====
REDIS_URL=redis://localhost:6379

# ===== 上下文裁剪配置 =====
CONTEXT_MAX_TOKENS=8000
CONTEXT_TRIMMING_STRATEGY=token_count   # token_count / none

# ===== MCP 服务配置 =====
MCP_CLS_URL=http://localhost:8003/mcp
MCP_MONITOR_URL=http://localhost:8004/mcp
```

## RAG 检索系统

项目实现了可插拔的 RAG 检索架构，通过 `RAG_MODE` 配置可在两种模式间切换。

### Basic 模式（默认）

单阶段 Dense 向量检索：

```
用户查询 → Embedding → Milvus L2 ANN 检索 → Top-K 文档
```

- 向量维度：1024（text-embedding-v4）
- 相似度：L2 距离
- 集合：`biz`

### Enhanced 模式

三阶段流水线：

```
用户查询
  → [1] 查询预处理（rewrite / none）
  → [2] 双向量混合检索（Dense COSINE + Sparse BM25 → RRF 融合）
  → [3] Cross-Encoder 精排（bge-reranker-v2-m3 / none）
  → Top-K 文档
```

**阶段一：查询预处理**

| 模式 | 说明 |
|------|------|
| `none` | 原始查询直接透传 |
| `rewrite` | 使用 ChatQwen 对查询进行语义改写，补全上下文、消除歧义 |

**阶段二：双向量混合检索**

- **Dense 向量**：text-embedding-v4，COSINE 相似度，捕捉语义相似性
- **Sparse 向量**：Milvus 内置 BM25 Function（Jieba 中文分词），捕捉关键词匹配
- **融合方式**：RRF (Reciprocal Rank Fusion)，k=60
- 粗排候选数由 `RERANK_COARSE_TOP_K` 控制（默认 20）
- 集合：`biz_enhanced`

**阶段三：Cross-Encoder 精排**

- 模型：`BAAI/bge-reranker-v2-m3`
- 使用原始查询对每个候选文档逐一打分
- 按分数降序排列，截断至 `RERANKER_TOP_K`

**降级策略**：
- 查询预处理失败 → 回退原始查询
- 精排模型加载失败 → 回退截断模式
- 混合检索失败 → 抛出异常（不降级，避免静默正确性问题）

### 检索耗时日志

Enhanced 模式下每次检索输出结构化耗时（可通过日志查看）：

```
[EnhancedRAG] trace=abc123 耗时: preprocess=0.00s|hybrid_search=0.12s|rerank=0.35s|total=0.47s
```

## AIOps 智能运维

基于 **Plan-Execute-Replan** 模式实现自动故障诊断。

### 核心特性

- 自动制定诊断计划（Planner，结合 RAG 知识库经验）
- 智能工具调用（Executor，调用 MCP 日志/监控工具）
- 动态调整步骤（Replanner，continue/replan/respond 决策）
- 最多 8 步执行，至少 5 步后禁止重新规划
- SSE 流式输出诊断全过程
- 生成结构化诊断报告
- 诊断报告自动持久化（Redis 7 天 TTL 或文件 JSON）

### 诊断流程

```
1. Planner 制定计划 → 结合知识库生成 4-6 个诊断步骤
2. Executor 执行步骤 → 调用 MCP 工具（日志查询、监控数据）
3. Replanner 评估结果 → 决定继续执行 / 调整计划 / 生成最终报告
4. 输出诊断报告 → 根因分析 + 运维建议 + 证据链
```

### SSE 事件类型

| 事件类型 | 说明 |
|----------|------|
| `status` | 状态更新（获取告警、初始化等） |
| `plan` | 诊断计划制定完成，含步骤列表 |
| `step_complete` | 单个步骤执行完成 |
| `report` | 最终诊断报告（Markdown 格式） |
| `complete` | 诊断流程完成 |
| `error` | 错误信息 |

### 快速测试

```bash
# 访问 Web 界面，点击"智能运维与诊断工具"
# 或使用 API
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test"}' \
  --no-buffer

# 查询历史诊断报告
curl "http://localhost:9900/api/aiops/diagnosis/test?limit=10"
```

## 评估体系

### RAG 评估

基于 RAGAs 框架的两阶段评估：

**评估指标**：

| 阶段 | 指标 | 说明 |
|------|------|------|
| 检索质量 | `context_precision` | 检索结果中相关文档的精确率 |
| 检索质量 | `context_recall` | 相关文档被检索到的召回率 |
| 生成质量 | `faithfulness` | 生成答案对检索上下文的忠实度 |
| 生成质量 | `answer_relevancy` | 生成答案与问题的相关度 |

**评估数据集**（78 条，v1.1.2）：

| 分类 | 数量 | 说明 |
|------|------|------|
| `exact_keyword` | 35 | 精确关键词匹配 |
| `colloquial` | 16 | 口语化表达 |
| `cross_doc` | 15 | 跨文档综合查询 |
| `edge_case` | 12 | 边界和异常场景 |

**运行评估**：

```bash
# 检索质量评估（始终执行）
python tests/evaluation/evaluate_rag.py

# 含生成质量评估
python tests/evaluation/evaluate_rag.py --with-generation

# 消融实验（12 组参数组合，basic/enhanced 各 6 组）
python tests/evaluation/run_ablation.py

# dataset 质量检查
python tests/evaluation/validate_dataset.py

# 生成候选评估问题（LLM 辅助）
python tests/evaluation/generate_questions.py --strategy keyword_based --count 20
```

### Agent 评估

覆盖 6 类诊断场景（12 条测试数据，v1.0.0）：

| 场景 | 说明 |
|------|------|
| 单工具路径 | 调用单个 MCP 工具即可完成诊断 |
| 多工具联合排查 | 需组合多个工具交叉验证 |
| 跨文档知识 | 答案涉及多篇知识库文档 |
| 误报/噪声 | 正常波动不应引发告警 |
| 多步推理 | 需要多步推理链的复杂场景 |
| 模糊输入 | 信息不足，需主动追问 |

**评估指标**：

- **工具调用准确率**：Exact Match / Precision / Recall（纯集合运算，无 LLM 依赖）
- **目标达成率**：LLM Judge 0/1/2 评分（3 次取平均）

```bash
# Agent 评估
python tests/evaluation/evaluate_agent.py

# 跳过目标达成率评估（仅工具准确率）
python tests/evaluation/evaluate_agent.py --skip-goal
```

## Memory 与上下文管理

### 会话持久化

- 配置 `REDIS_URL` 后自动启用 RedisSaver，支持跨重启的会话持久化
- 未配置则使用 MemorySaver（进程内内存，重启丢失）
- RAG Agent 和 AIOps Agent 均支持两种模式自动切换

### 上下文裁剪

替代固定数量的消息保留策略，使用 token 数量精确控制上下文窗口：

- 使用 `tiktoken`（`cl100k_base` 编码）计算消息 token 数
- 默认上限 8000 tokens，超出时从旧到新裁剪
- 始终保留首条 SystemMessage（包含系统指令和工具定义）
- 通过 `CONTEXT_TRIMMING_STRATEGY` 可关闭（设为 `none`）

### 诊断报告持久化

- 诊断完成后自动保存：session_id、原始输入、诊断计划、执行步骤、最终报告
- Redis 后端：7 天 TTL 自动过期
- 文件后端：`diagnosis_reports/` 目录下 JSON 文件
- 提供 `GET /api/aiops/diagnosis/{session_id}` 查询接口

## 开发指南

### 常用命令

```bash
# 项目管理
make init              # 一键初始化（Docker + 服务 + 文档）
make start             # 启动所有服务
make stop              # 停止所有服务
make restart           # 重启所有服务
make dev               # 开发模式（uvicorn --reload，热重载）

# 依赖管理
make install-dev       # 安装开发依赖
make sync              # 同步依赖

# Docker 管理
make up                # 启动 Milvus 容器
make down              # 停止 Milvus 容器

# 知识库管理
make upload            # 上传 aiops-docs/ 下所有文档

# 代码质量
make format            # ruff format + isort
make lint              # ruff check
make fix               # ruff --fix
make type-check        # mypy

# 测试
make test              # pytest with coverage
make test-quick        # pytest without coverage

# 服务监控
make check             # 健康检查
make status-mcp        # MCP 服务状态
make logs              # 查看服务日志
```

## 常见问题

### Windows 环境问题

**`make` 命令不可用**：使用批处理脚本 `.\start-windows.bat` / `.\stop-windows.bat`

**PowerShell 执行策略限制**：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

**端口被占用**：
```powershell
netstat -ano | findstr :9900
taskkill /F /PID <PID>
```

### 通用问题

**API Key 错误**：
```bash
cat .env | grep DASHSCOPE_API_KEY    # Linux/macOS
type .env | findstr DASHSCOPE_API_KEY  # Windows
```

**Milvus 连接失败**：
```bash
docker ps | grep milvus
docker compose -f vector-database.yml restart
```

**服务无法启动**：

Linux/macOS:
```bash
tail -f logs/app_$(date +%Y-%m-%d).log
lsof -i :9900   # FastAPI
lsof -i :8003   # CLS MCP
lsof -i :8004   # Monitor MCP
```

Windows:
```powershell
Get-ChildItem logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 50
netstat -ano | findstr :9900
```

**Enhanced RAG 首次启动较慢**：Cross-Encoder 模型（`BAAI/bge-reranker-v2-m3`）首次加载需下载模型文件，约 1-2 分钟。如不需要精排，设置 `RERANKER_TYPE=none` 即可跳过。
