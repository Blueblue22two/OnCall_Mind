# SuperBizAgent 技术问答（Q&A）文档

> 基于项目实际代码、配置和文档的深度技术分析
>
> 文档版本：v1.0 | 分析日期：2026-05-30

---

## 目录

- [Q1: Multi-Agent 选型依据](#q1-multi-agent-选型依据)
- [Q2: Agent 之间的交互与数据传递](#q2-agent-之间的交互与数据传递)
- [Q3: ReAct 范式及其在项目中的体现](#q3-react-范式及其在项目中的体现)
- [Q4: 为什么用 LangGraph 而不是简单的 LangChain](#q4-为什么用-langgraph-而不是简单的-langchain)
- [Q5: Memory 管理机制](#q5-memory-管理机制)
- [Q6: RAG 和 Memory 的本质区别](#q6-rag-和-memory-的本质区别)
- [Q7: Memory 持久化方案](#q7-memory-持久化方案)
- [Q8: Context Engineering 在该项目中的体现](#q8-context-engineering-在该项目中的体现)
- [Q9: 多轮对话上下文窗口处理机制](#q9-多轮对话上下文窗口处理机制)
- [Q10: 上下文压缩的时机和方法](#q10-上下文压缩的时机和方法)
- [Q11: State 管理与 Checkpoint 机制](#q11-state-管理与-checkpoint-机制)
- [Q12: Multi-Agent 与单 Agent 评估体系](#q12-multi-agent-与单-agent-评估体系)
- [Q13: Agent System 线上观测指标](#q13-agent-system-线上观测指标)
- [Q14: 如何约束 LLM 幻觉问题](#q14-如何约束-llm-幻觉问题)
- [Q15: MCP、Skill 和 Tools 的定义与区别](#q15-mcpskill-和-tools-的定义与区别)
- [Q16: 项目中 MCP 的使用与加载机制](#q16-项目中-mcp-的使用与加载机制)
- [Q17: 项目中 Tools 的使用](#q17-项目中-tools-的使用)
- [Q18: Skill 的运作机制](#q18-skill-的运作机制)

---

## Q1: Multi-Agent 选型依据

### Facts（项目事实）

项目中实现了 **两种独立的 Agent 架构**，分别服务于不同的业务场景：

**Agent 1：RAG Agent（ReAct 模式）**

```
File: app/services/rag_agent_service.py
Class: RagAgentService
```

- 使用 `langchain.agents.create_agent()` 创建 ReAct Agent
- 绑定工具列表：`[retrieve_knowledge, get_current_time]` + 动态加载的 MCP 工具
- 使用 `MemorySaver` 作为 checkpointer，支持多轮对话

```python
# File: app/services/rag_agent_service.py
# Class: RagAgentService
# Method: _initialize_agent()
self.agent = create_agent(
    self.model,
    tools=all_tools,
    checkpointer=self.checkpointer,
)
```

**Agent 2：AIOps Agent（Plan-Execute-Replan 模式）**

```
File: app/services/aiops_service.py
Class: AIOpsService
```

- 使用 LangGraph `StateGraph` 手动构建三节点工作流
- 节点：`planner` → `executor` → `replanner`（条件边循环）
- 每个节点是独立的异步函数，有独立的 Prompt 和 LLM 实例

```python
# File: app/services/aiops_service.py
# Class: AIOpsService
# Method: _build_graph()
workflow = StateGraph(PlanExecuteState)
workflow.add_node(NODE_PLANNER, planner)
workflow.add_node(NODE_EXECUTOR, executor)
workflow.add_node(NODE_REPLANNER, replanner)
```

**选型依据总结表：**

| 维度 | RAG Agent | AIOps Agent |
|------|-----------|-------------|
| 场景 | 对话式问答、知识检索 | 自动化故障诊断 |
| 架构 | ReAct（Reasoning + Acting 循环） | Plan-Execute-Replan（计划驱动） |
| 复杂度 | 单步推理，工具调用后直接生成回答 | 多步规划，需制定→执行→评估→调整 |
| 工具调用 | LLM 自主决定何时调用哪个工具 | Planner 规划工具使用，Executor 执行 |
| 状态管理 | 简单的消息列表 | 结构化状态（plan, past_steps, response） |

### Analysis（分析）

选型依据的核心逻辑：

1. **任务复杂度决定架构**：RAG 问答是"一问一答"模式，ReAct 的 Think-Act-Observe 循环足够；AIOps 诊断是"多步推理"任务，需要先规划后执行再评估，Plan-Execute-Replan 更合适。

2. **确定性 vs 灵活性**：RAG Agent 需要灵活地根据用户问题动态选择工具（可能不调工具、可能调一个、可能调多个）；AIOps Agent 需要更高的确定性——先制定诊断计划再逐步执行，避免 LLM 在复杂场景中"跳步"或"遗漏"。

3. **可观测性需求**：AIOps 诊断需要向前端推送诊断进度（规划完成→步骤1完成→步骤2完成→报告生成），Plan-Execute-Replan 的节点化设计天然支持 SSE 事件流式推送。

### Improvements（优化建议）

1. 当前两个 Agent 完全独立，未来可考虑引入 **Supervisor Agent** 做路由——用户输入先经过分类器，自动判断走 RAG 还是 AIOps 路径。
2. AIOps Agent 的 Planner 节点已经调用了 `retrieve_knowledge` 工具来获取知识库经验，但 RAG Agent 和 AIOps Agent 之间没有直接的数据交互，可以考虑共享知识库检索结果以减少重复调用。

---

## Q2: Agent 之间的交互与数据传递

### Facts（项目事实）

项目中两个 Agent（RAG Agent 和 AIOps Agent）是 **完全独立运行** 的，不存在 Agent 之间的直接交互或数据传递。它们通过以下间接方式共享基础设施：

**1. 共享 MCP 客户端（全局单例）**

```python
# File: app/agent/mcp_client.py
# 全局单例
_mcp_client: Optional[MultiServerMCPClient] = None

async def get_mcp_client(...) -> MultiServerMCPClient:
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = _create_mcp_client(servers, tool_interceptors)
    return _mcp_client
```

两个 Agent 共享同一个 `MultiServerMCPClient` 实例，通过它访问 CLS 日志服务和监控服务。

**2. 共享知识库（Milvus 向量库）**

- RAG Agent 通过 `retrieve_knowledge` 工具查询 Milvus
- AIOps Agent 的 Planner 节点也通过 `retrieve_knowledge` 工具查询同一 Milvus 实例

```python
# File: app/agent/aiops/planner.py
# Method: planner()
experience_docs = await retrieve_knowledge.ainvoke({"query": input_text})
```

**3. 共享配置（全局 Settings 单例）**

```python
# File: app/config.py
config = Settings()  # 全局配置单例
```

**4. AIOps 内部节点间的数据传递**

AIOps Agent 内部的三个节点通过 `PlanExecuteState`（TypedDict）传递数据：

```python
# File: app/agent/aiops/state.py
class PlanExecuteState(TypedDict, total=False):
    input: str                              # 用户输入
    plan: List[str]                         # 待执行步骤
    past_steps: Annotated[List[tuple], operator.add]  # 已执行步骤（追加模式）
    response: str                           # 最终响应
```

数据流向：

```
Planner → {"plan": ["步骤1", "步骤2", ...]}
    ↓
Executor → {"plan": plan[1:], "past_steps": [(task, result)]}
    ↓
Replanner → {"response": "..."} 或 {"plan": new_steps} 或 {}（继续）
```

关键点：`past_steps` 使用 `Annotated[List[tuple], operator.add]`，这意味着每次 Executor 返回的 `past_steps` 会被 **追加** 而非覆盖。

### Analysis（分析）

项目采用的是 **独立 Agent + 共享基础设施** 的模式，而非 Agent-to-Agent 通信模式。这种设计的优缺点：

**优点：**
- 简单可靠，无跨 Agent 通信的复杂性
- 两个 Agent 可以独立部署、独立扩缩容
- 不存在 Agent 间的状态竞争

**局限：**
- 无法实现 Agent 间的协作推理（如 RAG Agent 为 AIOps Agent 提供知识上下文）
- 没有统一的调度层来分配任务

### Improvements（优化建议）

1. 引入 **消息总线**（如 Redis Pub/Sub 或 Kafka）实现 Agent 间的松耦合通信
2. 考虑引入 **Orchestrator Agent** 做任务路由和多 Agent 协作编排
3. AIOps Agent 的诊断结果可以回写到 RAG 知识库，形成"经验积累"闭环

---

## Q3: ReAct 范式及其在项目中的体现

### Facts（项目事实）

**ReAct（Reasoning + Acting）范式定义：**

ReAct 是一种 Agent 架构范式，核心是将 LLM 的推理（Reasoning）与工具调用（Acting）交替进行，形成 Think → Act → Observe 的循环，直到 LLM 认为信息充足并生成最终回答。

**项目中的 ReAct 实现：**

RAG Agent 使用 `create_agent()` 创建的就是标准的 ReAct Agent：

```python
# File: app/services/rag_agent_service.py
# Method: _initialize_agent()
self.agent = create_agent(
    self.model,
    tools=all_tools,
    checkpointer=self.checkpointer,
)
```

ReAct 循环的执行流程：

```
用户问题 → LLM 推理（我应该调用什么工具？）
    → 工具调用（retrieve_knowledge / get_current_time / MCP 工具）
    → 观察工具返回结果
    → LLM 再次推理（信息是否充足？）
        → 不充足 → 继续调用工具
        → 充足 → 生成最终回答
```

**与 Workflow（工作流）的区别在项目中的体现：**

| 维度 | ReAct（RAG Agent） | Workflow（AIOps Agent） |
|------|---------------------|------------------------|
| 控制流 | LLM 自主决定（动态） | 代码定义的图（确定性） |
| 循环条件 | LLM 判断信息是否充足 | `should_continue()` 函数判断 state |
| 工具选择 | LLM 在每轮推理中自主选择 | Planner 规划，Executor 执行 |
| 入口 | `create_agent()` 一行创建 | `StateGraph` 手动构建节点和边 |
| 代码位置 | `rag_agent_service.py` | `aiops_service.py` + `app/agent/aiops/` |

**Workflow 在项目中的实现：**

```python
# File: app/services/aiops_service.py
# Method: _build_graph()
workflow.set_entry_point(NODE_PLANNER)
workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)
workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)
workflow.add_conditional_edges(
    NODE_REPLANNER,
    should_continue,
    {NODE_EXECUTOR: NODE_EXECUTOR, END: END}
)
```

### Analysis（分析）

**ReAct 适合 RAG Agent 的原因：**
- 用户问题多样化，无法预定义执行流程
- 有些问题不需要工具（如"你好"），有些需要一次工具调用，有些需要多次
- LLM 自主决策更灵活

**Workflow 适合 AIOps Agent 的原因：**
- 故障诊断有明确的流程：先规划→再执行→再评估
- 需要控制最大步骤数（`MAX_STEPS = 8`）防止无限循环
- 需要向前端推送结构化进度事件
- 需要强制约束（如"已执行 ≥ 5 步禁止 replan"）

### Improvements（优化建议）

1. RAG Agent 可以考虑加入 **Adaptive Workflow**：对于简单问题直接 ReAct，对于复杂问题（如需要多工具联合查询）切换到计划模式
2. AIOps Agent 的 Replanner 中硬编码了 `MAX_STEPS = 8`，可以根据任务复杂度动态调整

---

## Q4: 为什么用 LangGraph 而不是简单的 LangChain

### Facts（项目事实）

项目中 LangGraph 的使用体现在两个关键场景：

**场景 1：RAG Agent 使用 LangGraph 的 Checkpointer 机制**

```python
# File: app/services/rag_agent_service.py
from langgraph.checkpoint.memory import MemorySaver

self.checkpointer = MemorySaver()
self.agent = create_agent(
    self.model,
    tools=all_tools,
    checkpointer=self.checkpointer,
)
```

LangGraph 的 `MemorySaver` 提供了基于 `thread_id` 的会话隔离和状态持久化，这是纯 LangChain 不具备的。

**场景 2：AIOps Agent 使用 LangGraph 的 StateGraph 构建工作流**

```python
# File: app/services/aiops_service.py
from langgraph.graph import StateGraph, END

workflow = StateGraph(PlanExecuteState)
workflow.add_node(NODE_PLANNER, planner)
workflow.add_node(NODE_EXECUTOR, executor)
workflow.add_node(NODE_REPLANNER, replanner)
workflow.set_entry_point(NODE_PLANNER)
workflow.add_conditional_edges(NODE_REPLANNER, should_continue, {...})
compiled_graph = workflow.compile(checkpointer=self.checkpointer)
```

**LangGraph 提供的关键能力（LangChain 不具备）：**

| 能力 | LangGraph | 纯 LangChain |
|------|-----------|--------------|
| 状态图定义 | `StateGraph` + 节点 + 边 + 条件边 | ❌ 不支持 |
| 结构化状态管理 | `TypedDict` + `Annotated` reducer | 仅 `AgentState` |
| Checkpoint 持久化 | `MemorySaver` / `RedisSaver` | ❌ 无内置支持 |
| 流式输出模式 | `stream_mode="updates"` / `"messages"` | 仅 token 流 |
| 循环与条件分支 | `add_conditional_edges` | ❌ 仅线性 Chain |
| 节点级可观测性 | 每个节点独立日志和事件 | 不透明 |

**具体代码证据 —— 条件边（LangGraph 独有）：**

```python
# File: app/services/aiops_service.py
def should_continue(state: PlanExecuteState) -> str:
    if state.get("response"):
        return END
    plan = state.get("plan", [])
    if plan:
        return NODE_EXECUTOR
    return END

workflow.add_conditional_edges(
    NODE_REPLANNER,
    should_continue,
    {NODE_EXECUTOR: NODE_EXECUTOR, END: END}
)
```

这个条件边实现了"如果已生成最终响应则结束，否则继续执行下一步"的逻辑，这在纯 LangChain 中需要手写复杂的循环控制。

### Analysis（分析）

LangGraph 相比 LangChain 的核心优势在于 **状态图编程模型**：

1. **循环支持**：LangChain 的 Chain 是线性的（A → B → C），无法原生实现"执行→评估→再执行"的循环；LangGraph 通过条件边天然支持。

2. **状态持久化**：LangGraph 的 Checkpointer 机制可以在任意节点"快照"状态，支持会话恢复、时间旅行调试；LangChain 没有此能力。

3. **可组合性**：LangGraph 的节点是普通函数，可以自由组合、替换、测试；LangChain 的 Chain 嵌套过深时可读性和可测试性下降。

4. **流式粒度**：LangGraph 支持 `stream_mode="updates"`（按节点输出流式），可以精确知道当前执行到哪个节点；LangChain 仅支持 token 级流式。

### Improvements（优化建议）

1. 当前 AIOps Agent 没有利用 LangGraph 的 **Human-in-the-Loop**（HITL）能力，虽然 `PlanExecuteState` 中已预留了 `pending_approval`、`pending_tool_name` 等字段，但尚未实现。建议在危险操作（如重启服务、修改配置）前加入 HITL 审批。
2. 可以利用 LangGraph 的 `interrupt_before` / `interrupt_after` 功能实现断点续执行。

---

## Q5: Memory 管理机制

### Facts（项目事实）

项目中的 Memory 管理分为 **短期记忆（对话历史）** 和 **持久化存储** 两个层面，但 **未实现长期记忆（跨会话知识积累）**。

**1. 短期记忆 —— 会话级对话历史**

由 LangGraph 的 Checkpointer 管理，以 `thread_id`（即 `session_id`）为键隔离不同会话：

```python
# File: app/services/rag_agent_service.py
config_dict = {
    "configurable": {
        "thread_id": session_id
    }
}
result = await self.agent.ainvoke(
    input=agent_input,
    config=config_dict,
)
```

Checkpointer 后端选择：

```python
# File: app/services/rag_agent_service.py
if config.redis_url:
    from langgraph.checkpoint.redis import RedisSaver
    self.checkpointer = RedisSaver.from_conn_string(config.redis_url)
else:
    self.checkpointer = MemorySaver()
```

**2. 上下文窗口管理 —— Token 级裁剪**

```python
# File: app/services/rag_agent_service.py
# Function: trim_messages_by_tokens()
def trim_messages_by_tokens(
    messages: Sequence[BaseMessage],
    max_tokens: int = 8000,
    model_encoding: str = "cl100k_base",
) -> list[BaseMessage]:
    enc = tiktoken.get_encoding(model_encoding)
    first_msg = messages[0]  # 始终保留首条 SystemMessage
    first_tokens = len(enc.encode(str(first_msg.content or "")))
    kept: list[BaseMessage] = []
    remaining = max_tokens - first_tokens
    for msg in reversed(messages[1:]):
        msg_tokens = len(enc.encode(str(msg.content or "")))
        if remaining - msg_tokens < 0:
            break
        kept.insert(0, msg)
        remaining -= msg_tokens
    return [first_msg] + kept
```

配置参数：

```python
# File: app/config.py
context_max_tokens: int = 8000
context_trimming_strategy: Literal["token_count", "none"] = "token_count"
```

**3. 会话历史查询与清除**

```python
# File: app/services/rag_agent_service.py
# Method: get_session_history() — 从 Checkpointer 中读取消息历史
# Method: clear_session() — 调用 checkpointer.delete_thread()
```

**4. 诊断报告持久化（长期存储，但非"记忆"）**

```python
# File: app/services/diagnosis_store.py
class DiagnosisStore:
    # Redis 后端：7 天 TTL 自动过期
    self._redis_client.setex(record_id, 86400 * 7, json.dumps(record, ...))
    # 文件后端：JSON 文件永久保存
```

### Analysis（分析）

**短期记忆设计：**
- 短期记忆 = Checkpointer 中的消息历史 + token 裁剪
- 以 `session_id` 为粒度隔离，同一用户的不同会话互不影响
- 裁剪策略：保留首条 SystemMessage + 从最新到最旧填充到 8000 token 上限

**长期记忆：项目中未发现相关实现**

项目没有跨会话的长期记忆机制：
- 没有用户画像存储
- 没有跨会话的知识积累或经验学习
- 没有向量化的对话历史检索
- `DiagnosisStore` 虽然持久化了诊断报告，但仅用于历史查询，不参与后续推理

**Memory 分工总结：**

| 层级 | 实现 | 生命周期 | 用途 |
|------|------|----------|------|
| 短期记忆 | MemorySaver / RedisSaver | 会话内 | 多轮对话上下文 |
| 上下文窗口 | tiktoken 裁剪 | 单次请求 | 防止超 token 限制 |
| 诊断报告 | DiagnosisStore | 7天（Redis）/ 永久（文件） | 历史查询 |
| 知识库 | Milvus 向量库 | 永久 | RAG 检索 |
| 长期记忆 | ❌ 未实现 | — | — |

### Improvements（优化建议）

1. **引入长期记忆层**：将用户偏好、历史诊断经验、常见故障模式等持久化到独立的长期记忆存储中
2. **对话摘要压缩**：当对话历史过长时，使用 LLM 生成摘要替代早期消息，而非简单裁剪
3. **向量化的长期记忆**：将历史对话和诊断结果向量化后存入 Milvus，后续可以通过语义检索召回相关经验
4. **Reflection 机制**：让 Agent 在对话结束后"反思"本次交互，提取关键信息存入长期记忆

---

## Q6: RAG 和 Memory 的本质区别

### Facts（项目事实）

**RAG（Retrieval-Augmented Generation）在项目中的实现：**

```
File: app/retriever/basic.py → BasicRAGRetriever
File: app/retriever/enhanced.py → EnhancedRAGRetriever
File: app/tools/knowledge_tool.py → retrieve_knowledge
```

RAG 的核心数据流：

```
用户查询 → Embedding（text-embedding-v4, 1024维）→ Milvus ANN 检索 → Top-K 文档 → 注入 Prompt → LLM 生成回答
```

RAG 存储的是 **外部知识**（12 篇运维 SOP 文档）：

```
aiops-docs/
├── cpu_high_usage.md          # CPU 使用率过高排查
├── memory_high_usage.md       # 内存使用率过高排查
├── service_unavailable.md     # 服务不可用排查
├── cache_avalanche.md         # 缓存雪崩处理
└── ...（共 12 篇）
```

**Memory 在项目中的实现：**

```
File: app/services/rag_agent_service.py → MemorySaver / RedisSaver
```

Memory 存储的是 **对话历史**（用户和 Agent 之间的交互消息）。

### Analysis（分析）

**本质区别总结：**

| 维度 | RAG | Memory |
|------|-----|--------|
| **数据性质** | 外部知识（静态文档） | 交互历史（动态生成） |
| **数据来源** | 人工编写的运维 SOP 文档 | Agent 与用户的对话记录 |
| **写入时机** | 离线阶段（文档上传时） | 在线阶段（每次对话时） |
| **检索方式** | 语义相似度检索（向量 ANN） | 按 session_id 精确查找 |
| **更新频率** | 低频（知识库更新时重新 upload） | 高频（每轮对话都追加） |
| **生命周期** | 永久（除非手动删除） | 会话内 / 配置 TTL |
| **服务对象** | 所有用户共享同一知识库 | 按 session_id 隔离 |
| **在项目中的位置** | `app/retriever/` + `app/tools/knowledge_tool.py` | `MemorySaver` / `RedisSaver` |
| **技术实现** | Milvus 向量数据库 + Embedding | LangGraph Checkpointer |

**核心本质：**
- **RAG = 外部知识的检索增强**：解决 LLM 知识截止和领域知识不足的问题，是"给 LLM 提供参考书"
- **Memory = 交互历史的保持**：解决多轮对话中上下文丢失的问题，是"让 LLM 记住之前说了什么"

**一个类比：**
- RAG 像是考试时可以翻开的"教科书"
- Memory 像是考试时你脑海中记住的"之前做过的题目"

### Improvements（优化建议）

1. 项目当前 RAG 和 Memory 是完全独立的系统，可以考虑 **Memory-augmented RAG**：将用户历史对话中的关键信息也向量化存入 RAG 系统，实现"个性化检索"
2. 引入 **用户级 Memory**：记录每个用户的偏好、常用服务、历史故障模式等

---

## Q7: Memory 持久化方案

### Facts（项目事实）

项目中 Memory 持久化支持 **两种后端**，通过配置自动切换：

**方案 1：MemorySaver（进程内存，默认）**

```python
# File: app/services/rag_agent_service.py
from langgraph.checkpoint.memory import MemorySaver
self.checkpointer = MemorySaver()
```

- 数据存储在 Python 进程的内存中
- 进程重启后数据丢失
- 适用于开发和测试环境

**方案 2：RedisSaver（Redis 持久化，可选）**

```python
# File: app/services/rag_agent_service.py
if config.redis_url:
    from langgraph.checkpoint.redis import RedisSaver
    self.checkpointer = RedisSaver.from_conn_string(config.redis_url)
```

配置方式：

```bash
# File: .env
REDIS_URL=redis://localhost:6379
```

- 数据持久化到 Redis，进程重启后可恢复
- RAG Agent 和 AIOps Agent 均支持

**诊断报告的持久化（独立于对话 Memory）：**

```python
# File: app/services/diagnosis_store.py
class DiagnosisStore:
    def __init__(self):
        if config.redis_url:
            self._redis_client = redis_lib.from_url(config.redis_url)
        else:
            self._file_dir = Path("diagnosis_reports")
            self._file_dir.mkdir(exist_ok=True)

    def save(self, session_id, input_data, plan, past_steps, response):
        record_id = f"diagnosis:{session_id}:{int(time.time())}"
        if self._redis_client:
            self._redis_client.setex(record_id, 86400 * 7, json.dumps(record, ...))
        else:
            file_path = self._file_dir / f"{record_id.replace(':', '_')}.json"
            file_path.write_text(json.dumps(record, ...))
```

- Redis 后端：7 天 TTL 自动过期
- 文件后端：JSON 文件存储在 `diagnosis_reports/` 目录

### Analysis（分析）

**持久化架构层次：**

```
┌─────────────────────────────────────────────┐
│  对话 Memory（Checkpointer）                │
│  ├── MemorySaver（内存，重启丢失）           │
│  └── RedisSaver（Redis，持久化）             │
├─────────────────────────────────────────────┤
│  诊断报告（DiagnosisStore）                  │
│  ├── Redis（7 天 TTL）                      │
│  └── 文件 JSON（永久）                       │
├─────────────────────────────────────────────┤
│  知识库（Milvus 向量库）                     │
│  └── Docker Volume 持久化                    │
└─────────────────────────────────────────────┘
```

**当前方案的局限性：**
- `MemorySaver` 不支持多进程共享（多 worker 部署时会话不互通）
- Redis 单点故障时没有降级策略
- 诊断报告没有索引，按 session_id 查询时需要 `SCAN` 遍历

### Improvements（优化建议）

1. **引入 PostgreSQL Checkpointer**：LangGraph 官方支持 `PostgresSaver`，提供更强的一致性和查询能力
2. **会话归档**：对话结束后将历史压缩归档到冷存储（如 S3），支持后续审计和分析
3. **多租户隔离**：当前 Redis 中的 key 按 `session_id` 隔离，但没有租户前缀，建议加入 `tenant_id` 维度

---

## Q8: Context Engineering 在该项目中的体现

### Facts（项目事实）

Context Engineering（上下文工程）是指系统性地设计和管理输入给 LLM 的所有上下文信息，以优化模型输出质量。在该项目中，Context Engineering 体现在以下多个层面：

**1. System Prompt 设计**

RAG Agent 的系统提示词：

```python
# File: app/services/rag_agent_service.py
# Method: _build_system_prompt()
return dedent("""
    你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

    工作原则:
    1. 理解用户需求，选择合适的工具来完成任务
    2. 当需要获取实时信息或专业知识时，主动使用相关工具
    3. 基于工具返回的结果提供准确、专业的回答
    4. 如果工具无法提供足够信息，请诚实地告知用户

    回答要求:
    - 保持友好、专业的语气
    - 回答简洁明了，重点突出
    - 基于事实，不编造信息
    - 如有不确定的地方，明确说明
""").strip()
```

**2. AIOps Planner Prompt（含经验上下文注入）**

```python
# File: app/agent/aiops/planner.py
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", dedent("""
        作为一个专家级别的规划者，你需要将复杂的任务分解为可执行的步骤。

        可用工具列表（用于制定计划时参考）：
        {tools_description}

        {experience_context}

        对于给定的任务，请创建一个简单的、逐步的计划...
    """)),
])

# 知识库经验注入到上下文中
if experience_docs:
    experience_context = dedent(f"""
        ## 相关经验文档
        以下是从知识库中检索到的相关经验和最佳实践：
        {experience_docs}
    """)
```

**3. Replanner Prompt（决策优先级约束）**

```python
# File: app/agent/aiops/replanner.py
replanner_prompt = ChatPromptTemplate.from_messages([
    ("system", dedent("""
        你有三个选择（按优先级排序）：

        **1. 'respond' - 信息充足，立即生成最终响应** 【最高优先级】
        **2. 'continue' - 当前计划合理，继续执行** 【次优先级】
        **3. 'replan' - 当前计划有严重问题** 【最低优先级，谨慎使用】

        **决策优先级口诀：**
        "优先结束 > 保持不变 > 调整计划"
        "信息足够就响应，不要追求完美"
    """)),
])
```

**4. Executor Prompt（执行约束）**

```python
# File: app/agent/aiops/executor.py
SystemMessage(content="""你是一个能力强大的助手，负责执行具体的任务步骤。
注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务""")
```

**5. RAG 上下文格式化**

```python
# File: app/tools/knowledge_tool.py
# Function: format_docs()
formatted = f"【参考资料 {i}】"
formatted += f"\n标题: {header_str}"
formatted += f"\n来源: {source}"
formatted += f"\n内容:\n{doc.page_content}\n"
```

**6. 工具描述注入上下文**

```python
# File: app/agent/aiops/utils.py
def format_tools_description(tools: List) -> str:
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(tool_descriptions)
```

**7. AIOps 诊断任务的输出格式约束**

```python
# File: app/services/aiops_service.py
# Method: diagnose()
aiops_task = dedent("""诊断当前系统是否存在告警...
    诊断报告输出格式要求：
    ```
    # 告警分析报告
    ## 📋 活跃告警清单
    | 告警名称 | 级别 | 目标服务 | ... |
    ## 🔍 告警根因分析1 - [告警名称]
    ...
    ```
    **重要提醒**：
    - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
    - 所有内容必须基于工具查询的真实数据，严禁编造
""")
```

### Analysis（分析）

项目的 Context Engineering 包含以下层次：

| 层次 | 实现 | 目的 |
|------|------|------|
| **角色设定** | System Prompt | 定义 Agent 身份和行为准则 |
| **工具描述注入** | `format_tools_description()` | 让 LLM 了解可用工具 |
| **经验知识注入** | Planner 中的 `experience_context` | 利用历史经验指导规划 |
| **决策约束** | Replanner 的优先级口诀 | 控制 LLM 决策倾向 |
| **输出格式约束** | Structured Output（Pydantic） | 强制输出格式 |
| **反幻觉约束** | "不要编造"、"基于事实" | 减少幻觉 |
| **上下文裁剪** | `trim_messages_by_tokens()` | 控制上下文窗口大小 |
| **RAG 上下文格式化** | `format_docs()` | 结构化参考资料 |

### Improvements（优化建议）

1. **动态 Prompt 组装**：根据用户问题类型动态调整 System Prompt，而非使用固定模板
2. **Few-shot 示例注入**：在 Prompt 中加入高质量的诊断示例，引导 LLM 输出质量
3. **上下文预算管理**：为 System Prompt、工具描述、RAG 结果、对话历史分别分配 token 预算

---

## Q9: 多轮对话上下文窗口处理机制

### Facts（项目事实）

项目实现了基于 **Token 计数的上下文裁剪机制** 来处理多轮对话中的上下文窗口问题：

**核心函数：**

```python
# File: app/services/rag_agent_service.py
# Function: trim_messages_by_tokens()
def trim_messages_by_tokens(
    messages: Sequence[BaseMessage],
    max_tokens: int = 8000,
    model_encoding: str = "cl100k_base",
) -> list[BaseMessage]:
    """按 token 数裁剪消息历史，保留首条 system message + 从新到旧裁剪到 max_tokens"""
    enc = tiktoken.get_encoding(model_encoding)

    # 始终保留首条消息（通常为 SystemMessage）
    first_msg = messages[0]
    first_tokens = len(enc.encode(str(first_msg.content or "")))

    kept: list[BaseMessage] = []
    remaining = max_tokens - first_tokens

    # 从最新到最旧遍历（跳过第一条）
    for msg in reversed(messages[1:]):
        content = str(msg.content or "")
        msg_tokens = len(enc.encode(content))
        if remaining - msg_tokens < 0:
            break
        kept.insert(0, msg)
        remaining -= msg_tokens

    result = [first_msg] + kept
    return result
```

**配置参数：**

```python
# File: app/config.py
context_max_tokens: int = 8000   # 上下文窗口 token 上限
context_trimming_strategy: Literal["token_count", "none"] = "token_count"
```

**调用时机（每次请求前）：**

```python
# File: app/services/rag_agent_service.py
# Methods: query(), query_stream(), query_with_trace()
if config.context_trimming_strategy == "token_count":
    messages = trim_messages_by_tokens(messages, max_tokens=config.context_max_tokens)
```

**裁剪策略：**

```
消息列表: [SystemMessage, Msg1, Msg2, Msg3, Msg4, Msg5, Msg6, Msg7, Msg8]

裁剪过程（max_tokens=8000）：
1. 始终保留 SystemMessage（假设 500 tokens），剩余预算 7500
2. 从最新到最旧遍历：Msg8(200t) → Msg7(300t) → Msg6(400t) → ...
3. 当累计超过 7500 tokens 时停止
4. 最终结果: [SystemMessage, Msg5, Msg6, Msg7, Msg8]
```

### Analysis（分析）

**设计亮点：**

1. **始终保留 SystemMessage**：系统指令是 Agent 行为的基础，裁剪时不会被丢弃
2. **从新到旧优先**：最新的对话对当前问题最相关，优先保留
3. **Token 级精度**：比"保留最近 N 条消息"更精确，避免短消息浪费配额或长消息超出限制
4. **可配置开关**：通过 `context_trimming_strategy` 可以完全关闭裁剪

**潜在问题：**

1. 使用 `cl100k_base` 编码（GPT-4 的 tokenizer），但项目使用的模型是 Qwen，两者的 token 化方式可能不同，导致 token 计数不完全准确
2. 裁剪是在构建消息列表之后、发送给 LLM 之前执行的，但 LangGraph 的 Checkpointer 中保存的是裁剪前的完整历史——这意味着每次请求都会从 Checkpointer 加载完整历史，然后裁剪，存在性能浪费
3. 裁剪粒度是"消息级"，一条很长的消息要么完全保留要么完全丢弃，不会部分截断

### Improvements（优化建议）

1. 使用 Qwen 模型的 tokenizer 替代 `cl100k_base`，提高 token 计数准确性
2. 引入 **消息摘要机制**：当早期消息被裁剪掉时，先用 LLM 生成摘要，将摘要作为上下文的一部分
3. 考虑 **滑动窗口 + 重要消息锚定**：除了 SystemMessage，工具调用结果等关键消息也应优先保留

---

## Q10: 上下文压缩的时机和方法

### Facts（项目事实）

项目中的上下文压缩体现在以下两个层面：

**1. 消息级压缩 —— Token 裁剪（请求前）**

```python
# File: app/services/rag_agent_service.py
# 时机：每次请求构建消息列表后、发送给 LLM 前
if config.context_trimming_strategy == "token_count":
    messages = trim_messages_by_tokens(messages, max_tokens=config.context_max_tokens)
```

方法：从最新到最旧遍历消息，超出 token 上限时丢弃最旧的消息。

**2. 步骤结果截断（Replanner 中）**

```python
# File: app/agent/aiops/replanner.py
# Method: replanner()
steps_summary = "\n".join([
    f"步骤: {step}\n结果: {result[:300]}..."  # 截断到 300 字符
    for step, result in past_steps
])
```

Replanner 在构建已执行步骤的摘要时，将每个步骤的结果截断到 300 字符，避免执行历史占用过多上下文。

**3. 诊断报告持久化时的截断**

```python
# File: app/services/diagnosis_store.py
# Method: save()
"past_steps": [[step, str(result)[:200]] for step, result in past_steps],  # 截断到 200 字符
"response": response[:5000],  # 响应截断到 5000 字符
```

### Analysis（分析）

**压缩时机总结：**

| 时机 | 方法 | 目的 |
|------|------|------|
| 请求构建时 | Token 级消息裁剪 | 控制发送给 LLM 的总 token 数 |
| Replanner 构建输入时 | 结果截断（300 字符） | 避免执行历史膨胀 |
| 持久化存储时 | 结果截断（200 字符）+ 响应截断（5000 字符） | 控制存储大小 |

**项目中未发现以下压缩方法：**

- ❌ LLM 摘要压缩（用 LLM 对早期对话生成摘要替代原文）
- ❌ 语义去重（合并相似内容）
- ❌ 渐进式压缩（随对话轮次增加逐步提高压缩比）

**当前方案的局限：**

项目的压缩策略相对简单——"丢弃最旧的消息"和"截断长文本"。这会导致：
1. 被丢弃的消息中的关键信息完全丢失
2. 截断是无差别的前 300/200/5000 字符，可能截断掉重要结论

### Improvements（优化建议）

1. **引入 LLM 摘要压缩**：当消息数量超过阈值时，对早期消息生成摘要
2. **智能截断**：使用 LLM 提取步骤结果的关键信息，而非简单的前 N 字符截断
3. **分层压缩策略**：
   - 最近 3 条消息：保留完整内容
   - 4-10 条前：压缩为摘要
   - 10 条前：仅保留工具调用名称和结论关键词

---

## Q11: State 管理与 Checkpoint 机制

### Facts（项目事实）

**1. RAG Agent 的状态管理**

RAG Agent 使用简单的 `AgentState`：

```python
# File: app/services/rag_agent_service.py
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

`add_messages` 是 LangGraph 内置的 reducer，每次新消息会被追加到列表末尾。

**2. AIOps Agent 的状态管理**

```python
# File: app/agent/aiops/state.py
class PlanExecuteState(TypedDict, total=False):
    input: str                                    # 用户输入（不可变）
    plan: List[str]                               # 待执行步骤（覆盖式更新）
    past_steps: Annotated[List[tuple], operator.add]  # 已执行步骤（追加式更新）
    response: str                                 # 最终响应（覆盖式更新）
    # HITL 预留字段
    pending_approval: bool
    pending_tool_name: str
    pending_tool_args: dict
```

关键设计：`past_steps` 使用 `Annotated[List[tuple], operator.add]`，这意味着：

```python
# Executor 返回 {"past_steps": [(task, result)]}
# 不会覆盖已有的 past_steps，而是追加到列表末尾
# 第 1 步后: [(step1, result1)]
# 第 2 步后: [(step1, result1), (step2, result2)]
```

而 `plan` 是覆盖式更新：

```python
# Executor 返回 {"plan": plan[1:]}  # 移除已执行的步骤
# 第 1 步后: ["步骤2", "步骤3", "步骤4"]
# 第 2 步后: ["步骤3", "步骤4"]
```

**3. Checkpoint 机制**

```python
# File: app/services/aiops_service.py
compiled_graph = workflow.compile(checkpointer=self.checkpointer)

# 每次执行时通过 thread_id 关联 checkpoint
config_dict = {
    "configurable": {
        "thread_id": session_id
    }
}
async for event in self.graph.astream(
    input=initial_state,
    config=config_dict,
    stream_mode="updates"
):
```

LangGraph 在每个节点执行完后自动创建 checkpoint，保存当前 state 快照。

**4. 关于状态竞争**

项目中 **不存在多 Agent 同时运行时的状态竞争问题**，原因如下：

- RAG Agent 和 AIOps Agent 使用 **独立的 checkpointer 实例**
- 每个 Agent 的 checkpointer 按 `thread_id` 隔离
- 当前是单进程部署（uvicorn），没有多 worker 并发
- 即使同一 Agent 处理多个请求，不同的 `session_id` 对应不同的 checkpoint 链

```python
# File: app/services/rag_agent_service.py
rag_agent_service = RagAgentService(streaming=True)  # 全局单例

# File: app/services/aiops_service.py
aiops_service = AIOpsService()  # 全局单例
```

### Analysis（分析）

**状态管理的设计模式：**

```
RAG Agent:
  ┌────────────────────────────────────────────────┐
  │ AgentState                                     │
  │   messages: [Sys, H1, A1, H2, A2, ...]        │
  │   reducer: add_messages（追加）                │
  └────────────────────────────────────────────────┘

AIOps Agent:
  ┌────────────────────────────────────────────────┐
  │ PlanExecuteState                               │
  │   input: str（不可变）                         │
  │   plan: List[str]（覆盖式，逐步减少）          │
  │   past_steps: List[tuple]（追加式，逐步增加）  │
  │   response: str（覆盖式，最终写入一次）         │
  └────────────────────────────────────────────────┘
```

**关于状态竞争的深入分析：**

如果未来需要多进程部署（如 uvicorn 多 worker），当前的 `MemorySaver` 方案会出现问题：
1. 不同 worker 进程各自持有独立的 `MemorySaver`，同一 `session_id` 的请求可能被路由到不同 worker，导致上下文不连贯
2. 解决方案：使用 `RedisSaver` 作为共享 checkpoint 后端

### Improvements（优化建议）

1. **引入分布式锁**：如果使用 RedisSaver，对同一 `thread_id` 的并发操作加锁
2. **状态版本控制**：为 state 添加版本号，防止旧状态覆盖新状态
3. **状态清理策略**：长时间不活跃的 session 应自动清理 checkpoint，避免内存/存储膨胀

---

## Q12: Multi-Agent 与单 Agent 评估体系

### Facts（项目事实）

项目实现了 **两套独立的评估体系**：

**1. RAG 评估（检索质量 + 生成质量）**

```
File: tests/evaluation/evaluate_rag.py
File: tests/evaluation/rag_testset.py（78 条测试数据，v1.1.2）
```

评估指标：

| 阶段 | 指标 | 说明 |
|------|------|------|
| 检索评估 | `context_precision` | 检索结果中相关文档的精确率 |
| 检索评估 | `context_recall` | 相关文档被检索到的召回率 |
| 生成评估 | `faithfulness` | 生成答案对检索上下文的忠实度 |
| 生成评估 | `answer_relevancy` | 生成答案与问题的相关度 |

评估数据集分类：

| 分类 | 数量 | 说明 |
|------|------|------|
| `exact_keyword` | 35 | 精确关键词匹配 |
| `colloquial` | 16 | 口语化表达 |
| `cross_doc` | 15 | 跨文档综合查询 |
| `edge_case` | 12 | 边界和异常场景 |

**2. Agent 评估（工具调用准确率 + 目标达成率）**

```
File: tests/evaluation/evaluate_agent.py
File: tests/evaluation/agent_testset.py（12 条测试数据，v1.0.0）
```

**工具调用准确率（Tool Call Accuracy）：**

```python
# File: tests/evaluation/metrics/tool_call_accuracy.py
def compute_tool_call_accuracy(actual_calls, expected_calls):
    actual_names = {c.get("name", "") for c in actual_calls}
    expected_names = {c.get("name", "") for c in expected_calls}
    exact_match = actual_names == expected_names
    precision = len(actual_names & expected_names) / len(actual_names)
    recall = len(actual_names & expected_names) / len(expected_names)
    return {"exact_match": exact_match, "precision": precision, "recall": recall}
```

纯集合运算，不依赖 LLM。

**目标达成率（Goal Accuracy）：**

```python
# File: tests/evaluation/metrics/goal_accuracy.py
# LLM Judge 0/1/2 评分，3 次取平均
SCORE: 0 | 1 | 2
  0: 未达成，诊断方向错误
  1: 部分达成，有遗漏或轻微错误
  2: 完全达成，覆盖所有期望要点
```

Judge 使用独立的模型配置：

```python
# File: app/config.py
eval_judge_model: str = "qwen3.5-plus"
eval_judge_temperature: float = 0.0
```

**3. 消融实验**

```
File: tests/evaluation/run_ablation.py
```

支持 12 组参数组合的消融实验（basic/enhanced 各 6 组）。

**4. 评估数据集覆盖的 6 类 Agent 场景**

```python
# File: tests/evaluation/agent_testset.py
# 场景 1: 单工具路径（3 条）
# 场景 2: 多工具联合排查（3 条）
# 场景 3: 跨文档知识综合（2 条）
# 场景 4: 误报/噪声输入（2 条）
# 场景 5: 多步推理（1 条）
# 场景 6: 模糊/信息不足输入（1 条）
```

### Analysis（分析）

**评估体系分层：**

```
┌────────────────────────────────────────────┐
│  Multi-Agent System 评估                   │
│  ├── RAG Agent 评估                        │
│  │   ├── 检索质量: precision + recall       │
│  │   └── 生成质量: faithfulness + relevancy │
│  └── AIOps Agent 评估                      │
│      ├── 工具调用: exact match + P + R      │
│      └── 目标达成: LLM Judge 0/1/2          │
└────────────────────────────────────────────┘
```

**当前评估体系的优缺点：**

优点：
- RAG 评估使用标准 RAGAs 框架，指标体系完整
- Agent 评估覆盖了 6 类典型场景
- 工具调用准确率不依赖 LLM，计算确定性强
- Judge 模型与线上模型解耦，评估可复现

缺点：
- **Multi-Agent 整体评估缺失**：没有评估两个 Agent 的协作效果（虽然它们当前独立运行）
- **端到端延迟评估缺失**：没有记录 Agent 的响应时间分布
- **测试数据量偏少**：Agent 评估仅 12 条，覆盖度不够
- **AIOps Agent 评估缺失**：当前 `evaluate_agent.py` 仅评估 RAG Agent（通过 `query_with_trace`），AIOps 的 Plan-Execute-Replan 流程没有独立的评估脚本

### Improvements（优化建议）

1. **新增 AIOps Agent 专项评估**：评估 Planner 生成的计划质量、Executor 的工具调用正确性、Replanner 的决策合理性
2. **引入端到端延迟指标**：记录 P50/P95/P99 响应时间
3. **扩大 Agent 评估数据集**：至少 50 条，覆盖更多边界场景
4. **引入 A/B 测试框架**：支持不同模型/Prompt 版本的对比评估

---

## Q13: Agent System 线上观测指标

### Facts（项目事实）

项目中已实现的观测能力：

**1. 结构化日志（Loguru）**

```python
# File: app/services/rag_agent_service.py
logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")
logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")
```

**2. RAG 检索耗时追踪（Enhanced 模式）**

```python
# File: app/retriever/enhanced.py
meta: Dict[str, Any] = {
    "trace_id": trace_id,
    "preprocessor_type": config.query_preprocessor_type,
    "reranker_type": config.reranker_type,
    "degraded_stage": None,
    "fallback_reason": None,
    "candidate_count": 0,
    "final_count": 0,
    "total_time_ms": 0,
}
meta["hybrid_search_time_ms"] = int((time.time() - t_stage2) * 1000)
meta["reranker_time_ms"] = int((time.time() - t_stage3) * 1000)
```

**3. MCP 工具调用日志**

```python
# File: app/agent/mcp_client.py
logger.info(f"调用 MCP 工具: {request.name} (服务器: {request.server_name}, 第 {attempt + 1}/{max_retries} 次尝试)")
logger.info(f"MCP 工具 {request.name} 调用成功")
```

**4. AIOps 节点级事件流**

```python
# File: app/services/aiops_service.py
logger.info(f"节点 '{node_name}' 输出事件")
logger.info(f"已生成最终响应，结束流程")
logger.info(f"继续执行，剩余 {len(plan)} 个步骤")
```

**5. 健康检查接口**

```
GET /api/health → 服务状态 + Milvus 连接检查
```

**项目中未实现的观测能力：**

- ❌ Prometheus 指标导出
- ❌ OpenTelemetry 链路追踪
- ❌ Grafana 仪表盘
- ❌ 告警规则

### Analysis（分析）

**Agent System 线上观测应记录的核心指标体系：**

| 类别 | 指标 | 说明 | 当前状态 |
|------|------|------|----------|
| **请求级** | Request Count | 请求总数（按 Agent 类型分） | ❌ 未实现 |
| | Latency P50/P95/P99 | 响应延迟分布 | ❌ 未实现 |
| | Error Rate | 错误率（按错误类型分） | ❌ 未实现 |
| **Agent 级** | Tool Call Count | 工具调用次数/频率 | ⚠️ 仅日志 |
| | Tool Call Success Rate | 工具调用成功率 | ⚠️ 仅日志 |
| | Tool Call Latency | 工具调用延迟 | ⚠️ 仅 MCP 重试日志 |
| | Token Usage | Token 消耗量（input/output） | ❌ 未实现 |
| | ReAct Loop Count | ReAct 循环次数 | ❌ 未实现 |
| **RAG 级** | Retrieval Latency | 检索延迟 | ✅ Enhanced 模式有 |
| | Retrieval Hit Rate | 检索命中率 | ❌ 未实现 |
| | Context Precision | 线上检索精确率 | ❌ 仅评估时 |
| | Reranker Score Distribution | 精排分数分布 | ❌ 未实现 |
| **AIOps 级** | Plan Step Count | 计划步骤数 | ⚠️ 仅日志 |
| | Replan Rate | 重新规划率 | ❌ 未实现 |
| | Diagnosis Duration | 诊断总耗时 | ⚠️ 仅日志 |
| **系统级** | LLM API Latency | 模型调用延迟 | ❌ 未实现 |
| | LLM API Error Rate | 模型调用错误率 | ❌ 未实现 |
| | Milvus Query Latency | 向量库查询延迟 | ❌ 未实现 |
| | MCP Server Availability | MCP 服务可用性 | ⚠️ 有 health check |
| **业务级** | User Satisfaction | 用户满意度 | ❌ 未实现 |
| | Diagnosis Report Quality | 诊断报告质量评分 | ❌ 未实现 |
| | Session Length | 会话长度（轮次数） | ❌ 未实现 |

### Improvements（优化建议）

1. **引入 OpenTelemetry**：为每个 Agent 节点添加 span，实现端到端链路追踪
2. **接入 Prometheus + Grafana**：导出指标到 Prometheus，构建 Grafana 仪表盘
3. **Token 计量**：记录每次 LLM 调用的 input/output token 数，用于成本监控
4. **告警规则**：设定 P95 延迟、错误率、工具调用失败率的告警阈值
5. **Trace ID 贯穿**：将 Enhanced RAG 的 `trace_id` 扩展到整个请求链路

---

## Q14: 如何约束 LLM 幻觉问题

### Facts（项目事实）

项目中通过 **多层防线** 约束 LLM 幻觉：

**防线 1：System Prompt 约束**

```python
# File: app/services/rag_agent_service.py
"基于事实，不编造信息"
"如有不确定的地方，明确说明"
"如果工具无法提供足够信息，请诚实地告知用户"
```

```python
# File: app/agent/aiops/executor.py
"不要编造数据，只返回实际获取的信息"
"如果工具调用失败，请说明失败原因"
```

```python
# File: app/services/aiops_service.py
"所有内容必须基于工具查询的真实数据，严禁编造"
"如果某个步骤失败，在结论中如实说明，不要跳过"
```

**防线 2：RAG 知识锚定**

通过 `retrieve_knowledge` 工具将真实文档注入上下文，LLM 基于检索到的文档生成回答，而非凭空编造：

```python
# File: app/tools/knowledge_tool.py
@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    docs = get_rag_retriever().retrieve(query, top_k=effective_top_k)
    context = format_docs(docs)
    return context, docs
```

**防线 3：Structured Output 约束**

AIOps 的 Planner、Replanner 和最终响应生成器都使用 Pydantic 模型强制输出格式：

```python
# File: app/agent/aiops/planner.py
class Plan(BaseModel):
    steps: List[str] = Field(description="...")

planner_chain = planner_prompt | llm.with_structured_output(Plan)
```

```python
# File: app/agent/aiops/replanner.py
class Act(BaseModel):
    action: str = Field(description="必须是 continue/replan/respond 之一")
    new_steps: List[str] = Field(default_factory=list)

class Response(BaseModel):
    response: str = Field(description="对用户的最终响应")
```

Structured Output 通过函数调用（Function Calling）约束 LLM 输出必须符合 JSON Schema，减少自由发挥的空间。

**防线 4：RAGAs 忠实度评估（离线）**

```python
# File: tests/evaluation/evaluate_rag.py
# Phase 2 - 生成评估：faithfulness
# faithfulness 衡量生成答案是否忠实于检索上下文，不编造上下文中不存在的信息
```

**防线 5：MCP 工具调用重试（防止因工具失败导致的信息缺失）**

```python
# File: app/agent/mcp_client.py
async def retry_interceptor(request, handler, max_retries=3, delay=1.0):
    for attempt in range(max_retries):
        try:
            result = await handler(request)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
    # 所有重试都失败，返回错误结果而不是抛出异常
    return CallToolResult(
        content=[TextContent(type="text", text=error_msg)],
        isError=True
    )
```

工具调用失败时，返回包含错误信息的 `CallToolResult` 而非抛出异常，LLM 可以看到错误信息并据此决定如何处理，而不是在信息缺失的情况下编造答案。

**防线 6：Replanner 的"如实说明"约束**

```python
# File: app/agent/aiops/replanner.py
response_prompt = ChatPromptTemplate.from_messages([
    ("system", dedent("""
        根据原始任务和已执行步骤的结果，生成一个全面的最终响应。
        响应要求：
        - 基于实际数据，不要编造
        - 如果某些步骤失败，要诚实说明
    """)),
])
```

### Analysis（分析）

**幻觉防护层次图：**

```
┌─────────────────────────────────────────────┐
│  Prompt 层: "不要编造"、"基于事实"          │
├─────────────────────────────────────────────┤
│  知识层: RAG 注入真实文档作为上下文锚点      │
├─────────────────────────────────────────────┤
│  格式层: Structured Output 约束输出格式      │
├─────────────────────────────────────────────┤
│  工具层: 强制使用工具获取真实数据            │
├─────────────────────────────────────────────┤
│  容错层: 工具失败时返回错误信息而非静默失败  │
├─────────────────────────────────────────────┤
│  评估层: RAGAs faithfulness 离线检测幻觉     │
└─────────────────────────────────────────────┘
```

**仍然可能产生幻觉的场景：**
1. RAG 未检索到相关文档时，LLM 可能基于自身知识编造答案
2. 工具返回的数据不完整时，LLM 可能"补充"不存在的数据
3. 多步推理链中，LLM 可能在某一步骤做出不正确的推断

### Improvements（优化建议）

1. **引入 Citation 机制**：要求 LLM 在回答中标注每个结论的信息来源（哪个工具、哪篇文档）
2. **事实核查 Agent**：在最终输出前增加一个 Fact-Checker Agent，对照工具返回数据验证结论
3. **Confidence Score**：让 LLM 对每个结论输出置信度分数，低置信度结论标注为"待确认"
4. **Guardrails**：引入 Guardrails AI 或 NeMo Guardrails 进行输出校验

---

## Q15: MCP、Skill 和 Tools 的定义与区别

### Facts（项目事实）

**1. MCP（Model Context Protocol）**

MCP 是一种标准化的工具服务协议，项目中使用 `langchain-mcp-adapters` 连接 MCP 服务器：

```python
# File: app/agent/mcp_client.py
from langchain_mcp_adapters.client import MultiServerMCPClient

# MCP 服务器配置
DEFAULT_MCP_SERVERS = {
    "cls": {"transport": "streamable-http", "url": "http://localhost:8003/mcp"},
    "monitor": {"transport": "streamable-http", "url": "http://localhost:8004/mcp"}
}
```

MCP 服务器使用 `FastMCP` 框架实现：

```python
# File: mcp_servers/cls_server.py
from fastmcp import FastMCP
mcp = FastMCP("CLS")

@mcp.tool()
def search_log(topic_id, start_time, end_time, query=None, limit=100):
    ...
```

**2. Tools（LangChain 工具）**

项目中使用 `@tool` 装饰器定义的本地工具：

```python
# File: app/tools/knowledge_tool.py
from langchain_core.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题"""
    ...

# File: app/tools/time_tool.py
@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间"""
    ...
```

**3. Skill**

项目中 **未发现 Skill 的独立实现**。项目的 `.claude/skills/` 目录下有一个 `resume-writer` skill，但这是 Claude Code 的 skill 配置，与 Agent 系统无关。

```
File: .claude/skills/resume-writer/SKILL.md  # Claude Code 技能配置，非 Agent 运行时组件
```

### Analysis（分析）

**三者的定义与区别：**

| 维度 | MCP | Tools | Skill |
|------|-----|-------|-------|
| **本质** | 标准化协议 | 代码级函数 | 高级能力封装 |
| **定义方式** | `@mcp.tool()` + FastMCP 服务器 | `@tool` 装饰器 | 无标准定义 |
| **运行方式** | 独立进程/服务，通过 HTTP 通信 | 同进程内直接调用 | — |
| **可复用性** | 跨应用、跨语言复用 | 绑定到特定 Agent 框架 | — |
| **在项目中的例子** | CLS 日志服务、Monitor 监控服务 | `retrieve_knowledge`、`get_current_time` | ❌ 未实现 |
| **粒度** | 一个 MCP Server 暴露多个 Tool | 单个函数 | 通常包含多个 Tools + Prompt + 流程 |

**在项目中的层次关系：**

```
Agent
  ├── Tools（本地工具，同进程）
  │   ├── retrieve_knowledge → Milvus 向量检索
  │   └── get_current_time → 系统时间
  │
  └── MCP Tools（远程工具，跨进程）
      ├── CLS Server（端口 8003）
      │   ├── get_current_timestamp
      │   ├── get_region_code_by_name
      │   ├── get_topic_info_by_name
      │   ├── search_topic_by_service_name
      │   └── search_log
      └── Monitor Server（端口 8004）
          ├── query_cpu_metrics
          └── query_memory_metrics
```

**MCP 与 Tools 的统一：**

在项目中，MCP 工具和本地工具最终被合并为统一的工具列表，传递给 Agent：

```python
# File: app/services/rag_agent_service.py
all_tools = self.tools + self.mcp_tools  # 本地工具 + MCP 工具
self.agent = create_agent(self.model, tools=all_tools, ...)
```

对 LLM 来说，MCP 工具和本地工具的调用方式完全一致——都是 Function Calling。

### Improvements（优化建议）

1. **引入 Skill 层**：将多个相关 Tools + Prompt + 流程封装为 Skill（如"日志排查 Skill" = search_log + search_topic + 排查 Prompt）
2. **工具注册中心**：统一管理本地工具和 MCP 工具的注册、发现、版本控制

---

## Q16: 项目中 MCP 的使用与加载机制

### Facts（项目事实）

**1. MCP 服务器实现**

项目实现了两个 MCP 服务器：

**CLS 日志服务（端口 8003）：**

```
File: mcp_servers/cls_server.py
Transport: streamable-http
```

提供 5 个工具：

| 工具名 | 功能 |
|--------|------|
| `get_current_timestamp` | 获取当前毫秒时间戳 |
| `get_region_code_by_name` | 根据地区名查找地区代码 |
| `get_topic_info_by_name` | 根据主题名查找主题信息 |
| `search_topic_by_service_name` | 根据服务名搜索日志主题（支持模糊搜索） |
| `search_log` | 基于查询参数搜索日志 |

**Monitor 监控服务（端口 8004）：**

```
File: mcp_servers/monitor_server.py
Transport: streamable-http
```

提供 2 个工具：

| 工具名 | 功能 |
|--------|------|
| `query_cpu_metrics` | 查询服务 CPU 使用率监控数据 |
| `query_memory_metrics` | 查询服务内存使用监控数据 |

**2. MCP 客户端加载机制**

```python
# File: app/agent/mcp_client.py

# 步骤 1: 创建 MultiServerMCPClient
def _create_mcp_client(servers, tool_interceptors=None):
    kwargs = {}
    if tool_interceptors:
        kwargs["tool_interceptors"] = tool_interceptors
    return MultiServerMCPClient(servers, **kwargs)

# 步骤 2: 全局单例管理
async def get_mcp_client_with_retry(servers=None, tool_interceptors=None, force_new=False):
    interceptors = [retry_interceptor]  # 添加重试拦截器
    if tool_interceptors:
        interceptors.extend(tool_interceptors)
    return await get_mcp_client(servers=servers, tool_interceptors=interceptors, force_new=force_new)

# 步骤 3: Agent 初始化时加载 MCP 工具
# File: app/services/rag_agent_service.py
async def _initialize_agent(self):
    mcp_client = await get_mcp_client_with_retry()
    mcp_tools = await mcp_client.get_tools()  # 从 MCP 服务器获取所有工具
    all_tools = self.tools + self.mcp_tools
    self.agent = create_agent(self.model, tools=all_tools, ...)
```

**3. 重试拦截器（指数退避）**

```python
# File: app/agent/mcp_client.py
async def retry_interceptor(request: MCPToolCallRequest, handler, max_retries=3, delay=1.0):
    for attempt in range(max_retries):
        try:
            result = await handler(request)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # 1s → 2s → 4s
                await asyncio.sleep(wait_time)
    return CallToolResult(content=[TextContent(type="text", text=error_msg)], isError=True)
```

**4. 加载时机**

MCP 工具在 Agent 首次使用时 **延迟加载**（Lazy Initialization）：

```python
# File: app/services/rag_agent_service.py
async def _initialize_agent(self):
    if self._agent_initialized:
        return  # 只加载一次
    mcp_client = await get_mcp_client_with_retry()
    mcp_tools = await mcp_client.get_tools()
    ...
    self._agent_initialized = True
```

AIOps Agent 的 Planner/Executor/Replanner 每次执行时都会重新获取工具列表：

```python
# File: app/agent/aiops/planner.py
mcp_client = await get_mcp_client_with_retry()
mcp_tools = await mcp_client.get_tools()  # 每次都获取最新工具列表
```

### Analysis（分析）

**MCP 加载流程：**

```
应用启动 → FastAPI lifespan → Milvus 连接
    ↓
首次请求 → _initialize_agent() → get_mcp_client_with_retry()
    ↓
MultiServerMCPClient 创建 → 连接 CLS:8003 + Monitor:8004
    ↓
mcp_client.get_tools() → 返回 7 个 MCP 工具
    ↓
合并本地工具 → [retrieve_knowledge, get_current_time] + 7 MCP tools = 9 tools
    ↓
create_agent(model, tools=all_tools) → Agent 就绪
```

**当前方案的注意点：**
- MCP 服务器当前使用 Mock 数据（模拟日志和监控数据），非真实后端
- AIOps Agent 的每次节点执行都重新获取工具列表，存在性能浪费
- MCP 客户端是全局单例，但如果 MCP 服务器重启，客户端不会自动重连

### Improvements（优化建议）

1. **工具列表缓存**：避免每次节点执行都重新获取工具列表，设置合理的缓存 TTL
2. **健康检查与自动重连**：定期检查 MCP 服务器状态，断线时自动重连
3. **工具描述增强**：当前 MCP 工具的描述在服务器端定义，可以考虑在客户端增加领域特定的使用说明

---

## Q17: 项目中 Tools 的使用

### Facts（项目事实）

项目中使用的 Tools 分为 **本地工具** 和 **MCP 工具** 两类：

**本地工具（2 个）：**

**1. `retrieve_knowledge` — 知识检索工具**

```python
# File: app/tools/knowledge_tool.py
@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题"""
    docs = get_rag_retriever().retrieve(query, top_k=effective_top_k)
    context = format_docs(docs)
    return context, docs
```

- 使用 `response_format="content_and_artifact"` 模式
- 返回 `Tuple[str, List[Document]]`：第一个元素是给 LLM 看的文本，第二个是原始文档（Agent 框架自动处理）
- 根据 `RAG_MODE` 配置选择 Basic 或 Enhanced 检索器

**2. `get_current_time` — 时间工具**

```python
# File: app/tools/time_tool.py
@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间"""
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    return now.strftime('%Y-%m-%d %H:%M:%S')
```

**MCP 工具（7 个）：**

| 服务器 | 工具名 | 功能 |
|--------|--------|------|
| CLS:8003 | `get_current_timestamp` | 毫秒时间戳 |
| CLS:8003 | `get_region_code_by_name` | 地区代码查询 |
| CLS:8003 | `get_topic_info_by_name` | 日志主题查询 |
| CLS:8003 | `search_topic_by_service_name` | 服务名搜索日志主题 |
| CLS:8003 | `search_log` | 日志搜索 |
| Monitor:8004 | `query_cpu_metrics` | CPU 监控指标查询 |
| Monitor:8004 | `query_memory_metrics` | 内存监控指标查询 |

**工具的注册和使用：**

```python
# File: app/services/rag_agent_service.py
# 定义基础工具
self.tools = [retrieve_knowledge, get_current_time]

# 加载 MCP 工具
mcp_client = await get_mcp_client_with_retry()
mcp_tools = await mcp_client.get_tools()

# 合并并创建 Agent
all_tools = self.tools + self.mcp_tools
self.agent = create_agent(self.model, tools=all_tools, checkpointer=self.checkpointer)
```

```python
# File: app/agent/aiops/executor.py
# Executor 中同样加载所有工具
local_tools = [get_current_time, retrieve_knowledge]
mcp_client = await get_mcp_client_with_retry()
mcp_tools = await mcp_client.get_tools()
all_tools = local_tools + mcp_tools

# 使用 ToolNode 执行工具调用
tool_node = ToolNode(all_tools)
tool_messages = await tool_node.ainvoke({"messages": messages})
```

### Analysis（分析）

**工具使用统计：**

| 类别 | 数量 | 定义位置 | 加载方式 |
|------|------|----------|----------|
| 本地工具 | 2 | `app/tools/` | `@tool` 装饰器，直接导入 |
| MCP CLS 工具 | 5 | `mcp_servers/cls_server.py` | `MultiServerMCPClient.get_tools()` |
| MCP Monitor 工具 | 2 | `mcp_servers/monitor_server.py` | `MultiServerMCPClient.get_tools()` |
| **总计** | **9** | | |

**工具调用的完整链路：**

```
用户问题
  → LLM 推理（决定调用哪个工具）
  → 生成 tool_call（name + args）
  → Agent 框架路由到对应工具
    → 本地工具：直接调用 Python 函数
    → MCP 工具：通过 HTTP 发送到 MCP Server
  → 工具返回结果
  → LLM 基于工具结果生成回答
```

### Improvements（优化建议）

1. **工具参数校验增强**：当前 MCP 工具的参数校验依赖 LLM 正确传参，可以加入更严格的 schema 校验
2. **工具调用超时**：为 MCP 工具调用设置超时限制，避免长时间阻塞
3. **工具结果缓存**：对于短时间内重复查询相同参数的工具调用，可以缓存结果

---

## Q18: Skill 的运作机制

### Facts（项目事实）

**项目中未发现 Skill 的独立实现或运作机制。**

项目中存在一个 `.claude/skills/resume-writer/SKILL.md` 文件，但这是 Claude Code（IDE 插件）的技能配置，与 Agent 系统运行时无关：

```
File: .claude/skills/resume-writer/SKILL.md
用途: Claude Code 的简历写作辅助技能（IDE 层面）
与 Agent 系统的关系: 无
```

项目当前的架构中没有 Skill 层的抽象。所有的能力都直接通过 Tools（本地工具 + MCP 工具）+ Prompt 来实现。

### Analysis（分析）

**Skill 的概念定位：**

在 AI Agent 架构中，Skill 通常是介于 Tool 和 Agent 之间的抽象层：

```
Tool（原子能力）
  ↓ 组合
Skill（复合能力 = 多个 Tools + 专用 Prompt + 执行流程）
  ↓ 编排
Agent（自主决策 = 多个 Skills + 推理引擎 + 状态管理）
```

**以当前项目为例，如果要引入 Skill 层：**

| Skill 名称 | 包含的 Tools | 专用 Prompt | 场景 |
|------------|-------------|-------------|------|
| 日志排查 Skill | `search_topic_by_service_name` + `search_log` + `get_current_timestamp` | 日志排查专用 Prompt | 用户问"查一下日志" |
| 监控排查 Skill | `query_cpu_metrics` + `query_memory_metrics` | 监控分析专用 Prompt | 用户问"看看监控" |
| 知识问答 Skill | `retrieve_knowledge` | 知识问答专用 Prompt | 用户问专业知识 |
| 综合诊断 Skill | 所有 MCP 工具 + `retrieve_knowledge` | 故障诊断专用 Prompt | AIOps 自动诊断 |

**Skill 返回给大模型的形式（行业最佳实践）：**

1. **作为 Tool 返回**：Skill 被封装为一个高级 Tool，LLM 调用 `execute_log_investigation(service_name)` 这样的 Skill 级接口，内部自动编排多个子工具
2. **作为 Prompt 注入**：根据用户意图动态注入 Skill 对应的专用 Prompt 和工具子集
3. **作为 Sub-Agent**：Skill 是一个独立的 Agent，被 Supervisor Agent 调度

### Improvements（优化建议）

1. **引入 Skill 抽象层**：将相关的 Tools + Prompt 封装为 Skill，提高复用性和可维护性
2. **Skill 路由**：在用户输入后先用分类器判断走哪个 Skill，再将 Skill 对应的工具子集和 Prompt 注入 Agent
3. **Skill 评估**：为每个 Skill 建立独立的评估数据集和指标，而非整体评估
4. **Skill 编排**：支持 Skill 之间的组合调用（如"日志排查 Skill"完成后自动触发"监控排查 Skill"）

---

## 附录：项目技术栈速查

| 层级 | 技术 | 版本/说明 |
|------|------|-----------|
| Web 框架 | FastAPI | 0.109+ |
| Agent 框架 | LangGraph + LangChain | 最新版 |
| LLM | 阿里千问 ChatQwen | via DashScope API |
| Embedding | text-embedding-v4 | 1024 维 |
| 精排模型 | bge-reranker-v2-m3 | Cross-Encoder |
| 向量数据库 | Milvus | Docker 部署 |
| 工具协议 | MCP | streamable-http |
| 会话持久化 | MemorySaver / RedisSaver | 可选 Redis |
| 评估框架 | RAGAs | 两阶段评估 |
| 日志 | Loguru | 控制台 + 文件轮转 |
| Token 计算 | tiktoken | cl100k_base 编码 |
