# ReAct Agent 优化：Human-in-the-Loop 与工具路由

> 当前不建议作为优先事项推进。现阶段工具大多是查询型 mock 工具，真正的危险操作尚未接入，HITL 的实际收益偏低；工具总量也不大，路由层的复杂度和额外开销可能高于收益。建议先保持现状，待危险工具或明显的工具选择瓶颈出现后，再重新评估。

## 1. 功能和目的

对当前 RAG Chat Agent（基于 LangGraph `create_agent` 的 ReAct 范式）进行两项关键优化：

### I. Human-in-the-Loop (HITL)

运维场景中，某些工具调用（如重启服务、切换流量、执行高危命令）需要人工确认后才能执行。当前 Agent 完全自主运行，所有工具调用无人工审核环节。

改进目标：在工具调用前插入人工审批断点，Agent 暂停执行并等待操作者确认或拒绝，实现"确认后继续"的人机协同。

### II. MCP/Tools Routing（工具路由）

当前所有 9 个工具（2 个本地 + 7 个 MCP）全部绑定到 LLM。工具越多，LLM 的 Function Call 选择准确率越低，token 消耗越大。

改进目标：在 LLM 调用前加一层轻量级工具路由——用 embedding 匹配筛选与当前 query 最相关的 k 个工具，仅将筛选后的工具传给 LLM。

## 2. 抽象实现思路

### I. HITL 架构

```
用户提问 → Agent 开始执行
    ↓
LLM 决定调用工具 X
    ↓
【中断检查】工具 X 在 dangerous_tools 白名单中？
    ├── 否 → 直接执行工具，继续 ReAct 循环
    └── 是 → 暂停执行，SSE 发送 interrupt 事件
              ↓
         等待前端 POST /api/chat/approve
              ↓
         ┌── 批准 → graph.resume() 继续执行
         └── 拒绝 → 跳过该工具调用，告知 LLM 操作被拒绝
```

### II. 工具路由架构

```
用户 Query
    ↓
Embedding（DashScope text-embedding-v4）
    ↓
与所有工具的 name + description 做相似度匹配
    ↓
取 top-k（如 5 个）最相关工具
    ↓
保底工具（retrieve_knowledge, get_current_time）始终保留
    ↓
仅将筛选后的工具列表传给 LLM
```

### 关键约束

- **不破坏现有 API 接口**：`/api/chat` 和 `/api/chat_stream` 签名保持不变
- **HITL 为可选功能**：未配置 `dangerous_tools` 时，行为与当前完全一致
- **工具路由为可配置**：可通过 `tool_routing_enabled` 开关控制，默认关闭以保持向后兼容

## 3. 具体实现流程

### Step 1: HITL — 从 create_agent 迁移到 create_react_agent

**关键发现**：当前代码使用 `create_agent()`（第 130-134 行），这个高级封装**不暴露** `interrupt_before` 参数。必须切换到 `create_react_agent()`（LangGraph 的低级 API）。

在 [app/services/rag_agent_service.py](app/services/rag_agent_service.py) 的 `_initialize_agent()` 中：

```python
# 当前代码（第 130-134 行）
self.agent = create_agent(
    self.model,
    tools=all_tools,
    checkpointer=self.checkpointer,
)

# 改造后
from langgraph.prebuilt import create_react_agent

interrupt_config = ["tools"] if config.dangerous_tools else None

self.agent = create_react_agent(
    self.model,
    tools=all_tools,
    checkpointer=self.checkpointer,
    interrupt_before=interrupt_config,  # 关键参数
)
```

注意：`interrupt_before=["tools"]` 会在**每次**工具调用前暂停。如果需要更精细的控制（仅某些工具中断），需要在工具节点内部实现条件判断逻辑，或使用 LangGraph 的 `NodeInterrupt`。

### Step 2: HITL — 危险工具白名单配置

在 [app/config.py](app/config.py) 中新增：

```python
# HITL 配置
dangerous_tools: list[str] = []  # 需要人工确认的工具名列表
# 示例: ["restart_service", "execute_command", "clear_cache"]
```

对应 `.env`：
```env
DANGEROUS_TOOLS=restart_service,execute_command
```

注意：当前 MCP 工具全部是 mock 的查询类工具（CLS 日志查询、监控指标查询），没有真正的危险操作。此配置是为未来接入真实运维工具预留的。

### Step 3: HITL — 新增审批接口

在 [app/api/chat.py](app/api/chat.py) 中新增端点：

```python
@router.post("/chat/approve")
async def approve_tool_call(request: ApproveRequest):
    """
    批准或拒绝待确认的工具调用

    request.session_id: 会话 ID
    request.approved: True 批准 / False 拒绝
    """
    agent_service = get_rag_agent_service()
    graph = agent_service.get_graph()

    config = {"configurable": {"thread_id": request.session_id}}

    if request.approved:
        result = await graph.ainvoke(Command(resume={"approved": True}), config)
    else:
        result = await graph.ainvoke(Command(resume={"approved": False}), config)

    return {"status": "ok"}
```

新增模型 [app/models/request.py](app/models/request.py)：
```python
class ApproveRequest(BaseModel):
    session_id: str
    approved: bool
```

### Step 4: HITL — SSE 事件扩展

在 `query_stream()` 中，当 Agent 中断时发送新的 SSE 事件类型：

```python
# 新增 "interrupt" 事件
yield {
    "type": "interrupt",
    "data": {
        "tool_name": "restart_service",
        "tool_args": {"service_name": "data-sync-service"},
        "message": "Agent 请求执行 restart_service，等待审批...",
    }
}
```

前端收到 `"interrupt"` 事件后，展示确认对话框，用户点击批准/拒绝后调用 `/api/chat/approve`。

### Step 5: 工具路由 — Embedding 匹配层

在 [app/services/rag_agent_service.py](app/services/rag_agent_service.py) 中新增工具筛选逻辑：

```python
def _filter_tools_by_relevance(
    self, query: str, all_tools: list, top_k: int = 5
) -> list:
    """使用 embedding 匹配筛选与 query 最相关的工具"""
    from app.services.vector_embedding_service import get_vector_embedding_service

    embedding_service = get_vector_embedding_service()

    # 构建工具文本表示
    tool_texts = []
    for tool in all_tools:
        text = f"{tool.name}: {tool.description}"
        tool_texts.append(text)

    # 获取 query 和工具的 embedding
    query_emb = embedding_service.embed_query(query)
    tool_embs = embedding_service.embed_documents(tool_texts)

    # 计算余弦相似度
    from numpy import dot
    from numpy.linalg import norm
    similarities = [
        dot(query_emb, tool_emb) / (norm(query_emb) * norm(tool_emb))
        for tool_emb in tool_embs
    ]

    # 取 top-k 相关工具
    ranked = sorted(zip(all_tools, similarities), key=lambda x: x[1], reverse=True)

    # 始终保留保底工具
    core_tools = [t for t in all_tools if t.name in ("retrieve_knowledge", "get_current_time")]
    selected = [t for t, _ in ranked[:top_k] if t not in core_tools]

    return core_tools + selected
```

### Step 6: 工具路由 — 配置开关

在 [app/config.py](app/config.py) 中新增：

```python
# 工具路由配置
tool_routing_enabled: bool = False      # 是否启用工具路由
tool_routing_top_k: int = 5             # 相关工具数量（不含保底工具）
tool_routing_threshold: float = 0.3     # 相似度阈值（低于此值不纳入）
```

对应 `.env`：
```env
TOOL_ROUTING_ENABLED=false
TOOL_ROUTING_TOP_K=5
TOOL_ROUTING_THRESHOLD=0.3
```

### Step 7: 工具路由 — 集成到 _initialize_agent

```python
async def _initialize_agent(self) -> None:
    if self.agent is not None:
        return
    mcp_client = get_mcp_client_with_retry()
    self.mcp_tools = await mcp_client.get_tools()
    all_tools = self.tools + list(self.mcp_tools)

    # 工具路由：存储全量工具，运行时筛选
    if config.tool_routing_enabled:
        self.all_tools = all_tools  # 全量保存
        # 在 query()/query_stream() 中使用 _filter_tools_by_relevance 筛选
    else:
        # 原有行为
        self.agent = create_react_agent(
            self.model, tools=all_tools, checkpointer=self.checkpointer,
            interrupt_before=interrupt_config,
        )
```

当启用工具路由时，Agent 创建时使用筛选后的工具列表，而非全量。每次 `query()` 调用时都重新筛选。

### 迁移风险与注意点

1. **`create_agent` → `create_react_agent` 的 API 差异**：两者行为上基本等价（都是 ReAct 循环），但返回对象类型可能不同。需要验证 `astream`、`ainvoke` 的输出格式一致性
2. **`interrupt_before=["tools"]` 影响所有工具**：当前 `create_react_agent` 不支持"仅对部分工具中断"。要实现白名单模式，一个方案是在 tool 装饰器上打标记，然后在工具执行节点内检查标记并手动触发 `NodeInterrupt`
3. **工具路由的运行时开销**：每次 query 都需要额外做 n+1 次 embedding（query + n 个工具文本）+ 相似度计算。9 个工具时这个开销可忽略（< 50ms），工具数增加到 50+ 时需要缓存工具 embedding
4. **`@lru_cache` on retriever 与路由的交互**：工具路由不影响检索器，两者独立

## 4. 当前实现进度

### 已完成

- [x] `create_agent()` 基础的 ReAct Agent（使用 `ChatQwen` + 9 个工具 + RedisSaver/MemorySaver 自动切换）
- [x] `query()` 和 `query_stream()` 方法
- [x] MCP 工具动态加载（`get_mcp_client_with_retry()` → `get_tools()`）
- [x] `query_with_trace()` 方法（Plan 10 新增，用于 Agent 评估捕获工具调用）
- [x] RedisSaver/MemorySaver 自动切换（Plan 12 实施）
- [x] Token 计数驱动的上下文裁剪 `trim_messages_by_tokens()`（Plan 12 实施，替换旧 `trim_messages_middleware`）

### 尚未实现

- [ ] HITL / 工具路由：当前不建议推进，建议保持暂缓状态，待真实危险工具接入或工具误选率显著上升后再评估
- [ ] `create_agent` → `create_react_agent` 迁移：Plan 12 实施时发现当前 langgraph 版本的 `create_react_agent` 无 `middlewares` 参数，迁移无实际收益，因而保留 `create_agent`

### 跨计划联动（已落实）

- ~~`trim_messages_middleware` 接入问题~~ → Plan 12 已替换为 `trim_messages_by_tokens()`（tiktoken 计数，8000 token 阈值），接入 `query()`/`query_with_trace()`/`query_stream()` 三个方法
- ~~`create_react_agent` 迁移为 HITL 预留~~ → Plan 12 未迁移（当前版本 `create_react_agent` 无 `middlewares` 参数，且 HITL 暂不启用，迁移无收益）。未来若启用 HITL 需要 `create_react_agent` 的 `interrupt_before`，届时再做迁移
- Plan 12 扩展了 `PlanExecuteState`（`total=False`），为 HITL 预留了 `pending_approval`/`pending_tool_name`/`pending_tool_args` 字段

## 5. 为什么暂缓

### 5.1 HITL 的收益前提尚未满足

HITL 最适合接入高风险动作，比如重启、切流、清缓存、执行命令等。但当前系统中的工具几乎都是查询型工具: 日志查询、监控查询、知识检索和时间查询。这意味着即使加上审批流程，绝大多数调用也不会触发中断，收益很有限。

### 5.2 工具路由的复杂度高于当前收益

现在全量工具数量并不大，LLM 选择错误工具的成本还没有高到必须加路由层的程度。再加一层 embedding 匹配，会引入额外的查询延迟、缓存策略、相关性阈值调优和调试成本，但短期内不一定能显著改善结果。

### 5.3 这些能力会牵动一整串接口

一旦落地 HITL，就会联动 Agent 构建方式、SSE 事件协议、前端交互、审批 API、状态恢复机制。工具路由也会影响工具注册、初始化、trace 和评估逻辑。当前项目更需要先把检索、评估、记忆这些基础层做稳，再考虑这一层。

### 5.4 后续重新评估的触发条件

建议满足以下条件之一时再重新开启这项工作:
- 接入了真实危险工具
- 工具数量显著增加，且误选率开始影响体验
- 线上 trace 证明工具选择错误已经成为主要瓶颈
- 产品明确要求人工审批流程

## 6. Evidence（2026-05-24 核实）

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| 当前 agent 创建 | [app/services/rag_agent_service.py:144-148](app/services/rag_agent_service.py#L144) | `create_agent(model, tools=all_tools, checkpointer=self.checkpointer)` — 无 interrupt 参数，仍使用 `create_agent` |
| 旧 trim_middleware 已删除 | [app/services/rag_agent_service.py:38-81](app/services/rag_agent_service.py#L38) | Plan 12 替换为 `trim_messages_by_tokens`（tiktoken 计数） |
| Token 裁剪已接入 | [app/services/rag_agent_service.py:215](app/services/rag_agent_service.py#L215) | `query()` 中 `trim_messages_by_tokens(messages, ...)` |
| query_with_trace | [app/services/rag_agent_service.py:253](app/services/rag_agent_service.py#L253) | Plan 10 新增，捕获结构化 tool_calls trace |
| Checkpointer 切换 | [app/services/rag_agent_service.py:112-118](app/services/rag_agent_service.py#L112) | `config.redis_url` → RedisSaver / MemorySaver |
| Checkpointer 切换 | [app/services/aiops_service.py:27-33](app/services/aiops_service.py#L27) | AIOps 同步切换 |
| MCP 工具加载 | [app/services/rag_agent_service.py:135](app/services/rag_agent_service.py#L135) | `mcp_client.get_tools()` — 全量获取 |
| 工具合并 | [app/services/rag_agent_service.py:142](app/services/rag_agent_service.py#L142) | `all_tools = self.tools + self.mcp_tools` — 无筛选 |
| MCP 工具数量 | [mcp_servers/cls_server.py](mcp_servers/cls_server.py) + [mcp_servers/monitor_server.py](mcp_servers/monitor_server.py) | 5 个 CLS + 2 个 Monitor = 7 个 MCP 工具 |
| 本地工具数量 | [app/services/rag_agent_service.py:106](app/services/rag_agent_service.py#L106) | 2 个：`retrieve_knowledge`, `get_current_time` |
| 无 interrupt 使用 | 全项目 grep | 无 `interrupt_before`、`interrupt_after`、`Command(resume=...)` |
| 无审批接口 | [app/api/chat.py](app/api/chat.py) | 无 `/chat/approve` 端点 |
| 无工具路由配置 | [app/config.py](app/config.py) | 无 `tool_routing_enabled` 等字段 |
| PlanExecuteState HITL 预留 | [app/agent/aiops/state.py:26-29](app/agent/aiops/state.py#L26) | Plan 12 新增 `pending_approval` 等字段 |
