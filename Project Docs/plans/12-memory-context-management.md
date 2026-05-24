# Memory 管理与上下文窗口优化

## 1. 功能和目的

针对当前 RAG Chat Agent 和 AIOps Agent 在会话持久化和上下文管理方面的不足，实施三项改进：

### I. MemorySaver → RedisSaver（会话状态持久化）

当前使用 `MemorySaver`（进程内存）存储会话状态，服务重启后所有对话历史丢失。通过引入 RedisSaver 实现跨重启的会话持久化。采用可选依赖模式——配置了 `REDIS_URL` 时使用 RedisSaver，未配置时 fallback 到 MemorySaver。

### II. 上下文窗口 Token 计数裁剪

当前代码存在一个关键问题：`trim_messages_middleware` 函数在 [app/services/rag_agent_service.py:36-73](app/services/rag_agent_service.py#L36) **已定义但完全未使用**——`create_agent()` 调用未传入该参数。而且即使接入，也是硬编码的"保留最近 6 条消息"策略，不是 token 感知的裁剪。

改进策略二选一：
- **方案 A（推荐）**：使用 LangGraph 内置 `SummarizationMiddleware`，自动摘要历史消息，无需手动计算 token
- **方案 B**：使用 `tiktoken` 估算 token 数，动态裁剪到阈值（如 8000 token），而非硬编码消息条数

### III. 持久化诊断报告与执行 Trace

AIOps Agent 的 `past_steps` 和最终诊断 `response` 在服务重启后丢失。通过持久化存储（Redis 或文件）保存诊断报告和执行 trace，支持历史查询和回放。

## 2. 抽象实现思路

### I. RedisSaver 可选依赖

```
启动时检查 REDIS_URL 配置
    ├── 已配置 → 创建 RedisSaver.from_conn_string(REDIS_URL)
    └── 未配置 → 创建 MemorySaver()（保持当前行为）
```

`langgraph-checkpoint` 已内置 `RedisSaver`，`redis` 包已作为传递依赖存在于 `.venv` 中，无需额外安装。

### II-A. SummarizationMiddleware（推荐方案）

LangGraph 的 `SummarizationMiddleware` 在上下文超过 token 阈值时自动生成历史摘要：

```
消息历史
    ↓
Token 计数超过阈值？
    ├── 否 → 不做处理
    └── 是 → 调用 LLM 生成摘要
              ↓
         将早期消息替换为 SummaryMessage
              ↓
         保留近期消息 + 摘要
```

优势：不需要手动计算 token，摘要质量由 LLM 保证，且摘要本身比原始消息更省 token。

### II-B. Token 计数裁剪（备选方案）

```python
import tiktoken

def trim_by_token_count(messages, max_tokens=8000):
    """按 token 数裁剪消息历史"""
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    kept = []
    # 从最新到最旧遍历
    for msg in reversed(messages):
        tokens = len(enc.encode(msg.content))
        if total + tokens > max_tokens:
            break
        kept.insert(0, msg)
        total += tokens
    return kept
```

### III. 持久化诊断 Trace

在 `AIOpsService.execute()` 完成后，将 `state["past_steps"]` 和 `state["response"]` 序列化存储：

```python
diagnosis_record = {
    "session_id": session_id,
    "timestamp": datetime.now().isoformat(),
    "input": state["input"],
    "past_steps": state["past_steps"],
    "response": state["response"],
}

# Redis 模式
if redis_client:
    redis_client.set(f"diagnosis:{session_id}", json.dumps(diagnosis_record))

# 文件模式（fallback）
else:
    Path(f"diagnosis_reports/{session_id}.json").write_text(json.dumps(diagnosis_record))
```

## 3. 具体实现流程

### Step 1：RedisSaver 配置与集成

在 [app/config.py](app/config.py) 中新增：

```python
# Redis 配置（可选，不配置则使用 MemorySaver）
redis_url: str = ""  # 如 "redis://localhost:6379"
```

对应 `.env`：
```env
REDIS_URL=   # 留空则使用 MemorySaver
```

在 [app/services/rag_agent_service.py](app/services/rag_agent_service.py) 中的 `__init__`：

```python
# 当前代码（第 103-104 行）
self.checkpointer = MemorySaver()

# 改造后
if config.redis_url:
    from langgraph.checkpoint.redis import RedisSaver
    self.checkpointer = RedisSaver.from_conn_string(config.redis_url)
else:
    self.checkpointer = MemorySaver()
```

同样修改 [app/services/aiops_service.py](app/services/aiops_service.py) 中的第 25 行。

### Step 2：接入上下文裁剪（方案 A：SummarizationMiddleware）

在 [app/services/rag_agent_service.py](app/services/rag_agent_service.py) 中：

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import SummarizationMiddleware

# 创建 SummarizationMiddleware
summarization_middleware = SummarizationMiddleware(
    model=self.model,            # 复用 ChatQwen
    max_tokens_before_summary=8000,  # 超过 8000 token 时触发摘要
    messages_to_keep=10,         # 保留最近 10 条消息
)

self.agent = create_react_agent(
    self.model,
    tools=all_tools,
    checkpointer=self.checkpointer,
    interrupt_before=interrupt_config,  # 来自 HITL 配置（见文档 11）
    middlewares=[summarization_middleware],
)
```

注意：`create_agent`（当前的高层 API）已计划切换为 `create_react_agent`（作为 [11-react-agent-optimization.md](11-react-agent-optimization.md) HITL 实现的一部分），两者均支持 `middlewares` 参数。

### Step 3：接入上下文裁剪（方案 B：Token 计数）

如果选择方案 B（不依赖 SummarizationMiddleware），修改现有的 `trim_messages_middleware` 函数，从消息条数改为 token 计数：

```python
def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """Token 计数驱动的上下文裁剪"""
    messages = state["messages"]
    max_tokens = config.context_max_tokens or 8000

    # 估算当前 token 数
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(enc.encode(str(msg.content or ""))) for msg in messages)

    if total_tokens <= max_tokens:
        return None

    # 保留系统消息 + 从最新到最旧裁剪
    first_msg = messages[0]
    kept = [first_msg]
    remaining = max_tokens - len(enc.encode(str(first_msg.content or "")))

    for msg in reversed(messages[1:]):
        msg_tokens = len(enc.encode(str(msg.content or "")))
        if remaining - msg_tokens < 0:
            break
        kept.insert(1, msg)
        remaining -= msg_tokens

    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]
    }
```

新增配置项：

```python
context_max_tokens: int = 8000   # 上下文窗口 token 上限
context_trimming_strategy: Literal["token_count", "summarization", "none"] = "summarization"
```

### Step 4：持久化诊断报告

新增文件：[app/services/diagnosis_store.py](app/services/diagnosis_store.py)

```python
class DiagnosisStore:
    """诊断报告持久化存储"""

    def __init__(self):
        self.redis_client = None
        if config.redis_url:
            import redis
            self.redis_client = redis.from_url(config.redis_url)

    def save(self, session_id: str, state: PlanExecuteState) -> str:
        record = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "input": state["input"],
            "plan": state["plan"],
            "past_steps": state["past_steps"],
            "response": state["response"],
        }
        record_id = f"diagnosis:{session_id}:{int(time.time())}"

        if self.redis_client:
            self.redis_client.set(record_id, json.dumps(record, default=str))
        else:
            report_dir = Path("diagnosis_reports")
            report_dir.mkdir(exist_ok=True)
            (report_dir / f"{record_id.replace(':', '_')}.json").write_text(
                json.dumps(record, default=str, ensure_ascii=False, indent=2)
            )
        return record_id

    def get(self, record_id: str) -> dict | None:
        if self.redis_client:
            data = self.redis_client.get(record_id)
            return json.loads(data) if data else None
        else:
            path = Path("diagnosis_reports") / f"{record_id.replace(':', '_')}.json"
            if path.exists():
                return json.loads(path.read_text())
            return None
```

在 [app/services/aiops_service.py](app/services/aiops_service.py) 的 `execute()` 方法末尾，调用 `diagnosis_store.save(session_id, final_state)`。

### Step 5：扩展 PlanExecuteState

在 [app/agent/aiops/state.py](app/agent/aiops/state.py) 中扩展状态定义（为 HITL 和持久化做准备）：

```python
class PlanExecuteState(TypedDict, total=False):
    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple], operator.add]
    response: str
    # 新增字段（为 HITL 预留）
    pending_approval: bool          # 是否有待审批的工具调用
    pending_tool_name: str          # 待审批的工具名
    pending_tool_args: dict         # 待审批的工具参数
```

注意：使用 `total=False` 使所有字段变为可选，保持向后兼容。

### Step 6：新增 API 查询历史诊断

在 [app/api/aiops.py](app/api/aiops.py) 中新增：

```python
@router.get("/aiops/diagnosis/{session_id}")
async def get_diagnosis(session_id: str):
    """查询历史诊断报告"""
    store = get_diagnosis_store()
    records = store.list_by_session(session_id)
    return {"session_id": session_id, "records": records}
```

## 4. 当前实现进度

### 已完成

- [x] `MemorySaver` 用于 RAG Agent 和 AIOps Agent 的会话管理
- [x] `trim_messages_middleware` → `trim_messages_by_tokens`：基于 tiktoken 的 token 计数裁剪，替换硬编码消息条数策略
- [x] RedisSaver 集成：RAG Agent 和 AIOps Agent 的 checkpointer 均支持 `REDIS_URL` → RedisSaver / MemorySaver 自动切换
- [x] `REDIS_URL` 配置项：已添加到 `app/config.py` 和 `.env`
- [x] `context_max_tokens` / `context_trimming_strategy` 配置项：支持 `token_count` / `none`
- [x] 诊断报告持久化：`DiagnosisStore` 类，Redis + 文件双后端
- [x] `DiagnosisStore` 全局单例：`get_diagnosis_store()`
- [x] `PlanExecuteState` 扩展：`total=False` + HITL 预留字段 (`pending_approval`, `pending_tool_name`, `pending_tool_args`)
- [x] `GET /api/aiops/diagnosis/{session_id}` 历史查询接口
- [x] Token 裁剪接入 `query()` / `query_with_trace()` / `query_stream()` 三个方法
- [x] `langgraph-checkpoint-redis` 包已安装到 `.venv`

### 尚未实现

- [ ] `SummarizationMiddleware` 集成 — 当前 langgraph 版本不支持 `middlewares` 参数，预留到版本升级后
- [ ] 会话存储统一抽象层 — 当前通过 `config.redis_url` 在三处判断切换，抽象层留待未来多实例部署需求

### 实施决策

- 保持 `create_agent`（不迁移到 `create_react_agent`）：当前版本 `create_react_agent` 无 `middlewares` 参数，迁移无收益
- Token 裁剪在输入层进行：不在 checkpointer 内部裁剪，而是在每次 query 调用时裁剪输入消息。单轮场景（评估/独立查询）完全够用，多轮场景需要未来升级到 middleware 模式的裁剪
- `langgraph-checkpoint-redis` 通过 `ensurepip` + `pip install` 安装到 venv
- `tiktoken` 使用 `cl100k_base` 编码作为近似计数（对 Qwen 模型不完全精确但足够防止溢出）
- 诊断记录 past_steps 截断至 200 字符、response 截断至 5000 字符，避免存储膨胀
- DiagnosisStore Redis 记录 7 天自动过期

## 5. Evidence

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| RedisSaver + MemorySaver 切换 | [app/services/rag_agent_service.py:112-118](app/services/rag_agent_service.py#L112) | `config.redis_url` 判断 → RedisSaver / MemorySaver |
| RedisSaver + MemorySaver 切换 | [app/services/aiops_service.py:27-33](app/services/aiops_service.py#L27) | AIOps 同步切换 |
| trim_messages_by_tokens | [app/services/rag_agent_service.py:38-81](app/services/rag_agent_service.py#L38) | tiktoken 驱动的 token 计数裁剪，替换旧 trim_messages_middleware |
| Token 裁剪接入 query() | [app/services/rag_agent_service.py:214-215](app/services/rag_agent_service.py#L214) | 在 ainvoke 前裁剪输入消息 |
| Token 裁剪接入 query_with_trace() | [app/services/rag_agent_service.py:284-285](app/services/rag_agent_service.py#L284) | 同上 |
| Token 裁剪接入 query_stream() | [app/services/rag_agent_service.py:364-365](app/services/rag_agent_service.py#L364) | 同上 |
| REDIS_URL 配置 | [app/config.py:73-74](app/config.py#L73) | `redis_url: str = ""` |
| context_max_tokens / trimming_strategy | [app/config.py:77-78](app/config.py#L77) | `context_max_tokens=8000`, `context_trimming_strategy="token_count"` |
| .env 新增项 | [.env:42-47](.env#L42) | REDIS_URL, CONTEXT_MAX_TOKENS, CONTEXT_TRIMMING_STRATEGY |
| DiagnosisStore | [app/services/diagnosis_store.py](app/services/diagnosis_store.py) | Redis + 文件双后端，save/get/list_by_session |
| DiagnosisStore 集成 | [app/services/aiops_service.py:157-170](app/services/aiops_service.py#L157) | execute() 完成后自动保存诊断记录 |
| PlanExecuteState total=False | [app/agent/aiops/state.py:10](app/agent/aiops/state.py#L10) | `class PlanExecuteState(TypedDict, total=False)` |
| HITL 预留字段 | [app/agent/aiops/state.py:26-29](app/agent/aiops/state.py#L26) | pending_approval, pending_tool_name, pending_tool_args |
| 诊断历史查询 API | [app/api/aiops.py:156-175](app/api/aiops.py#L156) | `GET /aiops/diagnosis/{session_id}?limit=20` |
| langgraph-checkpoint-redis | `.venv/lib/python3.13/site-packages/langgraph/checkpoint/redis/` | 通过 ensurepip + pip install 安装 |

## 6. 设计问题与改进思路（实施后更新）

### 6.1 会话存储切换（已实施）

通过 `config.redis_url` 在 RAG Agent、AIOps Agent、DiagnosisStore 三处统一判断 Redis/fallback：
- 配置 `REDIS_URL` → `RedisSaver.from_conn_string(redis_url)` + Redis 客户端
- 未配置 → `MemorySaver()` + 文件存储
- 暂未做完整抽象层。三处判断逻辑一致，待多实例部署需求出现后再统一为 `CheckpointStore` 接口

### 6.2 Token 计数裁剪（已实施，方案 B）

- `trim_messages_middleware` 替换为 `trim_messages_by_tokens()`，使用 tiktoken `cl100k_base` 编码
- 保留首条 SystemMessage + 从新到旧裁剪到 `context_max_tokens`（默认 8000）
- 接入 `query()` / `query_with_trace()` / `query_stream()` 三个方法
- 裁剪策略通过 `context_trimming_strategy` 配置：`token_count` / `none`
- **已知局限**：裁剪在输入层进行，不修改 checkpointer 中存储的消息。对单轮场景完全有效；多轮长会话场景需未来 langgraph 版本升级后通过 middleware 在 checkpointer 层裁剪
- `SummarizationMiddleware` 预留为未来选项（需 langgraph 版本升级后支持 `middlewares` 参数）

### 6.3 历史查询（已部分实施）

- `get_session_history()` 保持现有实现（从 checkpointer 读取），未做重大修改
- 新增 `GET /api/aiops/diagnosis/{session_id}` 独立于 checkpoint 的诊断历史查询
- `DiagnosisStore.list_by_session()` 支持 Redis `SCAN` 和文件 glob 两种后端

### 6.4 诊断与会话分离（已实施）

- `DiagnosisStore` 独立存储诊断记录（input, plan, past_steps, response），不混入 checkpoint
- AIOps `execute()` 完成后自动保存，失败不影响主流程
- 诊断记录包含 timestamp 便于按时间查询
- past_steps 截断至 200 字符、response 截断至 5000 字符防止存储膨胀
- Redis 记录 7 天 TTL 自动过期
