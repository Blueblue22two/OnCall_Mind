# SuperBizAgent 优化实施方案

> 基于 `优化思路.md` 的全面分析，从可行性、实现难度、优先级三个维度评估，生成分阶段实施路线图。

---

## 📊 实施进度总览

| 阶段 | 状态 | 完成时间 | 说明 |
|------|------|----------|------|
| **P0** | ✅ 已完成 | 2026-05-31 | LLM 配置统一 + Fallback + RAG 参数优化 |
| **P1** | ✅ 已完成 | 2026-05-31 | Agent Eval / Error Handler / Redis / Metrics |
| **P2** | ⬜ 待开始 | — | HITL / 告警触发器 / Memory / RAG 并行化 |
| **P3** | ⬜ 待开始 | — | Supervisor / 自动修复 / 长期记忆 / RAG 深度优化 |

### P0 详细变更记录

| 编号 | 优化项 | 状态 | 变更文件 |
|------|--------|------|----------|
| 1.1 | LLM 配置统一 | ✅ | `config.py`, `planner.py`, `executor.py`, `replanner.py`, `rag_agent_service.py` |
| 1.2 | LLM Fallback 链路 | ✅ | `app/core/llm_factory.py`（重写）, `config.py`, 上述 4 个 agent 文件 |
| 1.3 | RAG 参数优化 | ✅ | `config.py`, `vector_search_service.py`, `.env` |

**配置变更摘要**（`.env` 新增字段）：
- `LLM_TEMPERATURE=0.0`, `LLM_CHAT_TEMPERATURE=0.7`, `LLM_TIMEOUT=60`, `LLM_MAX_RETRIES=2`
- `LLM_FALLBACK_MODEL=qwen-plus`
- `MILVUS_NPROBE=32`（10→32）
- `RERANK_COARSE_TOP_K=10`（20→10）

### P1 详细变更记录

| 编号 | 优化项 | 状态 | 变更文件 |
|------|--------|------|----------|
| 2.1 | Agent Eval 体系 | ✅ | 新增 `app/services/trace_store.py`；`planner.py`/`executor.py`/`replanner.py`/`aiops_service.py` 嵌入 trace 采集 |
| 2.2 | Error Handler/Retry | ✅ | `state.py`（error_count/max_errors/last_error）；`executor.py`（单步重试 max 2 次）；`aiops_service.py`（error_handler 节点 + 条件边） |
| 2.3 | Redis 默认启用 | ✅ | `vector-database.yml`（Redis 容器）；`.env`（REDIS_URL）；`aiops_service.py`/`rag_agent_service.py`（Redis→MemorySaver 优雅回退） |
| 2.4 | LLM Metrics | ✅ | 新增 `app/core/metrics.py`；`llm_factory.py` 嵌入 metrics 采集；`health.py` 新增 `/metrics` 端点 |

**P1 新增/变更文件**：
- 新增：`app/services/trace_store.py`（结构化 Trace 持久化）
- 新增：`app/core/metrics.py`（Prometheus 指标）
- 新增：`prometheus_client` 依赖
- 变更：`state.py`（trace_id + error 字段）
- 变更：`planner.py`、`executor.py`、`replanner.py`（trace + retry）
- 变更：`aiops_service.py`（trace_id 传递 + error_handler 节点 + Redis 回退）
- 变更：`llm_factory.py`（metrics 采集）
- 变更：`rag_agent_service.py`（Redis 回退）
- 变更：`health.py`（/metrics 端点）
- 变更：`vector-database.yml`（Redis 容器）
- 变更：`.env`（REDIS_URL=redis://localhost:6379）

---

## 评估维度说明

| 维度 | 定义 |
|------|------|
| **可行性** | 技术栈是否原生支持、是否需要引入新的基础设施、与现有架构的兼容程度 |
| **实现难度** | 预估工作量、涉及模块数量、是否需要跨层改动 |
| **优先级** | P0（立即）/ P1（短期）/ P2（中期）/ P3（长期），综合业务价值与实施成本 |

---

## 一、P0 — 立即实施 ✅ 已完成（2026-05-31）

### 1.1 LLM 配置统一与热更新 ✅（4.0-1, 4.0-4）

**现状**：`temperature` 等参数硬编码在各节点中（planner.py:129 `temperature=0`，executor.py:57 `temperature=0`，rag_agent_service.py:102 `temperature=0.7`），且缺少 `timeout` / `max_retries`。

**改进方案**：
- 在 `config.py` 中新增 `llm_temperature`、`llm_timeout`、`llm_max_retries` 字段
- 所有 LLM 实例化统一从 `config` 读取参数
- 为 `ChatQwen` 添加 `timeout`（默认 60s）和 `max_retries`（默认 2）

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 高 | ⭐ 低 | `config.py`, `planner.py`, `executor.py`, `replanner.py`, `rag_agent_service.py` |

**具体改动**：
```
config.py:     新增 llm_temperature=0.0, llm_timeout=60, llm_max_retries=2
planner.py:    ChatQwen(temperature=config.llm_temperature, timeout=config.llm_timeout, max_retries=config.llm_max_retries, ...)
executor.py:   同上
replanner.py:  同上（2处实例化）
rag_agent_service.py: 同上
```

---

### 1.2 LLM Fallback 链路 ✅（4.0-2, 4.0-5）

**现状**：单一依赖 `qwen-max`，存在单点风险。

**改进方案**：
- 在 `config.py` 新增 `llm_fallback_chain` 配置（如 `["qwen-max", "qwen-plus"]`）
- 封装 `LLMFactory` 或带 fallback 的 wrapper，自动降级
- 利用 LangChain 的 `with_fallbacks()` 机制

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 高 | ⭐⭐ 中 | `config.py`, 新增 `app/core/llm_factory.py`, 所有 LLM 实例化处 |

**具体改动**：
```python
# app/core/llm_factory.py (新增)
from langchain_qwq import ChatQwen
from app.config import config

def create_chat_qwen(temperature=None, streaming=False):
    """创建带 fallback 链的 ChatQwen 实例"""
    if temperature is None:
        temperature = config.llm_temperature
    
    primary = ChatQwen(
        model=config.dashscope_model,  # qwen-max
        api_key=config.dashscope_api_key,
        api_base=config.dashscope_api_base,
        temperature=temperature,
        timeout=config.llm_timeout,
        max_retries=config.llm_max_retries,
        streaming=streaming,
    )
    
    if config.llm_fallback_model:
        fallback = ChatQwen(
            model=config.llm_fallback_model,  # qwen-plus
            api_key=config.dashscope_api_key,
            api_base=config.dashscope_api_base,
            temperature=temperature,
            timeout=config.llm_timeout,
            max_retries=config.llm_max_retries,
            streaming=streaming,
        )
        return primary.with_fallbacks([fallback])
    
    return primary
```

---

### 1.3 RAG 检索参数快速优化 ✅（7.0-3, 7.0-5）

**现状**：`nprobe=10`（`vector_search_service.py:72`），`RERANK_COARSE_TOP_K=20`。

**改进方案**：
- `nprobe` 从 10 → 32（提升召回率，延迟影响可控）
- `RERANK_COARSE_TOP_K` 从 20 → 10（精排耗时减半）
- 在 `config.py` 中暴露为可配置参数

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 高 | ⭐ 低 | `config.py`, `vector_search_service.py`, `.env` |

---

## 二、P1 — 短期实施 ✅ 已完成（2026-05-31）

### 2.1 Agent Eval 体系建设 ✅（9.0）

**现状**：已有基础评估框架（`tests/evaluation/`），`query_with_trace()` 方法捕获 tool_calls。但 trace 信息不够完整。

**改进方案**：
- **Agent Trace 增强**：在 AIOps 的三个节点（planner/executor/replanner）中结构化记录：
  - 计划内容与变更
  - 每次工具调用的名称、参数、耗时、返回结果
  - 失败原因、重试次数
  - Token 消耗量（通过 LLM response 的 `usage_metadata`）
  - 最终状态
- **Trace 持久化**：Redis 存储完整 trace（利用现有 `DiagnosisStore` 扩展）
- **指标看板**：任务成功率、工具成功率、平均步数、超时率、人工介入率

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 高 | ⭐⭐ 中 | `executor.py`, `planner.py`, `replanner.py`, `diagnosis_store.py`, 新增 `app/services/trace_store.py` |

**关键设计**：
```python
# Trace 数据结构
{
    "trace_id": str,
    "session_id": str,
    "timestamp": str,
    "input": str,
    "nodes": [
        {
            "node": "planner",
            "plan": [...],
            "experience_docs_used": bool,
            "token_usage": {"input": int, "output": int},
            "duration_ms": int,
        },
        {
            "node": "executor",  # 每个步骤一条
            "step_index": int,
            "task": str,
            "tool_calls": [
                {"name": str, "args": dict, "result": str, "duration_ms": int, "success": bool}
            ],
            "token_usage": {...},
        },
        {
            "node": "replanner",
            "decision": "continue|replan|respond",
            "new_steps": [...],
            "token_usage": {...},
        }
    ],
    "final_response": str,
    "total_duration_ms": int,
    "total_tokens": {"input": int, "output": int},
    "status": "success|error|timeout"
}
```

---

### 2.2 Error Handler / Retry / Fallback 节点 ✅（2.0）

**现状**：AIOps 工作流中的错误处理在各节点内部 try-catch，缺乏统一的错误恢复机制。executor 失败后直接跳过该步骤（`executor.py:111-116`），没有重试逻辑。

**改进方案**：
- 在 LangGraph 图中增加 `error_handler` 节点，统一处理异常
- Executor 增加单步重试（max 2 次）
- 在 `should_continue` 条件边中增加错误计数判断
- 错误超过阈值时自动 fallback 到生成响应

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 高 | ⭐⭐ 中 | `aiops_service.py`, `executor.py`, `state.py`, 新增 `app/agent/aiops/error_handler.py` |

**状态扩展**（`state.py`）：
```python
class PlanExecuteState(TypedDict, total=False):
    # ... 现有字段 ...
    error_count: int              # 累计错误次数
    max_errors: int               # 错误上限（默认 3）
    last_error: str               # 最近一次错误信息
```

---

### 2.3 会话超时清理与 Redis 默认启用 ✅（5.0-2, 6.0-1）

**现状**：`MemorySaver` 无自动过期；Redis 可选但 Docker Compose 中未包含。

**改进方案**：
- 在 `vector-database.yml` 中增加 Redis 容器
- `.env` 中预设 `REDIS_URL=redis://localhost:6379`
- 使用 `RedisSaver` 的 TTL 特性自动清理过期会话
- 为 `MemorySaver` 模式增加定时清理逻辑（作为 fallback）

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 高 | ⭐ 低 | `vector-database.yml`, `.env`, `config.py`, `rag_agent_service.py` |

---

### 2.4 LLM API Metrics 上报 ✅（4.0-6）

**现状**：无 LLM 调用的可观测性指标。

**改进方案**：
- 在 `LLMFactory` 或 wrapper 中嵌入 metrics 采集
- 记录：调用延迟（P50/P95/P99）、成功率、重试次数
- 使用 `prometheus_client` 库暴露 `/metrics` 端点
- 可选：接入日志结构化输出（保留现有 loguru）

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 高 | ⭐⭐ 中 | `main.py`, 新增 `app/core/metrics.py`, `llm_factory.py` |

---

## 三、P2 — 中期实施（2-4 周，架构增强）

### 3.1 Human-in-the-Loop（HITL）（1.0）

**现状**：ReAct Agent 完全自主运行，高风险操作无人工确认机制。`state.py` 已预留 `pending_approval` / `pending_tool_name` / `pending_tool_args` 字段。

**改进方案**：
- RAG Agent 使用 `create_agent(interrupt_before=["tools"], ...)` 在工具调用前暂停
- WebSocket 端点：`/ws/chat/{session_id}` 用于双向通信
- 审批流程：Agent 暂停 → 推送待审批工具调用到前端 → 用户确认/拒绝 → `graph.resume()` 继续
- 保持现有 SSE 接口不变，WebSocket 作为 HITL 模式的补充

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 中 | ⭐⭐⭐ 高 | `rag_agent_service.py`, `chat.py` (新增 WebSocket 路由), `state.py`, 前端 |

**风险点**：
- WebSocket 与 SSE 双协议共存增加复杂度
- `graph.resume()` 的状态管理需要仔细处理
- 前端需要适配审批交互

**简化方案**（推荐先实施）：
- 仅对 `AIOps Agent` 的 executor 节点增加 HITL（已在 state 中预留字段）
- 利用 LangGraph 的 `interrupt_before` 在特定高风险工具前暂停
- 暂时使用轮询机制代替 WebSocket（客户端轮询 `/aiops/approval/{session_id}`）

---

### 3.2 告警自动触发器（3.0-1）

**现状**：依赖用户主动发起诊断请求（`POST /api/aiops`）。

**改进方案**：
- **Webhook 监听**：新增 `/api/aiops/webhook` 端点，接收外部告警系统（如 Prometheus AlertManager）的推送
- **定时巡检**：使用 APScheduler 或 asyncio 定时任务，周期性检查监控指标
- 触发后自动调用 `aiops_service.diagnose()`
- 诊断结果通过 webhook 回调或消息通知推送

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 中 | ⭐⭐ 中 | `aiops.py` (新增 webhook 路由), `main.py` (scheduler 生命周期), 新增 `app/services/alert_trigger.py` |

**架构设计**：
```
AlertManager/Webhook → POST /api/aiops/webhook → AlertTriggerService
    → 解析告警 → 构造诊断输入 → AIOpsService.diagnose()
    → 持久化结果 → 可选：回调通知
```

---

### 3.3 Memory 自适应策略（6.0-2, 6.0-3, 6.0-4）

**现状**：`trim_messages_by_tokens()` 使用简单的 token 计数裁剪，只保留首条 SystemMessage。

**改进方案**：
- **优先级裁剪**：按消息类型设定保留优先级——工具调用结果 > 用户问题 > 助手回复
- **自适应策略**：
  - < 20 轮：滑动窗口（当前逻辑）
  - 20-50 轮：在裁剪前用 LLM 生成早期对话摘要
  - > 50 轮：摘要 + 最近 10 轮完整对话
- 摘要生成使用轻量模型（`qwen-plus`）降低开销

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 中 | ⭐⭐ 中 | `rag_agent_service.py`, 新增 `app/services/memory_manager.py` |

**实现要点**：
```python
class AdaptiveMemoryManager:
    def __init__(self):
        self.summary_model = create_chat_qwen(temperature=0.3)  # 使用 fallback 模型
    
    async def optimize_context(self, messages, turn_count):
        if turn_count < 20:
            return trim_messages_by_tokens(messages)  # 滑动窗口
        elif turn_count < 50:
            summary = await self._summarize_early(messages[:turn_count//2])
            recent = messages[turn_count//2:]
            return [SystemMessage(summary)] + recent
        else:
            summary = await self._summarize_early(messages[:turn_count-10])
            return [SystemMessage(summary)] + messages[-10:]
```

---

### 3.4 Redis 序列化压缩（5.0-1）

**现状**：LangGraph 状态快照可能较大。

**改进方案**：
- 在写入 Redis 前使用 zlib 压缩状态快照
- 自定义 `RedisSaver` 的序列化/反序列化钩子

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 高 | ⭐ 低 | `rag_agent_service.py`, `aiops_service.py` |

---

### 3.5 RAG 异步并行化（7.0-6）

**现状**：Enhanced RAG Pipeline 中 Stage 0（Embedding）和 Stage 1（Rewrite）串行执行。

**改进方案**：
- Embedding 使用原始 query，Rewrite 同时并行执行
- Rewrite 完成后用改写文本做第二次 Embedding
- 两次 Embedding 结果合并（如取并集或加权融合）

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 中 | ⭐⭐ 中 | `app/retriever/enhanced.py` |

---

### 3.6 Judge 模型解耦（4.0-3）

**现状**：`config.py` 已有 `eval_judge_*` 配置字段（`eval_judge_model`, `eval_judge_api_base`, `eval_judge_api_key`），但 Judge 实际调用路径中可能未全部使用。

**改进方案**：
- 确认所有 Judge 调用统一从 `config.eval_judge_*` 读取
- 支持配置多个 Judge 后端，按场景选择

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 高 | ⭐ 低 | 评估模块 (已大部分完成，需验证) |

---

## 四、P3 — 长期演进（1-2 月，战略性优化）

### 4.1 Supervisor Agent 路由（8.0-1, 8.0-3）

**现状**：RAG Agent 和 AIOps Agent 完全独立，用户需手动选择接口。

**改进方案**：
- 新增 Supervisor Agent 节点（轻量分类器），判断用户意图：
  - 知识问答类 → RAG Agent
  - 故障诊断类 → AIOps Agent
- RAG Agent 增加 Adaptive Workflow：简单问题直接 ReAct，复杂问题切换 Plan 模式
- 统一入口：`POST /api/agent` 自动路由

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 中 | ⭐⭐⭐ 高 | 新增 `app/agent/supervisor.py`, `app/services/supervisor_service.py`, `app/api/agent.py` |

**架构设计**：
```
POST /api/agent
    → SupervisorAgent (分类)
        → "knowledge" → RagAgentService (ReAct)
        → "diagnosis" → AIOpsService (Plan-Execute-Replan)
        → "complex_qa" → RagAgentService (Plan mode via Adaptive Workflow)
```

---

### 4.2 诊断结果闭环（自动修复）（3.0-2）

**现状**：诊断完成后仅生成报告，不执行修复动作。

**改进方案**：
- 诊断报告结构化提取可操作的修复步骤
- 结合 HITL 审批流程，高风险操作需人工确认
- 低风险操作（如查询补充信息）自动执行

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ⚠️ 低 | ⭐⭐⭐ 高 | `replanner.py`, `executor.py`, 新增 `app/services/remediation.py` |

**风险警告**：自动修复存在误操作风险（如错误重启服务）。必须：
1. 严格的操作分级（read-only / low-risk / high-risk）
2. HITL 审批流程就位
3. 操作回滚能力

---

### 4.3 长期记忆与 Reflection 机制（6.0 optional）

**改进方案**：
- **向量化长期记忆**：将历史对话摘要和诊断结果向量化存入 Milvus
- **Reflection 机制**：对话/诊断结束后，LLM 反思并提取关键经验
- **经验检索**：新请求先检索历史经验，注入 Planner/RAG 的上下文

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 中 | ⭐⭐⭐ 高 | 新增 `app/services/long_term_memory.py`, `app/services/reflection.py`, `planner.py` |

---

### 4.4 RAG 深度优化（7.0-1, 7.0-2, 7.0-4, 7.0-7, 7.0-8, 7.0-9）

**Query 路由（7.0-1）**：
- 查询分类器：关键词查询 / 语义查询 / 混合查询
- 根据分类动态选择 retrieval 策略

**多路检索扩展（7.0-2）**：
- 增加 metadata 过滤通道（如按告警类型筛选）
- 在 `retrieve_knowledge` 工具中支持 filter 参数

**HNSW 索引（7.0-4）**：
- 评估当前 IVF_FLAT 切换到 HNSW 的收益
- 注意：HNSW 内存占用更高，适合 < 100K 向量的场景

**Cross-Encoder 量化（7.0-7）**：
- 使用 ONNX Runtime + INT8 量化
- 推理耗时预期降低 50%+

**Query Rewrite 增强（7.0-8, 7.0-9）**：
- 当前 `QUERY_PREPROCESSOR_TYPE=none`，启用 rewrite 模式
- 口语→规范术语映射 + 关键词补全

| 子项 | 可行性 | 难度 | ROI |
|------|--------|------|-----|
| Query 路由 | ✅ 高 | ⭐⭐ 中 | 高 |
| Metadata 过滤 | ✅ 高 | ⭐ 低 | 中 |
| HNSW 索引 | ✅ 中 | ⭐⭐ 中 | 低-中（取决于数据规模） |
| Cross-Encoder 量化 | ✅ 中 | ⭐⭐ 中 | 中 |
| Query Rewrite | ✅ 高 | ⭐ 低（功能已有，启用即可） | 高 |

**推荐顺序**：先启用 Query Rewrite（改动最小），再实施 Query 路由和 Metadata 过滤。

---

### 4.5 Agent 间知识共享（8.0-2）

**现状**：AIOps Planner 调用 `retrieve_knowledge`，但 RAG Agent 和 AIOps Agent 之间没有数据交互。

**改进方案**：
- 共享检索缓存层：短时间内相同 query 不重复检索
- 考虑用 Redis 缓存最近的检索结果（TTL 5 分钟）

| 可行性 | 难度 | 涉及文件 |
|--------|------|----------|
| ✅ 高 | ⭐ 低 | `knowledge_tool.py`, 新增缓存装饰器 |

---

## 五、实施路线图总览

```
Week 1 (P0) ✅ 已完成             Week 1 (P1) ✅ 已完成             Week 3-5 (P2)                    Week 6-12 (P3)
┌─────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌──────────────────────────────┐
│ 1.1 LLM 配置统一 ✅ │  │ 2.1 Agent Eval 体系 ✅  │  │ 3.1 HITL (简化方案)     │  │ 4.1 Supervisor Agent 路由    │
│ 1.2 LLM Fallback ✅ │  │ 2.2 Error Handler  ✅  │  │ 3.2 告警自动触发器      │  │ 4.2 诊断结果闭环（自动修复）   │
│ 1.3 RAG 参数优化 ✅ │  │ 2.3 Redis 默认启用  ✅  │  │ 3.3 Memory 自适应策略   │  │ 4.3 长期记忆 + Reflection    │
│                     │  │ 2.4 LLM Metrics    ✅  │  │ 3.4 Redis 序列化压缩    │  │ 4.4 RAG 深度优化              │
│                     │  │                         │  │ 3.5 RAG 异步并行化      │  │ 4.5 Agent 间知识共享          │
│                     │  │                         │  │ 3.6 Judge 模型解耦      │  │                              │
├─────────────────────┤  ├─────────────────────────┤  ├─────────────────────────┤  ├──────────────────────────────┤
│ 风险: 极低           │  │ 风险: 低                 │  │ 风险: 中                 │  │ 风险: 中-高                   │
└─────────────────────┘  └─────────────────────────┘  └─────────────────────────┘  └──────────────────────────────┘
```

---

## 六、优先级排序总表

| 编号 | 优化项 | 优先级 | 可行性 | 难度 | 预估工时 | 状态 | 来源 |
|------|--------|--------|--------|------|----------|------|------|
| 1.1 | LLM 配置统一与热更新 | **P0** | 高 | 低 | 0.5d | ✅ | 4.0-1,4 |
| 1.2 | LLM Fallback 链路 | **P0** | 高 | 中 | 1d | ✅ | 4.0-2,5 |
| 1.3 | RAG 检索参数快速优化 | **P0** | 高 | 低 | 0.5d | ✅ | 7.0-3,5 |
| 2.1 | Agent Eval 体系建设 | **P1** | 高 | 中 | 3d | ✅ | 9.0 |
| 2.2 | Error Handler/Retry/Fallback | **P1** | 高 | 中 | 2d | ✅ | 2.0 |
| 2.3 | 会话超时 + Redis 默认启用 | **P1** | 高 | 低 | 1d | ✅ | 5.0-2, 6.0-1 |
| 2.4 | LLM API Metrics 上报 | **P1** | 高 | 中 | 1.5d | ✅ | 4.0-6 |
| 3.1 | HITL（简化方案） | **P2** | 中 | 高 | 3d | ⬜ | 1.0 |
| 3.2 | 告警自动触发器 | **P2** | 中 | 中 | 2d | ⬜ | 3.0-1 |
| 3.3 | Memory 自适应策略 | **P2** | 中 | 中 | 2d | ⬜ | 6.0-2,3,4 |
| 3.4 | Redis 序列化压缩 | **P2** | 高 | 低 | 0.5d | ⬜ | 5.0-1 |
| 3.5 | RAG 异步并行化 | **P2** | 中 | 中 | 1.5d | ⬜ | 7.0-6 |
| 3.6 | Judge 模型解耦 | **P2** | 高 | 低 | 0.5d | ⬜ | 4.0-3 |
| 4.1 | Supervisor Agent 路由 | **P3** | 中 | 高 | 5d | ⬜ | 8.0-1,3 |
| 4.2 | 诊断结果闭环（自动修复） | **P3** | 低 | 高 | 5d+ | ⬜ | 3.0-2 |
| 4.3 | 长期记忆 + Reflection | **P3** | 中 | 高 | 5d | ⬜ | 6.0 opt |
| 4.4 | RAG 深度优化 | **P3** | 中-高 | 中-高 | 3-5d | ⬜ | 7.0-1,2,4,7,8,9 |
| 4.5 | Agent 间知识共享 | **P3** | 高 | 低 | 1d | ⬜ | 8.0-2 |

---

## 七、风险与依赖

| 风险 | 影响项 | 缓解措施 |
|------|--------|----------|
| LangGraph/LangChain 版本升级 API 变更 | 1.2, 2.2, 3.1 | 锁定版本，升级前在分支验证 |
| WebSocket 与 SSE 共存复杂度 | 3.1 | 先实施简化方案（轮询），再演进到 WebSocket |
| 自动修复误操作 | 4.2 | 严格操作分级 + HITL 审批 + 回滚能力 |
| Milvus 索引切换 (IVF_FLAT → HNSW) 数据丢失 | 4.4 | 重建索引前备份数据 |
| 模型 API 服务不可用 | 1.2, 2.4 | Fallback 链路 + 重试 + 降级策略 |
| Redis 引入增加运维复杂度 | 2.3, 3.4, 5.0 | Docker Compose 一键部署，保留 MemorySaver 回退 |
