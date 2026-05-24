# Agent 评估：工具调用准确率与目标达成率

## 1. 功能和目的

当前评估体系（[06-ragas-evaluation.md](06-ragas-evaluation.md) 和 [08-rag-evaluation-enhancement.md](08-rag-evaluation-enhancement.md)）仅评估检索质量（contexts），不评估 Agent 的工具调用行为。本模块补充 Agent 层面的评估：

- **Tool Call Accuracy**：Agent 是否调用了正确的工具（及正确的参数）
- **Agent Goal Accuracy**：Agent 是否成功达成了用户的目标（如正确诊断出根因）

该模块与检索评估是互补关系——检索评估衡量"找得对不对"，Agent 评估衡量"做得对不对"。

核心应用场景：当 Enhanced RAG 的检索指标优于 Basic RAG，但 Agent 整体诊断准确率没有相应提升时，说明瓶颈不在检索而在工具调用或推理链路，需要针对性地优化 Prompt 或工具描述。

## 2. 抽象实现思路

### 整体架构

```
tests/evaluation/
├── agent_testset.py          # 新增：Agent Trace 评测数据集
├── evaluate_agent.py          # 新增：Agent 评估主脚本
└── metrics/
    ├── tool_call_accuracy.py  # 新增：Tool Call Accuracy 计算
    └── goal_accuracy.py       # 新增：Agent Goal Accuracy（LLM Judge）
```

### Tool Call Accuracy

对每条测试数据，定义一个期望的工具调用序列。比较 Agent 实际调用与期望调用的匹配程度：

```python
{
    "scenario": "data-sync-service CPU 告警排查",
    "input": "data-sync-service 服务 CPU 使用率超过 80%，请排查原因",
    "expected_tools": [
        {"name": "query_cpu_metrics", "args": {"service_name": "data-sync-service"}},
        {"name": "search_log", "args": {"query": "ERROR"}},
    ],
    "expected_conclusion_contains": ["死循环", "流量高峰"],
    "forbidden_tools": ["restart_service"],  # 不应调用的危险工具
}
```

Tool Call Accuracy 子指标：
- **Exact Match**：实际调用工具集合 == 期望调用工具集合（忽略参数细节）
- **Precision**：调用对的工具数 / 实际调用工具总数
- **Recall**：调用对的工具数 / 期望调用工具总数
- 参数匹配可选（对于 mock 工具，参数有固定值，可以精确比对）

### Agent Goal Accuracy

使用 LLM Judge 评估 Agent 的最终输出是否达成了用户目标：

```python
GOAL_ACCURACY_PROMPT = """你是一个运维专家评估裁判。请判断以下 Agent 的诊断结论是否达到了用户的目标。

用户问题：{user_question}
期望结论要点：{expected_conclusion_contains}
Agent 实际输出：{agent_output}

评分标准：
- 2 分：完全达成，覆盖所有期望要点，诊断逻辑正确
- 1 分：部分达成，覆盖了部分要点但有遗漏或错误
- 0 分：未达成，诊断方向错误或结论与期望不符

请给出评分（0/1/2）和简要理由。
"""
```

注意：Agent Goal Accuracy 依赖 LLM Judge，需要确保 Judge 的 API 和模型可配置（与 [08-rag-evaluation-enhancement.md](08-rag-evaluation-enhancement.md) 中的 Judge 配置共用）。

### Agent Trace 数据集构建

这是本模块最核心、最耗时的部分。每条测试数据包含：

| 字段 | 说明 | 必填 |
|------|------|------|
| `scenario` | 场景名称 | 是 |
| `input` | 用户输入（模拟告警/提问） | 是 |
| `expected_tools` | 期望调用的工具列表 | 是 |
| `expected_conclusion_contains` | 期望结论应包含的要点 | 是 |
| `forbidden_tools` | 不应调用的危险工具 | 否 |
| `reference_docs` | 相关的知识库文档 | 否 |

建议初始构建 10-15 条高质量测试数据，覆盖以下场景：
1. CPU 告警排查（单工具路径）
2. 内存告警排查（单工具路径）
3. 服务不可用排查（多工具联合）
4. 慢响应排查（需跨文档知识综合）
5. 误报/噪声输入（Agent 应该能识别不需要调工具）
6. 多步推理场景（需要先查日志再查监控）

## 3. 具体实现流程

### Step 1：设计 Agent Trace 数据集

新增文件：[tests/evaluation/agent_testset.py](tests/evaluation/agent_testset.py)

手工构建 10-15 条测试数据。由于当前 MCP 工具返回的是 mock 数据，`expected_tools` 应基于 mock 数据的行为来定义。例如，对于 "data-sync-service CPU 告警"：
- Mock 的 `query_cpu_metrics("data-sync-service")` 返回模拟的 CPU 增长数据
- Mock 的 `search_log("topic-001", ..., query="error")` 返回模拟的错误日志
- 因此期望 Agent 调用这两个工具来诊断

### Step 2：实现 Tool Call Accuracy 计算

新增文件：[tests/evaluation/metrics/tool_call_accuracy.py](tests/evaluation/metrics/tool_call_accuracy.py)

```python
def compute_tool_call_accuracy(
    actual_calls: list[dict],      # Agent 实际调用的工具
    expected_calls: list[dict],    # 期望调用的工具
) -> dict:
    """计算 Tool Call Accuracy（Exact Match, Precision, Recall）"""
    actual_names = {call["name"] for call in actual_calls}
    expected_names = {call["name"] for call in expected_calls}

    exact_match = actual_names == expected_names
    precision = len(actual_names & expected_names) / len(actual_names) if actual_names else 0.0
    recall = len(actual_names & expected_names) / len(expected_names) if expected_names else 1.0

    return {
        "exact_match": exact_match,
        "tool_precision": precision,
        "tool_recall": recall,
    }
```

### Step 3：实现 Agent Goal Accuracy（LLM Judge）

新增文件：[tests/evaluation/metrics/goal_accuracy.py](tests/evaluation/metrics/goal_accuracy.py)

复用 [08-rag-evaluation-enhancement.md](08-rag-evaluation-enhancement.md) 中可配置的 Judge LLM，对每个测试用例的 Agent 输出进行评分（0/1/2），汇总为平均分。

### Step 4：实现 Agent 评估主脚本

新增文件：[tests/evaluation/evaluate_agent.py](tests/evaluation/evaluate_agent.py)

核心流程：
1. 加载 `agent_testset.py` 数据集
2. 初始化 `RagAgentService`（同步模式，非流式）
3. 对每条测试数据，调用 `agent.query(input, session_id=f"eval_agent_{i}")`
4. 从 Agent 执行过程中捕获工具调用序列（通过 `agent.astream` 的 tool_call 事件）
5. 计算 Tool Call Accuracy
6. 用 LLM Judge 评估 Agent Goal Accuracy
7. 输出 JSON + CSV 报告

捕获工具调用的关键：

```python
async for event in agent.agent.astream(messages, config=config, stream_mode="messages"):
    if isinstance(event, AIMessage) and event.tool_calls:
        tool_calls.extend(event.tool_calls)
```

### Step 5：与现有评估体系的关系

Agent 评估独立运行，不依赖 RAG 评估：

```bash
# 检索评估（已有）
RAG_MODE=enhanced python -m tests.evaluation.evaluate_rag

# Agent 评估（新增）
RAG_MODE=enhanced python -m tests.evaluation.evaluate_agent
```

两者互补：检索评估衡量检索质量，Agent 评估衡量端到端任务完成质量。

### 局限性与注意点

1. **Mock 数据的局限性**：当前 MCP 工具返回 mock 数据，Agent 行为可预测但不反映真实场景。连接真实 API 后，`expected_tools` 需要重新校准
2. **Tool Call 顺序敏感度**：当前设计只比对工具集合（名称集合），不比对调用顺序。如果需要顺序匹配，可以扩展为序列比对
3. **参数匹配复杂度**：mock 工具参数简单，易于比对。真实场景中参数可能为时间范围、正则表达式等，精确比对意义有限
4. **LLM Judge 一致性**：Goal Accuracy 的 0/1/2 评分依赖 LLM Judge 的稳定性。建议每个测试用例跑 3 次取平均值

## 4. 当前实现进度

### 已完成

- [x] Agent Trace 数据集 `agent_testset.py` — 12 条数据，覆盖 6 类场景（单工具×3 + 多工具联合×3 + 跨文档知识×2 + 误报/噪声×2 + 多步推理×1 + 模糊输入×1）
- [x] Tool Call Accuracy 计算函数 — `metrics/tool_call_accuracy.py`，Exact Match + Precision + Recall
- [x] Agent Goal Accuracy LLM Judge — `metrics/goal_accuracy.py`，0/1/2 评分 + 3 次取平均，复用 eval_judge_* 配置
- [x] Tool Call 捕获逻辑 — `RagAgentService.query_with_trace()` 新增方法，基于 ainvoke + 遍历消息历史提取 tool_calls
- [x] 评估主脚本 `evaluate_agent.py` — 完整异步评估流程，支持 `--skip-goal`、`--judge-model`、`--output-format`
- [x] JSON + CSV 双输出 — `_flatten_scores()` + `_save_csv()` 展平评估结果为单行 CSV
- [x] `metrics/__init__.py` — 导出新指标函数

### 尚未实现

无。Plan 10 所有计划项已完成。

### 实施决策

- **不接入 interrupt_before**：评估不需要真正的 HITL 断点，trace 捕获用 ainvoke 消息历史即可
- **参数不精确比对**：只比对工具名称集合（set），不比对参数和调用顺序
- **Judge 独立可配置**：复用现有 `eval_judge_model/temperature/api_base/api_key`，不绑定线上模型
- **每个 case 独立 session**：`session_id=f"agent_eval_{i}"` 避免跨 case 干扰
- **标注 mock 数据源**：评估报告显式标注 `"data_source": "mock"`，防止误读

## 5. Evidence

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| Agent 评估数据集 | [tests/evaluation/agent_testset.py](tests/evaluation/agent_testset.py) | 12 条数据，v1.0.0，6 类场景，validate_agent_testset() 校验通过 |
| Tool Call Accuracy | [tests/evaluation/metrics/tool_call_accuracy.py](tests/evaluation/metrics/tool_call_accuracy.py) | Exact Match + Precision + Recall，纯集合运算 |
| Goal Accuracy LLM Judge | [tests/evaluation/metrics/goal_accuracy.py](tests/evaluation/metrics/goal_accuracy.py) | 0/1/2 三级评分，默认 3 次取平均，复用 eval_judge_* 配置 |
| Agent 评估主脚本 | [tests/evaluation/evaluate_agent.py](tests/evaluation/evaluate_agent.py) | 异步执行 + 捕获 trace + JSON/CSV 双输出 |
| query_with_trace 方法 | [app/services/rag_agent_service.py:235](app/services/rag_agent_service.py#L235) | 新增方法，返回 {answer, tool_calls} 结构化 trace |
| 指标导出更新 | [tests/evaluation/metrics/__init__.py](tests/evaluation/metrics/__init__.py) | 导出 compute_tool_call_accuracy + goal_accuracy 函数 |
| 工具清单 | [mcp_servers/cls_server.py](mcp_servers/cls_server.py) + [mcp_servers/monitor_server.py](mcp_servers/monitor_server.py) | 5 个 CLS 工具 + 2 个监控工具，全部返回 mock 数据 |
| 本地工具 | [app/services/rag_agent_service.py:98](app/services/rag_agent_service.py#L98) | 2 个：retrieve_knowledge, get_current_time |

## 6. 设计问题与改进思路（实施后更新）

### 6.1 Trace 捕获（已实施）

通过新增 `RagAgentService.query_with_trace()` 方法实现结构化 trace 捕获。
- 使用 `ainvoke` + 遍历消息历史中的 `AIMessage.tool_calls` 提取所有工具调用。
- 返回结构化 dict：`{“answer”: str, “tool_calls”: [{“name”: str, “args”: dict}]}`。
- 不修改现有 `query()` / `query_stream()` 接口，作为独立方法添加。
- 当前 trace 仅包含 tool_name + tool_args，tool_result/timestamp/node 暂未纳入（待后续扩展）。

### 6.2 指标分层（已实施两层）

已实施的指标分层：
- **第一层**：Tool Call Accuracy（Exact Match + Precision + Recall），只比对工具名称集合。
- **第二层**：Agent Goal Accuracy（LLM Judge 0/1/2 评分），3 次取平均。
- **未纳入**：工具顺序和参数合理性评估 — 对 mock 工具意义有限，预留到真实工具接入后。

### 6.3 Mock 数据标注（已实施）

- 评估报告中显式标注 `”data_source”: “mock”`，避免误读 mock 高分。
- 12 条测试数据的 expected_tools 基于当前 mock 工具行为设计。
- 真实工具接入后，需要重新校准 expected_tools 和 expected_conclusion。

### 6.4 失败模式覆盖（已实施）

数据集 12 条中已包含：
- 误报/噪声输入 ×2（简单问候 + 无关闲聊），expected_tools=[] → 验证 Agent 不应盲目调工具
- 模糊输入 ×1（”修一下”），expected_tools=[] → 验证 Agent 应反问而非瞎操作
- `forbidden_tools` 字段标注了不应调用的工具（如知识查询场景禁止调用 MCP 工具）
- 评估脚本中异常 case 记录 error 字段继续下一个，不阻塞整体流程
