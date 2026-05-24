"""Agent Trace 评估数据集

手工构建的 Agent 测试数据集，覆盖 6 类场景共 12 条测试数据。
每条数据包含用户输入、期望工具调用序列、期望结论要点等信息。

可用工具（基于当前 MCP/Mock 行为）：
  MCP CLS 工具    : search_log, search_topic_by_service_name, get_topic_info_by_name,
                    get_region_code_by_name, get_current_timestamp
  MCP Monitor 工具 : query_cpu_metrics, query_memory_metrics
  本地工具         : retrieve_knowledge, get_current_time

场景分类：
  - 单工具路径        : 3 条（CPU 告警排查、内存告警排查、时间查询）
  - 多工具联合排查    : 3 条（服务不可用排查、日志错误排查、综合监控排查）
  - 跨文档知识综合    : 2 条（CPU 问题排查知识、内存泄漏排查知识）
  - 误报/噪声输入     : 2 条（简单问候、无关闲聊）
  - 多步推理          : 1 条（先查 topic 再搜日志再查 CPU）
  - 模糊/信息不足输入 : 1 条（极简指令"修一下"）
"""

from typing import Any

# ---------------------------------------------------------------------------
# 数据集版本号
# ---------------------------------------------------------------------------
DATASET_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Agent 评估测试数据集（12 条）
# ---------------------------------------------------------------------------
AGENT_EVAL_DATASET: list[dict[str, Any]] = [
    # =============================================
    # 场景 1: 单工具路径（3 条）
    # =============================================
    {
        "scenario": "CPU 告警排查",
        "input": "data-sync-service 的 CPU 使用率告警了，帮我查一下 CPU 情况",
        "expected_tools": [{"name": "query_cpu_metrics"}],
        "expected_conclusion_contains": [
            "CPU 使用率从 10% 逐渐升到 95%",
            "触发了告警",
        ],
        "forbidden_tools": [],
        "reference_docs": [],
    },
    {
        "scenario": "内存告警排查",
        "input": "data-sync-service 的内存使用率告警了，查一下内存情况",
        "expected_tools": [{"name": "query_memory_metrics"}],
        "expected_conclusion_contains": [
            "内存使用率从 30% 逐渐升到 85%",
            "触发了告警",
        ],
        "forbidden_tools": [],
        "reference_docs": [],
    },
    {
        "scenario": "时间查询",
        "input": "现在几点？",
        "expected_tools": [{"name": "get_current_time"}],
        "expected_conclusion_contains": [
            "当前时间",
        ],
        "forbidden_tools": [],
        "reference_docs": [],
    },
    # =============================================
    # 场景 2: 多工具联合排查（3 条）
    # =============================================
    {
        "scenario": "服务不可用排查",
        "input": "data-sync-service 挂了，帮我全面排查一下 CPU、内存和日志情况",
        "expected_tools": [
            {"name": "query_cpu_metrics"},
            {"name": "query_memory_metrics"},
            {"name": "search_log"},
        ],
        "expected_conclusion_contains": [
            "CPU 使用率",
            "内存使用率",
            "日志",
        ],
        "forbidden_tools": [],
        "reference_docs": [],
    },
    {
        "scenario": "日志错误排查",
        "input": "data-sync-service 有错误日志，帮我查一下日志主题和搜索错误信息",
        "expected_tools": [
            {"name": "search_topic_by_service_name"},
            {"name": "search_log"},
        ],
        "expected_conclusion_contains": [
            "日志主题",
            "日志信息",
        ],
        "forbidden_tools": [],
        "reference_docs": [],
    },
    {
        "scenario": "综合监控排查",
        "input": "帮我看看 data-sync-service 的 CPU 和内存情况，看看有没有异常",
        "expected_tools": [
            {"name": "query_cpu_metrics"},
            {"name": "query_memory_metrics"},
        ],
        "expected_conclusion_contains": [
            "CPU 使用率",
            "内存使用率",
        ],
        "forbidden_tools": [],
        "reference_docs": [],
    },
    # =============================================
    # 场景 3: 跨文档知识综合（2 条）
    # =============================================
    {
        "scenario": "CPU 问题排查知识",
        "input": "CPU 飙高一般是什么原因？怎么排查？",
        "expected_tools": [{"name": "retrieve_knowledge"}],
        "expected_conclusion_contains": [
            "CPU 飙高",
            "排查方法",
        ],
        "forbidden_tools": [
            "search_log",
            "search_topic_by_service_name",
            "query_cpu_metrics",
            "query_memory_metrics",
        ],
        "reference_docs": ["cpu_high_usage.md"],
    },
    {
        "scenario": "内存泄漏排查知识",
        "input": "内存泄漏怎么排查？有哪些常见原因？",
        "expected_tools": [{"name": "retrieve_knowledge"}],
        "expected_conclusion_contains": [
            "内存泄漏",
            "排查方法",
        ],
        "forbidden_tools": [
            "search_log",
            "search_topic_by_service_name",
            "query_cpu_metrics",
            "query_memory_metrics",
        ],
        "reference_docs": ["memory_high_usage.md"],
    },
    # =============================================
    # 场景 4: 误报/噪声输入（2 条）
    # =============================================
    {
        "scenario": "简单问候（不应调工具）",
        "input": "你好",
        "expected_tools": [],
        "expected_conclusion_contains": [
            "你好",
            "帮助",
        ],
        "forbidden_tools": [
            "search_log",
            "search_topic_by_service_name",
            "query_cpu_metrics",
            "query_memory_metrics",
            "retrieve_knowledge",
            "get_current_time",
        ],
        "reference_docs": [],
    },
    {
        "scenario": "无关闲聊（不应调工具）",
        "input": "今天天气怎么样？你会做什么？",
        "expected_tools": [],
        "expected_conclusion_contains": [
            "无法",
            "天气",
        ],
        "forbidden_tools": [
            "search_log",
            "search_topic_by_service_name",
            "query_cpu_metrics",
            "query_memory_metrics",
            "retrieve_knowledge",
            "get_current_time",
        ],
        "reference_docs": [],
    },
    # =============================================
    # 场景 5: 多步推理（1 条）
    # =============================================
    {
        "scenario": "多步推理：先查 topic 再搜日志再查 CPU",
        "input": "data-sync-service 出了什么问题？先查一下它的日志主题，然后搜一下错误日志，再看看 CPU 情况",
        "expected_tools": [
            {"name": "search_topic_by_service_name"},
            {"name": "search_log"},
            {"name": "query_cpu_metrics"},
        ],
        "expected_conclusion_contains": [
            "日志主题",
            "错误日志",
            "CPU 使用率",
        ],
        "forbidden_tools": [],
        "reference_docs": [],
    },
    # =============================================
    # 场景 6: 模糊/信息不足输入（1 条）
    # =============================================
    {
        "scenario": "信息不足输入（质问）",
        "input": "修一下",
        "expected_tools": [],
        "expected_conclusion_contains": [
            "信息",
            "什么",
        ],
        "forbidden_tools": [
            "search_log",
            "search_topic_by_service_name",
            "query_cpu_metrics",
            "query_memory_metrics",
        ],
        "reference_docs": [],
    },
]


def get_agent_eval_dataset() -> list[dict[str, Any]]:
    """获取 Agent 评估测试数据集。

    Returns:
        list[dict]: AGENT_EVAL_DATASET 的副本。
    """
    return list(AGENT_EVAL_DATASET)


def validate_agent_testset(dataset: list[dict[str, Any]]) -> list[str]:
    """校验数据集的完整性，返回错误列表。

    检查项：
      - 每个样本包含所有必填字段
      - 必填字段类型正确
      - expected_tools 每项至少含 name
      - expected_conclusion_contains 非空
      - forbidden_tools 为 list

    Args:
        dataset: Agent 评估数据集。

    Returns:
        list[str]: 错误信息列表，空列表表示通过。
    """
    errors: list[str] = []

    if not dataset:
        errors.append("数据集为空，至少需要一条样本")
        return errors

    required_fields = {
        "scenario": str,
        "input": str,
        "expected_tools": list,
        "expected_conclusion_contains": list,
        "forbidden_tools": list,
    }

    for i, sample in enumerate(dataset):
        prefix = f"[样本 {i}]"

        for field_name, field_type in required_fields.items():
            if field_name not in sample:
                errors.append(f"{prefix} 缺少必填字段 '{field_name}'")
                continue

            value = sample[field_name]
            if not isinstance(value, field_type):
                errors.append(
                    f"{prefix} 字段 '{field_name}' 类型应为 {field_type.__name__}，实际为 {type(value).__name__}"
                )

        # 校验 expected_tools 每项至少含 name
        expected_tools = sample.get("expected_tools", [])
        if isinstance(expected_tools, list):
            for j, tool in enumerate(expected_tools):
                if not isinstance(tool, dict) or "name" not in tool:
                    errors.append(f"{prefix} expected_tools[{j}] 缺少 'name' 字段")

        # 校验 expected_conclusion_contains 非空
        expected_conclusion = sample.get("expected_conclusion_contains", [])
        if isinstance(expected_conclusion, list) and not expected_conclusion:
            errors.append(f"{prefix} expected_conclusion_contains 为空列表")

        # 校验 forbidden_tools 为 list
        forbidden_tools = sample.get("forbidden_tools", [])
        if isinstance(forbidden_tools, list) and not forbidden_tools:
            # 允许为空，不做强制校验
            pass

    return errors
