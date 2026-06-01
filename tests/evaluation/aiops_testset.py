"""AIOps (Plan-Execute-Replan) Agent 评估测试数据集

AIOps Agent 使用 Plan-Execute-Replan 流程，Planner 动态生成步骤列表，
因此无法像 RAG Agent 一样指定精确的工具调用顺序。本数据集使用松散约束：

  - expected_tool_calls    : 无序的期望工具名称列表（应被调用的工具）
  - forbidden_tools        : 不应被调用的工具名称列表
  - expected_plan_patterns : Planner 生成的步骤中应包含的关键词
  - expected_conclusion_contains : 最终回答中应包含的字符串
  - min_steps / max_allowed_steps : 步骤数范围
  - scenario_type          : 场景分类（normal / error_recovery / ambiguous / multi_step / knowledge_lookup / noop）
  - expected_behavior      : 期望行为（full_diagnosis / quick_response / refuse_or_minimal）

可用工具（基于当前 MCP/Mock 行为）：
  MCP CLS 工具    : search_log, search_topic_by_service_name, get_topic_info_by_name,
                    get_region_code_by_name, get_current_timestamp
  MCP Monitor 工具 : query_cpu_metrics, query_memory_metrics, get_active_alerts
  本地工具         : retrieve_knowledge, get_current_time

场景分类（scenario_type）说明：
  - normal            : 标准告警诊断，Agent 按正常流程调用工具并给出结论
  - error_recovery    : Agent 调用工具时遇到错误（如服务不存在），应优雅降级并说明失败
  - ambiguous         : 用户输入信息不足，Agent 应主动询问澄清或给出系统级概览
  - multi_step        : 需要多维度排查（CPU + 内存 + 日志等），生成较长的步骤链
  - knowledge_lookup  : 纯知识查询，应使用 retrieve_knowledge 而非任何监控工具
  - noop              : 非运维类的闲聊输入，不应调用任何运维工具
"""

from typing import Any

# ---------------------------------------------------------------------------
# 数据集版本号
# ---------------------------------------------------------------------------
DATASET_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# 有效枚举值
# ---------------------------------------------------------------------------
VALID_SCENARIO_TYPES = {"normal", "error_recovery", "ambiguous", "multi_step", "knowledge_lookup", "noop"}
VALID_BEHAVIORS = {"full_diagnosis", "quick_response", "refuse_or_minimal"}

# ---------------------------------------------------------------------------
# AIOps 评估测试数据集（6 条）
# ---------------------------------------------------------------------------
AIOPS_EVAL_DATASET: list[dict[str, Any]] = [
    # =============================================
    # 场景 1: CPU 告警诊断 (normal)
    # =============================================
    {
        "scenario": "CPU 告警诊断",
        "scenario_type": "normal",
        "expected_behavior": "full_diagnosis",
        "input": "data-sync-service 的 CPU 使用率告警了，帮我诊断一下原因",
        "expected_tool_calls": ["query_cpu_metrics", "search_log"],
        "forbidden_tools": [],
        "expected_plan_patterns": ["CPU", "日志"],
        "min_steps": 2,
        "max_allowed_steps": 5,
        "expected_conclusion_contains": ["CPU", "使用率"],
    },
    # =============================================
    # 场景 2: 服务不可用综合排查 (multi_step)
    # =============================================
    {
        "scenario": "服务不可用综合排查",
        "scenario_type": "multi_step",
        "expected_behavior": "full_diagnosis",
        "input": "data-sync-service 完全不可用了，帮我全面排查 CPU、内存和错误日志",
        "expected_tool_calls": ["query_cpu_metrics", "query_memory_metrics", "search_log"],
        "forbidden_tools": [],
        "expected_plan_patterns": ["CPU", "内存", "日志"],
        "min_steps": 3,
        "max_allowed_steps": 7,
        "expected_conclusion_contains": ["CPU", "内存", "日志", "data-sync-service"],
    },
    # =============================================
    # 场景 3: 知识查询：CPU 飙高原因 (knowledge_lookup)
    # =============================================
    {
        "scenario": "知识查询：CPU 飙高原因",
        "scenario_type": "knowledge_lookup",
        "expected_behavior": "quick_response",
        "input": "CPU 飙高一般是什么原因？怎么排查？",
        "expected_tool_calls": ["retrieve_knowledge"],
        "forbidden_tools": [
            "query_cpu_metrics",
            "query_memory_metrics",
            "search_log",
            "search_topic_by_service_name",
        ],
        "expected_plan_patterns": ["知识", "文档"],
        "min_steps": 1,
        "max_allowed_steps": 3,
        "expected_conclusion_contains": ["CPU", "排查"],
    },
    # =============================================
    # 场景 4: 模糊输入 (ambiguous)
    # =============================================
    {
        "scenario": "模糊输入",
        "scenario_type": "ambiguous",
        "expected_behavior": "quick_response",
        "input": "帮我看看系统情况",
        "expected_tool_calls": [],       # 不强制要求调任何工具
        "forbidden_tools": [],
        "expected_plan_patterns": [],    # 不强制校验 plan 关键词
        "min_steps": 0,
        "max_allowed_steps": 3,
        "expected_conclusion_contains": ["系统", "信息"],
    },
    # =============================================
    # 场景 5: 简单问候 (noop)
    # =============================================
    {
        "scenario": "简单问候",
        "scenario_type": "noop",
        "expected_behavior": "refuse_or_minimal",
        "input": "你好，今天怎么样？",
        "expected_tool_calls": [],
        "forbidden_tools": [
            "query_cpu_metrics",
            "query_memory_metrics",
            "search_log",
            "search_topic_by_service_name",
            "get_active_alerts",
        ],
        "expected_plan_patterns": [],
        "min_steps": 0,
        "max_allowed_steps": 2,
        "expected_conclusion_contains": [],  # 问候语不检测结论关键词
    },
    # =============================================
    # 场景 6: 工具失败恢复 (error_recovery)
    # =============================================
    # Agent 行为: 先 search_topic_by_service_name → 服务不存在 → 立即终止报告
    # 预期工具可能不被调用(合理短路), 结论关键词反映"未找到"而非"失败"
    {
        "scenario": "工具失败恢复",
        "scenario_type": "error_recovery",
        "expected_behavior": "full_diagnosis",
        "input": "检查一下 nonexistent-service-xyz 的 CPU 和内存情况",
        "expected_tool_calls": ["query_cpu_metrics", "query_memory_metrics"],
        "forbidden_tools": [],
        "expected_plan_patterns": ["CPU", "内存"],
        "min_steps": 1,
        "max_allowed_steps": 5,
        "expected_conclusion_contains": ["没有找到", "不存在"],
    },
]


# ---------------------------------------------------------------------------
# 公共函数
# ---------------------------------------------------------------------------


def get_aiops_eval_dataset() -> list[dict[str, Any]]:
    """获取 AIOps 评估测试数据集的副本。

    Returns:
        list[dict]: AIOPS_EVAL_DATASET 的浅拷贝副本。
    """
    return list(AIOPS_EVAL_DATASET)


def validate_aiops_testset(dataset: list[dict[str, Any]]) -> list[str]:
    """校验 AIOps 评估数据集的完整性与类型正确性。

    检查项：
      - 数据集非空
      - 每个样本包含所有必填字段且类型正确
      - scenario_type 属于有效枚举值
      - expected_behavior 属于有效枚举值
      - expected_tool_calls 是字符串列表
      - forbidden_tools 是字符串列表
      - expected_conclusion_contains 是字符串列表且非空
      - expected_plan_patterns 是字符串列表
      - min_steps <= max_allowed_steps
      - min_steps >= 0
      - max_allowed_steps >= 0

    Args:
        dataset: AIOps 评估数据集。

    Returns:
        list[str]: 错误信息列表，空列表表示全部通过。
    """
    errors: list[str] = []

    if not dataset:
        errors.append("数据集为空，至少需要一条样本")
        return errors

    required_fields: dict[str, type] = {
        "scenario": str,
        "scenario_type": str,
        "expected_behavior": str,
        "input": str,
        "expected_tool_calls": list,
        "forbidden_tools": list,
        "expected_plan_patterns": list,
        "expected_conclusion_contains": list,
        "min_steps": int,
        "max_allowed_steps": int,
    }

    for i, sample in enumerate(dataset):
        prefix = f"[样本 {i}"

        # 提取 scenario 值用于更清晰的错误消息
        scenario_label = sample.get("scenario", f"index={i}")
        prefix = f"[样本 {i} ({scenario_label})]"

        # 检查所有必填字段存在且类型正确
        for field_name, field_type in required_fields.items():
            if field_name not in sample:
                errors.append(f"{prefix} 缺少必填字段 '{field_name}'")
                continue

            value = sample[field_name]
            if not isinstance(value, field_type):
                errors.append(
                    f"{prefix} 字段 '{field_name}' 类型应为 {field_type.__name__}，"
                    f"实际为 {type(value).__name__}"
                )

        # 校验 scenario_type
        scenario_type = sample.get("scenario_type", "")
        if isinstance(scenario_type, str) and scenario_type not in VALID_SCENARIO_TYPES:
            errors.append(
                f"{prefix} scenario_type='{scenario_type}' 不在有效值 {VALID_SCENARIO_TYPES} 中"
            )

        # 校验 expected_behavior
        expected_behavior = sample.get("expected_behavior", "")
        if isinstance(expected_behavior, str) and expected_behavior not in VALID_BEHAVIORS:
            errors.append(
                f"{prefix} expected_behavior='{expected_behavior}' 不在有效值 {VALID_BEHAVIORS} 中"
            )

        # 校验 expected_tool_calls 每个元素都是字符串
        expected_tool_calls = sample.get("expected_tool_calls", [])
        if isinstance(expected_tool_calls, list):
            for j, tool_name in enumerate(expected_tool_calls):
                if not isinstance(tool_name, str):
                    errors.append(f"{prefix} expected_tool_calls[{j}] 应为 str，实际为 {type(tool_name).__name__}")
        elif "expected_tool_calls" in sample:
            errors.append(f"{prefix} expected_tool_calls 应为 list")

        # 校验 forbidden_tools 每个元素都是字符串
        forbidden_tools = sample.get("forbidden_tools", [])
        if isinstance(forbidden_tools, list):
            for j, tool_name in enumerate(forbidden_tools):
                if not isinstance(tool_name, str):
                    errors.append(f"{prefix} forbidden_tools[{j}] 应为 str，实际为 {type(tool_name).__name__}")
        elif "forbidden_tools" in sample:
            errors.append(f"{prefix} forbidden_tools 应为 list")

        # 校验 expected_plan_patterns 每个元素都是字符串
        expected_plan_patterns = sample.get("expected_plan_patterns", [])
        if isinstance(expected_plan_patterns, list):
            for j, pattern in enumerate(expected_plan_patterns):
                if not isinstance(pattern, str):
                    errors.append(f"{prefix} expected_plan_patterns[{j}] 应为 str，实际为 {type(pattern).__name__}")
        elif "expected_plan_patterns" in sample:
            errors.append(f"{prefix} expected_plan_patterns 应为 list")

        # 校验 expected_conclusion_contains（noop/refuse_or_minimal 允许为空）
        expected_conclusion = sample.get("expected_conclusion_contains", [])
        if isinstance(expected_conclusion, list):
            scenario_type = sample.get("scenario_type", "")
            expected_behavior = sample.get("expected_behavior", "")
            if not expected_conclusion and scenario_type != "noop" and expected_behavior != "refuse_or_minimal":
                errors.append(f"{prefix} expected_conclusion_contains 为空列表，noop/refuse_or_minimal 场景除外")
            for j, keyword in enumerate(expected_conclusion):
                if not isinstance(keyword, str):
                    errors.append(
                        f"{prefix} expected_conclusion_contains[{j}] 应为 str，"
                        f"实际为 {type(keyword).__name__}"
                    )
        elif "expected_conclusion_contains" in sample:
            errors.append(f"{prefix} expected_conclusion_contains 应为 list")

        # 校验 min_steps / max_allowed_steps
        min_steps = sample.get("min_steps")
        max_allowed_steps = sample.get("max_allowed_steps")

        if isinstance(min_steps, int) and isinstance(max_allowed_steps, int):
            if min_steps < 0:
                errors.append(f"{prefix} min_steps={min_steps} 不能为负数")
            if max_allowed_steps < 0:
                errors.append(f"{prefix} max_allowed_steps={max_allowed_steps} 不能为负数")
            if min_steps > max_allowed_steps:
                errors.append(
                    f"{prefix} min_steps ({min_steps}) > max_allowed_steps ({max_allowed_steps})"
                )

    return errors
