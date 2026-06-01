"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现
"""

from typing import List, TypedDict, Annotated
import operator


class PlanExecuteState(TypedDict, total=False):
    """Plan-Execute-Replan 状态"""

    # 用户输入（任务描述）
    input: str

    # 执行计划（步骤列表）
    plan: List[str]

    # 已执行的步骤历史
    # 使用 operator.add 实现追加式更新（而非覆盖）
    past_steps: Annotated[List[tuple], operator.add]

    # 最终响应/报告
    response: str

    # --- P1-2.1: Agent Eval Trace ---
    trace_id: str                   # Trace 唯一标识，贯穿整个工作流

    # --- P1-2.2: Error Handler 字段 ---
    error_count: int                # 累计错误次数
    max_errors: int                 # 错误上限（默认 3）
    last_error: str                 # 最近一次错误信息

    # 为 HITL 预留的字段
    pending_approval: bool
    pending_tool_name: str
    pending_tool_args: dict
