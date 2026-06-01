"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现

P1-2.1: 增加结构化 Trace 采集（工具调用耗时、成功/失败、token 用量）
P1-2.2: 增加单步重试逻辑（max 2 次）
"""

import time
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.config import config
from app.core.llm_factory import create_chat_qwen
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from app.services.trace_store import get_trace_store, extract_token_usage
from .state import PlanExecuteState

# P1-2.2: 单步最大重试次数
MAX_STEP_RETRIES = 2


async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤

    使用 LangGraph 的 ToolNode 自动处理工具调用。
    P1-2.2: 失败时自动重试（最多 MAX_STEP_RETRIES 次）。
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])
    trace_id = state.get("trace_id", "")

    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    task = plan[0]
    logger.info(f"当前任务: {task}")

    t_start = time.monotonic()
    tool_calls_trace: list[dict[str, Any]] = []
    token_usage = {"input": 0, "output": 0, "total": 0}
    last_error = ""
    retry_count = 0

    # P1-2.2: 重试循环
    for attempt in range(MAX_STEP_RETRIES + 1):
        try:
            # 获取工具
            local_tools = [
                get_current_time,
                retrieve_knowledge
            ]
            mcp_client = await get_mcp_client_with_retry()
            mcp_tools = await mcp_client.get_tools()
            logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)} (尝试 {attempt + 1}/{MAX_STEP_RETRIES + 1})")

            all_tools = local_tools + mcp_tools

            llm = create_chat_qwen(temperature=0)
            llm_with_tools = llm.bind_tools(all_tools)
            tool_node = ToolNode(all_tools)

            messages = [
                SystemMessage(content="""你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务"""),
                HumanMessage(content=f"请执行以下任务: {task}")
            ]

            # 第一步：LLM 决定是否调用工具
            llm_response = await llm_with_tools.ainvoke(messages)
            token_usage["input"] += extract_token_usage(llm_response)["input"]
            token_usage["output"] += extract_token_usage(llm_response)["output"]
            logger.info(f"LLM 响应类型: {type(llm_response)}")

            # 第二步：如果有工具调用，执行工具
            if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
                logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")

                for tc in llm_response.tool_calls:
                    tc_name = tc.get("name", "unknown")
                    tc_args = tc.get("args", {})
                    tc_start = time.monotonic()

                    try:
                        messages.append(llm_response)
                        tool_messages = await tool_node.ainvoke({"messages": messages})
                        messages.extend(tool_messages["messages"])

                        tc_duration = int((time.monotonic() - tc_start) * 1000)
                        tool_calls_trace.append({
                            "name": tc_name,
                            "args": tc_args,
                            "duration_ms": tc_duration,
                            "success": True,
                        })
                        logger.info(f"工具 {tc_name} 执行成功 ({tc_duration}ms)")
                    except Exception as tool_err:
                        tc_duration = int((time.monotonic() - tc_start) * 1000)
                        tool_calls_trace.append({
                            "name": tc_name,
                            "args": tc_args,
                            "duration_ms": tc_duration,
                            "success": False,
                            "error": str(tool_err),
                        })
                        logger.warning(f"工具 {tc_name} 执行失败 ({tc_duration}ms): {tool_err}")
                        raise  # 抛出以触发步骤级重试

                # 第三步：将工具结果返回给 LLM 生成最终答案
                final_response = await llm_with_tools.ainvoke(messages)
                token_usage["output"] += extract_token_usage(final_response)["output"]
                token_usage["total"] = token_usage["input"] + token_usage["output"]
                result = final_response.content if hasattr(final_response, 'content') else str(final_response)
            else:
                logger.info("LLM 未调用工具，直接返回结果")
                result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            logger.info(f"步骤执行完成，结果长度: {len(result)}")

            # P1-2.1: 保存节点 trace（成功）
            _save_node_trace(trace_id, f"executor_{len(state.get('past_steps', []))}", {
                "step_index": len(state.get("past_steps", [])),
                "task": task,
                "tool_calls": tool_calls_trace,
                "token_usage": token_usage,
                "duration_ms": int((time.monotonic() - t_start) * 1000),
                "retry_count": retry_count,
                "status": "success",
            })

            return {
                "plan": plan[1:],
                "past_steps": [(task, result)],
            }

        except Exception as e:
            retry_count = attempt
            last_error = str(e)
            logger.error(f"执行步骤失败 (尝试 {attempt + 1}/{MAX_STEP_RETRIES + 1}): {last_error}")

            if attempt < MAX_STEP_RETRIES:
                wait_s = 2 ** attempt  # 指数退避: 1s, 2s
                logger.info(f"等待 {wait_s}s 后重试...")
                import asyncio
                await asyncio.sleep(wait_s)

    # 所有重试都失败
    total_duration = int((time.monotonic() - t_start) * 1000)
    logger.error(f"步骤在所有 {MAX_STEP_RETRIES + 1} 次尝试后仍然失败")

    # P1-2.1: 保存节点 trace（失败）
    _save_node_trace(trace_id, f"executor_{len(state.get('past_steps', []))}", {
        "step_index": len(state.get("past_steps", [])),
        "task": task,
        "tool_calls": tool_calls_trace,
        "token_usage": token_usage,
        "duration_ms": total_duration,
        "retry_count": retry_count,
        "status": "error",
        "error": last_error,
    })

    # P1-2.2: 返回错误信息到状态中
    return {
        "plan": plan[1:],
        "past_steps": [(task, f"执行失败 (已重试 {MAX_STEP_RETRIES} 次): {last_error}")],
        "error_count": 1,  # operator.add 会累加
        "last_error": last_error,
    }


def _save_node_trace(trace_id: str, node_name: str, data: dict) -> None:
    """保存单个节点的 trace 数据（P1-2.1）。"""
    if not trace_id:
        return
    try:
        get_trace_store().save_node_trace(trace_id, node_name, data)
    except Exception as e:
        logger.warning(f"保存节点 trace 失败 ({node_name}): {e}")
