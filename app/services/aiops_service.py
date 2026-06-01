"""
通用 Plan-Execute-Replan 服务
基于 LangGraph 官方教程实现

P1-2.1: 集成 TraceStore，生成 trace_id 并随状态传递，结束时保存完整 trace
P1-2.2: 增加 error_handler 节点，统一错误处理与 fallback
"""

import time
from typing import AsyncGenerator, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.agent.aiops import PlanExecuteState, planner, executor, replanner
from app.config import config


# 节点名称常量
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"
NODE_ERROR_HANDLER = "error_handler"  # P1-2.2


async def _error_handler(state: PlanExecuteState) -> Dict[str, Any]:
    """P1-2.2: 统一错误处理节点。

    当 executor 或 planner 抛出未捕获异常时进入此节点。
    根据 error_count 决定是重试还是强制生成响应。
    """
    error_count = state.get("error_count", 0) + 1
    last_error = state.get("last_error", "未知错误")
    max_errors = state.get("max_errors", 3)

    logger.warning(
        f"Error Handler 触发: error_count={error_count}/{max_errors}, "
        f"last_error={last_error}"
    )

    # 如果超过最大错误数，强制结束
    if error_count >= max_errors:
        logger.error(f"错误数达到上限 {max_errors}，强制终止")
        from textwrap import dedent
        return {
            "response": dedent(f"""# 诊断过程异常终止

## 错误摘要
- 累计错误次数: {error_count}
- 最近错误: {last_error}

## 说明
由于多次执行失败，诊断过程被强制终止。请检查：
1. MCP 服务（CLS / Monitor）是否正常运行
2. 网络连接是否正常
3. API Key 是否有效
"""),
            "error_count": error_count,
            "last_error": last_error,
        }

    # 未达上限，清除错误以便重试
    logger.info(f"错误 {error_count}/{max_errors}，尝试继续执行")
    return {
        "error_count": error_count,
        "last_error": last_error,
    }


class AIOpsService:
    """通用 Plan-Execute-Replan 服务（P1-2.1/2.2 增强）"""

    def __init__(self):
        """初始化服务"""
        # P1-2.3: Redis 优先，不可用时自动回退 MemorySaver
        self.checkpointer = self._create_checkpointer()
        self.graph = self._build_graph()
        logger.info("Plan-Execute-Replan Service 初始化完成 (P1 enhanced)")

    @staticmethod
    def _create_checkpointer():
        """创建 checkpointer，Redis 优先，不可用时回退 MemorySaver。

        P1-2.3: 兼容 LangGraph 新旧版本 RedisSaver API。
        """
        if config.redis_url:
            try:
                from langgraph.checkpoint.redis import RedisSaver
                # 新版本 from_conn_string 可能返回 async context manager
                maybe_saver = RedisSaver.from_conn_string(config.redis_url)
                # 检查是否为有效的 checkpointer 实例
                from langgraph.checkpoint.base import BaseCheckpointSaver
                if isinstance(maybe_saver, BaseCheckpointSaver):
                    logger.info(f"AIOps 使用 RedisSaver: {config.redis_url}")
                    return maybe_saver
                else:
                    logger.warning(
                        f"RedisSaver.from_conn_string 返回了 {type(maybe_saver)}，"
                        f"可能需要 async context manager。回退到 MemorySaver。"
                    )
            except Exception as e:
                logger.warning(f"RedisSaver 初始化失败 ({e})，回退到 MemorySaver")
        logger.info("AIOps 使用 MemorySaver（进程内存）")
        return MemorySaver()

    def _build_graph(self):
        """构建 Plan-Execute-Replan 工作流（P1-2.2: 增加 error_handler）"""
        logger.info("构建工作流图（含 error_handler）...")

        workflow = StateGraph(PlanExecuteState)

        workflow.add_node(NODE_PLANNER, planner)
        workflow.add_node(NODE_EXECUTOR, executor)
        workflow.add_node(NODE_REPLANNER, replanner)
        workflow.add_node(NODE_ERROR_HANDLER, _error_handler)

        workflow.set_entry_point(NODE_PLANNER)

        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)

        # P1-2.2: 条件边 — executor 出错时也走 error_handler
        def after_replanner(state: PlanExecuteState) -> str:
            """Replanner 之后的路由。

            如果已生成响应 → END
            如果 error_count 超过 max_errors → NODE_ERROR_HANDLER
            如果还有计划 → NODE_EXECUTOR
            否则 → END
            """
            if state.get("response"):
                logger.info("已生成最终响应，结束流程")
                return END

            error_count = state.get("error_count", 0)
            max_errors = state.get("max_errors", 3)
            if error_count >= max_errors:
                logger.info(f"错误数达标 ({error_count}/{max_errors})，进入 error_handler")
                return NODE_ERROR_HANDLER

            plan = state.get("plan", [])
            if plan:
                logger.info(f"继续执行，剩余 {len(plan)} 个步骤")
                return NODE_EXECUTOR

            logger.info("计划执行完毕，结束流程")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,
            after_replanner,
            {
                NODE_EXECUTOR: NODE_EXECUTOR,
                NODE_ERROR_HANDLER: NODE_ERROR_HANDLER,
                END: END,
            }
        )

        # P1-2.2: error_handler 之后的路由
        def after_error_handler(state: PlanExecuteState) -> str:
            """错误处理后的路由。

            如果已生成响应 → END
            否则尝试继续执行剩余计划。
            """
            if state.get("response"):
                return END

            plan = state.get("plan", [])
            if plan:
                return NODE_EXECUTOR
            return END

        workflow.add_conditional_edges(
            NODE_ERROR_HANDLER,
            after_error_handler,
            {
                NODE_EXECUTOR: NODE_EXECUTOR,
                END: END,
            }
        )

        compiled_graph = workflow.compile(checkpointer=self.checkpointer)
        logger.info("工作流图构建完成（含 P1-2.2 error_handler）")
        return compiled_graph

    async def execute(
        self,
        user_input: str,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行 Plan-Execute-Replan 流程

        Args:
            user_input: 用户的任务描述
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 流式事件
        """
        logger.info(f"[会话 {session_id}] 开始执行任务: {user_input}")

        # P1-2.1: 生成 trace_id
        trace_id = f"trace:{session_id}:{int(time.time() * 1000)}"
        t_total_start = time.monotonic()

        try:
            # P1-2.1: 初始状态包含 trace_id
            # P1-2.2: 初始状态包含 max_errors
            initial_state: PlanExecuteState = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "response": "",
                "trace_id": trace_id,
                "error_count": 0,
                "max_errors": 3,
                "last_error": "",
            }

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
                for node_name, node_output in event.items():
                    logger.info(f"节点 '{node_name}' 输出事件")

                    if node_name == NODE_PLANNER:
                        yield self._format_planner_event(node_output)

                    elif node_name == NODE_EXECUTOR:
                        yield self._format_executor_event(node_output)

                    elif node_name == NODE_REPLANNER:
                        yield self._format_replanner_event(node_output)

                    elif node_name == NODE_ERROR_HANDLER:
                        yield {
                            "type": "status",
                            "stage": "error_handler",
                            "message": f"错误处理: {node_output.get('last_error', '')}",
                            "error_count": node_output.get("error_count", 0),
                        }

            # 获取最终状态
            final_state = self.graph.get_state(config_dict)
            final_response = ""

            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")

            # P1-2.1: 保存完整 trace
            total_duration = int((time.monotonic() - t_total_start) * 1000)
            try:
                from app.services.trace_store import get_trace_store
                trace_store = get_trace_store()
                trace_store.save_trace({
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "input": user_input,
                    "plan": final_state.values.get("plan", []) if final_state and final_state.values else [],
                    "past_steps_count": len(final_state.values.get("past_steps", [])) if final_state and final_state.values else 0,
                    "error_count": final_state.values.get("error_count", 0) if final_state and final_state.values else 0,
                    "final_response": final_response[:5000] if final_response else "",
                    "total_duration_ms": total_duration,
                    "status": "completed" if final_response else "error",
                })
                logger.info(f"[会话 {session_id}] Trace 已保存: {trace_id}")
            except Exception as e:
                logger.error(f"[会话 {session_id}] 保存 Trace 失败: {e}")

            # 发送完成事件
            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成",
                "response": final_response,
                "trace_id": trace_id,
            }

            # 持久化诊断记录
            try:
                from app.services.diagnosis_store import get_diagnosis_store
                store = get_diagnosis_store()
                record_id = store.save(
                    session_id=session_id,
                    input_data=user_input,
                    plan=final_state.values.get("plan", []) if final_state and final_state.values else [],
                    past_steps=final_state.values.get("past_steps", []) if final_state and final_state.values else [],
                    response=final_response,
                )
                logger.info(f"[会话 {session_id}] 诊断记录已保存: {record_id}")
            except Exception as e:
                logger.error(f"[会话 {session_id}] 保存诊断记录失败: {e}")

            logger.info(f"[会话 {session_id}] 任务执行完成, trace_id={trace_id}")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "stage": "error",
                "message": f"任务执行出错: {str(e)}",
                "trace_id": trace_id,
            }

    async def diagnose(
        self,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        AIOps 诊断接口（兼容旧接口）

        Args:
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 诊断过程的流式事件
        """
        from textwrap import dedent
        aiops_task = dedent("""诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告，诊断报告输出格式要求：
                ```
                # 告警分析报告

                ---

                ## 📋 活跃告警清单

                | 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
                |---------|------|----------|-------------|-------------|------|
                | [告警1名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |
                | [告警2名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |

                ---

                ## 🔍 告警根因分析1 - [告警名称]

                ### 告警详情
                - **告警级别**: [级别]
                - **受影响服务**: [服务名]
                - **持续时间**: [X分钟]

                ### 症状描述
                [根据监控指标描述症状]

                ### 日志证据
                [引用查询到的关键日志]

                ### 根因结论
                [基于证据得出的根本原因]

                ---

                ## 🛠️ 处理方案执行1 - [告警名称]

                ### 已执行的排查步骤
                1. [步骤1]
                2. [步骤2]

                ### 处理建议
                [给出具体的处理建议]

                ### 预期效果
                [说明预期的效果]

                ---

                ## 🔍 告警根因分析2 - [告警名称]
                [如果有第2个告警，重复上述格式]

                ---

                ## 📊 结论

                ### 整体评估
                [总结所有告警的整体情况]

                ### 关键发现
                - [发现1]
                - [发现2]

                ### 后续建议
                1. [建议1]
                2. [建议2]

                ### 风险评估
                [评估当前风险等级和影响范围]
                ```

                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
                - 所有内容必须基于工具查询的真实数据，严禁编造
                - 如果某个步骤失败，在结论中如实说明，不要跳过""")

        async for event in self.execute(aiops_task, session_id):
            if event.get("type") == "complete":
                yield {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": "诊断流程完成",
                    "diagnosis": {
                        "status": "completed",
                        "report": event.get("response", ""),
                        "trace_id": event.get("trace_id", ""),
                    }
                }
            else:
                yield event

    def _format_planner_event(self, state: Dict | None) -> Dict:
        """格式化 Planner 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "planner",
                "message": "规划节点执行中"
            }

        plan = state.get("plan", [])

        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {len(plan)} 个步骤",
            "plan": plan
        }

    def _format_executor_event(self, state: Dict | None) -> Dict:
        """格式化 Executor 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "executor",
                "message": "执行节点运行中"
            }

        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        if past_steps:
            last_step, _ = past_steps[-1]
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"步骤执行完成 ({len(past_steps)}/{len(past_steps) + len(plan)})",
                "current_step": last_step,
                "remaining_steps": len(plan)
            }
        else:
            return {
                "type": "status",
                "stage": "executor",
                "message": "开始执行步骤"
            }

    def _format_replanner_event(self, state: Dict | None) -> Dict:
        """格式化 Replanner 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "replanner",
                "message": "评估节点运行中"
            }

        response = state.get("response", "")
        plan = state.get("plan", [])

        if response:
            return {
                "type": "report",
                "stage": "final_report",
                "message": "最终报告已生成",
                "report": response
            }
        else:
            return {
                "type": "status",
                "stage": "replanner",
                "message": f"评估完成，{'继续执行剩余步骤' if plan else '准备生成最终响应'}",
                "remaining_steps": len(plan)
            }


# 全局单例
aiops_service = AIOpsService()
