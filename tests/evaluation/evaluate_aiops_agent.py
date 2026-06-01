"""AIOps (Plan-Execute-Replan) Agent 评估主脚本

评估维度：
  1. Plan Quality Score — 规划质量（计划中包含多少预期关键词）
  2. Tool Recall — 工具调用召回率（预期工具被实际调用的比例）
  3. Tool Precision — 工具调用精确率（是否调用了禁止工具）
  4. Step Efficiency — 步骤效率（实际执行步骤是否在允许范围内）
  5. Error Recovery — 错误恢复（错误次数和重试次数）
  6. Conclusion Hits — 结论命中率（最终响应中包含多少预期关键词）

使用方法（在项目根目录执行）：

  # 完整评估
  python -m tests.evaluation.evaluate_aiops_agent

  # 指定输出路径
  python -m tests.evaluation.evaluate_aiops_agent --output reports/aiops_eval.json

  # 指定输出格式
  python -m tests.evaluation.evaluate_aiops_agent --output-format both

注意：MCP 服务（CLS + Monitor）必须运行，才能对 normal / multi_step / error_recovery
场景进行有意义的评估。knowledge_lookup 场景会使用 retrieve_knowledge 本地工具，
noop / ambiguous 场景通常不需要外部工具。脚本会捕获执行错误并记录，不会因为
单个 case 失败而中断整体评估。
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PARTIAL_DIR = Path("agent_traces")
"""文件系统上 agent_traces 目录的路径，用于读取 partial node trace。"""


# ---------------------------------------------------------------------------
# 数据加载与校验
# ---------------------------------------------------------------------------

def _load_and_validate_dataset() -> tuple[list[dict[str, Any]], str]:
    """加载 AIOps 评估数据集并校验完整性。

    Returns:
        tuple[list[dict], str]: (dataset, dataset_version)。

    Raises:
        SystemExit: 数据集校验失败时退出。
    """
    from tests.evaluation.aiops_testset import (
        AIOPS_EVAL_DATASET,
        DATASET_VERSION,
        validate_aiops_testset,
    )

    errors = validate_aiops_testset(AIOPS_EVAL_DATASET)
    if errors:
        logger.error(f"AIOps 测试数据集校验失败（{len(errors)} 条错误）:")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)

    logger.info(
        f"数据集校验通过: {len(AIOPS_EVAL_DATASET)} 条样本, version={DATASET_VERSION}"
    )
    return AIOPS_EVAL_DATASET, DATASET_VERSION


# ---------------------------------------------------------------------------
# 执行单个测试用例
# ---------------------------------------------------------------------------

async def _execute_single_case(
    case_index: int,
    case: dict[str, Any],
) -> dict[str, Any]:
    """执行单个 AIOps 测试用例，调用 aiops_service.execute() 并获取 trace_id。

    Args:
        case_index: 测试用例序号（用于 session_id 区分）。
        case: 测试用例字典（必须包含 "input" 键）。

    Returns:
        dict: 包含 trace_id, final_response, 或 error 的字典。
    """
    from app.services.aiops_service import aiops_service

    input_text: str = case.get("input", "")
    scenario: str = case.get("scenario", "unknown")
    session_id = f"aiops_eval_{case_index}"
    # 根据场景类型设超时：full_diagnosis 给 180s，其余 60s
    # 30s 不够 Planner（LLM）+ Executor（MCP工具含3次重试）+ Replanner 完成一轮
    timeout_s = 180 if case.get("expected_behavior") == "full_diagnosis" else 60

    logger.info(
        f"[{case_index + 1}] 执行 AIOps Agent: scenario={scenario}, "
        f"input='{input_text[:80]}', timeout={timeout_s}s"
    )

    trace_id = ""
    final_response = ""

    try:
        async def _consume_events():
            nonlocal trace_id, final_response
            async for event in aiops_service.execute(input_text, session_id=session_id):
                if event.get("type") == "complete":
                    trace_id = event.get("trace_id", "")
                    final_response = event.get("response", "")
                    logger.info(
                        f"[{case_index + 1}] 完成: trace_id={trace_id}, "
                        f"response_len={len(final_response)}"
                    )

        await asyncio.wait_for(_consume_events(), timeout=timeout_s)

        if not trace_id:
            return {
                "trace_id": "",
                "final_response": final_response or "",
                "error": "执行完成但未收到 complete 事件中的 trace_id",
            }

        return {
            "trace_id": trace_id,
            "final_response": final_response,
            "error": None,
        }

    except asyncio.TimeoutError:
        logger.warning(f"[{case_index + 1}] 执行超时 ({timeout_s}s)")
        return {
            "trace_id": trace_id or f"aiops_eval_{case_index}_timeout",
            "final_response": final_response or "",
            "error": f"执行超时 ({timeout_s}s)",
        }


# ---------------------------------------------------------------------------
# 读取 Trace 数据
# ---------------------------------------------------------------------------

def _read_complete_trace(trace_id: str) -> dict[str, Any] | None:
    """从 TraceStore 读取完整 trace。

    优先使用 TraceStore API，如果 Redis 不可用则从文件回退路径读取。

    Args:
        trace_id: 完整 trace 的 ID。

    Returns:
        dict | None: Trace 数据，不存在则返回 None。
    """
    try:
        from app.services.trace_store import get_trace_store

        store = get_trace_store()
        trace = store.get_trace(trace_id)
        if trace:
            return trace
    except Exception as e:
        logger.warning(f"TraceStore.get_trace() 失败 ({e})，尝试文件直接读取")

    # 文件回退路径
    safe_id = trace_id.replace(":", "_")
    file_path = PARTIAL_DIR / f"trace_{safe_id}.json"
    if file_path.exists():
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"直接读取 trace 文件失败: {e}")

    return None


def _read_executor_node_traces(trace_id: str) -> list[dict[str, Any]]:
    """读取所有 executor 节点的 partial trace 文件，收集工具调用信息。

    Executor 节点 trace 文件名格式:
        partial_node_{trace_id_safe}_executor_{step_index}.json

    Args:
        trace_id: 完整的 trace_id (如 "trace:aiops_eval_0:1717000000000")。

    Returns:
        list[dict]: 所有 executor 节点 trace 数据列表，按 step_index 排序。
    """
    safe_id = trace_id.replace(":", "_")
    pattern = f"partial_node_{safe_id}_executor_*.json"
    nodes: list[dict[str, Any]] = []

    for f in sorted(PARTIAL_DIR.glob(pattern)):
        try:
            node_data = json.loads(f.read_text(encoding="utf-8"))
            nodes.append(node_data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"读取 executor trace 文件失败 {f}: {e}")

    return nodes


def _read_planner_node_trace(trace_id: str) -> dict[str, Any] | None:
    """读取 planner 节点的 partial trace 文件，获取 plan 内容。

    Planner 节点 trace 文件名格式:
        partial_node_{trace_id_safe}_planner.json

    Args:
        trace_id: 完整的 trace_id。

    Returns:
        dict | None: planner trace 数据，不存在则返回 None。
    """
    safe_id = trace_id.replace(":", "_")
    file_path = PARTIAL_DIR / f"partial_node_{safe_id}_planner.json"
    if file_path.exists():
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"读取 planner trace 失败: {e}")
    return None


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def _compute_plan_quality(
    trace: dict[str, Any] | None,
    case: dict[str, Any],
) -> float:
    """计算 Plan Quality Score。

    从 trace["plan"] 或 planner partial trace 或 trace 中的 plan 字段提取计划文本，
    与 expected_plan_patterns 进行匹配。

    Args:
        trace: 完整 trace 数据（可能为 None）。
        case: 测试用例字典（含 expected_plan_patterns）。

    Returns:
        float: 0.0 ~ 1.0 的规划质量分数。
    """
    expected_patterns = case.get("expected_plan_patterns", [])
    if not expected_patterns:
        return 1.0

    # 从多个来源收集 plan 文本
    plan_sources: list[str] = []

    if trace:
        plan_list = trace.get("plan", [])
        if isinstance(plan_list, list):
            plan_sources.extend(plan_list)
        elif isinstance(plan_list, str):
            plan_sources.append(plan_list)

    # 也尝试从 planner partial trace 获取
    trace_id = (trace or {}).get("trace_id", "")
    if trace_id:
        planner_trace = _read_planner_node_trace(trace_id)
        if planner_trace:
            plan_steps = planner_trace.get("plan", [])
            if isinstance(plan_steps, list):
                plan_sources.extend(plan_steps)

    # 合并所有 plan 文本
    plan_text = " ".join(plan_sources)
    if not plan_text.strip():
        logger.warning("无法获取计划文本，plan_score 默认为 0.0")
        return 0.0

    hits = sum(1 for p in expected_patterns if p.lower() in plan_text.lower())
    score = hits / len(expected_patterns)
    logger.debug(f"  plan_score={score:.2f} ({hits}/{len(expected_patterns)})")
    return score


def _compute_tool_recall(
    all_tools_called: set[str],
    case: dict[str, Any],
) -> float:
    """计算工具调用召回率。

    在 expected_tool_calls 非空时，计算被实际调用的预期工具比例。

    Args:
        all_tools_called: 所有被调用工具的名称集合。
        case: 测试用例字典（含 expected_tool_calls）。

    Returns:
        float: 0.0 ~ 1.0 的召回率。
    """
    expected = set(case.get("expected_tool_calls", []))
    if not expected:
        return 1.0

    hits = expected & all_tools_called
    recall = len(hits) / len(expected)
    logger.debug(f"  tool_recall={recall:.2f} (hits={hits}, expected={expected})")
    return recall


def _compute_tool_precision(
    all_tools_called: set[str],
    case: dict[str, Any],
) -> float:
    """计算工具调用精确率（禁止工具检查）。

    如果调用了任何 forbidden_tools 中的工具，精确率为 0.0；否则为 1.0。

    Args:
        all_tools_called: 所有被调用工具的名称集合。
        case: 测试用例字典（含 forbidden_tools）。

    Returns:
        float: 0.0 或 1.0。
    """
    forbidden = set(case.get("forbidden_tools", []))
    violations = forbidden & all_tools_called
    precision = 0.0 if violations else 1.0
    if violations:
        logger.warning(f"  tool_precision=0.0 (违规工具: {violations})")
    return precision


def _compute_step_efficiency(
    trace: dict[str, Any] | None,
    case: dict[str, Any],
) -> tuple[int, int, bool]:
    """计算步骤效率。

    Args:
        trace: 完整 trace 数据。
        case: 测试用例字典（含 max_allowed_steps）。

    Returns:
        tuple[int, int, bool]: (step_count, max_allowed_steps, termination_ok)。
    """
    step_count = (trace or {}).get("past_steps_count", 0)
    max_allowed = case.get("max_allowed_steps", 8)
    termination_ok = step_count <= max_allowed
    logger.debug(
        f"  step_count={step_count}, max_allowed={max_allowed}, ok={termination_ok}"
    )
    return step_count, max_allowed, termination_ok


def _compute_error_recovery(
    trace: dict[str, Any] | None,
    trace_id: str,
) -> tuple[int, int]:
    """计算错误恢复指标。

    从 trace 获取 error_count，从 executor 节点 trace 汇总 retry_count。

    Args:
        trace: 完整 trace 数据。
        trace_id: trace ID（用于读取 executor 节点 traces）。

    Returns:
        tuple[int, int]: (error_count, retry_total)。
    """
    error_count = (trace or {}).get("error_count", 0)

    # 从 executor 节点 trace 汇总 retry_count
    retry_total = 0
    if trace_id:
        executor_nodes = _read_executor_node_traces(trace_id)
        for node in executor_nodes:
            retry_total += node.get("retry_count", 0)

    logger.debug(f"  error_count={error_count}, retry_total={retry_total}")
    return error_count, retry_total


def _compute_conclusion_hits(
    trace: dict[str, Any] | None,
    case: dict[str, Any],
) -> float:
    """计算结论命中率。

    从 trace["final_response"] 中检查 expected_conclusion_contains 关键词的出现比例。

    Args:
        trace: 完整 trace 数据。
        case: 测试用例字典（含 expected_conclusion_contains）。

    Returns:
        float: 0.0 ~ 1.0 的结论命中率。
    """
    expected_conclusion = case.get("expected_conclusion_contains", [])
    if not expected_conclusion:
        return 1.0

    response = (trace or {}).get("final_response", "")
    if not response.strip():
        logger.warning("final_response 为空，conclusion_score 默认为 0.0")
        return 0.0

    hits = sum(1 for c in expected_conclusion if c.lower() in response.lower())
    score = hits / len(expected_conclusion)
    logger.debug(f"  conclusion_score={score:.2f} ({hits}/{len(expected_conclusion)})")
    return score


def _compute_overall_pass(
    plan_score: float,
    tool_recall: float,
    tool_precision: float,
    termination_ok: bool,
    error_count: int,
    conclusion_score: float,
) -> bool:
    """计算整体通过指标。

    所有条件全部满足时返回 True：
      - plan_score >= 0.5
      - tool_recall >= 0.5
      - tool_precision == 1.0
      - termination_ok (step_count <= max_allowed_steps)
      - error_count <= 2
      - conclusion_score >= 0.5

    Returns:
        bool: 是否整体通过。
    """
    passed = (
        plan_score >= 0.5
        and tool_recall >= 0.5
        and tool_precision == 1.0
        and termination_ok
        and error_count <= 2
        and conclusion_score >= 0.5
    )
    return passed


# ---------------------------------------------------------------------------
# 收集工具调用信息
# ---------------------------------------------------------------------------

def _collect_all_tools_called(trace_id: str) -> set[str]:
    """从 executor 节点 trace 中收集所有被调用的工具名称。

    Args:
        trace_id: trace ID。

    Returns:
        set[str]: 所有被调用工具的名称集合。
    """
    all_tools: set[str] = set()
    if not trace_id:
        return all_tools

    executor_nodes = _read_executor_node_traces(trace_id)
    for node in executor_nodes:
        for tc in node.get("tool_calls", []):
            name = tc.get("name", "")
            if name:
                all_tools.add(name)

    return all_tools


# ---------------------------------------------------------------------------
# 评估流程主函数
# ---------------------------------------------------------------------------

async def run_aiops_evaluation(
    output_path: Optional[str] = None,
    output_format: str = "json",
) -> dict[str, Any]:
    """执行完整的 AIOps Agent 评估流程。

    步骤：
      1. 加载并校验 aiops_testset 数据集。
      2. 逐条调用 aiops_service.execute() 执行 Agent。
      3. 从 TraceStore 和文件系统读取 trace 数据。
      4. 计算每条测试用例的各维度指标。
      5. 聚合统计结果。
      6. 保存 JSON / CSV 文件。

    Args:
        output_path: 可选的结果输出文件路径。
        output_format: 输出格式 ("json", "csv", "both")。

    Returns:
        dict: 包含完整评估结果（含 per_case 明细和 summary 聚合）。
    """
    # 0. 加载并校验数据集
    dataset, dataset_version = _load_and_validate_dataset()
    num_cases = len(dataset)

    logger.info("=" * 60)
    logger.info("AIOps Agent 评估开始")
    logger.info(f"  数据集版本:     {dataset_version}")
    logger.info(f"  测试用例数:     {num_cases}")
    logger.info(
        "  场景类型:        "
        + ", ".join(sorted(set(c["scenario_type"] for c in dataset)))
    )
    logger.info("=" * 60)

    # 1. 逐条执行并计算指标
    per_case: list[dict[str, Any]] = []

    # 聚合用变量
    all_plan_scores: list[float] = []
    all_tool_recalls: list[float] = []
    all_tool_precisions: list[float] = []
    all_step_counts: list[int] = []
    all_termination_ok: list[bool] = []
    all_error_counts: list[int] = []
    all_retry_totals: list[float] = []
    all_conclusion_scores: list[float] = []
    all_duration_ms: list[int] = []
    all_passes: list[bool] = []

    for i, case in enumerate(dataset):
        scenario_label = case.get("scenario", f"index={i}")

        # 1a. 执行 Agent
        exec_result = await _execute_single_case(i, case)
        trace_id = exec_result["trace_id"]
        error = exec_result.get("error")

        # 如果执行出错，填充默认值
        if error or not trace_id:
            logger.error(f"  [{i + 1}/{num_cases}] {scenario_label}: 执行失败 - {error}")
            per_case_record: dict[str, Any] = {
                "index": i,
                "scenario": scenario_label,
                "scenario_type": case.get("scenario_type", ""),
                "trace_id": trace_id,
                "plan_score": 0.0,
                "tool_recall": 0.0,
                "tool_precision": 1.0,
                "step_count": 0,
                "max_allowed_steps": case.get("max_allowed_steps", 8),
                "termination_ok": False,
                "error_count": 0,
                "retry_total": 0,
                "conclusion_score": 0.0,
                "duration_ms": 0,
                "overall_pass": False,
                "all_tools_called": [],
                "status": "error",
                "error_message": error,
            }
            per_case.append(per_case_record)

            # 汇总
            all_plan_scores.append(0.0)
            all_tool_recalls.append(0.0)
            all_tool_precisions.append(1.0)
            all_step_counts.append(0)
            all_termination_ok.append(False)
            all_error_counts.append(0)
            all_retry_totals.append(0)
            all_conclusion_scores.append(0.0)
            all_duration_ms.append(0)
            all_passes.append(False)
            continue

        # 1b. 读取完整 trace
        trace = _read_complete_trace(trace_id)
        if trace is None:
            logger.warning(f"  [{i + 1}/{num_cases}] trace 未找到: {trace_id}，使用空字典")
            trace = {}

        # 1c. 收集工具调用信息
        all_tools_called = _collect_all_tools_called(trace_id)
        logger.info(
            f"  [{i + 1}/{num_cases}] {scenario_label}: "
            f"trace_id={trace_id[:40]}..., "
            f"tools={sorted(all_tools_called)}"
        )

        # 1d. 计算各维度指标
        plan_score = _compute_plan_quality(trace, case)
        tool_recall = _compute_tool_recall(all_tools_called, case)
        tool_precision = _compute_tool_precision(all_tools_called, case)
        step_count, max_allowed_steps, termination_ok = _compute_step_efficiency(trace, case)
        error_count, retry_total = _compute_error_recovery(trace, trace_id)
        conclusion_score = _compute_conclusion_hits(trace, case)
        duration_ms = trace.get("total_duration_ms", 0)
        status = trace.get("status", "unknown")
        overall_pass = _compute_overall_pass(
            plan_score, tool_recall, tool_precision,
            termination_ok, error_count, conclusion_score,
        )

        # 1e. 组装 per_case 记录
        per_case_record = {
            "index": i,
            "scenario": scenario_label,
            "scenario_type": case.get("scenario_type", ""),
            "trace_id": trace_id,
            "plan_score": round(plan_score, 4),
            "tool_recall": round(tool_recall, 4),
            "tool_precision": tool_precision,
            "step_count": step_count,
            "max_allowed_steps": max_allowed_steps,
            "termination_ok": termination_ok,
            "error_count": error_count,
            "retry_total": retry_total,
            "conclusion_score": round(conclusion_score, 4),
            "duration_ms": duration_ms,
            "overall_pass": overall_pass,
            "all_tools_called": sorted(all_tools_called),
            "status": status,
        }

        # 添加 error_message（如有）
        if error:
            per_case_record["error_message"] = error

        per_case.append(per_case_record)

        # 1f. 汇总
        all_plan_scores.append(plan_score)
        all_tool_recalls.append(tool_recall)
        all_tool_precisions.append(tool_precision)
        all_step_counts.append(step_count)
        all_termination_ok.append(termination_ok)
        all_error_counts.append(error_count)
        all_retry_totals.append(retry_total)
        all_conclusion_scores.append(conclusion_score)
        all_duration_ms.append(duration_ms)
        all_passes.append(overall_pass)

        # 打印进度
        pass_mark = "PASS" if overall_pass else "FAIL"
        logger.info(
            f"    plan={plan_score:.2f}, recall={tool_recall:.2f}, "
            f"prec={tool_precision}, steps={step_count}/{max_allowed_steps}, "
            f"errors={error_count}, retries={retry_total}, "
            f"conclusion={conclusion_score:.2f}, duration={duration_ms}ms "
            f"=> {pass_mark}"
        )

    # 2. 聚合统计
    passes = sum(1 for p in all_passes if p)
    summary: dict[str, Any] = {
        "evaluated_at": datetime.now().isoformat(),
        "dataset_version": dataset_version,
        "num_cases": num_cases,
        "pass_rate": round(passes / num_cases, 4) if num_cases else 0.0,
        "avg_plan_score": round(sum(all_plan_scores) / num_cases, 4) if num_cases else 0.0,
        "avg_tool_recall": round(sum(all_tool_recalls) / num_cases, 4) if num_cases else 0.0,
        "avg_tool_precision": round(sum(all_tool_precisions) / num_cases, 4) if num_cases else 0.0,
        "avg_step_count": round(sum(all_step_counts) / num_cases, 2) if num_cases else 0.0,
        "termination_ok_rate": round(sum(1 for v in all_termination_ok if v) / num_cases, 4)
        if num_cases
        else 0.0,
        "avg_error_count": round(sum(all_error_counts) / num_cases, 2) if num_cases else 0.0,
        "avg_retry_total": round(sum(all_retry_totals) / num_cases, 2) if num_cases else 0.0,
        "avg_conclusion_score": round(sum(all_conclusion_scores) / num_cases, 4)
        if num_cases
        else 0.0,
        "avg_duration_ms": round(sum(all_duration_ms) / num_cases, 1) if num_cases else 0.0,
        "per_case": per_case,
    }

    # 3. 打印摘要
    logger.info("=" * 60)
    logger.info("AIOps Agent 评估结果摘要")
    logger.info(f"  pass_rate:             {summary['pass_rate']:.4f} ({passes}/{num_cases})")
    logger.info(f"  avg_plan_score:        {summary['avg_plan_score']:.4f}")
    logger.info(f"  avg_tool_recall:       {summary['avg_tool_recall']:.4f}")
    logger.info(f"  avg_tool_precision:    {summary['avg_tool_precision']:.4f}")
    logger.info(f"  avg_step_count:        {summary['avg_step_count']:.2f}")
    logger.info(f"  termination_ok_rate:   {summary['termination_ok_rate']:.4f}")
    logger.info(f"  avg_error_count:       {summary['avg_error_count']:.2f}")
    logger.info(f"  avg_retry_total:       {summary['avg_retry_total']:.2f}")
    logger.info(f"  avg_conclusion_score:  {summary['avg_conclusion_score']:.4f}")
    logger.info(f"  avg_duration_ms:       {summary['avg_duration_ms']:.1f}")
    logger.info("=" * 60)

    # 4. 保存结果
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path:
        out_path = Path(output_path)
        json_path = out_path if output_format in ("json", "both") else None
        csv_path = out_path.with_suffix(".csv") if output_format in ("csv", "both") else None
    else:
        default_stem = Path(f"reports/aiops_eval_{timestamp}")
        json_path = default_stem.with_suffix(".json") if output_format in ("json", "both") else None
        csv_path = default_stem.with_suffix(".csv") if output_format in ("csv", "both") else None

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 结果已保存: {json_path}")

    if csv_path:
        _save_csv(summary, str(csv_path))

    return summary


# ---------------------------------------------------------------------------
# CSV 存储
# ---------------------------------------------------------------------------

def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """将嵌套的评估结果平铺为适合 CSV 的单层 dict。

    Args:
        summary: 评估结果字典（包含 summary 级别的聚合字段）。

    Returns:
        dict: 扁平化的单层字典。
    """
    flat: dict[str, Any] = {
        "evaluated_at": summary["evaluated_at"],
        "dataset_version": summary["dataset_version"],
        "num_cases": summary["num_cases"],
        "pass_rate": summary["pass_rate"],
    }

    # 聚合指标
    metric_keys = [
        "avg_plan_score",
        "avg_tool_recall",
        "avg_tool_precision",
        "avg_step_count",
        "termination_ok_rate",
        "avg_error_count",
        "avg_retry_total",
        "avg_conclusion_score",
        "avg_duration_ms",
    ]
    for key in metric_keys:
        if key in summary:
            flat[key] = summary[key]

    return flat


def _save_csv(summary: dict[str, Any], csv_path: str):
    """将评估结果保存为 CSV 文件（summary 一行 + per_case 明细表）。

    Args:
        summary: 评估结果字典。
        csv_path: CSV 输出路径。
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas 未安装，跳过 CSV 输出。请运行: pip install pandas")
        return

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    # summary 单行表
    flat = _flatten_summary(summary)
    summary_df = pd.DataFrame([flat])

    # per_case 明细表
    per_case = summary.get("per_case", [])
    if per_case:
        per_case_df = pd.DataFrame(per_case)
        # 全部写入同一个 CSV（summary 在前，明细在后，留空行分隔）
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            summary_df.to_csv(f, index=False)
            f.write("\n")  # 空行分隔
            f.write("# Per-Case Details\n")
            per_case_df.to_csv(f, index=False)
    else:
        summary_df.to_csv(csv_path, index=False, encoding="utf-8")

    logger.info(f"CSV 结果已保存: {csv_path}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIOps Agent 评估脚本")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出路径（默认: reports/aiops_eval_{timestamp}.json）",
    )
    parser.add_argument(
        "--output-format", "-f",
        type=str,
        choices=["json", "csv", "both"],
        default="json",
        help="输出格式 (default: json)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_aiops_evaluation(
            output_path=args.output,
            output_format=args.output_format,
        )
    )
