"""Agent 评估主脚本

评估维度：
  1. Tool Call Accuracy — 工具调用准确率（Exact Match, Precision, Recall）
  2. Goal Accuracy — 目标达成率（LLM Judge 0/1/2 评分，3 次取平均）

用法（在项目根目录执行）：

  # 完整评估（Tool Call Accuracy + Goal Accuracy）
  python -m tests.evaluation.evaluate_agent

  # 仅计算 Tool Call Accuracy（跳过 Goal Accuracy）
  python -m tests.evaluation.evaluate_agent --skip-goal

  # 覆盖 Judge 模型
  python -m tests.evaluation.evaluate_agent --judge-model qwen-max

  # 指定输出路径和格式
  python -m tests.evaluation.evaluate_agent --output reports/agent_eval.json
  python -m tests.evaluation.evaluate_agent --output-format both
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


def _build_judge_llm():
    """构建 Goal Accuracy 需要的 LangChain LLM。

    Judge 使用独立的 eval_judge_* 配置，与线上 RAG 模型解耦，确保评估可复现。
    使用 ChatQwen（OpenAI 兼容模式），支持 ainvoke 异步调用。
    """
    try:
        from langchain_qwq import ChatQwen
        from app.config import config

        api_key = config.eval_judge_api_key or config.dashscope_api_key
        api_base = config.eval_judge_api_base or config.dashscope_api_base

        llm = ChatQwen(
            model=config.eval_judge_model,
            temperature=config.eval_judge_temperature,
            api_key=api_key,
            api_base=api_base,
            timeout=config.llm_timeout,
            max_retries=config.llm_max_retries,
        )
        return llm

    except ImportError as e:
        logger.error(f"依赖未安装: {e}")
        sys.exit(1)


async def _execute_single_case(
    case_index: int,
    case: dict[str, Any],
    timeout_s: int = 60,
) -> dict[str, Any]:
    """执行单个 Agent 测试用例。

    调用 RagAgentService.query_with_trace() 获取工具调用 trace 和 answer。
    异常时记录 error 字段继续后续评估。

    Args:
        case_index: 测试用例序号。
        case: 测试用例字典。
        timeout_s: 单 case 超时（秒），默认 60s。

    Returns:
        dict: 包含执行结果（answer, actual_tools, 或 error）。
    """
    from app.services.rag_agent_service import rag_agent_service

    question: str = case.get("input", "")
    scenario: str = case.get("scenario", "unknown")

    logger.info(
        f"[{case_index + 1}] 执行 Agent: scenario={scenario}, "
        f"input='{question[:60]}', timeout={timeout_s}s"
    )

    try:
        result = await asyncio.wait_for(
            rag_agent_service.query_with_trace(
                question=question,
                session_id=f"agent_eval_{case_index}",
            ),
            timeout=timeout_s,
        )
        actual_tools = result.get("tool_calls", [])
        answer = result.get("answer", "")
        logger.info(
            f"[{case_index + 1}] 完成: tool_calls={len(actual_tools)}, "
            f"tools={[t.get('name', '?') for t in actual_tools]}"
        )
        return {
            "answer": answer,
            "actual_tools": actual_tools,
            "error": None,
        }
    except asyncio.TimeoutError:
        logger.warning(f"[{case_index + 1}] Agent 执行超时 ({timeout_s}s)")
        return {
            "answer": "",
            "actual_tools": [],
            "error": f"执行超时 ({timeout_s}s)",
        }
    except Exception as e:
        logger.error(f"[{case_index + 1}] Agent 执行失败: {e}")
        return {
            "answer": "",
            "actual_tools": [],
            "error": str(e),
        }


def _compute_tool_call_metrics(
    case: dict[str, Any],
    exec_result: dict[str, Any],
) -> dict[str, Any]:
    """计算单个 case 的工具调用准确率指标。

    Args:
        case: 测试用例字典。
        exec_result: Agent 执行结果。

    Returns:
        dict: 包含 tool_exact_match, tool_precision, tool_recall 的字典。
    """
    from tests.evaluation.metrics import compute_tool_call_accuracy

    expected_tools = case.get("expected_tools", [])
    actual_tools = exec_result.get("actual_tools", [])

    tca = compute_tool_call_accuracy(actual_calls=actual_tools, expected_calls=expected_tools)

    return {
        "tool_exact_match": tca["exact_match"],
        "tool_precision": tca["precision"],
        "tool_recall": tca["recall"],
    }


async def _compute_goal_metrics(
    case: dict[str, Any],
    exec_result: dict[str, Any],
    judge_llm: Any,
    skip_goal: bool = False,
    judge_timeout_s: int = 120,
) -> dict[str, Any]:
    """计算单个 case 的目标达成率。

    Args:
        case: 测试用例字典。
        exec_result: Agent 执行结果。
        judge_llm: LangChain LLM 实例。
        skip_goal: 是否跳过 Goal Accuracy 评估。
        judge_timeout_s: Judge 评分总超时（秒），默认 120s（3 次 × ~30s + 余量）。

    Returns:
        dict: 包含 goal_score 或 None 的字典。
    """
    if skip_goal:
        return {"goal_score": None}

    from tests.evaluation.metrics import compute_goal_accuracy

    error = exec_result.get("error")
    if error:
        return {"goal_score": 0.0}

    answer = exec_result.get("answer", "")
    try:
        ga = await asyncio.wait_for(
            compute_goal_accuracy(
                user_question=case.get("input", ""),
                expected_conclusion_contains=case.get("expected_conclusion_contains", []),
                agent_output=answer,
                judge_llm=judge_llm,
                num_trials=3,
            ),
            timeout=judge_timeout_s,
        )
        return {"goal_score": ga["score"]}
    except asyncio.TimeoutError:
        logger.warning(f"Goal Accuracy Judge 超时 ({judge_timeout_s}s)")
        return {"goal_score": 0.0}
    except Exception as e:
        logger.error(f"Goal Accuracy 计算失败: {e}")
        return {"goal_score": 0.0}


async def run_agent_evaluation(
    output_path: Optional[str] = None,
    output_format: str = "json",
    judge_model_override: Optional[str] = None,
    skip_goal: bool = False,
) -> dict[str, Any]:
    """执行完整的 Agent 评估流程。

    步骤：
      1. 加载并校验 agent_testset 数据集。
      2. 初始化 RagAgentService。
      3. 逐条执行 Agent，捕获工具调用序列。
      4. 计算 Tool Call Accuracy（Exact Match, Precision, Recall）。
      5. 可选：用 LLM Judge 评估 Goal Accuracy（每条 3 次取平均）。
      6. 汇总结果，保存 JSON / CSV 文件。

    Args:
        output_path: 可选，结果输出文件路径。
        output_format: 输出格式 ("json", "csv", "both")。
        judge_model_override: 覆盖 Judge 模型名称。
        skip_goal: 是否跳过 Goal Accuracy 评估。

    Returns:
        dict: 完整的评估结果。
    """
    from app.config import config
    from tests.evaluation.agent_testset import (
        AGENT_EVAL_DATASET,
        DATASET_VERSION,
        validate_agent_testset,
    )

    # 0. 校验数据集
    errors = validate_agent_testset(AGENT_EVAL_DATASET)
    if errors:
        logger.error(f"数据集校验失败（{len(errors)} 条错误）:")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)
    logger.info(
        f"数据集校验通过: {len(AGENT_EVAL_DATASET)} 条样本, version={DATASET_VERSION}"
    )

    # 覆盖 Judge 模型名
    if judge_model_override:
        original_model = config.eval_judge_model
        config.eval_judge_model = judge_model_override
        logger.info(f"Judge 模型覆盖: {original_model} -> {judge_model_override}")

    logger.info("=" * 60)
    logger.info("Agent 评估开始")
    logger.info(f"  数据集版本:         {DATASET_VERSION}")
    logger.info(f"  测试用例数:         {len(AGENT_EVAL_DATASET)}")
    logger.info(f"  Judge 模型:         {config.eval_judge_model}")
    logger.info(f"  Judge 温度:         {config.eval_judge_temperature}")
    logger.info(f"  Goal Accuracy:     {'跳过' if skip_goal else '启用'}")
    logger.info("=" * 60)

    # 1. 初始化 Agent 服务
    from app.services.rag_agent_service import rag_agent_service

    await rag_agent_service._initialize_agent()

    # 2. 构建 LLM Judge
    judge_llm = None
    if not skip_goal:
        judge_llm = _build_judge_llm()

    # 3. 逐条执行 Agent
    per_case: list[dict[str, Any]] = []
    all_tool_exact_match: list[bool] = []
    all_tool_precision: list[float] = []
    all_tool_recall: list[float] = []
    all_goal_scores: list[float] = []

    for i, case in enumerate(AGENT_EVAL_DATASET):
        exec_result = await _execute_single_case(i, case)

        # 工具调用指标
        tool_metrics = _compute_tool_call_metrics(case, exec_result)
        all_tool_exact_match.append(tool_metrics["tool_exact_match"])
        all_tool_precision.append(tool_metrics["tool_precision"])
        all_tool_recall.append(tool_metrics["tool_recall"])

        # 目标达成率
        goal_metrics = await _compute_goal_metrics(case, exec_result, judge_llm, skip_goal)
        goal_score = goal_metrics["goal_score"]
        if goal_score is not None:
            all_goal_scores.append(goal_score)

        # 组装 per_case
        per_case_record: dict[str, Any] = {
            "index": i,
            "scenario": case.get("scenario", ""),
            "input": case.get("input", ""),
            "actual_tools": exec_result.get("actual_tools", []),
            "expected_tools": case.get("expected_tools", []),
            **tool_metrics,
            **goal_metrics,
        }
        if exec_result.get("error"):
            per_case_record["error"] = exec_result["error"]

        per_case.append(per_case_record)

        # 打印进度
        logger.info(
            f"  [{i + 1}/{len(AGENT_EVAL_DATASET)}] {case['scenario']}: "
            f"em={tool_metrics['tool_exact_match']}, "
            f"p={tool_metrics['tool_precision']:.2f}, "
            f"r={tool_metrics['tool_recall']:.2f}"
        )
        if not skip_goal and goal_score is not None:
            logger.info(f"    goal_score={goal_score:.2f}")

    # 4. 汇总 Tool Call Accuracy
    num_cases = len(AGENT_EVAL_DATASET)
    tool_call_accuracy: dict[str, float] = {
        "exact_match_rate": round(sum(1 for v in all_tool_exact_match if v) / num_cases, 4)
        if num_cases
        else 0.0,
        "avg_precision": round(sum(all_tool_precision) / num_cases, 4) if num_cases else 0.0,
        "avg_recall": round(sum(all_tool_recall) / num_cases, 4) if num_cases else 0.0,
    }

    # 5. 汇总 Goal Accuracy
    goal_accuracy: Optional[dict[str, Any]] = None
    if not skip_goal and all_goal_scores:
        score_distribution = {"0": 0, "1": 0, "2": 0}
        for s in all_goal_scores:
            rounded = round(s)
            key = str(min(rounded, 2))
            score_distribution[key] = score_distribution.get(key, 0) + 1

        goal_accuracy = {
            "avg_score": round(sum(all_goal_scores) / len(all_goal_scores), 2),
            "score_distribution": score_distribution,
            "per_question": [
                {
                    "index": i,
                    "scenario": case["scenario"],
                    "score": all_goal_scores[i] if i < len(all_goal_scores) else None,
                }
                for i, case in enumerate(AGENT_EVAL_DATASET)
            ],
        }

    # 6. 组装结果
    judge_meta: dict[str, Any] = {
        "model": config.eval_judge_model,
        "temperature": config.eval_judge_temperature,
    }
    if config.eval_judge_api_base:
        judge_meta["api_base"] = config.eval_judge_api_base

    scores: dict[str, Any] = {
        "evaluated_at": datetime.now().isoformat(),
        "dataset_version": DATASET_VERSION,
        "num_test_cases": num_cases,
        "judge": judge_meta,
        "data_source": "mock",
        "tool_call_accuracy": tool_call_accuracy,
        "goal_accuracy": goal_accuracy,
        "per_case": per_case,
    }

    # 7. 打印摘要
    logger.info("=" * 60)
    logger.info("Agent 评估结果摘要")
    logger.info(f"  [Tool Call Accuracy]")
    logger.info(f"    exact_match_rate: {tool_call_accuracy['exact_match_rate']:.4f}")
    logger.info(f"    avg_precision:    {tool_call_accuracy['avg_precision']:.4f}")
    logger.info(f"    avg_recall:       {tool_call_accuracy['avg_recall']:.4f}")
    if goal_accuracy:
        logger.info(f"  [Goal Accuracy]")
        logger.info(f"    avg_score:        {goal_accuracy['avg_score']:.2f} / 2.0")
        logger.info(f"    分布: {goal_accuracy['score_distribution']}")
    logger.info("=" * 60)

    # 8. 保存结果
    ts: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    goal_suffix: str = "" if skip_goal else "_with_goal"

    if output_path:
        out_path = Path(output_path)
        json_path = out_path if output_format in ("json", "both") else None
        csv_path = out_path.with_suffix(".csv") if output_format in ("csv", "both") else None
    else:
        default_stem = Path(f"reports/agent_eval{goal_suffix}_{ts}")
        json_path = default_stem.with_suffix(".json") if output_format in ("json", "both") else None
        csv_path = default_stem.with_suffix(".csv") if output_format in ("csv", "both") else None

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 结果已保存: {json_path}")

    if csv_path:
        _save_csv(scores, str(csv_path))

    return scores


def _flatten_scores(scores: dict) -> dict:
    """将嵌套的评估结果平铺为适合 CSV 的单层 dict。

    Args:
        scores: 评估结果字典。

    Returns:
        dict: 扁平化的单层字典。
    """
    flat: dict[str, Any] = {
        "evaluated_at": scores["evaluated_at"],
        "dataset_version": scores["dataset_version"],
        "num_test_cases": scores["num_test_cases"],
        "data_source": scores["data_source"],
        "judge_model": scores["judge"].get("model", ""),
        "judge_temperature": scores["judge"].get("temperature", ""),
    }

    # 工具调用指标
    for k, v in scores.get("tool_call_accuracy", {}).items():
        if isinstance(v, (int, float)):
            flat[f"tca_{k}"] = v

    # 目标达成指标
    goal_accuracy = scores.get("goal_accuracy")
    if goal_accuracy:
        flat["ga_avg_score"] = goal_accuracy.get("avg_score", "")
        dist = goal_accuracy.get("score_distribution", {})
        flat["ga_score_0"] = dist.get("0", 0)
        flat["ga_score_1"] = dist.get("1", 0)
        flat["ga_score_2"] = dist.get("2", 0)

    return flat


def _save_csv(scores: dict, csv_path: str):
    """将评估结果保存为 CSV 文件。

    Args:
        scores: 评估结果字典。
        csv_path: CSV 输出路径。
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas 未安装，跳过 CSV 输出。请运行: pip install pandas")
        return

    flat = _flatten_scores(scores)
    df = pd.DataFrame([flat])
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"CSV 结果已保存: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 评估脚本")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径（默认：reports/agent_eval_{timestamp}.json）",
    )
    parser.add_argument(
        "--output-format", "-f",
        type=str,
        choices=["json", "csv", "both"],
        default="json",
        help="输出格式 (default: json)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="覆盖 Judge 模型名称",
    )
    parser.add_argument(
        "--skip-goal",
        action="store_true",
        default=False,
        help="跳过 Goal Accuracy 评估（仅计算 Tool Call Accuracy）",
    )
    args = parser.parse_args()

    asyncio.run(
        run_agent_evaluation(
            output_path=args.output,
            output_format=args.output_format,
            judge_model_override=args.judge_model,
            skip_goal=args.skip_goal,
        )
    )
