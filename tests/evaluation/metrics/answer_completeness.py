"""Answer Completeness（答案完整性）评估模块

使用 LLM Judge 对 Agent 生成答案进行 0/1/2 评分，评估答案是否覆盖了
所有期望的关键事实点。

评分标准：
  - 2: 完全覆盖，答案包含了所有期望事实，无遗漏
  - 1: 部分覆盖，答案覆盖了部分期望事实（≥50%），但有遗漏或表述不够精确
  - 0: 覆盖不足，答案遗漏了超过 50% 的期望事实，或关键事实错误

通过多次独立评分（默认 3 次）取平均得到最终分数。
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def build_completeness_prompt(
    question: str,
    expected_facts: list[str],
    agent_answer: str,
) -> str:
    """构建 Answer Completeness 评估的 LLM Judge prompt。

    Args:
        question: 用户的原始问题。
        expected_facts: 答案应包含的关键事实列表。
        agent_answer: Agent 的生成答案文本。

    Returns:
        str: 格式化的 LLM Judge prompt。
    """
    facts_text = "\n".join(f"  [{i+1}] {fact}" for i, fact in enumerate(expected_facts))

    prompt = f"""你是一个 RAG 系统答案质量评估专家。请评估 Agent 的回答是否完整覆盖了所有期望的关键事实。

## 用户问题
{question}

## 期望答案应包含的关键事实
{facts_text}

## Agent 的实际回答
{agent_answer}

## 评分标准
- 2 分（完全覆盖）：Agent 的回答覆盖了所有期望事实，无遗漏，且事实表述准确。
- 1 分（部分覆盖）：Agent 的回答覆盖了部分期望事实（≥50%），但有遗漏或部分事实表述不够精确。
- 0 分（覆盖不足）：Agent 的回答遗漏了超过 50% 的期望事实，或存在与期望事实相矛盾的关键错误。

## 输出格式
请严格按照以下格式输出，不要包含额外内容：
SCORE: <0|1|2>
COVERED: <已覆盖的事实编号列表，如 1,3,4>
MISSED: <未覆盖的事实编号列表，如 2,5>
REASON: <简短的中文理由>"""
    return prompt


def _parse_completeness_response(response_text: str) -> tuple[int, list[int], list[int], str]:
    """解析 LLM Judge 的响应，提取分数、覆盖清单和理由。

    Args:
        response_text: LLM 返回的原始文本。

    Returns:
        tuple: (score: int, covered: list[int], missed: list[int], reason: str)
    """
    score = 0
    covered: list[int] = []
    missed: list[int] = []
    reason = "解析失败"

    lines = response_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            score_str = line.split(":", 1)[1].strip()
            try:
                parsed = int(score_str)
                if parsed in (0, 1, 2):
                    score = parsed
            except ValueError:
                pass
        elif line.upper().startswith("COVERED:"):
            nums_str = line.split(":", 1)[1].strip()
            try:
                covered = [int(n.strip()) for n in nums_str.split(",") if n.strip().isdigit()]
            except ValueError:
                pass
        elif line.upper().startswith("MISSED:"):
            nums_str = line.split(":", 1)[1].strip()
            try:
                missed = [int(n.strip()) for n in nums_str.split(",") if n.strip().isdigit()]
            except ValueError:
                pass
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return score, covered, missed, reason


async def compute_answer_completeness(
    question: str,
    expected_facts: list[str],
    agent_answer: str,
    judge_llm: Any,  # LangChain LLM (e.g. ChatOpenAI) with ainvoke support
    num_trials: int = 3,
) -> dict[str, Any]:
    """对单个测试用例的生成答案进行完整性评分。

    调用 LLM Judge 多次（num_trials 次）取平均分，并记录每次的覆盖情况。

    Args:
        question: 用户的原始问题。
        expected_facts: 答案应包含的关键事实列表。
        agent_answer: Agent 的生成答案文本。
        judge_llm: LangChain LLM 实例（需支持 ainvoke）。
        num_trials: 独立评分次数，默认 3 次。

    Returns:
        dict: {
            "score": float,            # num_trials 次评分的平均分 (0.0 ~ 2.0)
            "scores": list[int],       # 每次评分的分数列表
            "covered_facts": list[int],# 最后一次评分的已覆盖事实编号
            "missed_facts": list[int], # 最后一次评分的未覆盖事实编号
            "reason": str,             # 最后一次评分的理由
            "num_expected": int,       # 期望事实总数
        }
    """
    if not agent_answer or not agent_answer.strip():
        logger.warning("Agent 输出为空，答案完整性为 0")
        return {
            "score": 0.0,
            "scores": [0],
            "covered_facts": [],
            "missed_facts": list(range(1, len(expected_facts) + 1)),
            "reason": "Agent 输出为空",
            "num_expected": len(expected_facts),
        }

    if not expected_facts:
        logger.warning("expected_facts 为空，跳过完整性评估，返回满分")
        return {
            "score": 2.0,
            "scores": [2],
            "covered_facts": [],
            "missed_facts": [],
            "reason": "无期望事实标注，默认满分",
            "num_expected": 0,
        }

    prompt = build_completeness_prompt(
        question=question,
        expected_facts=expected_facts,
        agent_answer=agent_answer,
    )

    scores: list[int] = []
    last_covered: list[int] = []
    last_missed: list[int] = []
    last_reason: str = ""

    for trial in range(num_trials):
        try:
            response = await judge_llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            score, covered, missed, reason = _parse_completeness_response(response_text)
            scores.append(score)
            last_covered = covered
            last_missed = missed
            last_reason = reason
            logger.debug(
                f"  Answer Completeness trial {trial + 1}/{num_trials}: "
                f"score={score}, covered={covered}, missed={missed}"
            )
        except Exception as e:
            logger.error(f"  Answer Completeness trial {trial + 1}/{num_trials} 失败: {e}")
            scores.append(0)
            last_reason = f"LLM Judge 调用失败: {e}"

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "score": round(avg_score, 2),
        "scores": scores,
        "covered_facts": last_covered,
        "missed_facts": last_missed,
        "reason": last_reason,
        "num_expected": len(expected_facts),
    }
