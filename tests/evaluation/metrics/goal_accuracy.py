"""Goal Accuracy（目标达成率）评估模块

使用 LLM Judge 对 Agent 输出进行 0/1/2 评分：
  - 2: 完全达成，覆盖所有期望要点，诊断逻辑正确
  - 1: 部分达成，覆盖部分要点但有遗漏或错误
  - 0: 未达成，诊断方向错误或结论与期望不符

通过多次独立评分（默认 3 次）取平均得到最终分数。
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def build_goal_accuracy_prompt(
    user_question: str,
    expected_conclusion_contains: list[str],
    agent_output: str,
) -> str:
    """构建 Goal Accuracy 评估的 LLM Judge prompt。

    Args:
        user_question: 用户的原始输入（告警/提问）。
        expected_conclusion_contains: 期望结论应包含的要点列表。
        agent_output: Agent 的最终输出（answer 文本）。

    Returns:
        str: 格式化的 LLM Judge prompt。
    """
    expected_points = "\n".join(f"  - {point}" for point in expected_conclusion_contains)

    prompt = f"""你是一个 AIOps Agent 评估专家。请根据以下标准对 Agent 的响应进行评分。

## 用户问题
{user_question}

## 期望结论应包含的要点
{expected_points}

## Agent 的响应
{agent_output}

## 评分标准
- 2 分（完全达成）：Agent 的响应覆盖了所有期望要点，诊断逻辑正确，结论合理。
- 1 分（部分达成）：Agent 的响应覆盖了部分期望要点，但有遗漏或存在轻微错误。
- 0 分（未达成）：Agent 的响应诊断方向错误，或结论与期望严重不符。

## 输出格式
请严格按照以下格式输出，不要包含额外内容：
SCORE: <0|1|2>
REASON: <简短的中文理由>"""
    return prompt


def _parse_judge_response(response_text: str) -> tuple[int, str]:
    """解析 LLM Judge 的响应，提取分数和理由。

    Args:
        response_text: LLM 返回的原始文本。

    Returns:
        tuple: (score: int, reason: str)，解析失败时 score 为 0，reason 为"解析失败"。
    """
    score = 0
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
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return score, reason


async def compute_goal_accuracy(
    user_question: str,
    expected_conclusion_contains: list[str],
    agent_output: str,
    judge_llm: Any,  # LangChain LLM (e.g. ChatTongyi) with ainvoke support
    num_trials: int = 3,
) -> dict[str, Any]:
    """对单个测试用例的 Agent 输出进行 0/1/2 评分。

    调用 LLM Judge 多次（num_trials 次）取平均分，并记录每次的分数和理由。

    Args:
        user_question: 用户的原始输入。
        expected_conclusion_contains: 期望结论应包含的要点列表。
        agent_output: Agent 的最终输出文本。
        judge_llm: LangChain LLM 实例（需支持 ainvoke），如 ChatTongyi。
        num_trials: 独立评分次数，默认 3 次。

    Returns:
        dict: {
            "score": float,        # num_trials 次评分的平均分 (0.0 ~ 2.0)
            "scores": list[int],   # 每次评分的分数列表
            "reason": str,         # 最后一次评分的理由
        }
    """
    if not agent_output or not agent_output.strip():
        logger.warning("Agent 输出为空，目标达成率为 0")
        return {
            "score": 0.0,
            "scores": [0],
            "reason": "Agent 输出为空",
        }

    prompt = build_goal_accuracy_prompt(
        user_question=user_question,
        expected_conclusion_contains=expected_conclusion_contains,
        agent_output=agent_output,
    )

    scores: list[int] = []
    last_reason: str = ""

    for trial in range(num_trials):
        try:
            response = await judge_llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            score, reason = _parse_judge_response(response_text)
            scores.append(score)
            last_reason = reason
            logger.debug(f"  Goal Accuracy trial {trial + 1}/{num_trials}: score={score}, reason={reason}")
        except Exception as e:
            logger.error(f"  Goal Accuracy trial {trial + 1}/{num_trials} 失败: {e}")
            scores.append(0)
            last_reason = f"LLM Judge 调用失败: {e}"

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "score": round(avg_score, 2),
        "scores": scores,
        "reason": last_reason,
    }
