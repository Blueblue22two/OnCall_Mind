"""Hallucination Score（幻觉检测）评估模块

使用 LLM Judge 检测 Agent 生成答案中是否包含检索上下文中不存在的事实断言。

评分标准：
  - 2: 无幻觉，所有事实均可追溯到检索上下文
  - 1: 轻微幻觉，1-2 处无依据的断言但不影响核心结论
  - 0: 严重幻觉，核心结论基于虚构事实或大面积编造
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def build_hallucination_prompt(
    question: str,
    agent_answer: str,
    retrieved_contexts: list[str],
) -> str:
    """构建 Hallucination 检测的 LLM Judge prompt。

    Args:
        question: 用户的原始问题。
        agent_answer: Agent 的生成答案文本。
        retrieved_contexts: 检索到的上下文列表（用于事实核查）。

    Returns:
        str: 格式化的 LLM Judge prompt。
    """
    contexts_text = "\n\n---\n\n".join(
        f"[上下文 {i+1}]\n{ctx[:1500]}{'...(截断)' if len(ctx) > 1500 else ''}"
        for i, ctx in enumerate(retrieved_contexts)
    )

    prompt = f"""你是一个 RAG 系统幻觉检测专家。请检查 Agent 的回答中是否有检索上下文中无法找到依据的事实断言（幻觉）。

## 用户问题
{question}

## 检索到的上下文（仅这些是可信信息来源）
{contexts_text}

## Agent 的实际回答
{agent_answer}

## 评分标准
- 2 分（无幻觉）：Agent 回答中的所有事实断言都可以追溯到上述检索上下文。回答严格基于检索结果。
- 1 分（轻微幻觉）：Agent 回答中有 1-2 处小断言未在检索上下文中出现，但不影响核心结论的正确性。
- 0 分（严重幻觉）：Agent 回答的核心结论基于虚构事实，或大面积编造了上下文中不存在的信息。

## 输出格式
请严格按照以下格式输出，不要包含额外内容：
SCORE: <0|1|2>
HALLUCINATED_CLAIMS: <幻觉断言的简要描述，多个用分号分隔；无幻觉时填"无">
REASON: <简短的中文理由>"""
    return prompt


def _parse_hallucination_response(response_text: str) -> tuple[int, list[str], str]:
    """解析 LLM Judge 的响应，提取分数、幻觉断言和理由。

    Args:
        response_text: LLM 返回的原始文本。

    Returns:
        tuple: (score: int, hallucinated_claims: list[str], reason: str)
    """
    score = 0
    claims: list[str] = []
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
        elif line.upper().startswith("HALLUCINATED_CLAIMS:"):
            claims_str = line.split(":", 1)[1].strip()
            if claims_str and claims_str != "无":
                claims = [c.strip() for c in claims_str.split(";") if c.strip()]
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return score, claims, reason


async def compute_hallucination_score(
    question: str,
    agent_answer: str,
    retrieved_contexts: list[str],
    judge_llm: Any,  # LangChain LLM (e.g. ChatOpenAI) with ainvoke support
    num_trials: int = 1,
) -> dict[str, Any]:
    """检测单个测试用例生成答案中的幻觉程度。

    Args:
        question: 用户的原始问题。
        agent_answer: Agent 的生成答案文本。
        retrieved_contexts: 检索到的上下文列表（用于事实核查）。
        judge_llm: LangChain LLM 实例（需支持 ainvoke）。
        num_trials: 独立评分次数，默认 1 次（幻觉检测对温度敏感，通常单次足够）。

    Returns:
        dict: {
            "score": float,                    # 评分 (0.0 ~ 2.0)
            "scores": list[int],               # 每次评分的分数列表
            "hallucinated_claims": list[str],  # 最后一次评分的幻觉断言列表
            "reason": str,                     # 最后一次评分的理由
        }
    """
    if not agent_answer or not agent_answer.strip():
        logger.warning("Agent 输出为空，幻觉评分为 0")
        return {
            "score": 0.0,
            "scores": [0],
            "hallucinated_claims": ["Agent 输出为空，无法评估"],
            "reason": "Agent 输出为空",
        }

    if not retrieved_contexts:
        logger.warning("检索上下文为空，无法进行幻觉检测")
        return {
            "score": 1.0,
            "scores": [1],
            "hallucinated_claims": [],
            "reason": "检索上下文为空，跳过幻觉检测",
        }

    prompt = build_hallucination_prompt(
        question=question,
        agent_answer=agent_answer,
        retrieved_contexts=retrieved_contexts,
    )

    scores: list[int] = []
    last_claims: list[str] = []
    last_reason: str = ""

    for trial in range(num_trials):
        try:
            response = await judge_llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            score, claims, reason = _parse_hallucination_response(response_text)
            scores.append(score)
            last_claims = claims
            last_reason = reason
            logger.debug(
                f"  Hallucination trial {trial + 1}/{num_trials}: "
                f"score={score}, claims={len(claims)}"
            )
        except Exception as e:
            logger.error(f"  Hallucination trial {trial + 1}/{num_trials} 失败: {e}")
            scores.append(0)
            last_reason = f"LLM Judge 调用失败: {e}"

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "score": round(avg_score, 2),
        "scores": scores,
        "hallucinated_claims": last_claims,
        "reason": last_reason,
    }
