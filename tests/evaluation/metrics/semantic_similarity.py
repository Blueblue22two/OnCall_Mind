"""Semantic Similarity（语义相似度）评估模块

计算生成答案与 ground_truth 参考文本之间的语义相似度。
使用 embedding cosine similarity，纯向量计算，无 LLM 依赖。

适用场景：
  - 快速评估（不需要 LLM Judge，成本低、速度快）
  - CI 集成中的回归检测
  - 作为 LLM Judge 指标的补充参考
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度。

    Args:
        vec_a: 向量 A。
        vec_b: 向量 B。

    Returns:
        float: 余弦相似度，范围 [0, 1]。
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_semantic_similarity(
    answer: str,
    ground_truth: str,
    embedding_service,  # DashScopeEmbeddings 实例
) -> float:
    """计算答案与 ground_truth 的语义相似度。

    将 answer 和 ground_truth 分别嵌入为向量，计算余弦相似度。
    纯向量计算，无 LLM 依赖，可快速批量运行。

    Args:
        answer: Agent 的生成答案文本。
        ground_truth: 参考 ground_truth 文本（多个要点已拼接为单个字符串）。
        embedding_service: 嵌入服务实例（如 DashScopeEmbeddings），
                           需支持 embed_query(str) -> list[float]。

    Returns:
        float: 余弦相似度，范围 [0, 1]。1 表示语义完全一致，0 表示完全无关。
    """
    if not answer or not answer.strip():
        logger.warning("Answer 为空，语义相似度为 0")
        return 0.0

    if not ground_truth or not ground_truth.strip():
        logger.warning("Ground truth 为空，语义相似度为 0")
        return 0.0

    try:
        answer_vec = embedding_service.embed_query(answer)
        truth_vec = embedding_service.embed_query(ground_truth)
        similarity = _cosine_similarity(answer_vec, truth_vec)
        # 将 [-1, 1] 范围映射到 [0, 1]
        normalized = max(0.0, min(1.0, similarity))
        return round(normalized, 4)
    except Exception as e:
        logger.error(f"语义相似度计算失败: {e}")
        return 0.0


def compute_batch_semantic_similarity(
    answers: list[str],
    ground_truths: list[str],
    embedding_service,
) -> tuple[float, list[float]]:
    """批量计算语义相似度。

    Args:
        answers: Agent 生成答案列表。
        ground_truths: 参考 ground_truth 文本列表。
        embedding_service: 嵌入服务实例。

    Returns:
        tuple: (avg_similarity: float, per_item_scores: list[float])
    """
    if len(answers) != len(ground_truths):
        raise ValueError(
            f"answers 和 ground_truths 长度不一致: {len(answers)} vs {len(ground_truths)}"
        )

    scores = []
    for answer, truth in zip(answers, ground_truths):
        score = compute_semantic_similarity(answer, truth, embedding_service)
        scores.append(score)

    avg = sum(scores) / len(scores) if scores else 0.0
    return round(avg, 4), scores
