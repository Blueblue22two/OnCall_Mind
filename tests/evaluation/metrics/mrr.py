"""MRR (Mean Reciprocal Rank) 计算模块

MRR = 所有问题的第一个相关文档排名的倒数的平均值。

不需要 LLM Judge，纯基于检索结果和 relevant_docs 标注计算。
"""

from typing import List


def compute_mrr(
    retrieved_file_names: List[List[str]],
    relevant_docs_map: List[List[str]],
) -> float:
    """计算 MRR

    对每个问题，找到第一个相关文档在检索结果中的排名（1-indexed），
    取倒数后求所有问题的平均值。未检索到相关文档的问题贡献 0。

    Args:
        retrieved_file_names: 每个问题的检索结果文件名列表
            e.g. [["disk_high_usage.md", "cpu_high_usage.md", ...], ...]
        relevant_docs_map: 每个问题的相关文档文件名列表
            e.g. [["cpu_high_usage.md"], ...]

    Returns:
        float: MRR (0.0 ~ 1.0)
    """
    if not retrieved_file_names or not relevant_docs_map:
        return 0.0

    reciprocal_ranks: list[float] = []

    for retrieved, relevant in zip(retrieved_file_names, relevant_docs_map):
        relevant_set = set(relevant)
        for rank, file_name in enumerate(retrieved, start=1):
            if file_name in relevant_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # 没有找到相关文档
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
