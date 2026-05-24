"""Hit Rate@k 计算模块

Hit Rate@k = 至少有一个相关文档出现在 top-k 中的问题占比。

不需要 LLM Judge，纯基于检索结果和 relevant_docs 标注的集合运算。
"""

from typing import List


def compute_hit_rate(
    retrieved_file_names: List[List[str]],
    relevant_docs_map: List[List[str]],
    k: int,
) -> float:
    """计算 Hit Rate@k

    Args:
        retrieved_file_names: 每个问题的检索结果文件名列表（已截断至 k）
            e.g. [["cpu_high_usage.md", "disk_high_usage.md"], ...]
        relevant_docs_map: 每个问题的相关文档文件名列表
            e.g. [["cpu_high_usage.md"], ...]
        k: top-k 截断值

    Returns:
        float: Hit Rate@k (0.0 ~ 1.0)
    """
    if not retrieved_file_names or not relevant_docs_map:
        return 0.0

    hits = 0
    for retrieved, relevant in zip(retrieved_file_names, relevant_docs_map):
        # 截断至 top-k
        top_k_names = retrieved[:k]
        # 检查是否有交集
        if set(top_k_names) & set(relevant):
            hits += 1

    return hits / len(retrieved_file_names)


def compute_hit_rate_multi_k(
    retrieved_file_names: List[List[str]],
    relevant_docs_map: List[List[str]],
    ks: List[int] = (3, 5, 10),
) -> dict[str, float]:
    """一次计算多个 k 值的 Hit Rate

    Returns:
        dict: {"hit_rate@3": 0.72, "hit_rate@5": 0.84, "hit_rate@10": 0.92}
    """
    return {f"hit_rate@{k}": compute_hit_rate(retrieved_file_names, relevant_docs_map, k) for k in ks}
