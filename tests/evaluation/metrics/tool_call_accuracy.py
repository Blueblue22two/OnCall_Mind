"""工具调用准确率（Tool Call Accuracy）计算模块

只比对工具名称集合（不比对参数、不比对顺序），计算三个子指标：
  - Exact Match : 实际调用的工具名集合与期望名集合完全一致
  - Precision   : |actual ∩ expected| / |actual|
  - Recall      : |actual ∩ expected| / |expected|

不需要 LLM Judge，纯集合运算。
"""

from typing import Any


def compute_tool_call_accuracy(
    actual_calls: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]],
) -> dict[str, float | bool]:
    """计算工具调用准确率的三项子指标。

    Args:
        actual_calls: Agent 实际调用的工具列表，每项至少含 {"name": str}。
        expected_calls: 期望调用的工具列表，每项至少含 {"name": str}。

    Returns:
        dict: {
            "exact_match": bool,    # 两个集合完全一致
            "precision": float,     # 精确率 (0.0 ~ 1.0)
            "recall": float,        # 召回率 (0.0 ~ 1.0)
        }
    """
    actual_names: set[str] = {c.get("name", "") for c in actual_calls if isinstance(c, dict)}
    expected_names: set[str] = {c.get("name", "") for c in expected_calls if isinstance(c, dict)}

    # Exact Match: 集合完全相等
    exact_match: bool = actual_names == expected_names

    # Precision: |actual ∩ expected| / |actual|
    # actual 为空时，若 expected 也为空则精确率 1.0（什么都没调 = 完美），否则 0.0
    if not actual_names:
        precision: float = 1.0 if not expected_names else 0.0
    else:
        precision = len(actual_names & expected_names) / len(actual_names)

    # Recall: |actual ∩ expected| / |expected|，expected 为空时返回 1.0
    if not expected_names:
        recall: float = 1.0
    else:
        recall = len(actual_names & expected_names) / len(expected_names)

    return {
        "exact_match": exact_match,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
