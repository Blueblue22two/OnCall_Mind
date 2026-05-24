"""评估指标计算模块

包含：
  - Hit Rate@k : 至少一个相关文档在 top-k 中的问题占比
  - MRR         : Mean Reciprocal Rank，第一个相关文档排名的倒数均值
  - Tool Call Accuracy : Agent 工具调用准确率（Exact Match, Precision, Recall）
  - Goal Accuracy      : Agent 目标达成率（LLM Judge 0/1/2 评分）
"""

from tests.evaluation.metrics.hit_rate import compute_hit_rate, compute_hit_rate_multi_k
from tests.evaluation.metrics.mrr import compute_mrr
from tests.evaluation.metrics.tool_call_accuracy import compute_tool_call_accuracy
from tests.evaluation.metrics.goal_accuracy import build_goal_accuracy_prompt, compute_goal_accuracy

__all__ = [
    "compute_hit_rate",
    "compute_hit_rate_multi_k",
    "compute_mrr",
    "compute_tool_call_accuracy",
    "build_goal_accuracy_prompt",
    "compute_goal_accuracy",
]
