"""Prometheus Metrics 模块（P1-2.4）

提供 LLM API 调用的可观测性指标：
- 调用延迟（P50/P95/P99 通过 Histogram 实现）
- 成功率（Counter）
- 重试次数（Counter）
- 当前活跃调用数（Gauge）
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY
from loguru import logger

# --- LLM 调用指标 ---

# Histogram: 自动计算 P50/P95/P99
llm_call_duration_seconds = Histogram(
    "llm_call_duration_seconds",
    "LLM API 调用延迟（秒）",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

# Counter: LLM 调用总量（按模型和状态）
llm_call_total = Counter(
    "llm_call_total",
    "LLM API 调用总次数",
    ["model", "status"],  # status: success / error / fallback
    registry=REGISTRY,
)

# Counter: LLM 重试次数
llm_retry_total = Counter(
    "llm_retry_total",
    "LLM API 重试总次数",
    ["model"],
    registry=REGISTRY,
)

# Gauge: 当前活跃调用数
llm_active_calls = Gauge(
    "llm_active_calls",
    "当前活跃的 LLM API 调用数",
    ["model"],
    registry=REGISTRY,
)

# --- Agent 执行指标 ---

agent_execution_total = Counter(
    "agent_execution_total",
    "Agent 执行总次数",
    ["agent_type", "status"],  # agent_type: rag / aiops, status: success / error / timeout
    registry=REGISTRY,
)

agent_step_duration_seconds = Histogram(
    "agent_step_duration_seconds",
    "AIOps Agent 单步执行耗时（秒）",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=REGISTRY,
)

agent_error_total = Counter(
    "agent_error_total",
    "Agent 错误总次数",
    ["agent_type", "node"],  # node: planner / executor / replanner
    registry=REGISTRY,
)


def track_llm_call(model: str, duration_s: float, status: str = "success"):
    """记录一次 LLM 调用指标。

    Args:
        model: 模型名称（如 qwen-max）。
        duration_s: 调用耗时（秒）。
        status: success / error / fallback。
    """
    try:
        llm_call_duration_seconds.observe(duration_s)
        llm_call_total.labels(model=model, status=status).inc()
    except Exception as e:
        logger.warning(f"Metrics 记录失败: {e}")


def track_llm_retry(model: str):
    """记录一次 LLM 重试。"""
    try:
        llm_retry_total.labels(model=model).inc()
    except Exception as e:
        logger.warning(f"Metrics 记录失败: {e}")


def track_agent_execution(agent_type: str, status: str):
    """记录一次 Agent 执行。

    Args:
        agent_type: rag / aiops。
        status: success / error / timeout。
    """
    try:
        agent_execution_total.labels(agent_type=agent_type, status=status).inc()
    except Exception as e:
        logger.warning(f"Metrics 记录失败: {e}")


def track_agent_error(agent_type: str, node: str):
    """记录一次 Agent 节点错误。

    Args:
        agent_type: rag / aiops。
        node: planner / executor / replanner。
    """
    try:
        agent_error_total.labels(agent_type=agent_type, node=node).inc()
    except Exception as e:
        logger.warning(f"Metrics 记录失败: {e}")


def get_metrics_response() -> bytes:
    """生成 Prometheus 格式的指标文本。

    Returns:
        bytes: Prometheus text format。
    """
    return generate_latest(REGISTRY)


logger.info("Prometheus Metrics 模块初始化完成")
