"""LLM 工厂模块（P0-1.1, P0-1.2, P1-2.4）

统一 ChatQwen 实例化入口，消除各节点中的硬编码参数。

提供：
- 集中读取 config 的 temperature / timeout / max_retries
- 可选的 qwen-max → qwen-plus 自动 Fallback 链路
- 对话 Agent 专用工厂方法（使用 chat_temperature）
- P1-2.4: Prometheus Metrics 采集
"""

from __future__ import annotations

import time
from typing import Optional

from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config


def create_chat_qwen(
    temperature: Optional[float] = None,
    streaming: bool = False,
    enable_fallback: bool = True,
) -> ChatQwen:
    """创建 ChatQwen 实例（带可选的 Fallback 链路）。

    用于规划/执行/评估类节点（默认 temperature=config.llm_temperature, streaming=False）。

    Args:
        temperature: LLM 温度，为 None 时使用 config.llm_temperature。
        streaming: 是否启用流式输出。
        enable_fallback: 是否启用 fallback 链路（qwen-max → qwen-plus）。

    Returns:
        ChatQwen 实例，如果启用 fallback 则返回带 with_fallbacks() 的实例。
    """
    if temperature is None:
        temperature = config.llm_temperature

    t_start = time.monotonic()
    model_name = config.dashscope_model

    primary = ChatQwen(
        model=model_name,
        api_key=config.dashscope_api_key,
        api_base=config.dashscope_api_base,
        temperature=temperature,
        timeout=config.llm_timeout,
        max_retries=config.llm_max_retries,
        streaming=streaming,
    )

    # P1-2.4: 记录创建耗时
    try:
        from app.core.metrics import track_llm_call
        creation_s = time.monotonic() - t_start
        track_llm_call(model_name, creation_s, "success")
    except Exception:
        pass

    logger.debug(
        f"创建 ChatQwen: model={model_name}, "
        f"temperature={temperature}, timeout={config.llm_timeout}s, "
        f"max_retries={config.llm_max_retries}"
    )

    if enable_fallback and config.llm_fallback_model:
        fallback = ChatQwen(
            model=config.llm_fallback_model,
            api_key=config.dashscope_api_key,
            api_base=config.dashscope_api_base,
            temperature=temperature,
            timeout=config.llm_timeout,
            max_retries=config.llm_max_retries,
            streaming=streaming,
        )
        logger.info(f"LLM Fallback 链路已启用: {model_name} → {config.llm_fallback_model}")
        # P1-2.4: 标记 fallback 配置
        try:
            from app.core.metrics import track_llm_call
            track_llm_call(config.llm_fallback_model, 0, "fallback")
        except Exception:
            pass
        return primary.with_fallbacks([fallback])

    return primary


def create_chat_qwen_for_chat(streaming: bool = True) -> ChatQwen:
    """创建用于对话 Agent 的 ChatQwen 实例。

    使用 config.llm_chat_temperature（默认 0.7），适合对话场景。

    Args:
        streaming: 是否启用流式输出。

    Returns:
        对话专用 ChatQwen 实例。
    """
    return create_chat_qwen(
        temperature=config.llm_chat_temperature,
        streaming=streaming,
        enable_fallback=True,
    )
