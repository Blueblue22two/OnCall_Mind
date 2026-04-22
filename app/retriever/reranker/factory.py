"""Reranker 工厂函数

根据 config.reranker_type 返回对应的精排器单例。
当前支持：
  - "none"         → PassthroughReranker（直接截断，不精排）
  - "cross_encoder" → CrossEncoderReranker（BGE bge-reranker-v2-m3）

扩展方式：
  1. 新建 app/retriever/reranker/your_reranker.py 并继承 BaseReranker
  2. 在 _build_registry() 中注册新的类型字符串
"""

from functools import lru_cache

from app.retriever.reranker.base import BaseReranker


def _build_registry() -> dict[str, type[BaseReranker]]:
    """延迟构建注册表，避免循环导入和模型提前加载"""
    from app.retriever.reranker.cross_encoder import CrossEncoderReranker
    from app.retriever.reranker.passthrough import PassthroughReranker

    return {
        "none": PassthroughReranker,
        "cross_encoder": CrossEncoderReranker,
        # 后续可在此追加：
        # "llm": LLMReranker,
    }


@lru_cache(maxsize=8)
def get_reranker(reranker_type: str, model_name: str = "") -> BaseReranker:
    """根据类型字符串返回对应的精排器单例

    Args:
        reranker_type: 精排器类型，如 "none" / "cross_encoder"
        model_name: 精排模型名称（仅 cross_encoder 等需要指定模型时生效）

    Returns:
        BaseReranker: 对应的精排器实例

    Raises:
        ValueError: 未知的 reranker_type 时抛出
    """
    registry = _build_registry()
    cls = registry.get(reranker_type)
    if cls is None:
        supported = list(registry.keys())
        raise ValueError(
            f"未知的精排器类型: '{reranker_type}'，"
            f"支持的类型: {supported}"
        )

    # 如果指定了 model_name 且 cls 构造器支持该参数，透传进去
    if model_name and reranker_type != "none":
        return cls(model_name=model_name)  # type: ignore[call-arg]
    return cls()
