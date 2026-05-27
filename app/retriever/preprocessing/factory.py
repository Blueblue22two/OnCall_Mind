"""查询预处理器工厂函数

根据 config.query_preprocessor_type 返回对应的预处理器单例。
当前支持：
  - "none"    → PassthroughPreprocessor（直接透传）
  - "rewrite" → QueryRewritePreprocessor（LLM 改写）

扩展方式：新增实现后，在 _REGISTRY 中注册对应类型字符串即可。
"""

from functools import lru_cache

from app.retriever.preprocessing.base import BaseQueryPreprocessor


# 已注册的预处理器类型 → 类映射（扩展时在此添加）
_REGISTRY: dict[str, type[BaseQueryPreprocessor]] = {}


def _build_registry() -> dict[str, type[BaseQueryPreprocessor]]:
    """延迟构建注册表，避免循环导入"""
    from app.retriever.preprocessing.passthrough import PassthroughPreprocessor
    from app.retriever.preprocessing.rewrite import QueryRewritePreprocessor

    return {
        "none": PassthroughPreprocessor,
        "rewrite": QueryRewritePreprocessor,
    }


@lru_cache(maxsize=8)
def get_query_preprocessor(preprocessor_type: str) -> BaseQueryPreprocessor:
    """根据类型字符串返回对应的预处理器单例

    Args:
        preprocessor_type: 预处理器类型，如 "none" / "rewrite"

    Returns:
        BaseQueryPreprocessor: 对应的预处理器实例

    Raises:
        ValueError: 未知的 preprocessor_type 时抛出
    """
    registry = _build_registry()
    cls = registry.get(preprocessor_type)
    if cls is None:
        supported = list(registry.keys())
        raise ValueError(
            f"未知的预处理器类型: '{preprocessor_type}'，"
            f"支持的类型: {supported}"
        )
    return cls()
