"""查询预处理模块

提供可插拔的查询预处理器，当前支持：
  - "none"    → PassthroughPreprocessor（直接透传）
  - "rewrite" → QueryRewritePreprocessor（LLM 改写）

使用方式：
    from app.retriever.preprocessing import get_query_preprocessor
    preprocessor = get_query_preprocessor("rewrite")
    rewritten_query = preprocessor.process(query)

扩展方式：
    1. 新建 app/retriever/preprocessing/your_method.py 并继承 BaseQueryPreprocessor
    2. 在 factory.py 的 _build_registry() 中注册新类型字符串
"""

from app.retriever.preprocessing.base import BaseQueryPreprocessor
from app.retriever.preprocessing.factory import get_query_preprocessor
from app.retriever.preprocessing.passthrough import PassthroughPreprocessor
from app.retriever.preprocessing.rewrite import QueryRewritePreprocessor

__all__ = [
    "BaseQueryPreprocessor",
    "get_query_preprocessor",
    "PassthroughPreprocessor",
    "QueryRewritePreprocessor",
]
