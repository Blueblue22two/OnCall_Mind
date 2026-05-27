"""Reranker 精排模块

提供可插拔的精排器，当前支持：
  - "none"          → PassthroughReranker（直接截断，不精排）
  - "cross_encoder" → CrossEncoderReranker（BAAI/bge-reranker-v2-m3）

使用方式：
    from app.retriever.reranker import get_reranker
    reranker = get_reranker("cross_encoder", "BAAI/bge-reranker-v2-m3")
    reranked_docs = reranker.rerank(original_query, candidate_docs, top_k=3)

扩展方式：
    1. 新建 app/retriever/reranker/your_reranker.py 并继承 BaseReranker
    2. 在 factory.py 的 _build_registry() 中注册新类型字符串
"""

from app.retriever.reranker.base import BaseReranker
from app.retriever.reranker.cross_encoder import CrossEncoderReranker
from app.retriever.reranker.factory import get_reranker
from app.retriever.reranker.passthrough import PassthroughReranker

__all__ = [
    "BaseReranker",
    "get_reranker",
    "PassthroughReranker",
    "CrossEncoderReranker",
]
