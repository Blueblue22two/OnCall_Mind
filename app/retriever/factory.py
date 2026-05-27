"""RAG 检索器工厂 - 根据配置返回对应实现"""

from functools import lru_cache

from app.retriever.base import BaseRAGRetriever


@lru_cache(maxsize=1)
def get_rag_retriever() -> BaseRAGRetriever:
    """根据 config.rag_mode 返回对应的 RAG 检索器单例

    Returns:
        BaseRAGRetriever: 检索器实例（懒初始化，首次调用时创建）
    """
    from app.config import config
    from app.retriever.basic import BasicRAGRetriever

    if config.rag_mode == "enhanced":
        from app.retriever.enhanced import EnhancedRAGRetriever
        return EnhancedRAGRetriever()

    return BasicRAGRetriever()
