"""BasicRAGRetriever - 封装现有 Dense 向量检索逻辑"""

from langchain_core.documents import Document
from loguru import logger

from app.retriever.base import BaseRAGRetriever
from app.services.vector_store_manager import vector_store_manager


class BasicRAGRetriever(BaseRAGRetriever):
    """基础 RAG 检索器，使用 Dense 向量检索（L2 距离），行为与重构前完全一致"""

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        """从 Milvus biz collection 检索相关文档

        Args:
            query: 查询文本
            top_k: 返回文档数量

        Returns:
            list[Document]: 相关文档列表
        """
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)
        logger.debug(f"BasicRAGRetriever 检索完成: query='{query}', 结果数={len(docs)}")
        return docs
