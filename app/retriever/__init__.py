"""RAG 检索模块 - 可插拔检索接口"""

from app.retriever.base import BaseRAGRetriever
from app.retriever.factory import get_rag_retriever

__all__ = ["BaseRAGRetriever", "get_rag_retriever"]
