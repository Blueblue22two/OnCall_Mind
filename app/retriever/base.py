"""RAG 检索器抽象基类"""

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseRAGRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        """统一检索接口，所有实现必须遵守此契约"""
        ...
