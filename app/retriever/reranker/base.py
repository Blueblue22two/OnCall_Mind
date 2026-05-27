"""Reranker 抽象基类"""

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseReranker(ABC):
    """精排器统一接口

    所有精排实现必须遵守此契约。
    输入原始查询 + 候选文档列表，输出经过精排的文档列表。
    """

    @abstractmethod
    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """对候选文档进行精排

        Args:
            query: 用于评分的查询字符串（始终使用用户原始查询，而非改写后的查询）
            documents: 待精排的候选文档列表
            top_k: 精排后返回的文档数量

        Returns:
            list[Document]: 按相关性降序排列的 top_k 个文档
        """
        ...
