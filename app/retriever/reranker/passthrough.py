"""PassthroughReranker — none 模式，不做精排直接透传

用于在禁用精排时保持 Pipeline 接口统一。
"""

from langchain_core.documents import Document
from loguru import logger

from app.retriever.reranker.base import BaseReranker


class PassthroughReranker(BaseReranker):
    """透传精排器：不做任何重排序，直接返回前 top_k 个文档"""

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        logger.debug(f"[Reranker:none] 透传，返回前 {top_k} 个文档")
        return documents[:top_k]
