"""PassthroughReranker — none 模式，不做精排直接透传

用于在禁用精排时保持 Pipeline 接口统一。
"""

from langchain_core.documents import Document
from loguru import logger

from app.retriever.reranker.base import BaseReranker


class PassthroughReranker(BaseReranker):
    """透传精排器：不做任何重排序，直接返回前 top_k 个文档

    为保持与 CrossEncoderReranker 的接口一致，也提供 last_scores 属性，
    但值始终为空列表（透传模式不产生分数）。
    """

    def __init__(self) -> None:
        self.last_scores: list[tuple[float, Document]] = []

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        logger.debug(f"[Reranker:none] 透传，返回前 {top_k} 个文档")
        self.last_scores = []
        return documents[:top_k]
