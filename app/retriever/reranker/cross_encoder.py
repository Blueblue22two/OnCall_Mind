"""CrossEncoderReranker — 使用 sentence-transformers Cross-Encoder 精排

模型: BAAI/bge-reranker-v2-m3
  - 多语言 Cross-Encoder，中英文表现优异
  - CPU 推理约 200–800ms（取决于候选数量和文本长度）
  - 模型大小约 560MB，首次使用时自动下载

精排原则：
  - 始终使用用户的「原始查询」对候选文档打分（而非改写后的查询）
  - 这样可以保证分数直接反映用户真实意图的相关性
"""

from langchain_core.documents import Document
from loguru import logger

from app.retriever.reranker.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """基于 sentence-transformers Cross-Encoder 的精排器（懒加载模型）"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name)
                logger.info(f"[Reranker:cross_encoder] 模型加载完成: {self._model_name}")
            except ImportError as e:
                raise ImportError(
                    "CrossEncoderReranker 需要安装 sentence-transformers"
                ) from e
        return self._model

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        if not documents:
            return []

        if len(documents) <= top_k:
            logger.debug(
                f"[Reranker:cross_encoder] 候选数({len(documents)}) <= top_k({top_k})，跳过精排"
            )
            return documents

        try:
            model = self._get_model()

            pairs = [[query, doc.page_content] for doc in documents]
            scores = model.predict(pairs, apply_softmax=True)

            scored_docs = sorted(
                zip(scores, documents),
                key=lambda x: x[0],
                reverse=True,
            )
            reranked = [doc for _, doc in scored_docs[:top_k]]

            logger.debug(
                f"[Reranker:cross_encoder] 精排完成: "
                f"候选={len(documents)}, 返回={len(reranked)}, "
                f"top_score={scored_docs[0][0]:.4f}"
            )
            return reranked

        except Exception as e:
            logger.error(f"[Reranker:cross_encoder] 精排失败，回退到截断: {e}")
            return documents[:top_k]
