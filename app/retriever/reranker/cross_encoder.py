"""CrossEncoderReranker — 使用 FlagEmbedding BGE Cross-Encoder 精排

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
    """基于 BGE Cross-Encoder 的精排器（懒加载模型）"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model_name = model_name
        self._model = None  # 懒初始化，避免导入时加载 560MB 模型

    def _get_model(self):
        """懒初始化 FlagReranker（首次调用时加载模型权重）"""
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker  # type: ignore[import-untyped]

                self._model = FlagReranker(
                    self._model_name,
                    use_fp16=True,  # 半精度推理，减少内存占用并加速 CPU 推理
                )
                logger.info(f"[Reranker:cross_encoder] 模型加载完成: {self._model_name}")
            except ImportError as e:
                raise ImportError(
                    "CrossEncoderReranker 需要安装 FlagEmbedding: "
                    "pip install FlagEmbedding>=1.2.0"
                ) from e
        return self._model

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """使用 Cross-Encoder 对文档列表重排序

        Args:
            query: 用户原始查询（用于评分，不用改写后的查询）
            documents: 候选文档列表
            top_k: 精排后返回数量

        Returns:
            list[Document]: 按 Cross-Encoder 分数降序排列的 top_k 文档
        """
        if not documents:
            return []

        if len(documents) <= top_k:
            # 候选数量不超过 top_k，无需精排
            logger.debug(
                f"[Reranker:cross_encoder] 候选数({len(documents)}) <= top_k({top_k})，跳过精排"
            )
            return documents

        try:
            model = self._get_model()

            # 构造 (query, passage) 对
            pairs = [[query, doc.page_content] for doc in documents]

            # 批量打分（分数越高越相关）
            scores = model.compute_score(pairs, normalize=True)

            # 按分数降序排列，取前 top_k
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
            # 精排失败时安全回退：返回原始顺序的前 top_k 个
            return documents[:top_k]
