"""CrossEncoderReranker — 使用 sentence-transformers Cross-Encoder 精排

模型: BAAI/bge-reranker-v2-m3
  - 多语言 Cross-Encoder，中英文表现优异
  - CPU 推理约 200–800ms（取决于候选数量和文本长度）
  - 模型大小约 560MB，首次使用时自动下载

模型加载策略:
  - 若配置了 RERANKER_MODEL_PATH（本地路径），优先从本地加载
  - 否则使用 RERANKER_MODEL 名称，由 sentence-transformers 自动下载（需联网）

精排原则：
  - 始终使用用户的「原始查询」对候选文档打分（而非改写后的查询）
  - 这样可以保证分数直接反映用户真实意图的相关性
"""

from langchain_core.documents import Document
from loguru import logger

from app.retriever.reranker.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """基于 sentence-transformers Cross-Encoder 的精排器（懒加载模型）

    每次调用 rerank() 后，可通过 last_scores 属性获取每个文档的分数，
    用于检索诊断和可观测性分析。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model_name = model_name
        self._model = None
        self.last_scores: list[tuple[float, Document]] = []

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                # 优先使用本地路径，避免 HF 下载超时
                from app.config import config

                model_path = config.reranker_model_path or self._model_name
                self._model = CrossEncoder(model_path)
                logger.info(
                    f"[Reranker:cross_encoder] 模型加载完成: {model_path}"
                )
            except ImportError as e:
                raise ImportError(
                    "CrossEncoderReranker 需要安装 sentence-transformers"
                ) from e
        return self._model

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        if not documents:
            self.last_scores = []
            return []

        if len(documents) <= top_k:
            logger.debug(
                f"[Reranker:cross_encoder] 候选数({len(documents)}) <= top_k({top_k})，跳过精排"
            )
            # 当候选数不足时，给每个文档一个占位分数（保持接口一致）
            self.last_scores = [(1.0, doc) for doc in documents]
            return documents

        try:
            model = self._get_model()

            pairs = [[query, doc.page_content] for doc in documents]
            scores = model.predict(pairs, apply_softmax=False)

            scored_docs = sorted(
                zip(scores, documents),
                key=lambda x: x[0],
                reverse=True,
            )
            self.last_scores = scored_docs  # 保存完整分数用于诊断
            reranked = [doc for _, doc in scored_docs[:top_k]]

            logger.debug(
                f"[Reranker:cross_encoder] 精排完成: "
                f"候选={len(documents)}, 返回={len(reranked)}, "
                f"top_score={scored_docs[0][0]:.4f}"
            )
            return reranked

        except Exception as e:
            logger.error(f"[Reranker:cross_encoder] 精排失败，回退到截断: {e}")
            self.last_scores = []  # 失败时清空分数
            return documents[:top_k]
