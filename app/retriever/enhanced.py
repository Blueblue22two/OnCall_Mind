"""EnhancedRAGRetriever — 完整增强检索 pipeline

执行流程：
  1. Query Preprocessing（查询预处理）
     - none:    直接使用原始查询
     - rewrite: 使用 LLM 对查询进行语义增强改写
  2. Hybrid Search（混合检索）
     - Dense ANN: DashScope text-embedding-v4（COSINE）
     - Sparse BM25: Milvus 内置 BM25 Function（中文 Jieba 分词）
     - RRF 融合（k=60），粗排候选数 = rerank_coarse_top_k
  3. Reranking（精排）
     - none:         直接截断到 top_k
     - cross_encoder: BGE bge-reranker-v2-m3 Cross-Encoder 精排

关键设计决策：
  - 预处理后的查询用于混合检索（dense + sparse），以获得更好的召回
  - 精排始终使用用户「原始查询」打分，确保分数直接反映用户真实意图
"""

from langchain_core.documents import Document
from loguru import logger

from app.retriever.base import BaseRAGRetriever


class EnhancedRAGRetriever(BaseRAGRetriever):
    """增强 RAG 检索器：Preprocessing → Hybrid Search → Reranking"""

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        """执行完整增强检索 pipeline

        Args:
            query: 用户原始查询
            top_k: 最终返回的文档数量

        Returns:
            list[Document]: 精排后的相关文档列表
        """
        from app.config import config
        from app.retriever.preprocessing.factory import get_query_preprocessor
        from app.retriever.reranker.factory import get_reranker
        from app.services.enhanced_vector_store_manager import enhanced_vector_store_manager

        original_query = query

        # ----------------------------------------------------------------
        # Step 1: Query Preprocessing
        # ----------------------------------------------------------------
        preprocessor = get_query_preprocessor(config.query_preprocessor_type)
        search_query = preprocessor.process(query)

        logger.info(
            f"[Enhanced] 开始检索: preprocessor={config.query_preprocessor_type}, "
            f"reranker={config.reranker_type}, top_k={top_k}"
        )

        # ----------------------------------------------------------------
        # Step 2: Hybrid Search（粗排）
        # ----------------------------------------------------------------
        coarse_top_k = config.rerank_coarse_top_k
        candidates = enhanced_vector_store_manager.hybrid_search(
            query=search_query,
            top_k=coarse_top_k,
            coarse_top_k=coarse_top_k,
        )

        if not candidates:
            logger.warning("[Enhanced] 混合检索未召回任何候选文档")
            return []

        logger.info(f"[Enhanced] 混合检索召回 {len(candidates)} 个候选文档")

        # ----------------------------------------------------------------
        # Step 3: Reranking（精排）
        # 注意：精排使用 original_query，而非 search_query（可能是改写后的查询）
        # ----------------------------------------------------------------
        reranker = get_reranker(config.reranker_type, config.reranker_model)
        final_docs = reranker.rerank(
            query=original_query,
            documents=candidates,
            top_k=top_k,
        )

        logger.info(
            f"[Enhanced] 检索完成: 候选={len(candidates)}, 精排后={len(final_docs)}"
        )
        return final_docs
