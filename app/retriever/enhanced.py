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

降级路径：
  - 预处理失败 → 回退到原始 query 进行检索
  - 精排失败 → 回退到粗排候选直接截断
  - 混合检索失败 → 抛出异常（基础设施问题不应静默降级）
"""

import time
import uuid
from typing import Any, Dict

from langchain_core.documents import Document
from loguru import logger

from app.retriever.base import BaseRAGRetriever


class EnhancedRAGRetriever(BaseRAGRetriever):
    """增强 RAG 检索器：Preprocessing → Hybrid Search → Reranking

    每次检索会生成 trace_id 并记录结构化日志，覆盖三阶段的输入输出。
    各阶段失败时有明确的降级策略，降级信息记录在 last_retrieval_meta 中。
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_retrieval_meta: Dict[str, Any] = {}

    def retrieve(self, query: str, top_k: int, debug: bool = False) -> list[Document]:
        """执行完整增强检索 pipeline

        Args:
            query: 用户原始查询
            top_k: 最终返回的文档数量
            debug: 是否输出中间结果摘要（记录到 debug 日志）

        Returns:
            list[Document]: 精排后的相关文档列表
        """
        from app.config import config
        from app.retriever.preprocessing.factory import get_query_preprocessor
        from app.retriever.reranker.factory import get_reranker
        from app.services.enhanced_vector_store_manager import enhanced_vector_store_manager

        trace_id = uuid.uuid4().hex[:8]
        t_start = time.time()
        original_query = query

        meta: Dict[str, Any] = {
            "trace_id": trace_id,
            "preprocessor_type": config.query_preprocessor_type,
            "reranker_type": config.reranker_type,
            "degraded_stage": None,
            "fallback_reason": None,
            "candidate_count": 0,
            "final_count": 0,
            "total_time_ms": 0,
        }

        # ----------------------------------------------------------------
        # Step 1: Query Preprocessing（降级：失败时回退原始 query）
        # ----------------------------------------------------------------
        preprocessor = get_query_preprocessor(config.query_preprocessor_type)
        try:
            search_query = preprocessor.process(query)
        except Exception as e:
            logger.warning(
                f"[Enhanced][{trace_id}] 预处理失败，回退到原始查询: {e}"
            )
            search_query = original_query
            meta["degraded_stage"] = "preprocessing"
            meta["fallback_reason"] = f"预处理失败: {e}"

        meta["search_query"] = search_query if debug else search_query[:80]

        logger.info(
            f"[Enhanced][{trace_id}] Stage1-预处理完成: "
            f"type={config.query_preprocessor_type}, "
            f"original='{original_query[:50]}...', "
            f"search='{search_query[:50]}...'"
        )

        # ----------------------------------------------------------------
        # Step 2: Hybrid Search（粗排，不下沉——基础设施问题必须暴露）
        # ----------------------------------------------------------------
        coarse_top_k = config.rerank_coarse_top_k
        t_stage2 = time.time()
        candidates = enhanced_vector_store_manager.hybrid_search(
            query=search_query,
            top_k=coarse_top_k,
            coarse_top_k=coarse_top_k,
        )
        meta["candidate_count"] = len(candidates)
        meta["hybrid_search_time_ms"] = int((time.time() - t_stage2) * 1000)

        if not candidates:
            logger.warning(
                f"[Enhanced][{trace_id}] Stage2-混合检索未召回任何候选文档, "
                f"耗时={meta['hybrid_search_time_ms']}ms"
            )
            meta["total_time_ms"] = int((time.time() - t_start) * 1000)
            self.last_retrieval_meta = meta
            return []

        logger.info(
            f"[Enhanced][{trace_id}] Stage2-混合检索完成: "
            f"candidates={len(candidates)}, "
            f"coarse_top_k={coarse_top_k}, "
            f"耗时={meta['hybrid_search_time_ms']}ms"
        )

        # ----------------------------------------------------------------
        # Step 3: Reranking（精排，降级：失败时回退粗排截断）
        # 注意：精排使用 original_query，而非 search_query（可能是改写后的查询）
        # ----------------------------------------------------------------
        t_stage3 = time.time()
        try:
            reranker = get_reranker(config.reranker_type, config.reranker_model)
            final_docs = reranker.rerank(
                query=original_query,
                documents=candidates,
                top_k=top_k,
            )
        except Exception as e:
            logger.warning(
                f"[Enhanced][{trace_id}] 精排失败，回退到粗排截断 top_k={top_k}: {e}"
            )
            final_docs = candidates[:top_k]
            if meta["degraded_stage"] is None:
                meta["degraded_stage"] = "reranker"
            else:
                meta["degraded_stage"] = f"{meta['degraded_stage']},reranker"
            prev = meta["fallback_reason"] or ""
            meta["fallback_reason"] = f"{prev}; 精排失败: {e}".strip("; ")

        meta["final_count"] = len(final_docs)
        meta["reranker_time_ms"] = int((time.time() - t_stage3) * 1000)
        meta["total_time_ms"] = int((time.time() - t_start) * 1000)
        self.last_retrieval_meta = meta

        # ----------------------------------------------------------------
        # 结构化摘要日志
        # ----------------------------------------------------------------
        if meta["degraded_stage"]:
            degraded_info = (
                f"degraded={meta['degraded_stage']}, "
                f"reason={meta['fallback_reason']}"
            )
        else:
            degraded_info = "degraded=无"
        logger.info(
            f"[Enhanced][{trace_id}] 检索完成: "
            f"preprocessor={config.query_preprocessor_type}, "
            f"reranker={config.reranker_type}, "
            f"candidates={len(candidates)}, final={len(final_docs)}, "
            f"total_ms={meta['total_time_ms']}, "
            f"{degraded_info}"
        )

        if debug:
            logger.debug(
                f"[Enhanced][{trace_id}] Debug: "
                f"original_query='{original_query}', "
                f"search_query='{search_query}', "
                f"candidate_sources={[d.metadata.get('_source','?')[:40] for d in candidates[:5]]}, "
                f"final_sources={[d.metadata.get('_source','?')[:40] for d in final_docs]}"
            )

        return final_docs
