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

    @staticmethod
    def _doc_dedupe_key(doc: Document) -> str:
        """生成文档去重 key，优先使用来源和标题，缺失时退回内容前缀。"""
        metadata = doc.metadata or {}
        source = metadata.get("_source") or metadata.get("_file_name") or ""
        headers = "|".join(
            str(metadata.get(key, "")) for key in ("h1", "h2", "h3") if metadata.get(key)
        )
        if source or headers:
            return f"{source}::{headers}::{doc.page_content[:80]}"
        return doc.page_content[:200]

    @classmethod
    def _merge_candidates(cls, *candidate_lists: list[Document]) -> list[Document]:
        """按召回顺序合并候选，并去除同一分片的重复结果。"""
        merged: list[Document] = []
        seen: set[str] = set()
        for candidates in candidate_lists:
            for doc in candidates:
                key = cls._doc_dedupe_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
        return merged

    @staticmethod
    def _doc_file_key(doc: Document) -> str:
        """生成文档来源 key，用于最终 Top-K 的来源多样性选择。"""
        metadata = doc.metadata or {}
        return str(
            metadata.get("_file_name")
            or metadata.get("_source")
            or metadata.get("source")
            or ""
        )

    @classmethod
    def _diversify_by_file(cls, documents: list[Document], top_k: int) -> list[Document]:
        """按 rerank 顺序优先覆盖不同来源文件，再补充同文件后续分片。

        该策略不改变 reranker 给出的相对顺序，只在最终截断前减少同一
        _file_name 连续占满 Top-K 的情况，用于改善 cross_doc 查询的证据覆盖。
        """
        if len(documents) <= top_k:
            return documents

        selected: list[Document] = []
        deferred: list[Document] = []
        seen_files: set[str] = set()

        for doc in documents:
            file_key = cls._doc_file_key(doc)
            if file_key and file_key not in seen_files:
                selected.append(doc)
                seen_files.add(file_key)
                if len(selected) >= top_k:
                    return selected[:top_k]
            else:
                deferred.append(doc)

        for doc in deferred:
            if len(selected) >= top_k:
                break
            selected.append(doc)

        return selected[:top_k]

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
            "rewritten_candidate_count": 0,
            "original_candidate_count": 0,
            "rerank_pool_k": 0,
            "diversify_by_file": config.rag_diversify_by_file,
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
        rewritten_candidates = enhanced_vector_store_manager.hybrid_search(
            query=search_query,
            top_k=coarse_top_k,
            coarse_top_k=coarse_top_k,
        )
        candidates = rewritten_candidates

        # rewrite 可能提升语义召回，但也可能稀释原始关键词。额外保留原始 query
        # 的召回结果，再去重合并，兼顾口语化查询和精确术语查询。
        original_candidates: list[Document] = []
        if search_query.strip() != original_query.strip():
            original_candidates = enhanced_vector_store_manager.hybrid_search(
                query=original_query,
                top_k=coarse_top_k,
                coarse_top_k=coarse_top_k,
            )
            candidates = self._merge_candidates(rewritten_candidates, original_candidates)

        meta["rewritten_candidate_count"] = len(rewritten_candidates)
        meta["original_candidate_count"] = len(original_candidates)
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
            f"rewrite_candidates={len(rewritten_candidates)}, "
            f"original_candidates={len(original_candidates)}, "
            f"coarse_top_k={coarse_top_k}, "
            f"耗时={meta['hybrid_search_time_ms']}ms"
        )

        # ----------------------------------------------------------------
        # Step 3: Reranking（精排，降级：失败时回退粗排截断）
        # 注意：精排使用 original_query，而非 search_query（可能是改写后的查询）
        # ----------------------------------------------------------------
        t_stage3 = time.time()
        rerank_pool_k = top_k
        if config.rag_diversify_by_file:
            multiplier = max(1, config.rag_diversify_candidate_multiplier)
            rerank_pool_k = min(len(candidates), max(top_k, top_k * multiplier))
        meta["rerank_pool_k"] = rerank_pool_k

        try:
            reranker = get_reranker(config.reranker_type, config.reranker_model)
            reranked_docs = reranker.rerank(
                query=original_query,
                documents=candidates,
                top_k=rerank_pool_k,
            )
            if config.rag_diversify_by_file:
                final_docs = self._diversify_by_file(reranked_docs, top_k=top_k)
            else:
                final_docs = reranked_docs[:top_k]
        except Exception as e:
            logger.warning(
                f"[Enhanced][{trace_id}] 精排失败，回退到粗排截断 top_k={top_k}: {e}"
            )
            fallback_docs = candidates[:rerank_pool_k]
            if config.rag_diversify_by_file:
                final_docs = self._diversify_by_file(fallback_docs, top_k=top_k)
            else:
                final_docs = fallback_docs[:top_k]
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
            f"candidates={len(candidates)}, rerank_pool={rerank_pool_k}, "
            f"diversify_by_file={config.rag_diversify_by_file}, "
            f"final={len(final_docs)}, "
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
