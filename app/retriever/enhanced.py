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

import statistics
import time
import uuid
from typing import Any, Dict, Optional

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

    @classmethod
    def _rank_with_section_prior(
        cls,
        reranker: Any,
        fallback_docs: list[Document],
        target_section_types: tuple[str, ...],
        prior: float,
        normalize_scores: bool,
    ) -> list[Document]:
        """Normalize per-query scores and optionally boost intent-matched sections."""
        if not normalize_scores or not getattr(reranker, "last_scores", None):
            return fallback_docs
        raw_scores = [float(score) for score, _ in reranker.last_scores]
        low, high = min(raw_scores), max(raw_scores)
        span = high - low
        adjusted: list[tuple[float, Document]] = []
        for raw_score, doc in reranker.last_scores:
            normalized = (float(raw_score) - low) / span if span > 1e-12 else 1.0
            section_type = str((doc.metadata or {}).get("section_type", "general"))
            boost = prior if section_type in target_section_types else 0.0
            adjusted_score = normalized + boost
            doc.metadata["_rerank_raw_score"] = float(raw_score)
            doc.metadata["_rerank_normalized_score"] = normalized
            doc.metadata["_rerank_adjusted_score"] = adjusted_score
            adjusted.append((adjusted_score, doc))
        adjusted.sort(key=lambda item: item[0], reverse=True)
        reranker.last_scores = adjusted
        return [doc for _, doc in adjusted]

    @classmethod
    def _guarded_cross_doc_diversity(
        cls,
        ranked_docs: list[Document],
        top_k: int,
        max_per_file: int,
        score_margin: float,
    ) -> list[Document]:
        """Replace excessive same-file chunks only when relevance loss is bounded."""
        selected = list(ranked_docs[:top_k])
        outside = list(ranked_docs[top_k:])

        def score(doc: Document) -> float:
            return float((doc.metadata or {}).get("_rerank_adjusted_score", 0.0))

        while True:
            counts: dict[str, int] = {}
            for doc in selected:
                key = cls._doc_file_key(doc)
                counts[key] = counts.get(key, 0) + 1
            overfull = {key for key, count in counts.items() if key and count > max_per_file}
            if not overfull:
                break
            replace_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if cls._doc_file_key(selected[index]) in overfull
                ),
                None,
            )
            candidate_index = next(
                (
                    index
                    for index, doc in enumerate(outside)
                    if counts.get(cls._doc_file_key(doc), 0) < max_per_file
                ),
                None,
            )
            if replace_index is None or candidate_index is None:
                break
            replacement = outside[candidate_index]
            if score(selected[replace_index]) - score(replacement) > score_margin:
                break
            outside.append(selected[replace_index])
            selected[replace_index] = outside.pop(candidate_index)

        rank_map = {id(doc): rank for rank, doc in enumerate(ranked_docs)}
        return sorted(selected, key=lambda doc: rank_map[id(doc)])

    @staticmethod
    def _expand_parent_context(
        documents: list[Document], max_chars: int, max_tokens: int = 3000
    ) -> list[Document]:
        """Expand one child per parent while enforcing a shared context token budget."""
        def truncate(text: str, budget: int) -> tuple[str, int]:
            """Offline token estimate: CJK≈1 token, other text≈4 chars/token."""
            used = 0.0
            end = 0
            for index, char in enumerate(text, 1):
                cost = 1.0 if "\u4e00" <= char <= "\u9fff" else 0.25
                if used + cost > budget:
                    break
                used += cost
                end = index
            return text[:end], max(1, int(used + 0.999))

        expanded: list[Document] = []
        seen_parents: set[str] = set()
        remaining_tokens = max_tokens
        for doc in documents:
            if remaining_tokens <= 0:
                break
            metadata = dict(doc.metadata or {})
            parent_id = str(metadata.get("parent_id", ""))
            parent_content = str(metadata.get("_parent_content", ""))
            if parent_id and parent_content and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                metadata["context_expanded"] = "parent"
                candidate = parent_content[:max_chars]
            else:
                metadata["context_expanded"] = "child"
                candidate = doc.page_content
            content, used_tokens = truncate(candidate, remaining_tokens)
            remaining_tokens -= used_tokens
            metadata["estimated_context_tokens"] = used_tokens
            expanded.append(Document(page_content=content, metadata=metadata))
        return expanded

    @staticmethod
    def _build_chunk_diagnostics(
        reranker: Any,
        rerank_source: str,
        candidates: list[Document],
        final_docs: list[Document],
        top_k: int,
        diversify_by_file: bool,
    ) -> Dict[str, Any]:
        """构建分片级诊断信息，用于定位精排、分块或截断环节的瓶颈。

        记录了候选池中每个分片的分数、元数据、是否被选中及截断原因。
        """
        Content_PREVIEW_LEN = 120

        # CrossEncoder.last_scores 已按得分降序；透传/降级时保持粗排顺序。
        ranked_candidates: list[Document] = list(candidates)
        rerank_scores: Dict[int, float] = {}
        if hasattr(reranker, "last_scores") and reranker.last_scores:
            for score, doc in reranker.last_scores:
                rerank_scores[id(doc)] = float(score)
            ranked_candidates = [doc for _, doc in reranker.last_scores]

        chunk_entries: list[Dict[str, Any]] = []

        # 标记最终选中的文档 ID
        final_ids: set[int] = {id(doc) for doc in final_docs}
        final_order = {id(doc): rank for rank, doc in enumerate(final_docs, 1)}

        # 确定截断原因
        def _truncation_reason(doc: Document, rank: int, selected: bool) -> Optional[str]:
            if selected:
                return None
            if diversify_by_file and rank <= top_k:
                return "diversity_skip"
            return "below_top_k"

        for rank, doc in enumerate(ranked_candidates, 1):
            metadata = doc.metadata or {}
            score = rerank_scores.get(id(doc))
            selected = id(doc) in final_ids

            entry: Dict[str, Any] = {
                "rank": rank,
                "score": round(score, 6) if score is not None else None,
                "raw_score": metadata.get("_rerank_raw_score"),
                "adjusted_score": metadata.get("_rerank_adjusted_score"),
                "selected": selected,  # backward-compatible alias
                "selected_in_pool": selected,
                "selected_in_context": False,  # evaluation layer fills this
                "output_rank": final_order.get(id(doc)),
                "truncation_reason": _truncation_reason(doc, rank, selected),
                "file_name": metadata.get("_file_name", metadata.get("_source", "")),
                "source": metadata.get("_source", ""),
                "h1": metadata.get("h1", "") or "",
                "h2": metadata.get("h2", "") or "",
                "h3": metadata.get("h3", "") or "",
                "content_preview": doc.page_content[:Content_PREVIEW_LEN].replace("\n", " "),
                "chunk_id": metadata.get("chunk_id", ""),
            }
            chunk_entries.append(entry)

        # 汇总统计
        valid_scores = [e["score"] for e in chunk_entries if e["score"] is not None]
        score_range: Optional[Dict[str, float]] = None
        if valid_scores:
            score_range = {
                "min": round(min(valid_scores), 6),
                "max": round(max(valid_scores), 6),
                "mean": round(statistics.mean(valid_scores), 6),
                "n": len(valid_scores),
            }

        selected_count = sum(1 for e in chunk_entries if e["selected"])
        diversity_skipped = sum(
            1 for e in chunk_entries if e.get("truncation_reason") == "diversity_skip"
        )
        below_top_k_count = sum(
            1 for e in chunk_entries if e.get("truncation_reason") == "below_top_k"
        )

        return {
            "rerank_source": rerank_source,
            "total_candidates": len(candidates),
            "final_selected": selected_count,
            "dropped_below_top_k": below_top_k_count,
            "dropped_by_diversity": diversity_skipped,
            "score_range": score_range,
            "chunks": chunk_entries,
        }

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
        from app.retriever.query_router import classify_query
        from app.retriever.reranker.factory import get_reranker
        from app.services.enhanced_vector_store_manager import enhanced_vector_store_manager

        trace_id = uuid.uuid4().hex[:8]
        t_start = time.time()
        original_query = query
        route = classify_query(query) if config.rag_query_routing else None
        query_type = route.query_type if route else "general"
        effective_top_k = (
            max(top_k, config.rag_cross_doc_top_k)
            if query_type == "cross_doc"
            else top_k
        )

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
            "query_type": query_type,
            "requested_top_k": top_k,
            "effective_top_k": effective_top_k,
            "total_time_ms": 0,
        }

        # ----------------------------------------------------------------
        # Step 1: Query Preprocessing（降级：失败时回退原始 query）
        # ----------------------------------------------------------------
        t_stage1 = time.time()
        if route and route.skip_rewrite:
            search_query = original_query
            meta["rewrite_skipped_by_router"] = True
        else:
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
        meta["preprocessing_time_ms"] = int((time.time() - t_stage1) * 1000)

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
        rerank_pool_k = effective_top_k
        if route and route.query_type in {"procedural", "cross_doc"}:
            # Section prior and guarded diversity need candidates outside final Top-K.
            rerank_pool_k = len(candidates)
        elif config.rag_diversify_by_file:
            multiplier = max(1, config.rag_diversify_candidate_multiplier)
            rerank_pool_k = min(len(candidates), max(top_k, top_k * multiplier))
        meta["rerank_pool_k"] = rerank_pool_k

        rerank_source: str = config.reranker_type  # 实际使用的精排来源
        reranker = get_reranker(config.reranker_type, config.reranker_model)
        try:
            reranked_docs = reranker.rerank(
                query=original_query,
                documents=candidates,
                top_k=rerank_pool_k,
            )
            target_sections = route.target_section_types if route else ()
            ranked_docs = self._rank_with_section_prior(
                reranker,
                fallback_docs=reranked_docs,
                target_section_types=target_sections,
                prior=config.rag_section_prior if config.rag_query_routing else 0.0,
                normalize_scores=(
                    config.rag_query_routing
                    and (bool(target_sections) or query_type == "cross_doc")
                ),
            )
            if query_type == "cross_doc" and config.rag_query_routing:
                final_docs = self._guarded_cross_doc_diversity(
                    ranked_docs,
                    top_k=effective_top_k,
                    max_per_file=config.rag_max_chunks_per_file,
                    score_margin=config.rag_diversity_score_margin,
                )
            elif config.rag_diversify_by_file:
                final_docs = self._diversify_by_file(
                    ranked_docs, top_k=effective_top_k
                )
            else:
                final_docs = ranked_docs[:effective_top_k]
        except Exception as e:
            logger.warning(
                f"[Enhanced][{trace_id}] 精排失败，回退到粗排截断 top_k={top_k}: {e}"
            )
            fallback_docs = candidates[:rerank_pool_k]
            if config.rag_diversify_by_file:
                final_docs = self._diversify_by_file(
                    fallback_docs, top_k=effective_top_k
                )
            else:
                final_docs = fallback_docs[:effective_top_k]
            rerank_source = "coarse_truncation"
            if meta["degraded_stage"] is None:
                meta["degraded_stage"] = "reranker"
            else:
                meta["degraded_stage"] = f"{meta['degraded_stage']},reranker"
            prev = meta["fallback_reason"] or ""
            meta["fallback_reason"] = f"{prev}; 精排失败: {e}".strip("; ")

        meta["final_count"] = len(final_docs)
        meta["reranker_time_ms"] = int((time.time() - t_stage3) * 1000)
        meta["total_time_ms"] = int((time.time() - t_start) * 1000)

        # ----------------------------------------------------------------
        # Step 3b: 构建分片级诊断信息（P0-1）
        # ----------------------------------------------------------------
        chunk_diagnostics = self._build_chunk_diagnostics(
            reranker=reranker,
            rerank_source=rerank_source,
            candidates=candidates,
            final_docs=final_docs,
            top_k=effective_top_k,
            diversify_by_file=(
                config.rag_diversify_by_file
                or (query_type == "cross_doc" and config.rag_query_routing)
            ),
        )
        meta["chunk_diagnostics"] = chunk_diagnostics
        meta["rerank_source"] = rerank_source

        output_docs = final_docs
        if config.rag_parent_context:
            output_docs = self._expand_parent_context(
                final_docs,
                max_chars=config.rag_parent_context_max_chars,
                max_tokens=config.rag_parent_context_max_tokens,
            )
        meta["parent_context_expanded"] = config.rag_parent_context

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
                f"final_sources={[d.metadata.get('_source','?')[:40] for d in output_docs]}"
            )

        return output_docs
