"""BasicRAGRetriever - 封装现有 Dense 向量检索逻辑"""

import time
import uuid
from typing import Any, Dict

from langchain_core.documents import Document
from loguru import logger

from app.retriever.base import BaseRAGRetriever
from app.services.vector_store_manager import vector_store_manager


class BasicRAGRetriever(BaseRAGRetriever):
    """基础 RAG 检索器，使用 Dense 向量检索（L2 距离），行为与重构前完全一致"""

    def __init__(self) -> None:
        super().__init__()
        self.last_retrieval_meta: Dict[str, Any] = {}

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        """从 Milvus biz collection 检索相关文档

        Args:
            query: 查询文本
            top_k: 返回文档数量

        Returns:
            list[Document]: 相关文档列表
        """
        trace_id = uuid.uuid4().hex[:8]
        t_start = time.time()
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)

        Content_PREVIEW_LEN = 120
        chunk_entries = []
        for rank, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            chunk_entries.append({
                "rank": rank,
                "score": None,  # Basic 模式无精排分数
                "selected": True,
                "selected_in_pool": True,
                "selected_in_context": False,
                "output_rank": rank,
                "truncation_reason": None,
                "file_name": metadata.get("_file_name", metadata.get("_source", "")),
                "source": metadata.get("_source", ""),
                "h1": metadata.get("h1", "") or "",
                "h2": metadata.get("h2", "") or "",
                "h3": metadata.get("h3", "") or "",
                "content_preview": doc.page_content[:Content_PREVIEW_LEN].replace("\n", " "),
                "chunk_id": metadata.get("chunk_id", ""),
            })

        self.last_retrieval_meta = {
            "trace_id": trace_id,
            "rag_mode": "basic",
            "candidate_count": len(docs),
            "final_count": len(docs),
            "rerank_source": "none",
            "total_time_ms": int((time.time() - t_start) * 1000),
            "chunk_diagnostics": {
                "rerank_source": "none",
                "total_candidates": len(docs),
                "final_selected": len(docs),
                "dropped_below_top_k": 0,
                "dropped_by_diversity": 0,
                "score_range": None,
                "chunks": chunk_entries,
            },
        }

        logger.debug(f"BasicRAGRetriever 检索完成: query='{query[:50]}', 结果数={len(docs)}")
        return docs
