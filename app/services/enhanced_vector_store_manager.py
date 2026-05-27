"""增强向量存储管理器 - 负责 biz_enhanced collection 的写入与检索

biz_enhanced collection 使用双向量 schema：
  - dense_vector: DashScope text-embedding-v4（1024 维, COSINE）
  - sparse_vector: Milvus 内置 BM25 Function 自动生成（无需客户端写入）
  - content_text: 原始文本（BM25 的输入来源，同时作为 langchain Document.page_content）
  - metadata: JSON 元数据
"""

import time
import uuid
from typing import Any, Dict, List

from langchain_core.documents import Document
from loguru import logger
from pymilvus import AnnSearchRequest, Collection, RRFRanker, WeightedRanker

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


class EnhancedVectorStoreManager:
    """biz_enhanced collection 的写入与混合检索管理器"""

    COLLECTION_NAME = "biz_enhanced"

    def __init__(self) -> None:
        self._collection: Collection | None = None

    def _get_collection(self) -> Collection:
        """懒获取 collection 句柄（connect 后才可用）"""
        if self._collection is None:
            self._collection = milvus_manager.get_enhanced_collection()
        return self._collection

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> List[str]:
        """批量写入文档到 biz_enhanced

        流程：
          1. 调用 DashScope Embedding API 获取 dense_vector
          2. 构造写入数据（sparse_vector 由 Milvus BM25 Function 自动填充）
          3. 插入 Collection

        Args:
            documents: LangChain Document 列表

        Returns:
            List[str]: 生成的文档 ID 列表
        """
        if not documents:
            return []

        try:
            start_time = time.time()

            # 1. 生成 dense 向量（批量调用 DashScope）
            texts = [doc.page_content for doc in documents]
            dense_vectors = vector_embedding_service.embed_documents(texts)

            # 2. 构造插入数据
            ids = [str(uuid.uuid4()) for _ in documents]
            entities: List[Dict[str, Any]] = []
            for doc_id, doc, dense_vec in zip(ids, documents, dense_vectors):
                entity: Dict[str, Any] = {
                    "id": doc_id,
                    "dense_vector": dense_vec,
                    "content_text": doc.page_content,
                    # sparse_vector 由 Milvus BM25 Function 自动填充，不需要提供
                    "metadata": doc.metadata,
                }
                entities.append(entity)

            # 3. 插入 collection
            collection = self._get_collection()
            collection.insert(entities)
            collection.flush()

            elapsed = time.time() - start_time
            logger.info(
                f"[Enhanced] 批量写入 {len(documents)} 个文档完成, "
                f"耗时: {elapsed:.2f}秒"
            )
            return ids

        except Exception as e:
            logger.error(f"[Enhanced] 写入文档失败: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """删除指定文件来源的所有文档

        Args:
            file_path: 文件路径（与 metadata._source 对应）

        Returns:
            int: 删除数量
        """
        try:
            collection = self._get_collection()
            expr = f'metadata["_source"] == "{file_path}"'
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            logger.info(f"[Enhanced] 删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count
        except Exception as e:
            logger.warning(f"[Enhanced] 删除旧数据失败（可能是首次索引）: {e}")
            return 0

    # ------------------------------------------------------------------
    # 混合检索（Dense ANN + Sparse BM25 → RRF 融合）
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int,
        coarse_top_k: int | None = None,
    ) -> List[Document]:
        """执行混合检索（dense + sparse 双路，RRF 融合）

        Args:
            query: 原始查询文本
            top_k: 最终返回文档数量
            coarse_top_k: 每路检索的候选数量（默认为 top_k * 5，供 Reranker 使用）

        Returns:
            List[Document]: 融合后的文档列表（按 RRF 分数排序）
        """
        coarse_k = coarse_top_k or max(top_k * 5, 20)

        try:
            # 生成 dense query 向量
            query_dense_vec = vector_embedding_service.embed_query(query)

            # 构造 Dense ANN 检索请求
            dense_req = AnnSearchRequest(
                data=[query_dense_vec],
                anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=coarse_k,
            )

            # 构造 Sparse BM25 检索请求（文本传入，Milvus 内部做 BM25 编码）
            sparse_req = AnnSearchRequest(
                data=[query],
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=coarse_k,
            )

            # RRF 融合（k=60 是 BEIR 推荐默认值）
            ranker = RRFRanker(k=60)

            collection = self._get_collection()
            results = collection.hybrid_search(
                reqs=[dense_req, sparse_req],
                rerank=ranker,
                limit=top_k,
                output_fields=["content_text", "metadata"],
            )

            # 将 pymilvus Hits 转换为 LangChain Document
            docs: List[Document] = []
            for hit in results[0]:
                entity = hit.entity
                page_content = entity.get("content_text", "")
                metadata = entity.get("metadata", {})
                docs.append(Document(page_content=page_content, metadata=metadata))

            logger.debug(
                f"[Enhanced] hybrid_search 完成: query='{query[:30]}...', "
                f"coarse_k={coarse_k}, 返回={len(docs)}"
            )
            return docs

        except Exception as e:
            logger.error(f"[Enhanced] hybrid_search 失败: {e}")
            raise


# 全局单例（懒初始化，首次调用 _get_collection 时才真正连接）
enhanced_vector_store_manager = EnhancedVectorStoreManager()
