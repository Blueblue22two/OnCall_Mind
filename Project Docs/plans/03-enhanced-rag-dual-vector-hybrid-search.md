# 增强版 RAG 双路向量与混合检索

## 1. 功能和目的

在基础 Dense 向量检索之上，新增 Sparse（BM25）稀疏向量检索能力，并通过 RRF（Reciprocal Rank Fusion）将双路检索结果融合，兼顾语义相似度和精确关键词匹配。

该模块解决的核心问题：
- Dense 检索擅长语义匹配但无法处理精确关键词（专有名词如 `HighCPUUsage`、`OOMKilled`）
- Sparse BM25 检索擅长精确关键词匹配但缺乏语义泛化能力
- 双路互补 + RRF 融合可同时提升召回率（recall）和精准率（precision）

与整体 RAG 系统的关系：
- 位于 Enhanced RAG Pipeline 的第二阶段（Query Preprocessing → **Hybrid Search** → Reranking）
- 依赖 Milvus `biz_enhanced` 集合的双向量 Schema
- 检索结果作为 Reranker 的候选输入

## 2. 抽象实现思路

### 技术路线：Milvus 2.5 内置 BM25 Function

实际实现采用了 Milvus 2.5 原生的 `FunctionType.BM25`，由 Milvus 服务端在文档插入时自动生成和维护稀疏向量，完全规避了 Python 侧 BM25 模型的 fit/refit 问题。

这与 plans.md 附录中推荐的方案四（Milvus 内置全文检索）一致，是推荐的长期演进方向。

### 双向量 Schema 设计

在 [app/core/milvus_client.py](app/core/milvus_client.py) 中定义 `biz_enhanced` 集合：

```python
fields = [
    FieldSchema(name="id", dtype=VARCHAR, max_length=100, is_primary=True),
    FieldSchema(name="dense_vector", dtype=FLOAT_VECTOR, dim=1024),      # Dense 向量
    FieldSchema(name="content_text", dtype=VARCHAR, max_length=8000,
                enable_analyzer=True,
                analyzer_params={"type": "chinese"}),                    # BM25 输入（Jieba 中文分词）
    FieldSchema(name="sparse_vector", dtype=SPARSE_FLOAT_VECTOR),        # BM25 自动生成
    FieldSchema(name="metadata", dtype=JSON),
]

# Milvus 内置 BM25 Function
bm25_function = Function(
    name="bm25",
    function_type=FunctionType.BM25,
    input_field_names=["content_text"],
    output_field_names=["sparse_vector"],
)
```

Dense 索引使用 COSINE 距离（对归一化向量更稳定），Sparse 索引使用 SPARSE_INVERTED_INDEX。

### 混合检索 + RRF 融合

在 [app/services/enhanced_vector_store_manager.py](app/services/enhanced_vector_store_manager.py) 中实现 `hybrid_search()`：

```python
from pymilvus import AnnSearchRequest, RRFRanker

dense_req = AnnSearchRequest(
    data=[dense_query_vector],
    anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=rerank_coarse_top_k,
)
sparse_req = AnnSearchRequest(
    data=[query_text],          # 直接传原始文本，Milvus 内部 BM25 编码
    anns_field="sparse_vector",
    param={"metric_type": "BM25"},
    limit=rerank_coarse_top_k,
)

results = collection.hybrid_search(
    reqs=[dense_req, sparse_req],
    rerank=RRFRanker(k=60),
    limit=rerank_coarse_top_k,
    output_fields=["content_text", "metadata"],
)
```

RRF 算法原理：$$\text{RRF\_score}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + r(d)}$$

### 与 plans.md 设计的差异

| 维度 | plans.md 原始设计 | 实际实现 |
|------|-------------------|----------|
| Sparse 编码方式 | Python 侧 `BM25EmbeddingFunction.fit()` + `encode_queries()` | Milvus 2.5 内置 `FunctionType.BM25`，服务端自动编码 |
| BM25 持久化 | `data/bm25_model.pkl` 文件序列化 | 无需持久化，Milvus 内部管理 |
| BM25 refit | 文档更新后需 Python 侧 refit | 无需 refit，Milvus 自动维护统计 |
| 中文分词 | 需额外配置 Jieba | `analyzer_params={"type": "chinese"}` 在 Schema 中配置 |
| Sparse 查询编码 | `bm25.encode_queries([query])` | 直接传原始文本，`metric_type="BM25"` |

## 3. 具体实现流程

### Step 1：扩展 Milvus Schema

在 [app/core/milvus_client.py](app/core/milvus_client.py) 中新增 `biz_enhanced` 集合的 Schema 定义（约 177 行起），包含 `dense_vector`、`content_text`（带 Jieba 分析器）、`sparse_vector`（BM25 Function 自动填充）和 `metadata` 字段。

索引配置：
- `dense_vector`: `IVF_FLAT`, `metric_type=COSINE`, `nlist=128`
- `sparse_vector`: `SPARSE_INVERTED_INDEX`, `metric_type=BM25`

### Step 2：实现 EnhancedVectorStoreManager

在 [app/services/enhanced_vector_store_manager.py](app/services/enhanced_vector_store_manager.py) 中实现：

- `add_documents()`：手动生成 Dense 向量（DashScope），Sparse 向量由 Milvus BM25 Function 自动生成
- `delete_by_source()`：按 `metadata["_source"]` 删除
- `hybrid_search()`：Dense ANN + Sparse BM25 双路检索 + RRF 融合

### Step 3：更新文档入库流程

在 [app/services/vector_index_service.py](app/services/vector_index_service.py) 中实现双写逻辑：
- `index_single_file()` 在写入 `biz` 集合后，同步写入 `biz_enhanced` 集合
- 增强集合写入失败不阻塞基础集合（try/except 包裹）

### Step 4：集成到 Enhanced Pipeline

`EnhancedRAGRetriever.retrieve()` 调用 `enhanced_vector_store_manager.hybrid_search()` 获取候选文档集，再传给 Reranker 精排。

## 4. 当前实现进度

### 已完成

- [x] `biz_enhanced` 集合 Schema 定义（双向量 + Milvus BM25 Function）
- [x] Dense 向量索引（COSINE, IVF_FLAT）
- [x] Sparse 向量索引（SPARSE_INVERTED_INDEX, BM25 metric）
- [x] `EnhancedVectorStoreManager` 完整实现（add_documents, delete_by_source, hybrid_search）
- [x] 混合检索 `hybrid_search()`：`AnnSearchRequest` × 2 + `RRFRanker(k=60)`
- [x] 文档入库双写逻辑（`biz` + `biz_enhanced`）
- [x] 与 `EnhancedRAGRetriever` 集成
- [x] pymilvus monkey-patch 兼容性处理（`_patch_pymilvus_milvus_client_orm_alias()`）

### 尚未完成

无。此阶段已 100% 完成。

### 依赖其他模块

- 依赖 Milvus 2.5+ 运行环境（`vector-database.yml` 中的 `milvus-standalone` 镜像）
- 查询预处理结果影响 Dense 检索的查询向量（但 Sparse 始终用原始文本，BM25 编码由 Milvus 处理）

## 5. Evidence

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| Schema 定义 | [app/core/milvus_client.py:177](app/core/milvus_client.py#L177) | `biz_enhanced` 集合字段定义 |
| BM25 Function | [app/core/milvus_client.py:218](app/core/milvus_client.py#L218) | `FunctionType.BM25` 自动生成稀疏向量 |
| Jieba 分词器 | [app/core/milvus_client.py](app/core/milvus_client.py) | `analyzer_params={"type": "chinese"}` |
| Dense 索引 | [app/core/milvus_client.py](app/core/milvus_client.py) | `IVF_FLAT`, `metric_type=COSINE`, `nlist=128` |
| Sparse 索引 | [app/core/milvus_client.py](app/core/milvus_client.py) | `SPARSE_INVERTED_INDEX`, `metric_type=BM25` |
| Hybrid Search | [app/services/enhanced_vector_store_manager.py:118](app/services/enhanced_vector_store_manager.py#L118) | `AnnSearchRequest` + `RRFRanker` + `hybrid_search()` |
| 双写逻辑 | [app/services/vector_index_service.py:174-181](app/services/vector_index_service.py#L174) | `biz` 和 `biz_enhanced` 同步写入 |
| 增强集成 | [app/retriever/enhanced.py:46](app/retriever/enhanced.py#L46) | `hybrid_search` 在 `retrieve()` 中调用 |
| Monkey-patch | [app/core/milvus_client.py](app/core/milvus_client.py) | `_patch_pymilvus_milvus_client_orm_alias()` |
| Docker Compose | [vector-database.yml](vector-database.yml) | Milvus standalone + etcd + minio |
| Git 提交 | `f1f48be` | `feat: Phase 2 - 实现 Enhanced RAG（双向量混合检索 + 可插拔精排）` |
