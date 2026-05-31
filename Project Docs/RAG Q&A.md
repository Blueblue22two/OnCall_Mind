# RAG 系统技术问答（Q&A）文档

> 基于项目实际代码、配置、评估报告的深度技术分析
>
> 文档版本：v1.0 | 分析日期：2026-05-30

---

## 目录

- [Q1: RAG 的流程](#q1-rag-的流程)
- [Q2: RAG 系统中文档数量有多大](#q2-rag-系统中文档数量有多大)
- [Q3: 分块策略与 chunk_size](#q3-分块策略与-chunk_size)
- [Q4: 向量检索和关键词检索的区别](#q4-向量检索和关键词检索的区别)
- [Q5: 向量数据库选型](#q5-向量数据库选型)
- [Q6: 向量数据库准确性降低的因素](#q6-向量数据库准确性降低的因素)
- [Q7: 检索优化机制](#q7-检索优化机制)
- [Q8: 为什么要在 RAG 中加入 Query Rewrite](#q8-为什么要在-rag-中加入-query-rewrite)
- [Q9: Query Rewrite 的实现方式](#q9-query-rewrite-的实现方式)
- [Q10: RAG 流程的召回速度](#q10-rag-流程的召回速度)
- [Q11: RAG 部分的 Agent 是否自动选择是否需要 RAG](#q11-rag-部分的-agent-是否自动选择是否需要-rag)
- [Q12: RAG 评估核心指标与结果](#q12-rag-评估核心指标与结果)
- [Q13: RAG 对比实验设计](#q13-rag-对比实验设计)
- [Q14: 最终检索准确率与优化思路](#q14-最终检索准确率与优化思路)
- [Q15: PDF 扫描件、OCR、表格结构化的处理思考](#q15-pdf-扫描件ocr表格结构化的处理思考)

---

## Q1: RAG 的流程

### Facts（项目事实）

项目实现了 **两套可切换的 RAG Pipeline**，通过 `RAG_MODE` 环境变量在 `basic` 和 `enhanced` 之间切换。

**整体架构：可插拔工厂模式**

```python
# File: app/retriever/factory.py
@lru_cache(maxsize=1)
def get_rag_retriever() -> BaseRAGRetriever:
    if config.rag_mode == "enhanced":
        from app.retriever.enhanced import EnhancedRAGRetriever
        return EnhancedRAGRetriever()
    return BasicRAGRetriever()
```

所有检索器都遵守统一抽象接口：

```python
# File: app/retriever/base.py
class BaseRAGRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        ...
```

**入口：Agent 通过 Tool 触发检索**

```python
# File: app/tools/knowledge_tool.py
@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    docs = get_rag_retriever().retrieve(query, top_k=effective_top_k)
    context = format_docs(docs)
    return context, docs
```

---

#### Pipeline 1: Basic 模式（单阶段 Dense 检索）

```
用户查询
  → Embedding（text-embedding-v4, 1024 维）
  → Milvus ANN 检索（L2 距离, IVF_FLAT 索引）
  → Top-K 文档
```

代码实现：

```python
# File: app/retriever/basic.py
class BasicRAGRetriever(BaseRAGRetriever):
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)
        return docs
```

- 集合：`biz`
- 向量字段：`vector`（FLOAT_VECTOR, 1024 维）
- 索引类型：IVF_FLAT
- 距离度量：L2（欧氏距离）
- 默认 Top-K：3

---

#### Pipeline 2: Enhanced 模式（三阶段增强检索）

```
用户查询
  → Stage 1: 查询预处理（none / rewrite）
  → Stage 2: 双向量混合检索（Dense COSINE + Sparse BM25 → RRF 融合）
  → Stage 3: Cross-Encoder 精排（none / cross_encoder）
  → Top-K 文档
```

代码实现：

```python
# File: app/retriever/enhanced.py
class EnhancedRAGRetriever(BaseRAGRetriever):
    def retrieve(self, query: str, top_k: int, debug: bool = False) -> list[Document]:
        # Stage 1: Query Preprocessing
        preprocessor = get_query_preprocessor(config.query_preprocessor_type)
        search_query = preprocessor.process(query)

        # Stage 2: Hybrid Search (Dense + Sparse → RRF)
        candidates = enhanced_vector_store_manager.hybrid_search(
            query=search_query, top_k=coarse_top_k, coarse_top_k=coarse_top_k,
        )

        # Stage 3: Reranking（使用 original_query 打分）
        reranker = get_reranker(config.reranker_type, config.reranker_model)
        final_docs = reranker.rerank(query=original_query, documents=candidates, top_k=top_k)
        return final_docs
```

- 集合：`biz_enhanced`
- Dense 向量字段：`dense_vector`（FLOAT_VECTOR, 1024 维, COSINE）
- Sparse 向量字段：`sparse_vector`（SPARSE_FLOAT_VECTOR, Milvus 内置 BM25 Function 自动生成）
- 融合方式：RRF（Reciprocal Rank Fusion, k=60）
- 粗排候选数：`RERANK_COARSE_TOP_K`（默认 20）
- 精排后返回数：`RERANKER_TOP_K`（默认 3）

**混合检索核心代码：**

```python
# File: app/services/enhanced_vector_store_manager.py
# Method: hybrid_search()
dense_req = AnnSearchRequest(
    data=[query_dense_vec],
    anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
    limit=coarse_k,
)
sparse_req = AnnSearchRequest(
    data=[query],
    anns_field="sparse_vector",
    param={"metric_type": "BM25"},
    limit=coarse_k,
)
ranker = RRFRanker(k=60)
results = collection.hybrid_search(
    reqs=[dense_req, sparse_req], rerank=ranker, limit=top_k,
    output_fields=["content_text", "metadata"],
)
```

---

#### 完整数据流（从用户提问到 LLM 回答）

```
用户: "CPU 飙高怎么排查？"
  │
  ▼
[FastAPI POST /api/chat_stream]
  │
  ▼
[RagAgentService.query_stream()]
  │ 构建消息: [SystemMessage, HumanMessage]
  │ Token 裁剪 (tiktoken, max 8000 tokens)
  │
  ▼
[LangGraph ReAct Agent]
  │ LLM 推理 → 决定调用 retrieve_knowledge 工具
  │
  ▼
[retrieve_knowledge(query="CPU 飙高怎么排查")]
  │
  ▼
[get_rag_retriever().retrieve(query, top_k=3)]
  │
  ├─ Basic 模式:
  │   Embedding → Milvus L2 ANN → Top-3 文档
  │
  └─ Enhanced 模式:
      Stage 1: rewrite → "CPU 使用率过高排查方法与步骤"
      Stage 2: Dense COSINE + Sparse BM25 → RRF → 20 候选
      Stage 3: Cross-Encoder 精排 → Top-3 文档
  │
  ▼
[format_docs(docs)] → 格式化为 "【参考资料 1】...【参考资料 2】..."
  │
  ▼
[LLM 基于检索上下文生成回答]
  │
  ▼
SSE 流式返回给用户
```

### Analysis（分析）

项目的 RAG 设计有以下亮点：

1. **可插拔架构**：通过工厂模式 + 抽象基类，Basic 和 Enhanced 模式可以无缝切换，新增检索策略只需实现 `BaseRAGRetriever` 接口
2. **双集合设计**：`biz`（Basic）和 `biz_enhanced`（Enhanced）两个 Milvus 集合独立存储，文档上传时双写，避免模式切换时重新入库
3. **降级策略**：Enhanced 模式的每个阶段都有降级路径——预处理失败回退原始查询，精排失败回退直接截断
4. **精排使用原始查询**：Stage 3 始终使用用户原始查询（而非改写后的查询）打分，确保精排分数反映用户真实意图

### Improvements（优化建议）

1. **引入 HyDE（Hypothetical Document Embeddings）**：先让 LLM 生成一个"假想答案"，用假想答案的 Embedding 去检索，可能比直接检索用户问题更有效
2. **检索结果缓存**：对高频查询的检索结果进行缓存，减少重复检索开销
3. **自适应模式选择**：根据查询复杂度自动选择 Basic 或 Enhanced 模式——简单关键词查询走 Basic，复杂语义查询走 Enhanced

---

## Q2: RAG 系统中文档数量有多大

### Facts（项目事实）

知识库由 **12 篇运维故障排查 SOP 文档** 组成，存储在项目根目录的 `aiops-docs/` 下：

| 文件 | 大小（字节） | 行数 | 主题 |
|------|-------------|------|------|
| `cpu_high_usage.md` | 3,585 | 135 | CPU 使用率过高 |
| `memory_high_usage.md` | 5,427 | 180 | 内存使用率过高 |
| `disk_high_usage.md` | 7,716 | 343 | 磁盘使用率过高 |
| `service_unavailable.md` | 7,486 | 289 | 服务不可用 |
| `slow_response.md` | 6,615 | 257 | 服务响应慢 |
| `network_high_latency.md` | 5,965 | 169 | 网络延迟高 |
| `api_error_rate_spike.md` | 6,670 | 245 | API 错误率飙升 |
| `cache_avalanche.md` | 6,439 | 241 | 缓存雪崩 |
| `certificate_expiry.md` | 5,969 | 232 | 证书过期 |
| `container_oom_killed.md` | 5,520 | 164 | 容器 OOM |
| `database_connection_pool_exhaustion.md` | 5,669 | 188 | 数据库连接池耗尽 |
| `message_queue_backlog.md` | 6,906 | 248 | 消息队列积压 |
| **合计** | **73,967** | **2,691** | — |

**知识库总规模：约 74 KB 纯文本（12 篇 Markdown 文档）。**

上传后同时写入两个 Milvus 集合：

```python
# File: app/services/vector_index_service.py
# Method: index_single_file()
# 写入基础 biz collection
vector_store_manager.add_documents(documents)
# 同步写入 biz_enhanced collection
enhanced_vector_store_manager.add_documents(documents)
```

### Analysis（分析）

**规模评估：**

- 74 KB / 12 篇文档属于 **极小规模** 的知识库
- 在这样小的数据量下，向量检索的优势不明显——甚至全文检索也能覆盖大部分场景
- 评估数据集中 Hit Rate@3 达到 97%，很大程度上是因为文档总量只有 12 篇，top-3 就覆盖了 25% 的文档

**分块后的实际向量数量：**

根据分块配置（`chunk_max_size=800`, `chunk_overlap=100`），每篇文档约产生 5–15 个分块。估计总向量数在 **80–150 个** 之间。

### Improvements（优化建议）

1. **扩展知识库**：增加更多运维场景文档（如 Kubernetes Pod 异常、网络 DNS 故障、负载均衡配置等），目标至少 100 篇以上
2. **引入多级知识库**：按领域（基础设施 / 应用层 / 网络层 / 安全层）组织文档，检索时先定位领域再细粒度检索
3. **知识库更新机制**：当前文档上传后需要手动重新 upload，建议增加文件变更监控自动重新索引

---

## Q3: 分块策略与 chunk_size

### Facts（项目事实）

项目使用 **两阶段分割 + 小片段合并** 策略，由 `DocumentSplitterService` 实现。

**配置参数：**

```python
# File: app/config.py
chunk_max_size: int = 800      # 基础 chunk 大小（token 数参考）
chunk_overlap: int = 100       # 重叠大小
```

**分割器实现：**

```python
# File: app/services/document_splitter_service.py
class DocumentSplitterService:

    # Stage 1: Markdown 标题分割器
    self.markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            # 不按三级标题分割，避免过度碎片化
        ],
        strip_headers=False,  # 保留标题在内容中
    )

    # Stage 2: 递归字符分割器（二次分割）
    self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=self.chunk_size * 2,    # 1600 字符
        chunk_overlap=self.chunk_overlap,   # 100 字符
        length_function=len,
        is_separator_regex=False,
    )
```

**三阶段处理流程：**

```
原始 Markdown 文档
  │
  ▼ Stage 1: MarkdownHeaderTextSplitter
  │  按 # 和 ## 标题分割，保留标题层级信息到 metadata
  │  输出：多个以标题为边界的文档块
  │
  ▼ Stage 2: RecursiveCharacterTextSplitter
  │  对过长的块进行二次分割
  │  chunk_size = 800 * 2 = 1600 字符
  │  chunk_overlap = 100 字符
  │
  ▼ Stage 3: 小片段合并（_merge_small_chunks）
  │  合并 < 300 字符的小片段
  │  合并后不超过 chunk_size * 2（1600 字符）
  │
  ▼ 最终分块列表（每个分块附带 metadata: _source, _file_name, h1, h2）
```

**小片段合并逻辑：**

```python
# File: app/services/document_splitter_service.py
# Method: _merge_small_chunks()
def _merge_small_chunks(self, documents, min_size=300):
    for doc in documents:
        doc_size = len(doc.page_content)
        if current_doc is None:
            current_doc = doc
        elif doc_size < min_size and len(current_doc.page_content) < self.chunk_size * 2:
            current_doc.page_content += "\n\n" + doc.page_content  # 合并小片段
        else:
            merged_docs.append(current_doc)
            current_doc = doc
```

**Metadata 保留：**

```python
# 每个分块都附带来源信息
doc.metadata["_source"] = file_path           # 文件完整路径
doc.metadata["_extension"] = ".md"            # 文件扩展名
doc.metadata["_file_name"] = Path(file_path).name  # 文件名
# MarkdownHeaderTextSplitter 还会自动添加 h1, h2 等标题层级
```

### Analysis（分析）

**分块策略选型分析：**

| 策略 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| Markdown 标题分割 | `MarkdownHeaderTextSplitter` | 保持文档结构语义完整性 | 可能导致过大的块 |
| 递归字符分割 | `RecursiveCharacterTextSplitter` | 控制块大小上限 | 可能切断语义完整的段落 |
| 小片段合并 | `_merge_small_chunks(min_size=300)` | 避免过短的无意义碎片 | 可能引入跨主题合并 |

**关键设计决策：**

1. **不按三级标题（###）分割**：注释中明确说明"避免过度碎片化"——运维 SOP 文档的步骤详情通常在三级标题下，如果按 ### 分割，每个步骤会成为独立分块，失去上下文
2. **二级分割的 chunk_size 设为 `800 * 2 = 1600`**：比基础的 800 大一倍，目的是在标题分割的基础上允许更大的块，减少分片数
3. **overlap 设为 100**：确保相邻块之间有上下文衔接，避免关键信息恰好在边界处被截断

### Improvements（优化建议）

1. **语义分块（Semantic Chunking）**：使用 Embedding 相似度判断分块边界，而非仅依赖字符数和标题层级
2. **动态 chunk_size**：根据文档类型动态调整——SOP 步骤类文档适合保持步骤完整性，FAQ 类文档适合更小的 chunk
3. **父子分块索引**：维护小块（检索用）和父块（上下文用）的映射关系，检索命中小块时返回完整的父块上下文
4. **chunk_overlap 调优**：当前 100 字符的 overlap 在 1600 字符的 chunk 中占比仅 6.25%，对于跨段落引用的场景可能不够

---

## Q4: 向量检索和关键词检索的区别

### Facts（项目事实）

项目中 **同时使用了向量检索和关键词检索**，在 Enhanced 模式中将两者结合为混合检索。

**向量检索（Dense Retrieval）：**

```python
# File: app/services/enhanced_vector_store_manager.py
# Dense ANN 检索请求
dense_req = AnnSearchRequest(
    data=[query_dense_vec],                    # text-embedding-v4 生成的 1024 维向量
    anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
    limit=coarse_k,
)
```

- 模型：DashScope `text-embedding-v4`（1024 维）
- 距离度量：COSINE（余弦相似度）
- 索引类型：IVF_FLAT（倒排文件 + 暴力搜索）
- 集合：`biz_enhanced` 的 `dense_vector` 字段

**关键词检索（Sparse / BM25 Retrieval）：**

```python
# File: app/services/enhanced_vector_store_manager.py
# Sparse BM25 检索请求
sparse_req = AnnSearchRequest(
    data=[query],                              # 原始文本直接传入
    anns_field="sparse_vector",
    param={"metric_type": "BM25"},
    limit=coarse_k,
)
```

- 实现：Milvus 内置 BM25 Function（无需外部 BM25 服务）
- 分词器：Jieba 中文分词（Milvus 内置 `chinese` 分析器）
- 距离度量：BM25（基于词频 TF-IDF 变体）
- 索引类型：SPARSE_INVERTED_INDEX
- 集合：`biz_enhanced` 的 `sparse_vector` 字段

**BM25 字段 Schema 定义：**

```python
# File: app/core/milvus_client.py
# Method: _create_enhanced_collection()
# BM25 输入字段，启用 Jieba 中文分析器
FieldSchema(
    name="content_text",
    dtype=DataType.VARCHAR,
    max_length=8000,
    enable_analyzer=True,
    analyzer_params={"type": "chinese"},  # Milvus 内置 Jieba
),
# 稀疏向量由 BM25 Function 自动填充
FieldSchema(
    name="sparse_vector",
    dtype=DataType.SPARSE_FLOAT_VECTOR,
),
# Milvus 内置 BM25：从 content_text 自动生成 sparse_vector
bm25_function = Function(
    name="biz_bm25",
    function_type=FunctionType.BM25,
    input_field_names=["content_text"],
    output_field_names=["sparse_vector"],
)
```

**融合方式 —— RRF（Reciprocal Rank Fusion）：**

```python
# File: app/services/enhanced_vector_store_manager.py
ranker = RRFRanker(k=60)  # k=60 是 BEIR 推荐默认值
results = collection.hybrid_search(
    reqs=[dense_req, sparse_req],
    rerank=ranker,
    limit=top_k,
)
```

### Analysis（分析）

**向量检索 vs 关键词检索对比表：**

| 维度 | 向量检索（Dense） | 关键词检索（BM25） |
|------|-------------------|-------------------|
| **表示方式** | 连续浮点向量（1024 维） | 稀疏词频向量 |
| **匹配方式** | 语义相似度（COSINE） | 精确关键词匹配（BM25 打分） |
| **优势场景** | 同义词、语义相近的查询 | 专有名词、技术术语、告警名称 |
| **劣势场景** | 精确术语匹配可能被"稀释" | 无法理解同义词和语义改写 |
| **分词依赖** | 无（端到端 Embedding） | 依赖分词器质量（Jieba） |
| **索引结构** | IVF_FLAT | SPARSE_INVERTED_INDEX |
| **项目中的例子** | "电脑卡了"→ 匹配 CPU/内存相关文档 | "ContainerOOMKilled"→ 精确匹配告警名 |

**项目中的实际互补效果：**

以评估数据集为例，问题分为 4 类：

- `exact_keyword`（精确关键词，如"APIErrorRateSpike 告警触发条件"）→ BM25 优势
- `colloquial`（口语化，如"CPU 飙高可能是死循环吗"）→ Dense 优势
- `cross_doc`（跨文档综合）→ 两者互补
- `edge_case`（边界场景）→ 两者互补

### Improvements（优化建议）

1. **加权融合替代 RRF**：RRF 对两路检索等权处理，可以根据查询类型动态调整权重（如检测到专有名词时增加 BM25 权重）
2. **自定义分词词典**：为 Jieba 添加运维领域自定义词典（如 "OOM", "Killed", "5xx", "Pod" 等），提升 BM25 的中文分词质量
3. **学习排序（Learning to Rank）**：使用标注数据训练一个排序模型替代固定的 RRF 融合

---

## Q5: 向量数据库选型

### Facts（项目事实）

项目使用 **Milvus v2.5.10**（Standalone 模式），通过 Docker Compose 部署。

**部署配置：**

```yaml
# File: vector-database.yml
services:
  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.5.10
    command: ["milvus", "run", "standalone"]
    ports:
      - "19530:19530"
    depends_on:
      - etcd        # 元数据存储
      - minio       # 对象存储（持久化向量数据）
  attu:
    container_name: milvus-attu
    image: zilliz/attu:v2.5
    ports:
      - "8000:3000"  # Web UI 管理界面
```

**双集合 Schema 设计：**

Basic 集合 `biz`：

```python
# File: app/core/milvus_client.py
fields = [
    FieldSchema(name="id", dtype=VARCHAR, max_length=100, is_primary=True),
    FieldSchema(name="vector", dtype=FLOAT_VECTOR, dim=1024),
    FieldSchema(name="content", dtype=VARCHAR, max_length=8000),
    FieldSchema(name="metadata", dtype=JSON),
]
# 索引: IVF_FLAT, metric=L2, nlist=128
```

Enhanced 集合 `biz_enhanced`：

```python
fields = [
    FieldSchema(name="id", dtype=VARCHAR, max_length=100, is_primary=True),
    FieldSchema(name="dense_vector", dtype=FLOAT_VECTOR, dim=1024),
    FieldSchema(name="content_text", dtype=VARCHAR, max_length=8000,
                enable_analyzer=True, analyzer_params={"type": "chinese"}),
    FieldSchema(name="sparse_vector", dtype=SPARSE_FLOAT_VECTOR),
    FieldSchema(name="metadata", dtype=JSON),
]
# Dense 索引: IVF_FLAT, metric=COSINE, nlist=128
# Sparse 索引: SPARSE_INVERTED_INDEX, metric=BM25
```

**LangChain 集成：**

```python
# File: app/services/vector_store_manager.py
self.vector_store = Milvus(
    embedding_function=vector_embedding_service,
    collection_name="biz",
    connection_args={"host": config.milvus_host, "port": config.milvus_port},
    auto_id=False,
    text_field="content",
    vector_field="vector",
    primary_field="id",
    metadata_field="metadata",
)
```

### Analysis（分析）

**选择 Milvus 的原因分析：**

| 维度 | Milvus 优势 | 项目中的体现 |
|------|------------|-------------|
| **混合检索原生支持** | 内置 BM25 Function + Dense + RRF | `biz_enhanced` 集合同时支持 Dense ANN 和 Sparse BM25 |
| **中文分词内置** | 内置 Jieba 中文分析器 | `analyzer_params={"type": "chinese"}` |
| **LangChain 生态集成** | `langchain-milvus` 官方适配器 | `from langchain_milvus import Milvus` |
| **可扩展性** | 支持从 Standalone 到 Distributed 无缝升级 | 当前 Standalone，未来可升级为集群 |
| **向量维度灵活** | 支持多种维度（128–32768） | 1024 维（text-embedding-v4） |
| **可视化管理** | Attu Web UI | 端口 8000 提供图形化管理界面 |
| **开源免费** | Apache 2.0 协议 | 无许可成本 |

**与其他向量数据库对比：**

| 数据库 | 混合检索 | 中文 BM25 | LangChain 集成 | 适合本项目 |
|--------|----------|-----------|---------------|-----------|
| **Milvus** | ✅ 原生支持 | ✅ 内置 Jieba | ✅ langchain-milvus | ✅ 最佳匹配 |
| Pinecone | ⚠️ 需要 Sparse-Dense 方案 | ❌ 无内置 | ✅ | ⚠️ 云服务，成本高 |
| Weaviate | ✅ 原生支持 | ⚠️ 需自定义 | ✅ | ⚠️ 中文分词需额外配置 |
| Qdrant | ⚠️ 需要 Sparse Vector | ❌ 无内置 | ✅ | ⚠️ BM25 支持有限 |
| ChromaDB | ❌ 不支持 | ❌ | ✅ | ❌ 功能不足 |
| FAISS | ❌ 仅向量 | ❌ | ✅ | ❌ 无 BM25，无持久化 |

### Improvements（优化建议）

1. **升级为 Milvus Distributed**：当文档量增长到百万级时，从 Standalone 升级为分布式部署
2. **启用 GPU 加速**：Milvus 支持 GPU 索引（如 GPU_IVF_FLAT），在大规模数据下显著提升检索速度
3. **Collection 分区**：按文档领域（CPU / 内存 / 网络等）创建 Partition，检索时限定分区范围

---

## Q6: 向量数据库准确性降低的因素

### Facts（项目事实）

基于项目代码和配置，以下因素已被识别为可能影响检索准确性：

**1. Embedding 模型与 Tokenizer 不匹配**

项目使用 `text-embedding-v4` 生成向量，但上下文裁剪使用 `cl100k_base`（GPT-4 的 tokenizer）：

```python
# File: app/services/rag_agent_service.py
def trim_messages_by_tokens(messages, max_tokens=8000, model_encoding="cl100k_base"):
    enc = tiktoken.get_encoding(model_encoding)
```

这不影响检索准确性，但影响上下文窗口管理精度。

**2. 向量维度配置**

```python
# File: app/core/milvus_client.py
VECTOR_DIM: int = 1024  # 统一使用 1024 维
```

项目中 Embedding 模型和 Milvus 集合的维度一致（1024），不存在维度不匹配的问题。但代码中专门处理了维度不匹配的边界情况：

```python
# File: app/core/milvus_client.py
if existing_dim != self.VECTOR_DIM:
    logger.warning(f"检测到向量维度不匹配！当前: {existing_dim}, 配置: {self.VECTOR_DIM}")
    # 自动删除旧 collection 并重建
    utility.drop_collection(self.COLLECTION_NAME)
    self._create_collection()
```

**3. 分块质量**

分块过小或过大都会影响检索准确性。当前配置 `chunk_size * 2 = 1600` 字符，`overlap = 100` 字符。

**4. BM25 分词质量**

```python
# File: app/core/milvus_client.py
analyzer_params={"type": "chinese"},  # Milvus 内置 Jieba 中文分析器
```

Jieba 对运维领域专有名词（如 "OOM", "Killed", "Pod", "Kafka"）的分词效果可能不理想。

**5. IVF_FLAT 索引参数**

```python
# 创建索引时 nlist=128
index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}

# 搜索时 nprobe=16
param={"metric_type": "COSINE", "params": {"nprobe": 16}}
```

`nprobe=16` 表示搜索时只检查 128 个聚类中的 16 个，可能导致跨聚类的相似文档被遗漏。

### Analysis（分析）

**系统性梳理——向量数据库准确性降低的六大因素：**

| 因素 | 影响机制 | 项目中的风险等级 |
|------|----------|-----------------|
| **1. Embedding 模型质量** | 模型对语义的编码能力直接决定向量空间的区分度 | 🟡 中（text-embedding-v4 表现良好，但非最先进） |
| **2. 分块策略** | 块过大引入噪声，块过小丢失上下文 | 🟡 中（1600 字符 + 100 overlap 可能不够精细） |
| **3. 向量维度** | 维度越低信息压缩越严重，越高检索越慢 | 🟢 低（1024 维是合理选择） |
| **4. 索引近似度** | IVF_FLAT 是近似搜索，nprobe 过低会遗漏 | 🟡 中（nprobe=16，nlist=128，覆盖率 12.5%） |
| **5. 数据质量** | 文档内容质量、格式一致性、元数据完整性 | 🟢 低（SOP 文档结构统一） |
| **6. BM25 分词质量** | 中文分词错误导致关键词匹配失败 | 🟡 中（Jieba 通用词典可能不适配运维领域） |

**特别值得关注的因素：**

`nprobe/nlist = 16/128 = 12.5%` 意味着每次搜索只检查约 12.5% 的聚类。对于当前极小规模的数据集（~100 个向量），这影响不大；但数据量增长到数万时，可能导致显著的召回率下降。

### Improvements（优化建议）

1. **提升 nprobe**：将 `nprobe` 从 16 提升到 32 或 64，以牺牲少量延迟换取更高召回率
2. **自定义 Jieba 词典**：为运维领域添加自定义词汇，如 `OOMKilled`、`5xx`、`Pod`、`Kafka`、`Redis`
3. **引入 HNSW 索引**：对于小规模数据集，HNSW 索引比 IVF_FLAT 有更高的召回率（接近精确搜索）
4. **定期重建索引**：文档大量增删后，IVF 聚类中心可能偏离最优，需要 `collection.compact()` 或重建索引

---

## Q7: 检索优化机制

### Facts（项目事实）

项目实现了 **多层次的检索优化机制**，核心体现在 Enhanced RAG Pipeline 的三阶段设计中。

**优化 1：可插拔的检索器工厂**

```python
# File: app/retriever/factory.py
@lru_cache(maxsize=1)
def get_rag_retriever() -> BaseRAGRetriever:
    if config.rag_mode == "enhanced":
        return EnhancedRAGRetriever()
    return BasicRAGRetriever()
```

**优化 2：查询预处理（Query Preprocessing）**

可插拔的预处理器工厂：

```python
# File: app/retriever/preprocessing/factory.py
_REGISTRY = {
    "none": PassthroughPreprocessor,      # 透传，不做任何处理
    "rewrite": QueryRewritePreprocessor,   # LLM 语义改写
}
```

**优化 3：双向量混合检索（Hybrid Search）**

Dense（语义）+ Sparse（关键词）双路检索，RRF 融合：

```python
# File: app/services/enhanced_vector_store_manager.py
dense_req = AnnSearchRequest(data=[query_dense_vec], anns_field="dense_vector", ...)
sparse_req = AnnSearchRequest(data=[query], anns_field="sparse_vector", ...)
ranker = RRFRanker(k=60)
results = collection.hybrid_search(reqs=[dense_req, sparse_req], rerank=ranker, ...)
```

**优化 4：Cross-Encoder 精排（Reranking）**

可插拔的精排器工厂：

```python
# File: app/retriever/reranker/factory.py
_REGISTRY = {
    "none": PassthroughReranker,        # 直接截断
    "cross_encoder": CrossEncoderReranker,  # BGE-Reranker 精排
}
```

精排模型：`BAAI/bge-reranker-v2-m3`

```python
# File: app/retriever/reranker/cross_encoder.py
class CrossEncoderReranker(BaseReranker):
    def rerank(self, query, documents, top_k):
        model = self._get_model()  # 懒加载 CrossEncoder
        pairs = [[query, doc.page_content] for doc in documents]
        scores = model.predict(pairs, apply_softmax=True)
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return scored_docs[:top_k]
```

**优化 5：粗排-精排两阶段（Coarse-to-Fine）**

```python
# File: app/config.py
rerank_coarse_top_k: int = 20    # 粗排召回 20 个候选
reranker_top_k: int = 3          # 精排后只保留 3 个
```

混合检索先粗排召回 20 个候选文档，Cross-Encoder 再从中精选 3 个，避免精排模型处理过多文档。

**优化 6：降级策略（Graceful Degradation）**

```python
# File: app/retriever/enhanced.py
# 预处理失败 → 回退原始查询
try:
    search_query = preprocessor.process(query)
except Exception as e:
    search_query = original_query
    meta["degraded_stage"] = "preprocessing"

# 精排失败 → 回退直接截断
try:
    final_docs = reranker.rerank(query=original_query, documents=candidates, top_k=top_k)
except Exception as e:
    final_docs = candidates[:top_k]
    meta["degraded_stage"] = "reranker"
```

**优化 7：结构化检索日志（Trace）**

每次检索生成唯一 `trace_id`，记录三阶段的输入输出和耗时：

```python
# File: app/retriever/enhanced.py
trace_id = uuid.uuid4().hex[:8]
meta = {
    "trace_id": trace_id,
    "preprocessor_type": config.query_preprocessor_type,
    "reranker_type": config.reranker_type,
    "degraded_stage": None,
    "candidate_count": 0,
    "final_count": 0,
    "total_time_ms": 0,
}
# 结构化日志输出
logger.info(f"[Enhanced][{trace_id}] 检索完成: preprocessor=rewrite, reranker=cross_encoder, candidates=20, final=3, total_ms=470")
```

### Analysis（分析）

**优化机制层次图：**

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: 查询优化                                      │
│  └── Query Rewrite（LLM 语义增强改写）                  │
├─────────────────────────────────────────────────────────┤
│  Layer 2: 检索优化                                      │
│  ├── Dense 向量检索（语义匹配）                         │
│  ├── Sparse BM25 检索（关键词匹配）                     │
│  └── RRF 融合（互补增强）                               │
├─────────────────────────────────────────────────────────┤
│  Layer 3: 排序优化                                      │
│  └── Cross-Encoder 精排（逐对打分重排序）               │
├─────────────────────────────────────────────────────────┤
│  Layer 4: 工程优化                                      │
│  ├── 粗排-精排两阶段（20 候选 → 3 精选）               │
│  ├── 懒加载模型（首次调用时才加载 Cross-Encoder）       │
│  ├── 降级策略（失败时不中断流程）                       │
│  └── 结构化日志（trace_id + 分阶段耗时）                │
└─────────────────────────────────────────────────────────┘
```

### Improvements（优化建议）

1. **引入查询路由**：根据查询类型（关键词查询 / 语义查询 / 混合查询）动态选择检索策略
2. **多路检索扩展**：除 Dense 和 BM25 外，增加 metadata 过滤（如按告警类型筛选）
3. **在线学习**：根据用户反馈（点击/未点击、有用/无用）动态调整 RRF 权重或 Reranker 模型
4. **Embedding 缓存**：对高频查询的 Embedding 向量进行缓存，减少 API 调用成本

---

## Q8: 为什么要在 RAG 中加入 Query Rewrite

### Facts（项目事实）

Query Rewrite 是 Enhanced RAG Pipeline 的 Stage 1，通过 `QUERY_PREPROCESSOR_TYPE=rewrite` 配置启用。

**Rewrite 的目标（来自 Prompt 定义）：**

```python
# File: app/retriever/preprocessing/rewrite.py
_REWRITE_PROMPT_TEMPLATE = """\
你是一名专业的信息检索优化专家。请将下面的用户问题改写为更适合向量数据库语义检索的表述。

改写要求：
1. 保留原始问题的核心意图，不要改变问题的含义
2. 补充可能缺失的关键词和专业术语
3. 将口语化表述转化为书面化、规范化的表述
4. 输出只包含改写后的问题，不需要解释或多余内容

用户原始问题：{query}

改写后的问题："""
```

**Rewrite 解决的实际问题（从评估数据集提取的真实案例）：**

| 类型 | 原始查询（口语化） | 改写后（预期效果） |
|------|-------------------|-------------------|
| colloquial | "CPU 飙高可能是代码里写了死循环吗？" | "CPU使用率过高的原因分析：死循环或无限递归排查方法" |
| colloquial | "接口不通了，服务挂了怎么看进程还在不在？" | "服务不可用时的进程状态检查方法" |
| colloquial | "内存满了怎么抓堆栈分析？是打 dump 吗？" | "内存使用率过高时的堆栈分析方法：JVM heap dump" |
| colloquial | "怎么看哪个文件夹把磁盘占满了？" | "磁盘使用率过高的排查方法：目录空间占用分析" |

**评估数据集中的问题分类：**

```python
# File: tests/evaluation/rag_testset.py
# exact_keyword: 17 条 — 已经是规范表述，Rewrite 收益低
# colloquial: 36 条 — 口语化严重，Rewrite 收益高
# cross_doc: 7 条 — 跨文档综合，Rewrite 可能帮助补全关键词
# edge_case: 7 条 — 边界场景，Rewrite 需谨慎避免改变意图
```

### Analysis（分析）

**加入 Query Rewrite 的核心原因：**

1. **弥合用户表述与文档表述的语义鸿沟**：运维人员通常用口语描述问题（"服务挂了"），但知识库文档使用规范术语（"服务不可用"）。Rewrite 将口语映射到规范表述，提升向量检索的匹配度。

2. **补全缺失关键词**：用户可能只描述了症状（"接口变慢了"），Rewrite 可以补充相关技术术语（"RT 升高、慢 SQL、缓存击穿"），增加检索到正确文档的概率。

3. **消除歧义**：简短的口语可能有多种理解，Rewrite 可以结合上下文补充限定信息。

**但 Rewrite 也有风险：**

1. **改写引入噪声**：LLM 可能错误理解用户意图，将 "内存满了" 改写为 "磁盘空间不足"
2. **增加延迟**：每次检索前多一次 LLM 调用，增加 200–500ms 延迟
3. **评估数据表明 Rewrite 效果不明显**：对比报告显示 Enhanced（含 Rewrite）的 context_precision 反而低于 Basic（详见 Q12）

### Improvements（优化建议）

1. **条件性 Rewrite**：仅对 `colloquial` 类型查询启用改写，对 `exact_keyword` 查询直接透传
2. **多次改写投票**：生成 2-3 个改写版本，分别检索后合并结果
3. **改写质量评估**：加入改写前后的相似度检查，差异过大时回退到原始查询

---

## Q9: Query Rewrite 的实现方式

### Facts（项目事实）

**实现架构：**

Query Rewrite 通过工厂模式实现可插拔切换：

```python
# File: app/retriever/preprocessing/factory.py
@lru_cache(maxsize=8)
def get_query_preprocessor(preprocessor_type: str) -> BaseQueryPreprocessor:
    registry = {
        "none": PassthroughPreprocessor,
        "rewrite": QueryRewritePreprocessor,
    }
    cls = registry.get(preprocessor_type)
    return cls()
```

**Passthrough 实现（none 模式）：**

```python
# File: app/retriever/preprocessing/passthrough.py
class PassthroughPreprocessor(BaseQueryPreprocessor):
    def process(self, query: str) -> str:
        return query  # 直接透传
```

**LLM Rewrite 实现（rewrite 模式）：**

```python
# File: app/retriever/preprocessing/rewrite.py
class QueryRewritePreprocessor(BaseQueryPreprocessor):
    def __init__(self) -> None:
        self._llm = None  # 懒初始化

    def _get_llm(self):
        if self._llm is None:
            from langchain_community.chat_models import ChatTongyi
            self._llm = ChatTongyi(
                model=config.rag_model,      # qwen-max
                temperature=0,               # 确定性输出
                dashscope_api_key=config.dashscope_api_key,
            )
        return self._llm

    def process(self, query: str) -> str:
        try:
            prompt = _REWRITE_PROMPT_TEMPLATE.format(query=query)
            response = self._get_llm().invoke(prompt)
            rewritten = response.content.strip()
            if not rewritten:
                return query  # 空响应回退
            return rewritten
        except Exception as e:
            return query  # 失败回退到原始查询
```

**关键设计要素：**

| 要素 | 值 | 原因 |
|------|-----|------|
| 模型 | `ChatTongyi`（qwen-max） | 复用项目已有的 DashScope API |
| temperature | `0` | 确保改写结果稳定、可复现 |
| 初始化方式 | 懒加载（`_get_llm()`） | 避免 `none` 模式下不必要的模型加载 |
| 降级策略 | `try/except → return query` | 改写失败时不中断检索流程 |
| Prompt 约束 | "输出只包含改写后的问题" | 避免 LLM 输出多余解释 |

### Analysis（分析）

**Rewrite 实现方式对比：**

| 方式 | 项目中使用 | 优点 | 缺点 |
|------|-----------|------|------|
| **LLM 改写** | ✅ 使用 | 语义理解强，可补充专业术语 | 延迟高（200-500ms），成本高 |
| 规则改写 | ❌ 未使用 | 速度快，确定性高 | 泛化能力差，维护成本高 |
| 小模型改写 | ❌ 未使用 | 速度快，可本地部署 | 需要训练数据 |
| 查询扩展（QE） | ❌ 未使用 | 不改变原始查询，增加检索维度 | 可能引入噪声 |

**项目选择 LLM 改写的原因：**
1. 项目已经集成了 DashScope API（qwen-max），无需额外部署
2. 运维领域的口语-书面语映射需要较强的语义理解能力
3. 知识库规模小（12 篇），Rewrite 的延迟增加在可接受范围内

### Improvements（优化建议）

1. **本地部署轻量改写模型**：使用微调的 BERT 或小参数 LLM（如 Qwen-1.8B）做改写，消除 API 调用延迟
2. **改写结果缓存**：对相似查询的改写结果缓存，避免重复调用 LLM
3. **引入 Few-shot 示例**：在 Prompt 中加入 2-3 个高质量的改写示例，提升改写质量一致性
4. **Rewrite + 原始查询双路检索**：同时用原始查询和改写后查询检索，合并结果取并集

---

## Q10: RAG 流程的召回速度

### Facts（项目事实）

项目在 Enhanced 模式下记录了 **分阶段结构化耗时日志**：

```python
# File: app/retriever/enhanced.py
t_start = time.time()

# Stage 1: Query Preprocessing
preprocessor = get_query_preprocessor(config.query_preprocessor_type)
search_query = preprocessor.process(query)

# Stage 2: Hybrid Search
t_stage2 = time.time()
candidates = enhanced_vector_store_manager.hybrid_search(...)
meta["hybrid_search_time_ms"] = int((time.time() - t_stage2) * 1000)

# Stage 3: Reranking
t_stage3 = time.time()
final_docs = reranker.rerank(query=original_query, documents=candidates, top_k=top_k)
meta["reranker_time_ms"] = int((time.time() - t_stage3) * 1000)

meta["total_time_ms"] = int((time.time() - t_start) * 1000)

# 日志格式
logger.info(
    f"[Enhanced][{trace_id}] 检索完成: "
    f"preprocessor=rewrite, reranker=cross_encoder, "
    f"candidates=20, final=3, total_ms=470"
)
```

README 中的示例日志：

```
[EnhancedRAG] trace=abc123 耗时: preprocess=0.00s|hybrid_search=0.12s|rerank=0.35s|total=0.47s
```

**各阶段耗时分解（基于代码注释和日志示例）：**

| 阶段 | 操作 | 预估耗时 | 依赖 |
|------|------|----------|------|
| Stage 0: Embedding | text-embedding-v4 API 调用 | ~100-200ms | 网络 + DashScope API |
| Stage 1: Preprocessing | ChatTongyi LLM 改写（rewrite 模式） | ~200-500ms | 网络 + DashScope API |
| Stage 2: Hybrid Search | Milvus Dense ANN + Sparse BM25 + RRF | ~50-150ms | 本地 Milvus |
| Stage 3: Reranking | Cross-Encoder 推理（20 个候选对） | ~200-800ms | 本地 CPU 推理 |
| **总计（Enhanced + Rewrite + Reranker）** | | **~550-1650ms** | |
| **总计（Basic，无改写无精排）** | | **~100-200ms** | |

### Analysis（分析）

**耗时瓶颈分析：**

```
Enhanced 模式耗时分布（估算）：

  Preprocessing (LLM Rewrite)  ████████████████  ~30%  (~350ms)
  Embedding API                ████████          ~15%  (~150ms)
  Hybrid Search (Milvus)       ██████            ~12%  (~100ms)
  Reranking (Cross-Encoder)    ██████████████████████  ~43%  (~500ms)
```

**关键发现：**

1. **Cross-Encoder 精排是最大瓶颈**：约占总耗时的 43%。Cross-Encoder 需要对每个 (query, doc) 对做完整的 Transformer 推理，20 个候选就是 20 次推理
2. **LLM Rewrite 是第二瓶颈**：约占 30%，每次检索额外增加一次 LLM API 调用
3. **Milvus 检索本身很快**：在 ~100 个向量的规模下，Dense + Sparse + RRF 仅需 ~100ms
4. **Basic 模式快 3-8 倍**：仅需 Embedding + Milvus ANN，约 100-200ms

### Improvements（优化建议）

1. **减少精排候选数**：将 `RERANK_COARSE_TOP_K` 从 20 降到 10，精排耗时减半
2. **异步并行化**：Stage 1（Rewrite）和 Stage 0（Embedding）可以并行执行——Rewrite 用原始文本，Embedding 也用原始文本，改写完成后再用改写文本做第二次 Embedding
3. **Cross-Encoder 模型量化**：使用 INT8 或 FP16 量化减少推理耗时
4. **Bi-Encoder 快速精排**：用 Bi-Encoder（如 BGE-Large）替代 Cross-Encoder 做初筛，Cross-Encoder 仅对 top-5 精排

---

## Q11: RAG 部分的 Agent 是否自动选择是否需要 RAG

### Facts（项目事实）

**是的，Agent 自动决定是否调用 RAG。**

RAG Agent 使用 LangGraph 的 ReAct Agent，LLM 自主决定何时调用 `retrieve_knowledge` 工具：

```python
# File: app/services/rag_agent_service.py
self.tools = [retrieve_knowledge, get_current_time]
# MCP tools also added
all_tools = self.tools + self.mcp_tools
self.agent = create_agent(self.model, tools=all_tools, checkpointer=self.checkpointer)
```

LLM 看到的工具描述：

```python
# File: app/tools/knowledge_tool.py
@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。
    """
```

**Agent 的自动决策行为（基于评估数据集的观察）：**

```python
# File: tests/evaluation/agent_testset.py
# 场景 4: 误报/噪声输入 — Agent 不应调用任何工具
{
    "scenario": "简单问候（不应调工具）",
    "input": "你好",
    "expected_tools": [],
    "forbidden_tools": ["retrieve_knowledge", "search_log", ...],
},
{
    "scenario": "无关闲聊（不应调工具）",
    "input": "今天天气怎么样？你会做什么？",
    "expected_tools": [],
    "forbidden_tools": ["retrieve_knowledge", ...],
},

# 场景 3: 知识类查询 — Agent 应调用 retrieve_knowledge
{
    "scenario": "CPU 问题排查知识",
    "input": "CPU 飙高一般是什么原因？怎么排查？",
    "expected_tools": [{"name": "retrieve_knowledge"}],
    "forbidden_tools": ["search_log", "query_cpu_metrics", ...],
},
```

System Prompt 中的工具使用指导：

```python
# File: app/services/rag_agent_service.py
# Method: _build_system_prompt()
"当需要获取实时信息或专业知识时，主动使用相关工具"
"如果工具无法提供足够信息，请诚实地告知用户"
```

### Analysis（分析）

**Agent 自动选择 RAG 的决策机制：**

```
用户输入 → LLM 推理
  │
  ├─ "你好" → 不需要工具 → 直接回答
  │
  ├─ "CPU 飙高怎么排查" → 需要专业知识 → 调用 retrieve_knowledge
  │
  ├─ "现在几点" → 需要实时信息 → 调用 get_current_time
  │
  ├─ "data-sync-service CPU 告警" → 需要监控数据 → 调用 query_cpu_metrics (MCP)
  │
  └─ "帮我全面排查" → 需要多种信息 → 调用多个工具
```

**决策质量取决于：**

1. **工具描述的清晰度**：`retrieve_knowledge` 的描述明确指出"当用户的问题涉及专业知识、文档内容或需要参考资料时使用"
2. **System Prompt 的引导**："当需要获取实时信息或专业知识时，主动使用相关工具"
3. **LLM 的推理能力**：qwen-max 需要正确理解用户意图并匹配到合适的工具

### Improvements（优化建议）

1. **工具选择评估**：在 Agent 评估中增加"工具选择准确率"指标——不仅评估调用了哪些工具，还评估是否在不该调用时避免了调用
2. **意图分类前置**：在 Agent 之前加一个轻量意图分类器，预先判断是否需要 RAG，减少 LLM 自主决策的不确定性
3. **动态工具集**：根据用户历史行为或会话上下文动态调整可用工具集

---

## Q12: RAG 评估核心指标与结果

### Facts（项目事实）

项目使用 **RAGAs 框架** 实现了两阶段评估，并补充了非 LLM 的检索指标。

**评估脚本：**

```python
# File: tests/evaluation/evaluate_rag.py
# 两阶段评估：
# Phase 1: 检索评估（始终执行）— context_precision + context_recall + Hit Rate + MRR
# Phase 2: 生成评估（可选）— faithfulness + answer_relevancy
```

**评估数据集：**

```python
# File: tests/evaluation/rag_testset.py
DATASET_VERSION = "1.1.2"  # 67 条样本
```

| 分类 | 数量 | 说明 |
|------|------|------|
| `exact_keyword` | 17 | 精确关键词/技术术语查询 |
| `colloquial` | 36 | 口语化查询 |
| `cross_doc` | 7 | 跨文档综合查询 |
| `edge_case` | 7 | 边界/反事实/关联影响查询 |

**核心指标体系：**

| 指标 | 类型 | 计算方式 | 说明 |
|------|------|----------|------|
| `context_precision` | LLM Judge | RAGAs 框架 | 检索结果中相关文档的精确率 |
| `context_recall` | LLM Judge | RAGAs 框架 | 相关文档被检索到的召回率 |
| `hit_rate@k` | 非 LLM | 集合运算 | 至少有一个相关文档出现在 top-k 中的问题占比 |
| `mrr` | 非 LLM | 集合运算 | 第一个相关文档排名的倒数的平均值 |
| `faithfulness` | LLM Judge | RAGAs 框架 | 生成答案对检索上下文的忠实度 |
| `answer_relevancy` | LLM Judge | RAGAs 框架 | 生成答案与问题的相关度 |

**实际评估结果（来自 `reports/basic.json`）：**

```json
// File: reports/basic.json
// Basic 模式 (Dense L2, no preprocessing, no reranker, top_k=3)
{
    "rag_mode": "basic",
    "retrieval_metrics": {
        "context_precision": 0.5797,
        "context_recall": 0.5437
    },
    "non_llm_metrics": {
        "hit_rate@3": 0.9701,
        "hit_rate@5": 0.9701,
        "hit_rate@10": 0.9701,
        "mrr": 0.9254
    }
}
```

**实际评估结果（来自 `reports/enhanced.json`）：**

```json
// File: reports/enhanced.json
// Enhanced 模式 (Dense+Sparse+RRF, rewrite preprocessing, cross_encoder reranker, top_k=3)
{
    "rag_mode": "enhanced",
    "retrieval_metrics": {
        "context_precision": 0.5324,
        "context_recall": 0.4852
    },
    "non_llm_metrics": {
        "hit_rate@3": 0.9701,
        "hit_rate@5": 0.9701,
        "hit_rate@10": 0.9701,
        "mrr": 0.8955
    }
}
```

**对比结果（来自 `reports/comparison.json`）：**

```json
// File: reports/comparison.json
{
    "comparison": {
        "context_precision": {
            "basic": 0.5797,
            "enhanced": 0.5324,
            "delta": -0.0473
        },
        "context_recall": {
            "basic": 0.5437,
            "enhanced": 0.4852,
            "delta": -0.0585
        }
    }
}
```

**目标基线（来自评估脚本注释）：**

```python
# File: tests/evaluation/evaluate_rag.py
# 目标基线（basic 模式）：context_precision ≥ 0.70, context_recall ≥ 0.70
# Enhanced 模式目标：context_precision ≥ 0.80, context_recall ≥ 0.80
```

### Analysis（分析）

**结果解读：**

| 指标 | Basic | Enhanced | Delta | 目标 | 达标 |
|------|-------|----------|-------|------|------|
| `context_precision` | 0.5797 | 0.5324 | **-0.0473** | ≥ 0.70 | ❌ 均未达标 |
| `context_recall` | 0.5437 | 0.4852 | **-0.0585** | ≥ 0.70 | ❌ 均未达标 |
| `hit_rate@3` | **0.9701** | **0.9701** | 0 | — | ✅ 极高 |
| `mrr` | **0.9254** | 0.8955 | -0.0299 | — | ✅ 高 |

**关键发现：**

1. **Hit Rate 极高（97%），但 Precision/Recall 偏低（~55%）**：这说明检索系统能够找到正确的文档（Hit Rate 高），但 LLM Judge 认为检索到的文档片段中"与问题最相关的部分"覆盖不够充分（Precision 低），以及 ground_truth 中的要点未被检索上下文完全覆盖（Recall 低）。

2. **Enhanced 模式反而低于 Basic 模式**：这可能是因为：
   - 知识库只有 12 篇文档，数据量太小，混合检索和精排的优势无法体现
   - Query Rewrite 引入了噪声（改写后的查询可能偏离了原始意图）
   - Cross-Encoder 精排可能将某些 LLM Judge 认为重要的片段排到了后面

3. **MRR 高（0.93）**：说明第一个相关文档通常排在第 1 位（1/0.93 ≈ 1.08），排序质量很好

4. **生成评估未执行**：`generation_metrics: null`，faithfulness 和 answer_relevancy 没有数据

### Improvements（优化建议）

1. **执行生成评估**：运行 `--with-generation` 获取 faithfulness 和 answer_relevancy 数据
2. **分析低分样本**：找出 context_precision/recall 最低的问题，分析失败原因
3. **扩充知识库**：12 篇文档太小，增加文档数量后重新评估
4. **调整 LLM Judge**：当前 Judge 使用 `qwen3.5-plus`，不同 Judge 模型对评分的影响可能很大

---

## Q13: RAG 对比实验设计

### Facts（项目事实）

项目通过 **消融实验（Ablation Study）** 和 **Basic vs Enhanced 对比报告** 两种方式进行对比实验。

**1. 消融实验（10 组参数组合）**

```python
# File: tests/evaluation/run_ablation.py
ABLATION_COMBINATIONS = [
    # --- basic 模式基线 ---
    {"label": "basic (baseline k=3)",        "RAG_MODE": "basic", "RAG_TOP_K": "3"},
    {"label": "basic k=5",                   "RAG_MODE": "basic", "RAG_TOP_K": "5"},
    {"label": "basic k=10",                  "RAG_MODE": "basic", "RAG_TOP_K": "10"},

    # --- enhanced: hybrid, no reranker, no preprocessor ---
    {"label": "enhanced (hybrid, no-rerank) k=3",  "RAG_MODE": "enhanced",
     "QUERY_PREPROCESSOR_TYPE": "none", "RERANKER_TYPE": "none",
     "RERANK_COARSE_TOP_K": "20", "RERANKER_TOP_K": "3"},
    {"label": "enhanced (hybrid, no-rerank) k=5",  ...},

    # --- enhanced: cross_encoder reranker, no preprocessor ---
    {"label": "enhanced (cross_encoder) k=3",       "RAG_MODE": "enhanced",
     "QUERY_PREPROCESSOR_TYPE": "none", "RERANKER_TYPE": "cross_encoder",
     "RERANK_COARSE_TOP_K": "20", "RERANKER_TOP_K": "3"},
    {"label": "enhanced (cross_encoder) k=5",       ...},
    {"label": "enhanced (cross_encoder) k=10",      ...},

    # --- enhanced: cross_encoder + query rewrite ---
    {"label": "enhanced (cross_encoder+rewrite) k=3", "RAG_MODE": "enhanced",
     "QUERY_PREPROCESSOR_TYPE": "rewrite", "RERANKER_TYPE": "cross_encoder",
     "RERANK_COARSE_TOP_K": "20", "RERANKER_TOP_K": "3"},
    {"label": "enhanced (cross_encoder+rewrite) k=5", ...},
]
```

消融参数维度：

| 维度 | 取值 | 组合数 |
|------|------|--------|
| `RAG_MODE` | basic / enhanced | 2 |
| `top_k` | 3 / 5 / 10 | 3 |
| `RERANKER_TYPE` | none / cross_encoder | 2 |
| `QUERY_PREPROCESSOR_TYPE` | none / rewrite | 2 |

**实验设计逻辑：渐进式叠加**

```
baseline (basic k=3)
  → 增加 top_k (k=5, k=10)
    → 切换 enhanced + 混合检索 (no reranker)
      → 增加 Cross-Encoder 精排
        → 增加 Query Rewrite
```

每组实验通过子进程独立运行，注入环境变量覆盖配置：

```python
# File: tests/evaluation/run_ablation.py
def _run_one_ablation(combo, idx, total):
    env = os.environ.copy()
    env.update({k: v for k, v in combo.items() if k != "label"})
    result = subprocess.run(
        [sys.executable, "-m", "tests.evaluation.evaluate_rag", "--output", tmp_path],
        env=env, capture_output=True, text=True,
    )
```

**2. Basic vs Enhanced 对比报告**

```python
# File: tests/evaluation/compare_reports.py
IMPROVEMENT_THRESHOLD = 0.10  # Enhanced 必须比 Basic 提升 10% 以上
PASS_THRESHOLD = 0.70         # 绝对值达标线
```

对比报告使用独立的评估结果文件：

```bash
# 先分别运行两次评估
RAG_MODE=basic    python -m tests.evaluation.evaluate_rag --output reports/basic.json
RAG_MODE=enhanced python -m tests.evaluation.evaluate_rag --output reports/enhanced.json
# 生成对比报告
python -m tests.evaluation.compare_reports --basic reports/basic.json --enhanced reports/enhanced.json
```

### Analysis（分析）

**消融实验设计评估：**

| 方面 | 评价 | 说明 |
|------|------|------|
| **变量控制** | ✅ 良好 | 每次只变化一个维度（如 top_k 或 reranker_type） |
| **基线对照** | ✅ 良好 | 以 basic k=3 为基线 |
| **渐进式叠加** | ✅ 良好 | 逐步增加复杂度，可量化每个组件的贡献 |
| **独立性** | ✅ 良好 | 每个组合独立子进程，避免状态污染 |
| **覆盖度** | ⚠️ 部分不足 | `chunk_size` 未纳入消融（需重新入库文档） |
| **统计显著性** | ❌ 缺失 | 每组实验只运行一次，无多次运行取平均 |

**实验的局限性：**

1. 评估脚本注释中明确说明："chunk_size 不纳入消融（需重新入库文档，不适合脚本自动化）"
2. 每组实验只运行一次，没有多次运行取均值和标准差
3. `RERANK_COARSE_TOP_K` 固定为 20，未做消融

### Improvements（优化建议）

1. **多次运行取均值**：每组实验至少运行 3 次取平均，减少 LLM Judge 的随机性
2. **chunk_size 消融**：增加 400/800/1600 三种 chunk_size 的消融组
3. **统计显著性检验**：使用配对 t 检验或 Wilcoxon 检验评估差异是否显著
4. **可视化**：生成消融实验的折线图/热力图，直观展示参数-指标关系

---

## Q14: 最终检索准确率与优化思路

### Facts（项目事实）

**最终检索准确率汇总：**

| 指标 | Basic 模式 | Enhanced 模式 | 说明 |
|------|-----------|--------------|------|
| **Hit Rate@3** | **97.01%** | **97.01%** | 在 top-3 中命中相关文档的概率 |
| **Hit Rate@5** | 97.01% | 97.01% | 在 top-5 中命中相关文档的概率 |
| **Hit Rate@10** | 97.01% | 97.01% | 在 top-10 中命中相关文档的概率 |
| **MRR** | **92.54%** | 89.55% | 第一个相关文档的排名质量 |
| **Context Precision** | 57.97% | 53.24% | LLM Judge 评估的检索精确率 |
| **Context Recall** | 54.37% | 48.52% | LLM Judge 评估的检索召回率 |

**达标情况：**

```python
# 目标基线
# Basic:  context_precision ≥ 0.70, context_recall ≥ 0.70 → ❌ 均未达标
# Enhanced: context_precision ≥ 0.80, context_recall ≥ 0.80 → ❌ 均未达标
```

### Analysis（分析）

**准确率解读：**

1. **Hit Rate 97% 很高**：67 条测试问题中，约 65 条能在 top-3 检索结果中找到正确的文档。这主要得益于知识库只有 12 篇文档，top-3 就覆盖了 25% 的文档空间。

2. **Context Precision/Recall ~55% 偏低**：虽然检索到了正确的文档，但 LLM Judge 认为返回的 3 个文档片段中，与问题最相关的信息覆盖不够充分。可能原因：
   - top_k=3 太少，有些问题需要更多上下文
   - 分块策略导致关键信息被分割到不同的 chunk 中
   - LLM Judge 的评分标准严格

3. **Enhanced 不如 Basic**：在小规模知识库上，复杂的检索策略反而引入了噪声。

**优化思路体系：**

```
┌─────────────────────────────────────────────────────────┐
│  优化方向 1: 提升 Context Precision / Recall             │
│  ├── 增大 top_k（从 3 增到 5-8）                        │
│  ├── 优化分块策略（语义分块 / 父子分块）                │
│  └── 增加知识库文档数量和覆盖度                         │
├─────────────────────────────────────────────────────────┤
│  优化方向 2: 提升检索质量                                │
│  ├── 替换 Embedding 模型（如 bge-large-zh-v1.5）       │
│  ├── 引入 HyDE（假想文档 Embedding）                   │
│  ├── 自适应 top_k（根据置信度动态调整）                 │
│  └── 多路检索 + 学习排序                                │
├─────────────────────────────────────────────────────────┤
│  优化方向 3: 提升知识库质量                              │
│  ├── 扩充文档数量（从 12 篇到 100+）                   │
│  ├── 文档结构化（添加标签、分类、优先级）              │
│  └── 定期更新和维护                                    │
├─────────────────────────────────────────────────────────┤
│  优化方向 4: 优化评估体系                                │
│  ├── 增加评估数据集规模（从 67 条到 200+）             │
│  ├── 执行生成评估（faithfulness + answer_relevancy）    │
│  └── 多次运行取均值，减少 Judge 随机性                 │
└─────────────────────────────────────────────────────────┘
```

### Improvements（优化建议）

**短期（低成本高收益）：**

1. 将 `top_k` 从 3 增加到 5，给 LLM 更多上下文
2. 为 Jieba 添加运维领域自定义词典
3. 执行 `--with-generation` 获取完整的生成质量评估

**中期（中等投入）：**

4. 扩充知识库到 50+ 篇文档，覆盖更多运维场景
5. 引入语义分块（Semantic Chunking）替代当前的标题+字符分割
6. 实现父子分块索引（小块检索 + 大块上下文）

**长期（战略投入）：**

7. 微调领域专用的 Embedding 模型
8. 引入 Graph RAG（将运维知识构建为知识图谱）
9. 建立持续评估和反馈闭环

---

## Q15: PDF 扫描件、OCR、表格结构化的处理思考

### Facts（项目事实）

**当前 PDF 处理方式：**

项目仅实现了 **基础 PDF 文本提取**，使用 `pypdf` 库：

```python
# File: app/services/vector_index_service.py
# Method: _extract_pdf_text()
def _extract_pdf_text(self, path: Path) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)
```

**支持的文件类型：**

```python
# File: app/api/file.py
ALLOWED_EXTENSIONS = ["txt", "md", "pdf"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
```

**当前方案的局限性：**

| 场景 | 当前处理 | 效果 |
|------|----------|------|
| **原生 PDF**（文字可选） | `pypdf.extract_text()` | ✅ 可提取文字 |
| **PDF 扫描件**（图片型） | `pypdf.extract_text()` | ❌ 提取为空或极少文字 |
| **OCR 需求** | ❌ 未实现 | — |
| **表格结构化** | ❌ 未实现 | — |
| **图表/流程图** | ❌ 未实现 | — |
| **多栏排版** | `pypdf.extract_text()` | ⚠️ 提取顺序可能错乱 |

### Analysis（分析）

**当前实现的不足：**

`pypdf.extract_text()` 只能处理"原生 PDF"（即文字以字符编码存储的 PDF）。对于：

1. **扫描件 PDF**：页面是图片，`extract_text()` 返回空字符串
2. **表格数据**：提取为无结构的纯文本，丢失行列关系
3. **多栏排版**：文本提取顺序可能从左栏跳到右栏，破坏语义

**运维知识库中的实际需求分析：**

运维 SOP 文档（如项目中的 `cpu_high_usage.md`）通常是 Markdown 格式，包含：
- 步骤说明（有序列表）
- 命令行示例（代码块）
- 参数表格（Markdown 表格）
- 架构图（图片/流程图）

当前所有文档都是 `.md` 格式，PDF 支持是预留的扩展能力。

**行业解决方案对比：**

| 方案 | 扫描件 OCR | 表格结构化 | 多栏排版 | 集成复杂度 |
|------|-----------|-----------|----------|-----------|
| **pypdf**（当前） | ❌ | ❌ | ⚠️ | 低 |
| **PyMuPDF (fitz)** | ❌ | ⚠️ 基础 | ✅ | 低 |
| **Unstructured.io** | ✅ | ✅ | ✅ | 中 |
| **LlamaParse** | ✅ | ✅ | ✅ | 中 |
| **PaddleOCR + 自定义** | ✅ | ✅（需定制） | ⚠️ | 高 |
| **Azure AI Document Intelligence** | ✅ | ✅ | ✅ | 中（云服务） |
| **Marker（开源）** | ✅ | ✅ | ✅ | 中 |

### Improvements（优化建议）

**方案 1: 引入 Unstructured.io（推荐）**

```python
# 建议的实现方式
from unstructured.partition.auto import partition

def extract_document(self, path: Path) -> list[Document]:
    elements = partition(filename=str(path))
    # 自动识别文档类型，处理 OCR、表格、多栏等
    # 返回结构化的元素列表（标题、段落、表格、图片描述等）
```

优点：一站式处理所有文档格式，LangChain 生态集成好
缺点：依赖较重，OCR 需要安装 Tesseract

**方案 2: PaddleOCR（中文优化）**

```python
# 针对中文扫描件的 OCR
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')
result = ocr.ocr(image_path)
```

优点：中文 OCR 准确率高
缺点：模型大（~1GB），需要 GPU 加速

**方案 3: 表格结构化处理**

```python
# 使用 camelot 或 tabula-py 提取表格
import camelot
tables = camelot.read_pdf(file_path, pages='all')
for table in tables:
    df = table.df  # pandas DataFrame
    # 将表格转换为 Markdown 格式存入 Document
```

**方案 4: 多模态 Embedding（前沿方向）**

对于包含图表的文档，使用多模态 Embedding（如 CLIP）将图片也向量化，实现图文混合检索。

**优先级建议：**

1. **短期**：替换 `pypdf` 为 `PyMuPDF`，提升原生 PDF 的提取质量
2. **中期**：引入 `Unstructured.io`，处理扫描件和表格
3. **长期**：探索多模态 RAG，支持图表和架构图的检索

---

## 附录：RAG 系统技术栈速查

| 组件 | 技术 | 配置 |
|------|------|------|
| Embedding 模型 | DashScope text-embedding-v4 | 1024 维 |
| 向量数据库 | Milvus v2.5.10 Standalone | Docker 部署 |
| Basic 集合 | `biz` | L2 距离, IVF_FLAT |
| Enhanced 集合 | `biz_enhanced` | COSINE + BM25, RRF 融合 |
| 查询改写 | ChatTongyi (qwen-max) | temperature=0 |
| 精排模型 | BAAI/bge-reranker-v2-m3 | Cross-Encoder, 懒加载 |
| 文档分割 | MarkdownHeader + RecursiveCharacter | chunk_size=1600, overlap=100 |
| 评估框架 | RAGAs | 两阶段（检索 + 生成） |
| Judge 模型 | qwen3.5-plus | temperature=0.0 |
| 知识库 | 12 篇 Markdown SOP | ~74 KB |
| 评估数据集 | 67 条 Q&A | v1.1.2, 4 类场景 |
