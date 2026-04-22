# SuperBizAgent RAG 优化方案文档

> 基于项目代码深度分析，面向三个优化目标的可实施方案设计。
> 项目路径：`super_biz_agent_py-release-2026-03-21/`

---

## 背景与现状分析

### 当前 RAG 实现局限性

| 维度 | 当前状态 | 问题 |
|---|---|---|
| 检索方式 | 纯 Dense 向量检索（L2 距离） | 无法处理精确关键词匹配，专有名词召回率低 |
| 向量索引 | IVF_FLAT，nlist=128，nprobe=10 | 仅单路稠密向量字段 `vector`(1024-dim) |
| 召回数量 | top_k=3，无分数阈值过滤 | 可能召回不相关文档，无质量把控 |
| 后处理 | 无任何 Reranking | 粗排即最终结果，排序质量有限 |
| 可扩展性 | `retrieve_knowledge` 工具硬耦合实现 | 无法在不修改工具代码的情况下切换策略 |

### 关键现存文件（后续修改基础）

```
app/config.py                             # Pydantic Settings，需新增 RAG 模式配置项
app/tools/knowledge_tool.py               # retrieve_knowledge @tool，入口点
app/services/vector_store_manager.py      # LangChain Milvus 包装，get_vector_store()
app/services/vector_embedding_service.py  # DashScopeEmbeddings，embed_query/embed_documents
app/services/vector_search_service.py     # 直接 PyMilvus 搜索路径（低层）
app/core/milvus_client.py                 # Milvus 集合 Schema 定义，需扩展为双向量字段
app/services/rag_agent_service.py         # LangGraph ReAct Agent，query 注入点
app/agent/aiops/planner.py                # AIOps Planner，直接调用 retrieve_knowledge
```

---

## 问题一：RAG 可插拔接口设计

### 目标

在保留原有 RAG 功能的前提下，将 RAG 检索抽象为可插拔接口，通过配置项决定使用哪个实现。

### 设计思路

#### Step 1：抽象检索接口

定义抽象基类，统一所有 RAG 实现的调用契约：

```
app/retriever/
├── __init__.py
├── base.py          # 抽象基类 BaseRAGRetriever
├── basic.py         # 原始 Dense 检索实现（现有逻辑迁移）
├── enhanced.py      # 增强版 RAG 实现（问题二实现）
└── factory.py       # 工厂函数，根据配置返回对应实现
```

**核心抽象接口设计：**

```python
# app/retriever/base.py
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseRAGRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int) -> list[Document]:
        """统一检索接口，所有实现必须遵守此契约"""
        ...
```

**工厂函数（依据配置动态返回）：**

```python
# app/retriever/factory.py
def get_rag_retriever() -> BaseRAGRetriever:
    if config.rag_mode == "enhanced":
        return EnhancedRAGRetriever()
    return BasicRAGRetriever()
```

#### Step 2：重构 `retrieve_knowledge` 工具

将工具与具体实现解耦，工具只依赖抽象接口：

```python
# app/tools/knowledge_tool.py（重构后）
retriever = get_rag_retriever()   # 工厂注入

@tool(response_format="content_and_artifact")
async def retrieve_knowledge(query: str) -> tuple[str, list[Document]]:
    docs = await retriever.retrieve(query, top_k=config.rag_top_k)
    return format_docs(docs), docs
```

#### Step 3：在配置中新增开关

```python
# app/config.py 新增字段
rag_mode: Literal["basic", "enhanced"] = "basic"
```

对应 `.env` 中：

```
RAG_MODE=basic    # 或 enhanced
```

#### Step 4：BasicRAGRetriever（迁移现有逻辑）

将 `vector_store_manager.get_vector_store().as_retriever(search_kwargs={"k": top_k})` 封装进 `BasicRAGRetriever`，确保原有行为 100% 保留。

### 关键约束

- `retrieve_knowledge` 的 `@tool(response_format="content_and_artifact")` 签名不变，对上层 LangGraph Agent 透明
- `BasicRAGRetriever` 的行为与当前完全一致，确保不破坏现有功能
- 工厂函数使用模块级单例缓存，避免每次请求重复初始化

---

## 问题二：增强版 RAG 系统设计

### 2.1 Query Preprocessing（查询预处理）

#### 架构设计

在 `EnhancedRAGRetriever.retrieve()` 内部、Embedding 之前注入预处理层，对上层工具签名完全透明：

```
app/retriever/
└── preprocessing/
    ├── __init__.py
    ├── base.py           # 抽象基类 BaseQueryPreprocessor
    ├── passthrough.py    # 直通（不处理）
    ├── rewrite.py        # LLM Query Rewriting
    ├── hyde.py           # HyDE（假设文档嵌入）
    ├── multi_query.py    # Multi-Query（多路查询）
    └── factory.py        # 依据配置返回实现
```

**抽象接口：**

```python
# app/retriever/preprocessing/base.py
from dataclasses import dataclass, field

@dataclass
class ProcessedQuery:
    queries: list[str]       # 一个或多个待检索的 query 字符串
    use_hyde: bool = False   # 是否使用 HyDE 向量（已是假设文档文本）

class BaseQueryPreprocessor(ABC):
    @abstractmethod
    async def process(self, query: str) -> ProcessedQuery:
        """将原始 query 转换为一个或多个检索用 query"""
        ...
```

`ProcessedQuery.queries` 统一返回 `list[str]`，支持单 query 和多 query 两种情况，`EnhancedRAGRetriever` 只需遍历列表分别检索后合并去重。

#### 四种实现

**① PassthroughPreprocessor（直通）**

`query_preprocessor_type=none` 时使用，Zero overhead，默认策略。

```python
class PassthroughPreprocessor(BaseQueryPreprocessor):
    async def process(self, query: str) -> ProcessedQuery:
        return ProcessedQuery(queries=[query])
```

---

**② QueryRewritePreprocessor（查询改写）**

**原理**：让 LLM 将用户的口语化/模糊问题改写为更适合向量检索的形式，消除歧义词、补全缩写、统一术语。

**适用场景**：用户输入是自然语言提问（"服务挂了怎么排查"），但知识库中使用专业术语（"ServiceUnavailable 告警处置"）。

```python
REWRITE_PROMPT = """你是一个信息检索专家，负责将用户的问题改写为更适合在运维知识库中检索的查询语句。

改写原则：
1. 保留所有关键技术术语（服务名、指标名、告警名）
2. 展开缩写（如 "CPU 高" → "CPU 使用率高 HighCPUUsage 告警"）
3. 补充同义词或相关术语（如 "内存不够" → "内存使用率过高 HighMemoryUsage OOM"）
4. 去除与检索无关的礼貌用语和主观描述
5. 输出1句话，不超过50字

原始问题：{query}
改写后的检索语句："""
```

LLM 调用：temperature=0（确定性输出），返回单个改写后的 query 字符串。

---

**③ HyDEPreprocessor（假设文档嵌入）**

**原理**（Precise Zero-Shot Dense Retrieval, Gao et al. 2022）：

```
原始 query → LLM 生成假设文档片段（模拟知识库中的答案段落）
                    ↓
假设文档文本 → Dense Embedding → 向量
                    ↓
与真实知识库向量做相似度检索（向量空间更接近真实文档分布）
```

比直接对 query 做 embedding 更有效，因为假设文档的向量分布与真实文档的向量分布更相近，减少了 query-document 语义鸿沟。

```python
HYDE_PROMPT = """请根据以下运维问题，撰写一段可能出现在运维知识库中的参考文档片段（约100字）。
直接给出文档内容，不要包含"根据知识库"等前缀。

问题：{query}
文档片段："""
```

返回 `ProcessedQuery(queries=[hypothetical_doc], use_hyde=True)`。

**与混合检索结合时的关键细节**：

```python
# Dense embedding 使用假设文档文本向量
dense_vec = embeddings.embed_query(hypothetical_doc)

# Sparse（BM25）仍使用原始 query（关键词匹配不适合假设文档）
sparse_vec = bm25.encode_queries([original_query])
```

Dense 用假设文档向量，Sparse 用原始 query 关键词，两路互补。

---

**④ MultiQueryPreprocessor（多路查询）**

**原理**：从多个角度生成同一问题的不同表述，对每个变体分别检索，通过 RRF 或去重合并候选集，提升召回覆盖率。

```python
MULTI_QUERY_PROMPT = """请为以下运维问题生成{n}个不同角度的检索查询，每行一个，不要编号。
要求：
- 保持核心意图不变
- 每个查询角度或侧重点略有不同（如：症状表述、原因表述、操作动词变化）
- 可混合中英文技术术语

原始问题：{query}
查询变体："""
```

始终将原始 query 加入列表首位，确保原意不丢失；保序去重后返回 `n+1` 个 query。

#### 与 EnhancedRAGRetriever 集成

```python
async def retrieve(self, query: str, top_k: int) -> list[Document]:
    # 1. Query Preprocessing
    processed = await self.preprocessor.process(query)

    # 2. 对每个 query 变体分别做混合检索
    all_candidates: list[Document] = []
    for q in processed.queries:
        dense_text = q
        # HyDE 时 sparse 回退使用原始 query
        sparse_text = query if processed.use_hyde else q

        dense_vec = self.embeddings.embed_query(dense_text)
        sparse_vec = self.bm25.encode_queries([sparse_text])

        hits = self._hybrid_search(dense_vec, sparse_vec, coarse_top_k=20)
        all_candidates.extend(hits)

    # 3. 多路结果合并去重
    candidates = self._deduplicate(all_candidates)

    # 4. Reranking（始终使用原始 query 打分，确保与用户意图一致）
    if self.reranker:
        candidates = self.reranker.rerank(query, candidates, top_k=top_k)
    else:
        candidates = candidates[:top_k]

    return candidates
```

**重要原则**：Reranker 阶段始终使用**原始 query** 打分，而非改写后的文本。

#### 各策略对比

| 策略 | 额外 LLM 调用 | 适用场景 | 额外延迟（估计） |
|---|---|---|---|
| `none` | 0 次 | 默认，查询已足够清晰 | 无 |
| `rewrite` | 1 次 | 用户输入口语化、含歧义 | +100~300ms |
| `hyde` | 1 次 | 知识库与提问风格差距大 | +200~500ms |
| `multi_query` | 1 次 | 需提升多角度召回覆盖率 | +200~500ms + N 路检索 |

#### 配置项

```python
# app/config.py 新增
query_preprocessor_type: Literal["none", "rewrite", "hyde", "multi_query"] = "none"
multi_query_count: int = 3           # multi_query 模式下生成变体数量
preprocessor_temperature: float = 0.0  # 预处理 LLM 温度（建议 0，确定性输出）
```

对应 `.env`：

```
QUERY_PREPROCESSOR_TYPE=none   # none | rewrite | hyde | multi_query
MULTI_QUERY_COUNT=3
PREPROCESSOR_TEMPERATURE=0.0
```

---

### 2.2 Embedding 策略：双路编码

#### Dense Embeddings（保留现有）

- 模型：`text-embedding-v4`（1024-dim，DashScope）
- 捕获深层语义，处理同义词、模糊表达
- 存储字段：`dense_vector` (FLOAT_VECTOR, 1024-dim)

#### Sparse Embeddings（新增 BM25）

- 工具：`pymilvus.model.sparse.BM25EmbeddingFunction`（Milvus 2.5+ 内置）
  - 备选：`rank_bm25` 库 + 自定义稀疏格式转换
- 捕获精确关键词权重，处理专有名词（服务名、告警名、指标名）
- 存储字段：`sparse_vector` (SPARSE_FLOAT_VECTOR)
- BM25 需要在全量文档语料上 `fit()` 后才能编码，需在文档入库阶段同步训练

**推荐框架：** `pymilvus>=2.4.0`（原生 Sparse Vector 支持）

#### 双向量 Schema 设计

需在 `app/core/milvus_client.py` 中扩展 Milvus Collection Schema：

```python
# 新增字段（创建新集合 biz_enhanced，与原 biz 集合并存）
FieldSchema(name="dense_vector",  dtype=DataType.FLOAT_VECTOR,  dim=1024),
FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
```

**集合迁移策略：**
- 创建新集合 `biz_enhanced`，与原 `biz` 集合并存
- `MILVUS_COLLECTION` 配置项控制当前使用哪个集合
- 文档重新入库时同时写入两个向量字段

**注意事项：**
- `BM25EmbeddingFunction` 需要保存拟合后的模型状态（序列化到磁盘），以便重启后复用
- 推荐路径：`data/bm25_model.pkl`，在 `VectorIndexService.upload_documents()` 中管理生命周期

### 2.3 粗排召回：混合检索 + RRF 融合

#### 技术路线：Milvus Native Hybrid Search

Milvus 2.5+ 提供原生混合检索 API，推荐直接使用：

```python
from pymilvus import AnnSearchRequest, RRFRanker, Collection

dense_req = AnnSearchRequest(
    data=[dense_query_vector],
    anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=coarse_top_k,   # 粗排 top_k，建议 20~50
)
sparse_req = AnnSearchRequest(
    data=[sparse_query_vector],
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=coarse_top_k,
)

results = collection.hybrid_search(
    reqs=[dense_req, sparse_req],
    rerank=RRFRanker(k=60),   # RRF 融合，k=60 为经验值
    limit=rerank_top_k,        # 精排输入候选集大小，建议 10~20
    output_fields=["content", "metadata"],
)
```

**RRF 算法原理：**

$$\text{RRF\_score}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + r(d)}$$

稀疏检索命中的专有名词（如 `HighCPUUsage`）和稠密检索召回的语义相关内容，通过 RRF 互补融合，同时提升查全率和查准率。

**索引配置更新（`milvus_client.py`）：**

```python
# dense_vector 字段索引
{"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
# sparse_vector 字段索引（Milvus 2.5+）
{"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {"drop_ratio_build": 0.2}}
```

注意：将 metric_type 从 `L2` 改为 `COSINE`（对归一化向量更稳定）

#### 备选方案（如 Milvus 版本不支持）

使用 `langchain_milvus` 的 `hybrid_search` 扩展 + 手动 RRF 实现：

```python
# 分别执行 dense 和 sparse 搜索，手动实现 RRF 合并
dense_docs = dense_store.similarity_search_with_score(query, k=coarse_k)
sparse_docs = sparse_store.similarity_search_with_score(query, k=coarse_k)
merged = rrf_merge(dense_docs, sparse_docs, k=60)
```

### 2.4 精排重排：Reranker

粗排后对候选集（建议 10~20 个）进行深度语义排序，输出最终 top_k。

#### 方案 A：Cross-Encoder Reranker（推荐，本地推理）

- 模型推荐：`BAAI/bge-reranker-v2-m3`（支持中英双语，HuggingFace）
- 框架：`FlagEmbedding` 库（`FlagReranker`）或 `sentence-transformers` (`CrossEncoder`)
- 原理：将 `(query, doc)` 对拼接后经 BERT 类模型打分，捕获细粒度语义差异
- 延迟：本地推理约 50~200ms（GPU）或 200~800ms（CPU，取决于候选集大小）

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
scores = reranker.compute_score([(query, doc.page_content) for doc in candidates])
```

#### 方案 B：LLM Reranker（无需额外模型，利用现有 Qwen）

- 使用项目已有的 `ChatQwen` 模型
- 让 LLM 对每个候选文档打 0~10 分并给出理由
- 优点：无需部署额外模型；缺点：延迟高（每个候选都需 LLM 调用）
- 适合候选集较小（≤5）时使用

#### 可插拔 Reranker 配置

```python
# app/config.py 新增
reranker_type: Literal["none", "cross_encoder", "llm"] = "none"
reranker_model: str = "BAAI/bge-reranker-v2-m3"
reranker_top_k: int = 3          # 最终输出 top_k
rerank_coarse_top_k: int = 20    # 粗排候选集大小
```

### 2.5 EnhancedRAGRetriever 完整流程

```
原始 Query
  ↓
【Query Preprocessing】（依据 query_preprocessor_type 配置）
  · none       → 直接传入原始 query
  · rewrite    → LLM 改写为检索友好语句
  · hyde       → LLM 生成假设文档片段（Dense 用假设文档，Sparse 仍用原始 query）
  · multi_query→ LLM 生成 N 个变体，原始 query 加入列表首位
  ↓
【双路 Embedding】（对每个 query 变体）
  1. Dense Embedding: DashScopeEmbeddings.embed_query() → 1024-dim float vector
  2. Sparse Embedding: BM25EmbeddingFunction.encode_queries() → sparse dict
  ↓
【Hybrid Search + RRF】（Milvus AnnSearchRequest × 2 + RRFRanker(k=60)）
  → coarse_top_k 候选文档（默认 20），多路变体结果合并去重
  ↓
【Reranking】（使用原始 query 打分，依据 reranker_type 配置）
  → reranker_top_k 精排结果（默认 3）
  ↓
返回 List[Document]（与 BasicRAGRetriever 接口一致）
```

### 2.6 文档入库流程变化

原有 `VectorIndexService.upload_documents()` 需同时生成并写入两种向量：

```
Document Chunks
  ↓
DashScopeEmbeddings.embed_documents()    → dense_vectors  (List[List[float]])
BM25EmbeddingFunction.encode_documents() → sparse_vectors (List[dict])
  ↓
pymilvus insert: {id, content, metadata, dense_vector, sparse_vector}
```

**推荐框架版本：**
- `pymilvus>=2.4.6`（Sparse Vector + Hybrid Search 完整支持）
- `FlagEmbedding>=1.2.0`（bge-reranker-v2-m3）
- `rank_bm25>=0.2.2`（备选 BM25 实现）

---

## 问题三：RAGAs 评估体系设计

### 3.1 整体思路

利用 RAGAs 框架，在可插拔接口基础上，对 `BasicRAGRetriever` 和 `EnhancedRAGRetriever` 分别评估，量化两者性能差异。

**推荐框架：** `ragas>=0.1.0`（最新版，支持 LangChain 集成）

### 3.2 评估指标选择

| 指标 | 评估维度 | 需要 | 说明 |
|---|---|---|---|
| `context_precision` | 检索精准率 | query + contexts + ground_truth | 召回文档中有多少是真正相关的 |
| `context_recall` | 检索召回率 | query + contexts + ground_truth | 相关文档中有多少被召回 |
| `faithfulness` | 生成忠实度 | query + answer + contexts | 回答是否完全基于检索内容，无幻觉 |
| `answer_relevancy` | 回答相关性 | query + answer | 回答与问题的相关程度 |

对于本项目重点关注**检索质量**（context_precision + context_recall），因为这是两个 RAG 系统最直接的差异所在。

### 3.3 评估数据集构建

项目中已有 5 个 aiops-docs Markdown 知识库文档，基于这些文档构建标准评估集：

**数据集格式：**

```python
# tests/evaluation/rag_testset.py
EVAL_DATASET = [
    {
        "question": "CPU使用率超过80%时应该如何排查？",
        "ground_truth": "首先使用 query_cpu_metrics 工具查看具体进程，...",
        "reference_docs": ["aiops-docs/cpu_high_usage.md"],
    },
    {
        "question": "内存使用率超过85%的紧急处理步骤是什么？",
        "ground_truth": "...",
        "reference_docs": ["aiops-docs/memory_high_usage.md"],
    },
    # 覆盖 5 个文档的多样化问题，建议每文档 3~5 个问题，共 15~25 条
]
```

**数据集扩充建议：**
- 使用 RAGAs `TestsetGenerator`（基于 LLM 自动生成测试问题）：

```python
from ragas.testset.generator import TestsetGenerator
generator = TestsetGenerator.from_langchain(llm, embeddings)
testset = generator.generate_with_langchain_docs(docs, test_size=25)
```

### 3.4 评估执行架构

```
tests/evaluation/
├── __init__.py
├── rag_testset.py           # 标准评估数据集
├── evaluate_rag.py          # 主评估脚本
├── ragas_metrics.py         # 指标配置
└── report/                  # 评估报告输出目录
    ├── basic_rag_report.json
    └── enhanced_rag_report.json
```

**核心评估逻辑：**

```python
# tests/evaluation/evaluate_rag.py
async def run_evaluation(rag_mode: str):
    os.environ["RAG_MODE"] = rag_mode

    retriever = get_rag_retriever()
    agent = RagAgentService()

    results = []
    for item in EVAL_DATASET:
        docs = await retriever.retrieve(item["question"], top_k=config.rag_top_k)
        answer = await agent.query(item["question"], session_id="eval")
        results.append({
            "question": item["question"],
            "answer": answer,
            "contexts": [doc.page_content for doc in docs],
            "ground_truth": item["ground_truth"],
        })

    from datasets import Dataset
    dataset = Dataset.from_list(results)
    scores = evaluate(dataset, metrics=[
        context_precision, context_recall, faithfulness, answer_relevancy
    ])
    scores.to_pandas().to_json(f"tests/evaluation/report/{rag_mode}_rag_report.json")
    return scores
```

### 3.5 LLM Judge 配置

RAGAs 内部需要 LLM 作为 judge，项目现有 `ChatQwen` 可直接复用：

```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ragas_llm = LangchainLLMWrapper(ChatQwen(model=config.rag_model))
ragas_embeddings = LangchainEmbeddingsWrapper(vector_embedding_service)
```

### 3.6 对比评估流程

```bash
# 评估 basic RAG
RAG_MODE=basic python -m tests.evaluation.evaluate_rag

# 评估 enhanced RAG
RAG_MODE=enhanced python -m tests.evaluation.evaluate_rag

# 生成对比报告
python -m tests.evaluation.compare_reports
```

**对比报告示例输出：**

| 指标 | Basic RAG | Enhanced RAG | Delta |
|---|---|---|---|
| context_precision | 0.72 | 0.85 | +0.13 ↑ |
| context_recall | 0.68 | 0.82 | +0.14 ↑ |
| faithfulness | 0.88 | 0.91 | +0.03 ↑ |
| answer_relevancy | 0.79 | 0.83 | +0.04 ↑ |

---

## 实施顺序建议

```
Phase 1：可插拔接口（问题一）
  → 新增 app/retriever/ 模块（base + basic + factory）
  → 重构 app/tools/knowledge_tool.py
  → 添加 rag_mode 配置项
  → 验证：RAG_MODE=basic 时行为与现有完全一致

Phase 2：增强 RAG 系统（问题二）
  → 实现 app/retriever/preprocessing/ 预处理模块（4 种策略）
  → 扩展 Milvus Schema（新建 biz_enhanced 集合）
  → 实现 BM25 稀疏向量编码和入库
  → 实现 Hybrid Search（AnnSearchRequest + RRFRanker）
  → 实现 Reranker 模块（Cross-Encoder 优先）
  → 组装 EnhancedRAGRetriever（预处理 → 双路检索 → Rerank）
  → 重新入库文档，验证混合检索效果

Phase 3：RAGAs 评估（问题三）
  → 构建 tests/evaluation/rag_testset.py 评估数据集
  → 实现 evaluate_rag.py 评估脚本
  → 分别在 basic 和 enhanced 模式下运行评估
  → 生成对比报告
```

## 依赖新增清单

```toml
# pyproject.toml 需新增
"pymilvus>=2.4.6"         # Sparse Vector + Hybrid Search 支持
"FlagEmbedding>=1.2.0"    # Cross-Encoder Reranker（BAAI/bge-reranker-v2-m3）
"rank_bm25>=0.2.2"        # 备选 BM25 实现
"ragas>=0.1.0"            # RAG 评估框架
"datasets>=2.0.0"         # RAGAs 依赖
```

## 注意事项

1. **向量维度兼容性**：Dense metric 从 `L2` 改为 `COSINE` 后，Milvus 集合需重建（或创建新集合 `biz_enhanced`），建议保留原有 `biz` 集合以支持 `basic` 模式
2. **BM25 状态持久化**：BM25 拟合后的词频统计需序列化保存，否则重启后无法复用已入库的稀疏向量
3. **Cross-Encoder 资源需求**：`bge-reranker-v2-m3` 约 560MB，需预先下载并考虑内存开销；无 GPU 时可选更小的 `bge-reranker-base`
4. **RAGAs LLM 配额**：评估时 RAGAs 会大量调用 LLM（每条数据多次），注意 DashScope API 调用成本和限速
5. **可插拔性验证**：切换 `RAG_MODE` 或 `QUERY_PREPROCESSOR_TYPE` 后无需重启，通过延迟初始化（lazy singleton）实现
6. **Reranker 使用原始 query**：预处理后 Reranker 打分始终使用原始用户 query，避免改写文本影响最终排序相关性
7. **HyDE + 混合检索**：Dense 检索用假设文档向量，Sparse 检索回退用原始 query，两路策略不能混用同一文本
