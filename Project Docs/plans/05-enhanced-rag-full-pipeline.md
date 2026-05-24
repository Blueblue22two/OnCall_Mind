# 增强版 RAG 完整流程与文档入库

## 1. 功能和目的

将查询预处理、双路混合检索、精排重排三个独立模块串联为完整的 Enhanced RAG Pipeline，并确保文档入库流程同时支持基础集合（`biz`）和增强集合（`biz_enhanced`）的双写。

该模块是 Phase 2（增强版 RAG）的集成层，将各独立组件组装为可运行的端到端检索流程。同时确保新文档上传后两个集合保持数据一致。

与整体系统的关系：
- `EnhancedRAGRetriever.retrieve()` 是增强检索的入口，封装完整三阶段 Pipeline
- `VectorIndexService.index_single_file()` 负责文档入库时的双集合同步写入
- 对上层 `retrieve_knowledge` 工具完全透明（通过可插拔接口切换）

## 2. 抽象实现思路

### Enhanced RAG 完整流程图

```
原始 Query
  ↓
【Stage 1: Query Preprocessing】（依据 query_preprocessor_type 配置）
  · none       → 直接传入原始 query
  · rewrite    → LLM 改写为检索友好语句
  · hyde       → 计划中，尚未实现
  · multi_query→ 计划中，尚未实现
  ↓
【Stage 2: Hybrid Search + RRF】（Milvus AnnSearchRequest × 2 + RRFRanker(k=60)）
  → Dense ANN: DashScopeEmbeddings.embed_query() → COSINE 检索
  → Sparse BM25: 原始文本 → Milvus 内置 BM25 检索
  → RRF 融合 → coarse_top_k 候选文档（默认 20）
  ↓
【Stage 3: Reranking】（使用原始 query 打分，依据 reranker_type 配置）
  → reranker_top_k 精排结果（默认 3）
  ↓
返回 List[Document]
```

### 文档入库双写流程

```
Document Chunks
  ↓
DashScopeEmbeddings.embed_documents() → dense_vectors
  ↓
┌─────────────────────────────────────────────────────┐
│ 写入 biz 集合（基础）                                 │
│ fields: id, vector(dense), content, metadata         │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ 写入 biz_enhanced 集合（增强）                        │
│ fields: id, dense_vector, content_text, metadata     │
│ sparse_vector 由 Milvus BM25 Function 自动生成       │
│ （写入失败不阻塞基础集合，try/except 包裹）            │
└─────────────────────────────────────────────────────┘
```

### 关键设计决策

1. **合并去重策略**：多路查询变体结果基于 `doc.metadata` 去重（为 multi_query 策略预留）
2. **原始 query 传递**：预处理后的 query 用于检索，但 Reranker 始终接收原始 query 打分
3. **空结果保护**：如果 Hybrid Search 返回 0 条候选文档，直接返回空列表，跳过 Reranker
4. **增强集合写入容错**：`biz_enhanced` 写入失败不阻塞 `biz` 写入，确保基础功能可用

## 3. 具体实现流程

### Step 1：EnhancedRAGRetriever 完整流程

文件：[app/retriever/enhanced.py](app/retriever/enhanced.py)

`retrieve(query, top_k)` 方法的执行顺序：
1. 调用 `self.preprocessor.process(query)` 获取预处理后的查询文本
2. 调用 `self.enhanced_store.hybrid_search(processed_query, ...)` 获取粗排候选集
3. 若候选集为空，直接返回空列表
4. 调用 `self.reranker.rerank(query, candidates, top_k=top_k)` 进行精排
5. 返回最终结果

### Step 2：文档入库双写

文件：[app/services/vector_index_service.py](app/services/vector_index_service.py)

`index_single_file()` 方法的执行顺序：
1. 读取文件（支持 txt, md, pdf）
2. 删除该文件之前写入的所有 chunks（两个集合都删除）
3. 使用 `DocumentSplitterService` 分块
4. 写入 `biz` 集合（`vector_store_manager.add_documents()`）
5. 写入 `biz_enhanced` 集合（`enhanced_vector_store_manager.add_documents()`，try/except 包裹）

### Step 3：服务初始化

在 [app/main.py](app/main.py) 的 lifespan 中：
- Milvus 连接建立 → 创建/加载 `biz` 和 `biz_enhanced` 集合
- 各 Service singleton 在首次调用时惰性初始化

### Step 4：配置项整合

在 [app/config.py](app/config.py) 中，Enhanced RAG 相关配置项：

```python
rag_mode: Literal["basic", "enhanced"] = "basic"
rag_top_k: int = 3
query_preprocessor_type: Literal["none", "rewrite", "hyde", "multi_query"] = "none"
reranker_type: Literal["none", "cross_encoder", "llm"] = "cross_encoder"
reranker_model: str = "BAAI/bge-reranker-v2-m3"
reranker_top_k: int = 3
rerank_coarse_top_k: int = 20
```

对应 `.env` 配置：

```env
RAG_MODE=enhanced
QUERY_PREPROCESSOR_TYPE=rewrite
RERANKER_TYPE=cross_encoder
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_TOP_K=3
RERANK_COARSE_TOP_K=20
```

## 4. 当前实现进度

### 已完成

- [x] `EnhancedRAGRetriever.retrieve()` 完整三阶段流程串联
- [x] 预处理结果接入检索流程
- [x] Hybrid Search 候选集获取
- [x] Reranker 精排 + 原始 query 打分
- [x] 空候选集保护（early return）
- [x] 文档入库双写逻辑（`biz` + `biz_enhanced`）
- [x] 增强集合写入失败容错（try/except 非阻塞）
- [x] 所有 Enhanced RAG 配置项在 `.env` 和 `config.py` 中定义
- [x] 通过 `rag_mode` 配置项在 Basic 和 Enhanced 模式间切换

### 尚未完成

此阶段的主流程已完成并可运行。当前仍保留的子模块扩展，属于下一阶段能力：
- Query Preprocessing 的 `hyde` 和 `multi_query` 策略（依赖 `02-enhanced-rag-query-preprocessing`）
- Reranker 的 `llm` 策略（依赖 `04-enhanced-rag-reranker`）

### 设计改进（已完成 2026-05-22）

以下六项改进已实施，覆盖可观测性、降级路径、双写反馈和配置收敛：

- [x] **6.3 检索链路可观测性** — trace_id + 三阶段结构化日志 + debug 模式
- [x] **6.4 降级路径** — 预处理/精排失败 fallback + retrieval_meta 存储
- [x] **6.2 双写结果反馈** — `SingleFileIndexResult` + API 返回值增强
- [x] **6.1 配置语义收敛** — `rag_top_k` / `reranker_top_k` / `rerank_coarse_top_k` 三层语义 + enhanced 模式自动选择

### 依赖其他模块

- 依赖 `02-enhanced-rag-query-preprocessing` 的 `hyde`/`multi_query` 补齐（影响多查询变体逻辑）
- 依赖 `04-enhanced-rag-reranker` 的 `llm` 策略补齐

## 5. Evidence

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| 完整流程 | [app/retriever/enhanced.py:46-70](app/retriever/enhanced.py#L46) | `retrieve()` 三阶段流水线 |
| 预处理集成 | [app/retriever/enhanced.py:49](app/retriever/enhanced.py#L49) | `processed_query = self.preprocessor.process(query)` |
| Hybrid Search | [app/retriever/enhanced.py:54](app/retriever/enhanced.py#L54) | `self.enhanced_store.hybrid_search(...)` |
| 空结果保护 | [app/retriever/enhanced.py:57](app/retriever/enhanced.py#L57) | `if not candidates: return []` |
| Reranker 原始query | [app/retriever/enhanced.py:62](app/retriever/enhanced.py#L62) | `self.reranker.rerank(query, candidates, top_k=top_k)` |
| 双写逻辑 | [app/services/vector_index_service.py:174-181](app/services/vector_index_service.py#L174) | `biz` 写入后 `biz_enhanced` 同步写入 |
| 容错处理 | [app/services/vector_index_service.py:181](app/services/vector_index_service.py#L181) | try/except 包裹增强集合写入 |
| RAG 模式配置 | [app/config.py:40](app/config.py#L40) | `rag_mode: Literal["basic", "enhanced"]` |
| 增强配置 | [app/config.py:43-47](app/config.py#L43) | 所有 Enhanced RAG 配置项 |
| 环境变量 | [.env:21-28](.env#L21) | `RAG_MODE` 及相关配置 |
| Git 提交 | `f1f48be` | `feat: Phase 2 - 实现 Enhanced RAG（双向量混合检索 + 可插拔精排）` |
| 可观测性 trace_id | [app/retriever/enhanced.py:63](app/retriever/enhanced.py#L63) | `trace_id = uuid.uuid4().hex[:8]` |
| 三阶段日志 | [app/retriever/enhanced.py:89-97](app/retriever/enhanced.py#L89) | Stage1/Stage2/Stage3 结构化日志 + 耗时 |
| 降级 - 预处理 | [app/retriever/enhanced.py:81-87](app/retriever/enhanced.py#L81) | try/except 预处理失败 → 回退原始 query |
| 降级 - 精排 | [app/retriever/enhanced.py:125-136](app/retriever/enhanced.py#L125) | try/except 精排失败 → 回退粗排截断 |
| retrieval_meta | [app/retriever/enhanced.py:46-47](app/retriever/enhanced.py#L46) | `self.last_retrieval_meta` 存储降级信息 |
| 双写结果 dataclass | [app/services/vector_index_service.py:16-35](app/services/vector_index_service.py#L16) | `SingleFileIndexResult` 包含双集合状态 |
| 上传 API 增强 | [app/api/file.py:71-88](app/api/file.py#L71) | 返回值含 `basic_index_status` / `enhanced_index_status` |
| 配置三层语义 | [app/config.py:41-59](app/config.py#L41) | `rag_top_k` / `reranker_top_k` / `rerank_coarse_top_k` 注释 |
| enhanced 选 top_k | [app/tools/knowledge_tool.py:28-36](app/tools/knowledge_tool.py#L28) | `rag_mode` 自动选择 `reranker_top_k` 或 `rag_top_k` |

## 6. 设计问题与改进（状态：✅ 已实施 2026-05-22）

### 6.1 配置语义需要收敛 ✅

**原问题**：`rag_top_k`、`rerank_coarse_top_k`、`reranker_top_k` 的职责边界不够清晰。`reranker_top_k` 虽然配置存在，却没有真正进入主链路。

**已实施方案**：
- 在 `config.py` 中明确三层语义注释：`rag_top_k`（basic 模式最终返回数）、`reranker_top_k`（enhanced 模式精排后截断数）、`rerank_coarse_top_k`（混合检索粗排候选数）
- 在 `knowledge_tool.py` 中根据 `rag_mode` 自动选择对应的 top_k：basic 模式使用 `rag_top_k`，enhanced 模式使用 `reranker_top_k`
- 评估脚本（`tests/evaluation/evaluate_rag.py`）继续使用 `rag_top_k`，与线上配置保持独立

### 6.2 双写流程需要更明确的结果反馈 ✅

**原问题**：增强集合写入失败采用吞错策略，缺少对外反馈，增强集合缺数据的问题可能长期隐藏。

**已实施方案**：
- 新增 `SingleFileIndexResult` dataclass，包含 `basic_index_status`、`enhanced_index_status`、`enhanced_index_error`、`partial_success` 四个状态字段
- `index_single_file()` 返回值从 None 改为 `SingleFileIndexResult`
- 上传 API 返回值新增 `chunks_count`、`basic_index_status`、`enhanced_index_status`，部分成功时附加 `partial_success_reason`
- 增强集合写入失败时输出结构化 WARNING 日志（含文件路径和分片数）
- 基础集合写入失败改为直接抛出异常（之前是静默吞错）

### 6.3 检索链路缺少观测点 ✅

**原问题**：增强检索只有整体开始和结束日志，缺少对每个阶段质量的记录。

**已实施方案**：
- 每次检索生成 `trace_id`（uuid4 hex[:8]），统一贯穿三阶段日志
- 三阶段各有独立的结构化日志：预处理（原始 query/改写 query）、混合检索（候选数/coarse_top_k/耗时）、精排（最终数/耗时）
- 结束时输出汇总日志：preprocessor 类型、reranker 类型、候选数、最终数、总耗时、降级状态
- 新增 `debug=True` 参数，输出中间结果摘要（原始 query、改写 query、候选来源列表、最终来源列表）
- 降级信息存储在实例的 `last_retrieval_meta` 字典中，便于外部查询

### 6.4 需要明确增强能力的降级路径 ✅

**原问题**：预处理、reranker、Milvus BM25 任意一层出错都可能让结果退化到截断或空结果，没有统一降级策略。

**已实施方案**：
- 预处理失败 → 回退原始 query 进行检索（WARNING 日志 + meta 标记 `degraded_stage=”preprocessing”`）
- 精排失败 → 回退粗排候选直接截断 top_k（WARNING 日志 + meta 标记 `degraded_stage=”reranker”`）
- 混合检索失败 → 不降级，直接抛出异常（基础设施问题不应静默）
- 所有降级信息记录在 `self.last_retrieval_meta` 中，字段包括 `degraded_stage`、`fallback_reason`、`trace_id`
- 支持多阶段连续降级（如预处理和精排同时失败时 `degraded_stage=”preprocessing,reranker”`）
