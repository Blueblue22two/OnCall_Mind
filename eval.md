# SuperBizAgent RAG 优化方案评估

> 基于 plans.md 中所有优化方案，从实施难度、推荐框架、注意事项、评估与测试四个维度进行综合评估。

---

## 问题一：RAG 可插拔接口设计

### 实施难度

**低** — 纯代码重构，不涉及基础设施变更。

- 新增 `app/retriever/` 模块（base + basic + factory），约 100 行代码
- 重构 `knowledge_tool.py`：将 `vector_store_manager.as_retriever()` 调用替换为工厂注入
- `app/config.py` 新增 `rag_mode` 字段（1 行）
- `BasicRAGRetriever` 只是对现有逻辑的薄封装，行为 100% 保留
- 无数据库变更、无依赖新增

### 推荐框架

| 组件 | 推荐 | 说明 |
|---|---|---|
| 抽象接口 | Python `abc.ABC` + `@abstractmethod` | 标准库，无额外依赖 |
| 工厂模式 | 模块级单例 + `functools.lru_cache` | 避免每次请求重复初始化 |
| 配置切换 | `pydantic-settings`（已有） | `Literal["basic", "enhanced"]` 类型安全 |

### 需要注意的点

1. **工具签名不变**：`@tool(response_format="content_and_artifact")` 装饰器和函数签名必须保持不变，LangGraph Agent 依赖此格式
2. **同步 vs 异步**：现有 `retrieve_knowledge` 是同步函数，`BasicRAGRetriever.retrieve()` 需保持同步兼容，或统一改为 `async`（需同步修改 `planner.py` 中的调用方式）
3. **单例初始化时机**：工厂函数在模块导入时执行，Milvus 连接必须已建立；需确保 `milvus_manager.connect()` 在 `get_rag_retriever()` 之前完成（FastAPI lifespan 已保证）
4. **AIOps Planner 路径**：`app/agent/aiops/planner.py` 直接调用 `retrieve_knowledge.ainvoke()`，重构后行为不变，但需回归测试

### 评估与测试

```bash
# 单元测试：验证工厂函数返回正确实现
RAG_MODE=basic pytest tests/retriever/test_factory.py

# 集成测试：验证 BasicRAGRetriever 与原有行为一致
pytest tests/retriever/test_basic_retriever.py

# 回归测试：完整 RAG 对话流程
RAG_MODE=basic pytest tests/integration/test_rag_chat.py
```

**验证标准**：`RAG_MODE=basic` 时，相同 query 的检索结果与重构前完全一致（文档内容、顺序、数量）。

---

## 问题二-A：双路 Embedding（Dense + BM25 Sparse）

### 实施难度

**中** — 涉及 Milvus Schema 变更和文档重新入库。

- 新建 `biz_enhanced` collection（双向量 Schema）
- `pymilvus` 升级：`>=2.3.5` → `>=2.4.6`（需验证 monkey-patch 兼容性）
- `VectorIndexService.index_single_file()` 需同时生成 dense + sparse 向量
- BM25 模型需要 `fit()` + 序列化持久化（`data/bm25_model.pkl`）
- 所有已有文档需重新入库到新集合

### 推荐框架

| 组件 | 推荐 | 备选 |
|---|---|---|
| Sparse Embedding | `pymilvus.model.sparse.BM25EmbeddingFunction` | `rank_bm25>=0.2.2` + 自定义格式转换 |
| Dense Embedding | 现有 `DashScopeEmbeddings`（保留） | — |
| 向量存储 | `pymilvus>=2.4.6` 原生 Sparse Vector | — |
| BM25 持久化 | `pickle` 序列化到 `data/bm25_model.pkl` | `bm25s` 库（更快的序列化） |

### 需要注意的点

1. **pymilvus monkey-patch 风险**：`_patch_pymilvus_milvus_client_orm_alias()` 在升级后需重新验证，该 patch 强制 `MilvusClient._using="default"` 以兼容 langchain_milvus 的 ORM 别名
2. **BM25 fit 时机**：必须在所有文档入库完成后执行 `fit()`，不能在单文件入库时 fit（语料不完整）；推荐在 `index_directory()` 完成后统一 fit
3. **metric_type 变更**：Dense 从 `L2` 改为 `COSINE`，旧 `biz` 集合不受影响，新 `biz_enhanced` 集合独立创建
4. **BM25 模型与 Milvus 数据一致性**：若 `bm25_model.pkl` 丢失或与当前 Milvus 数据不匹配，稀疏向量检索结果将不可靠；需在启动时校验模型版本
5. **中文分词**：`BM25EmbeddingFunction` 默认使用空格分词，对中文效果差；需配置 Jieba 分词器或使用 Milvus 内置中文 analyzer

### 评估与测试

```bash
# 验证双向量入库
pytest tests/services/test_vector_index_enhanced.py

# 验证 BM25 编码正确性（专有名词命中）
pytest tests/retriever/test_bm25_encoding.py

# 手动验证：专有名词检索
python -c "
from app.retriever.enhanced import EnhancedRAGRetriever
r = EnhancedRAGRetriever()
docs = r.retrieve('HighCPUUsage 告警', top_k=3)
print([d.metadata.get('_file_name') for d in docs])
"
```

**验证标准**：包含精确专有名词（如 `HighCPUUsage`、`OOMKilled`）的 query，Enhanced 模式的召回率高于 Basic 模式。

---

## 问题二-B：混合检索 + RRF 融合

### 实施难度

**中** — Milvus 2.5.10 已原生支持，API 调用层面改动不大，但需要正确配置双路 AnnSearchRequest。

- 实现 `_hybrid_search()` 方法（约 30 行）
- 配置 `AnnSearchRequest × 2 + RRFRanker(k=60)`
- 处理 Milvus `hybrid_search` 返回格式到 `List[Document]` 的转换
- 需要 `biz_enhanced` 集合已建立（依赖双路 Embedding）

### 推荐框架

| 组件 | 推荐 | 说明 |
|---|---|---|
| 混合检索 | `pymilvus.AnnSearchRequest` + `RRFRanker` | Milvus 2.5+ 原生，无需手动 RRF |
| RRF 参数 | `k=60`（经验值） | 可通过 RAGAs 评估调优 |
| 备选（低版本 Milvus） | 手动 RRF：分别检索后合并 | 代码量约 50 行 |

### 需要注意的点

1. **`coarse_top_k` 设置**：建议 20~50，过小会导致 Reranker 候选集不足，过大增加 Reranker 延迟
2. **`output_fields` 必须包含 `content` 和 `metadata`**：否则无法构建 `Document` 对象
3. **RRF k 值调优**：k=60 是经验值，实际效果取决于语料分布；可通过 RAGAs `context_precision` 指标调优
4. **Sparse 向量为空的情况**：若 query 中所有词均不在 BM25 词表中（如纯英文 query 而词表为中文），稀疏向量为全零，需做降级处理（回退到纯 Dense 检索）

### 评估与测试

```bash
# 对比 Dense-only vs Hybrid 检索结果
pytest tests/retriever/test_hybrid_search.py

# 专有名词召回率对比
python tests/evaluation/compare_retrieval.py --query "HighCPUUsage 排查" --mode basic,enhanced
```

**验证标准**：对包含专有名词的 query，Hybrid 模式的 `context_recall` 高于纯 Dense 模式至少 10%。

---

## 问题二-C：Query Preprocessing（查询预处理）

### 实施难度

| 策略 | 难度 | 说明 |
|---|---|---|
| Passthrough（none） | 极低 | 直通，零代码 |
| QueryRewrite（rewrite） | 低 | 单次 LLM 调用，prompt 工程 |
| HyDE（hyde） | 低-中 | 单次 LLM 调用，但需处理 Dense/Sparse 分离逻辑 |
| MultiQuery（multi_query） | 中 | 多路检索 + 去重合并，需处理结果聚合 |

整体模块难度：**低-中**，主要工作量在 `EnhancedRAGRetriever.retrieve()` 中的多路聚合逻辑。

### 推荐框架

| 组件 | 推荐 | 说明 |
|---|---|---|
| LLM 调用 | 现有 `ChatQwen`（复用） | temperature=0，确定性输出 |
| 抽象接口 | `abc.ABC` + `@dataclass ProcessedQuery` | 统一返回格式 |
| 多路去重 | 基于 `doc.metadata["_source"] + chunk_index` | 避免相同文档重复进入 Reranker |

### 需要注意的点

1. **HyDE + 混合检索的分离原则**：Dense 用假设文档向量，Sparse 必须回退用原始 query（关键词匹配不适合假设文档文本）
2. **Reranker 始终用原始 query 打分**：无论使用哪种预处理策略，Reranker 阶段的 query 参数必须是用户原始输入
3. **AIOps Planner 应使用 Passthrough**：`planner.py` 传入的是完整任务描述字符串，不适合 LLM 改写；通过配置 `query_preprocessor_type=none` 保证
4. **额外 LLM 调用成本**：rewrite/hyde/multi_query 均需额外 LLM 调用（+100~500ms），在高并发场景下需评估 DashScope API 限速影响
5. **multi_query 的原始 query 保序**：原始 query 必须加入变体列表首位，防止语义漂移

### 评估与测试

```bash
# 各策略单元测试
pytest tests/retriever/preprocessing/

# 对比不同预处理策略的检索质量
RAG_MODE=enhanced QUERY_PREPROCESSOR_TYPE=none python -m tests.evaluation.evaluate_rag
RAG_MODE=enhanced QUERY_PREPROCESSOR_TYPE=rewrite python -m tests.evaluation.evaluate_rag
RAG_MODE=enhanced QUERY_PREPROCESSOR_TYPE=hyde python -m tests.evaluation.evaluate_rag
RAG_MODE=enhanced QUERY_PREPROCESSOR_TYPE=multi_query python -m tests.evaluation.evaluate_rag
```

**验证标准**：对口语化 query（如"服务挂了怎么排查"），`rewrite` 策略的 `context_precision` 高于 `none` 策略；对知识库风格差异大的 query，`hyde` 策略的 `context_recall` 更优。

---

## 问题二-D：Reranker（精排重排）

### 实施难度

| 方案 | 难度 | 说明 |
|---|---|---|
| Cross-Encoder（bge-reranker-v2-m3） | 中 | 需下载 560MB 模型，本地推理，无额外 API 成本 |
| LLM Reranker（ChatQwen） | 低 | 复用现有模型，但延迟高（每候选一次 LLM 调用） |
| none（不使用） | 极低 | 直接截取 coarse_top_k 前 N 个 |

### 推荐框架

| 组件 | 推荐 | 备选 |
|---|---|---|
| Cross-Encoder | `FlagEmbedding>=1.2.0`（`FlagReranker`） | `sentence-transformers`（`CrossEncoder`） |
| 模型 | `BAAI/bge-reranker-v2-m3`（中英双语） | `BAAI/bge-reranker-base`（更小，无 GPU 时） |
| LLM Reranker | 现有 `ChatQwen`（复用） | — |

### 需要注意的点

1. **模型下载与内存**：`bge-reranker-v2-m3` 约 560MB，首次启动需预下载；无 GPU 时 CPU 推理延迟约 200~800ms（取决于候选集大小）
2. **`rerank_coarse_top_k` 与 `reranker_top_k` 的比例**：建议 coarse=20，final=3，比例约 7:1；比例过小会导致 Reranker 优化空间不足
3. **LLM Reranker 的并发限制**：候选集 20 个时需 20 次 LLM 调用，延迟不可接受；LLM Reranker 仅适合 coarse_top_k ≤ 5 的场景
4. **Reranker 与 HuggingFace Hub 的网络依赖**：生产环境需提前下载模型到本地，避免运行时网络请求

### 评估与测试

```bash
# 对比有无 Reranker 的最终结果质量
RERANKER_TYPE=none python -m tests.evaluation.evaluate_rag
RERANKER_TYPE=cross_encoder python -m tests.evaluation.evaluate_rag

# 延迟基准测试
pytest tests/retriever/test_reranker_latency.py --benchmark
```

**验证标准**：启用 Cross-Encoder Reranker 后，`context_precision` 提升 ≥ 5%，端到端延迟增加 ≤ 500ms（CPU 环境）。

---

## 问题二-E：BM25 增量更新缓解

### 实施难度

| 方案 | 难度 | 说明 |
|---|---|---|
| 全量重建（从 Milvus content） | 低 | 利用现有数据，约 20 行代码 |
| 批量延迟 Refit | 低-中 | 需引入 dirty flag 或定时任务 |
| 增量 DF 近似 | 高 | 需自定义 BM25 编码逻辑，与 pymilvus 格式不兼容 |
| Milvus 内置全文检索 | 中 | Schema 变更 + 中文 analyzer 配置，但长期零维护 |
| SPLADE | 高 | 需额外 500MB+ 模型，中文支持有限 |

### 推荐框架

| 方案 | 推荐框架 |
|---|---|
| 全量重建 | `pymilvus` collection.query() + `BM25EmbeddingFunction.fit()` |
| 定时任务 | `APScheduler>=3.10.0` 或 FastAPI lifespan background task |
| Milvus 内置 BM25 | `pymilvus>=2.5.0` + `FunctionType.BM25` + Jieba analyzer |

### 需要注意的点

1. **当前语料规模极小**（约 20-50 chunk），全量重建耗时 < 1 秒，短期内无需复杂方案
2. **BM25 模型与 Milvus 数据版本一致性**：每次 refit 后需更新模型文件的版本戳，启动时校验
3. **Milvus 内置 BM25 的中文分词**：需配置 `analyzer_params={"type": "chinese"}`，否则中文分词效果差
4. **增量 DF 近似方案的误差积累**：长期运行后 IDF 值偏差会累积，建议每周执行一次全量 refit 校正

### 评估与测试

```bash
# 验证 refit 后稀疏向量检索结果一致性
pytest tests/services/test_bm25_refit.py

# 模拟文档更新后的检索质量
python tests/evaluation/test_incremental_update.py \
  --add-doc aiops-docs/new_alert.md \
  --query "新告警处置步骤"
```

**验证标准**：新文档入库并 refit 后，针对新文档内容的 query 能正确召回新文档（`context_recall` = 1.0）。

---

## 问题三：RAGAs 评估体系

### 实施难度

**中** — 主要工作量在构建高质量评估数据集，框架集成本身较简单。

- RAGAs 框架集成：约 50 行代码
- 评估数据集构建：每文档 3~5 个问题，共 15~25 条，需人工标注 `ground_truth`（主要工作量）
- LLM Judge 配置：复用现有 `ChatQwen`，约 10 行
- 对比报告脚本：约 30 行

### 推荐框架

| 组件 | 推荐 | 说明 |
|---|---|---|
| 评估框架 | `ragas>=0.1.0` | 支持 LangChain 集成，4 个核心指标 |
| 数据集格式 | `datasets>=2.0.0`（HuggingFace） | RAGAs 依赖 |
| LLM Judge | `LangchainLLMWrapper(ChatQwen(...))` | 复用现有模型，无额外成本 |
| 自动生成测试集 | `ragas.testset.generator.TestsetGenerator` | 减少人工标注工作量 |

### 需要注意的点

1. **`ground_truth` 质量决定评估有效性**：`context_recall` 和 `context_precision` 的计算依赖 ground_truth，标注质量差会导致指标失真
2. **RAGAs LLM 调用成本**：每条评估数据需多次 LLM 调用（faithfulness 约 3~5 次），25 条数据约消耗 75~125 次 API 调用，注意 DashScope 限速
3. **`TestsetGenerator` 的局限性**：自动生成的问题可能过于简单或与知识库内容高度重叠，建议人工审核后使用
4. **评估环境隔离**：评估时使用独立 `session_id="eval"`，避免污染生产对话历史
5. **RAGAs 版本兼容性**：`ragas>=0.1.0` API 变化较大，需锁定版本；`ragas>=0.2.0` 引入了新的 metric 接口

### 评估与测试

```bash
# 构建并验证评估数据集
python -m tests.evaluation.validate_testset

# 运行完整对比评估
RAG_MODE=basic python -m tests.evaluation.evaluate_rag
RAG_MODE=enhanced python -m tests.evaluation.evaluate_rag

# 生成对比报告
python -m tests.evaluation.compare_reports
```

**验证标准**：
- `context_precision` 和 `context_recall` 均 ≥ 0.7（Basic 模式基线）
- Enhanced 模式相比 Basic 模式，`context_precision` 提升 ≥ 0.10，`context_recall` 提升 ≥ 0.10

---

## 综合优先级建议

| 优化模块 | 实施难度 | 预期收益 | 建议优先级 |
|---|---|---|---|
| 可插拔接口（问题一） | 低 | 架构基础，无直接性能提升 | P0（必须先做） |
| 双路 Embedding + Hybrid Search | 中 | 专有名词召回率显著提升 | P1 |
| Query Preprocessing（rewrite） | 低 | 口语化 query 精准率提升 | P1 |
| Cross-Encoder Reranker | 中 | 排序质量提升，延迟可控 | P2 |
| RAGAs 评估体系 | 中 | 量化验证，指导调优 | P2 |
| HyDE / MultiQuery | 低-中 | 特定场景有效，通用性一般 | P3 |
| BM25 增量更新优化 | 低（当前规模） | 工程健壮性 | P3 |
| Milvus 内置 BM25 | 中 | 长期零维护 | P3（长期演进） |
