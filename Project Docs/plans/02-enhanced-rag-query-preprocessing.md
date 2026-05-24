# 增强版 RAG 查询预处理

## 1. 功能和目的

在 `EnhancedRAGRetriever.retrieve()` 内部、Embedding 之前注入查询预处理层，对原始用户查询进行改写或扩展，提升后续检索的召回率和精准率。

该模块解决的核心问题：用户输入往往是口语化、模糊的自然语言（如"服务挂了怎么排查"），而知识库使用专业术语（如"ServiceUnavailable 告警处置"）。预处理层通过 LLM 将用户查询转化为更适合向量检索和关键词匹配的形式，弥合 query-document 之间的语义鸿沟。

与整体 RAG 系统的关系：
- 位于 Enhanced RAG Pipeline 的第一阶段（Query Preprocessing → Hybrid Search → Reranking）
- 通过 `query_preprocessor_type` 配置项选择策略
- 对上层 `retrieve_knowledge` 工具签名完全透明

## 2. 抽象实现思路

### 架构设计

```
app/retriever/preprocessing/
├── __init__.py
├── base.py           # 抽象基类 BaseQueryPreprocessor
├── passthrough.py    # 直通（不处理），query_preprocessor_type=none
├── rewrite.py        # LLM Query Rewriting，query_preprocessor_type=rewrite
├── hyde.py           # HyDE 假设文档嵌入（计划中，尚未实现）
├── multi_query.py    # Multi-Query 多路查询（计划中，尚未实现）
└── factory.py        # 工厂函数，依据配置返回实现
```

### 抽象接口

实际实现中采用了简化接口——`process(query) -> str`（返回单个改写后的查询字符串），而非 plans.md 中设计的 `ProcessedQuery` 数据类（支持多查询变体和 HyDE 标志位）。这是因为当前仅实现了单查询策略（none 和 rewrite）。

```python
# app/retriever/preprocessing/base.py（实际实现）
class BaseQueryPreprocessor(ABC):
    @abstractmethod
    def process(self, query: str) -> str:
        """预处理查询文本，返回优化后的查询字符串"""
        ...
```

### 四种策略（plans.md 设计）

| 策略 | 额外 LLM 调用 | 适用场景 | 实现状态 |
|------|--------------|----------|----------|
| `none` | 0 次 | 默认，查询已足够清晰 | 已实现 |
| `rewrite` | 1 次 | 用户输入口语化、含歧义 | 已实现 |
| `hyde` | 1 次 | 知识库与提问风格差距大 | 计划中，尚未实现 |
| `multi_query` | 1 次 | 需提升多角度召回覆盖率 | 计划中，尚未实现 |

### 工厂模式

```python
# app/retriever/preprocessing/factory.py
_processor_registry = {
    "none": PassthroughPreprocessor,
    "rewrite": QueryRewritePreprocessor,
}

@lru_cache
def get_query_preprocessor() -> BaseQueryPreprocessor:
    processor_type = config.query_preprocessor_type
    cls = _processor_registry.get(processor_type)
    if cls is None:
        return PassthroughPreprocessor()
    return cls()
```

## 3. 具体实现流程

### 已实现：PassthroughPreprocessor（直通）

文件：[app/retriever/preprocessing/passthrough.py](app/retriever/preprocessing/passthrough.py)

`query_preprocessor_type=none` 时使用。直接返回原始 query，零开销。

### 已实现：QueryRewritePreprocessor（LLM 改写）

文件：[app/retriever/preprocessing/rewrite.py](app/retriever/preprocessing/rewrite.py)

使用 `ChatTongyi`（DashScope）调用 LLM，通过改写 prompt 将用户口语化问题改写为更适合向量检索的形式：
- 保留所有关键技术术语（服务名、指标名、告警名）
- 展开缩写（如 "CPU 高" → "CPU 使用率高 HighCPUUsage 告警"）
- 补充同义词或相关术语
- 去除与检索无关的礼貌用语

LLM 采用懒惰初始化，失败时回退到原始 query，确保降级可用。

### 计划中：HyDEPreprocessor（假设文档嵌入）

状态：**计划中，尚未实现。**

原理（Gao et al. 2022）：让 LLM 根据用户问题生成一段假设的知识库文档片段，对该假设文档做 Dense Embedding 后进行向量检索。因为假设文档的向量分布与真实文档更接近，可以减少 query-document 语义鸿沟。

与混合检索结合的关键设计：
- Dense Embedding 使用假设文档文本向量
- Sparse (BM25) 仍使用原始 query（关键词匹配不适合假设文档）

### 计划中：MultiQueryPreprocessor（多路查询）

状态：**计划中，尚未实现。**

原理：从多个角度生成同一问题的不同表述变体，对每个变体分别检索，通过去重合并候选集，提升召回覆盖率。始终将原始 query 加入列表首位保证原意不丢失。

### 与 EnhancedRAGRetriever 的集成

文件：[app/retriever/enhanced.py](app/retriever/enhanced.py)

```python
# 实际集成逻辑（简化版）
def retrieve(self, query: str, top_k: int) -> list[Document]:
    # 1. 预处理查询
    processed_query = self.preprocessor.process(query)  # 返回单个 str

    # 2. 使用处理后查询进行混合检索
    candidates = self._hybrid_search(processed_query, ...)

    # 3. 使用原始 query 进行 Reranking
    if self.reranker:
        candidates = self.reranker.rerank(query, candidates, top_k=top_k)
    return candidates
```

重要设计原则：Reranker 阶段始终使用原始 query 打分，而非改写后的文本。

### 配置项

在 [app/config.py](app/config.py) 中：

```python
query_preprocessor_type: Literal["none", "rewrite", "hyde", "multi_query"] = "none"
```

对应 `.env`：`QUERY_PREPROCESSOR_TYPE=none`。

注意：虽然 `Literal` 类型声明了全部四种策略，但 `hyde` 和 `multi_query` 在 factory 注册表中尚未注册。

## 4. 当前实现进度

### 已完成

- [x] 抽象基类 `BaseQueryPreprocessor` 定义完成
- [x] `PassthroughPreprocessor`（none 策略）实现完成
- [x] `QueryRewritePreprocessor`（rewrite 策略）实现完成，使用 `ChatTongyi` LLM 改写
- [x] 工厂函数 `get_query_preprocessor()` 实现完成，带 `@lru_cache` 和注册表模式
- [x] 配置项 `query_preprocessor_type` 添加完成
- [x] 与 `EnhancedRAGRetriever.retrieve()` 集成完成

### 部分完成

- [ ] `config.py` 中 `Literal` 类型声明了 `hyde` 和 `multi_query`，但 factory 注册表中未注册

### 尚未完成

- [ ] `hyde.py` — HyDE 假设文档嵌入预处理（计划中，尚未实现）
- [ ] `multi_query.py` — Multi-Query 多路查询预处理（计划中，尚未实现）
- [ ] plans.md 中设计的 `ProcessedQuery` 数据类（支持 `use_hyde` 标志和多查询变体列表）未实现
- [ ] `multi_query_count` 和 `preprocessor_temperature` 配置项未添加

### 依赖其他模块

- `hyde` 和 `multi_query` 的实现会影响 `EnhancedRAGRetriever.retrieve()` 的聚合逻辑（需要支持多查询变体遍历和 HyDE 的 Dense/Sparse 分离）

## 5. Evidence

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| 抽象基类 | [app/retriever/preprocessing/base.py](app/retriever/preprocessing/base.py) | `BaseQueryPreprocessor` 抽象类，`process() -> str` |
| 直通实现 | [app/retriever/preprocessing/passthrough.py](app/retriever/preprocessing/passthrough.py) | `PassthroughPreprocessor`，返回原始 query |
| 改写实现 | [app/retriever/preprocessing/rewrite.py](app/retriever/preprocessing/rewrite.py) | `QueryRewritePreprocessor`，使用 `ChatTongyi` LLM |
| 工厂函数 | [app/retriever/preprocessing/factory.py](app/retriever/preprocessing/factory.py) | `get_query_preprocessor()`，仅注册 `none` 和 `rewrite` |
| 配置项 | [app/config.py:43](app/config.py#L43) | `query_preprocessor_type: Literal["none", "rewrite", "hyde", "multi_query"]` |
| 环境变量 | [.env:24](.env#L24) | `QUERY_PREPROCESSOR_TYPE=none` |
| 集成点 | [app/retriever/enhanced.py:49](app/retriever/enhanced.py#L49) | `processed_query = self.preprocessor.process(query)` |
| hyde.py 不存在 | `app/retriever/preprocessing/` | 目录中无 `hyde.py` 文件 |
| multi_query.py 不存在 | `app/retriever/preprocessing/` | 目录中无 `multi_query.py` 文件 |
| Git 提交 | `f1f48be` | `feat: Phase 2 - 实现 Enhanced RAG（双向量混合检索 + 可插拔精排）` |
