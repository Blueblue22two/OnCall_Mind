# 增强版 RAG 精排重排（Reranker）

## 1. 功能和目的

在混合检索粗排之后，对候选文档集进行深度语义排序，输出最终 top_k 结果。Reranker 通过 Cross-Encoder 等模型对 `(query, doc)` 对进行细粒度语义打分，弥补向量检索粗排的排序质量不足。

该模块解决的核心问题：向量检索（无论是 Dense 还是 Sparse）的排序是近似且粗糙的，可能将部分相关但非最佳的文档排在前面。Reranker 对候选集逐对打分，输出更精确的相关性排序。

与整体 RAG 系统的关系：
- 位于 Enhanced RAG Pipeline 的第三阶段（Query Preprocessing → Hybrid Search → **Reranking**）
- 接收 Hybrid Search 的粗排候选集（默认 20 个），输出精排后的 top_k 结果（默认 3 个）
- 始终使用原始用户 query 打分（而非预处理改写后的 query）

## 2. 抽象实现思路

### 架构设计

```
app/retriever/reranker/
├── __init__.py
├── base.py           # 抽象基类 BaseReranker
├── passthrough.py    # 直通（不重排），reranker_type=none
├── cross_encoder.py  # Cross-Encoder 模型重排，reranker_type=cross_encoder
└── factory.py        # 工厂函数，依据配置返回实现
```

### 抽象接口

```python
# app/retriever/reranker/base.py
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """对候选文档进行重排序，返回 top_k 个最相关文档"""
        ...
```

### 三种方案（plans.md 设计）

| 方案 | 实现状态 | 说明 |
|------|----------|------|
| `none`（PassthroughReranker） | 已实现 | 直接截取前 top_k 个文档，无额外计算 |
| `cross_encoder`（CrossEncoderReranker） | 已实现 | 使用 BAAI/bge-reranker-v2-m3 本地推理（~560MB） |
| `llm`（LLM Reranker） | 计划中，尚未实现 | 使用现有 ChatQwen 打分，无需额外模型但延迟高 |

### 工厂模式

```python
# app/retriever/reranker/factory.py
_reranker_registry = {
    "none": PassthroughReranker,
    "cross_encoder": CrossEncoderReranker,
}

@lru_cache
def get_reranker() -> BaseReranker:
    reranker_type = config.reranker_type
    cls = _reranker_registry.get(reranker_type)
    if cls is None:
        return PassthroughReranker()
    if issubclass(cls, CrossEncoderReranker):
        return cls(model_name=config.reranker_model)
    return cls()
```

### 配置项

```python
# app/config.py
reranker_type: Literal["none", "cross_encoder", "llm"] = "cross_encoder"
reranker_model: str = "BAAI/bge-reranker-v2-m3"
reranker_top_k: int = 3           # 最终输出 top_k
rerank_coarse_top_k: int = 20     # 粗排候选集大小
```

## 3. 具体实现流程

### 已实现：PassthroughReranker（直通）

文件：[app/retriever/reranker/passthrough.py](app/retriever/reranker/passthrough.py)

`reranker_type=none` 时使用。直接返回 `documents[:top_k]`，零开销。适用于调试或仅依赖粗排质量的场景。

### 已实现：CrossEncoderReranker（Cross-Encoder 模型重排）

文件：[app/retriever/reranker/cross_encoder.py](app/retriever/reranker/cross_encoder.py)

使用 `FlagEmbedding.FlagReranker` 加载 `BAAI/bge-reranker-v2-m3` 模型：
- 模型约 560MB，首次启动时从 HuggingFace Hub 下载
- 使用 `use_fp16=True` 加速推理
- 采用懒惰初始化（`@property` + 首次调用时才加载模型）
- 对每个 `(query, doc.page_content)` 对计算相关性分数
- 按分数降序排列，返回 top_k 文档
- 失败时回退到截断列表（降级可用）

### 计划中：LLM Reranker

状态：**计划中，尚未实现。**

使用项目已有的 `ChatQwen` 模型，让 LLM 对每个候选文档进行打分（0-10 分）。优点是不需要部署额外模型，缺点是延迟高（每个候选都需一次 LLM 调用），仅适合候选集极小（≤5）的场景。

### 与 EnhancedRAGRetriever 集成

文件：[app/retriever/enhanced.py](app/retriever/enhanced.py)

```python
# Reranking（始终使用原始 query 打分）
if self.reranker:
    candidates = self.reranker.rerank(original_query, candidates, top_k=top_k)
else:
    candidates = candidates[:top_k]
```

重要设计原则：Reranker 始终使用**原始 query** 打分，确保分数反映用户真实意图，而非改写后的文本。

### 配置语义注意点

当前实现已经把 `reranker_top_k` 接入 enhanced 主链路：`knowledge_tool.py` 在 `rag_mode=enhanced` 时会优先使用 `reranker_top_k`，`tests/evaluation/evaluate_rag.py` 也以该值作为 enhanced 评估的有效 `top_k`。因此现在的语义是：
- `rag_top_k`：basic 模式最终返回数
- `reranker_top_k`：enhanced 模式精排后最终返回数
- `rerank_coarse_top_k`：enhanced 模式粗排候选数

这里不再是“配置存在但没进主链路”的状态，而是已完成语义收敛。

## 4. 当前实现进度

### 已完成

- [x] 抽象基类 `BaseReranker` 定义完成
- [x] `PassthroughReranker`（none 策略）实现完成
- [x] `CrossEncoderReranker`（cross_encoder 策略）实现完成
- [x] 使用 `FlagEmbedding.FlagReranker` + `BAAI/bge-reranker-v2-m3`
- [x] 懒惰初始化模型加载（避免启动时阻塞）
- [x] 工厂函数 `get_reranker()` 实现完成，带 `@lru_cache`
- [x] 配置项 `reranker_type`、`reranker_model`、`reranker_top_k`、`rerank_coarse_top_k` 添加完成
- [x] 与 `EnhancedRAGRetriever.retrieve()` 集成完成
- [x] 原始 query 打分原则已落实

### 部分完成

- [ ] `llm` reranker 仍未实现；现有链路已完成 `reranker_top_k` / `rag_top_k` 的语义分离

### 尚未完成

- [ ] LLM Reranker（`llm` 策略）—— 计划中，尚未实现。`Literal` 类型声明了但 factory 注册表中未注册

### 依赖其他模块

- 依赖 HuggingFace Hub 网络连接（模型首次下载）
- CPU 推理延迟约 200-800ms（取决于候选集大小），GPU 推理约 50-200ms
- `FlagEmbedding` 已在 [pyproject.toml:35](pyproject.toml#L35) 中声明

## 5. Evidence

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| 抽象基类 | [app/retriever/reranker/base.py](app/retriever/reranker/base.py) | `BaseReranker` 抽象类 |
| 直通实现 | [app/retriever/reranker/passthrough.py](app/retriever/reranker/passthrough.py) | `PassthroughReranker`，截取 top_k |
| Cross-Encoder | [app/retriever/reranker/cross_encoder.py](app/retriever/reranker/cross_encoder.py) | `CrossEncoderReranker`，`FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)` |
| 工厂函数 | [app/retriever/reranker/factory.py](app/retriever/reranker/factory.py) | `get_reranker()`，注册 `none` 和 `cross_encoder` |
| LLM Reranker 不存在 | `app/retriever/reranker/` | 目录中无 `llm.py` 文件 |
| 配置项 | [app/config.py:44-47](app/config.py#L44) | `reranker_type`, `reranker_model`, `reranker_top_k`, `rerank_coarse_top_k` |
| 环境变量 | [.env:25-28](.env#L25) | `RERANKER_TYPE=cross_encoder`, `RERANKER_MODEL=BAAI/bge-reranker-v2-m3` |
| 集成点 | [app/retriever/enhanced.py](app/retriever/enhanced.py) | Reranker 使用原始 query 打分 |
| 集成点 | [app/retriever/enhanced.py:62](app/retriever/enhanced.py#L62) | `self.reranker.rerank(query, candidates, top_k=top_k)` — 注意 top_k 参数来源 |
| 依赖声明 | [pyproject.toml:35](pyproject.toml#L35) | `FlagEmbedding>=1.2.0` |
| Git 提交 | `f1f48be` | `feat: Phase 2 - 实现 Enhanced RAG（双向量混合检索 + 可插拔精排）` |
