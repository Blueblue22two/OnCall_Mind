# RAG 可插拔检索接口

## 1. 功能和目的

将 RAG 检索抽象为可插拔接口，通过配置项决定使用哪种检索实现（基础 Dense 检索或增强混合检索），在不修改工具代码的情况下切换检索策略。

该模块是整个 RAG 系统可扩展性的架构基础。后续所有检索优化（混合检索、精排、查询预处理）都基于此接口扩展。

它与整体系统的关系：
- 向上为 `retrieve_knowledge` 工具和 LangGraph Agent 提供统一的检索契约
- 向下封装具体的 Milvus 检索逻辑（basic 或 enhanced）
- 通过 `app/config.py` 中的 `rag_mode` 配置项实现运行时切换

## 2. 抽象实现思路

### 架构设计

```
app/retriever/
├── __init__.py      # 模块导出
├── base.py          # 抽象基类 BaseRAGRetriever
├── basic.py         # 基础 Dense 检索实现（现有逻辑迁移）
├── enhanced.py      # 增强版 RAG 实现（Phase 2）
└── factory.py       # 工厂函数，依据 rag_mode 返回对应实现
```

### 核心接口

```python
# app/retriever/base.py
class BaseRAGRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        """统一检索接口，所有实现必须遵守此契约"""
        ...
```

### 工厂模式

```python
# app/retriever/factory.py
@lru_cache
def get_rag_retriever() -> BaseRAGRetriever:
    if config.rag_mode == "enhanced":
        return EnhancedRAGRetriever()
    return BasicRAGRetriever()
```

使用 `@lru_cache` 实现模块级单例，避免每次请求重复初始化检索器。

### 工具解耦

`retrieve_knowledge` 工具通过工厂注入检索器，只依赖抽象接口，不依赖具体实现：

```python
# app/tools/knowledge_tool.py
retriever = get_rag_retriever()  # 工厂注入

@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> tuple[str, list[Document]]:
    docs = retriever.retrieve(query, top_k=config.rag_top_k)
    return format_docs(docs), docs
```

### 配置切换

```python
# app/config.py
rag_mode: Literal["basic", "enhanced"] = "basic"
```

对应 `.env` 中 `RAG_MODE=basic` 或 `RAG_MODE=enhanced`。

## 3. 具体实现流程

### Step 1：定义抽象基类

在 [app/retriever/base.py](app/retriever/base.py) 中定义 `BaseRAGRetriever(ABC)`，声明抽象方法 `retrieve(query, top_k) -> list[Document]`。所有 RAG 检索实现必须继承此类。

### Step 2：迁移现有逻辑到 BasicRAGRetriever

在 [app/retriever/basic.py](app/retriever/basic.py) 中实现 `BasicRAGRetriever`，将原有 `vector_store_manager.get_vector_store().as_retriever(search_kwargs={"k": top_k})` 调用封装进 `retrieve()` 方法，确保原有行为 100% 保留。

### Step 3：创建工厂函数

在 [app/retriever/factory.py](app/retriever/factory.py) 中实现 `get_rag_retriever()`，使用 `@lru_cache` 缓存实例，根据 `config.rag_mode` 返回 `BasicRAGRetriever` 或 `EnhancedRAGRetriever`。

### Step 4：重构 knowledge_tool

在 [app/tools/knowledge_tool.py](app/tools/knowledge_tool.py) 中，将直接调用 `vector_store_manager` 的逻辑替换为调用 `get_rag_retriever()` 工厂。保持 `@tool(response_format="content_and_artifact")` 签名不变，对上层 LangGraph Agent 完全透明。

### Step 5：添加配置项

在 [app/config.py](app/config.py) 中新增 `rag_mode: Literal["basic", "enhanced"]` 字段，默认值为 `"basic"`。在 [.env](.env) 中新增 `RAG_MODE=basic`。

## 4. 当前实现进度

### 已完成

- [x] 抽象基类 `BaseRAGRetriever` 定义完成
- [x] `BasicRAGRetriever` 实现完成，封装原有 Dense 检索逻辑
- [x] `EnhancedRAGRetriever` 框架就绪（具体增强逻辑由 Phase 2 实现）
- [x] 工厂函数 `get_rag_retriever()` 实现完成，带 `@lru_cache` 单例缓存
- [x] `retrieve_knowledge` 工具重构完成，通过工厂注入检索器
- [x] `rag_mode` 配置项添加完成（`Literal["basic", "enhanced"]`）
- [x] `RAG_MODE=basic` 时行为与重构前完全一致

### 尚未完成

无。此阶段已 100% 完成。

### 依赖其他模块的内容

- `EnhancedRAGRetriever` 的具体增强逻辑由后续阶段实现，但其接口框架已在本阶段定义

## 5. Evidence

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| 抽象基类 | [app/retriever/base.py](app/retriever/base.py) | `BaseRAGRetriever` 抽象类，定义 `retrieve()` 契约 |
| 基础实现 | [app/retriever/basic.py](app/retriever/basic.py) | `BasicRAGRetriever`，封装 Dense 检索 |
| 增强实现框架 | [app/retriever/enhanced.py](app/retriever/enhanced.py) | `EnhancedRAGRetriever`，Phase 2 增强检索 |
| 工厂函数 | [app/retriever/factory.py](app/retriever/factory.py) | `get_rag_retriever()` 带 `@lru_cache` 单例 |
| 工具重构 | [app/tools/knowledge_tool.py:13](app/tools/knowledge_tool.py#L13) | `retriever = get_rag_retriever()` 工厂注入 |
| 工具重构 | [app/tools/knowledge_tool.py:29](app/tools/knowledge_tool.py#L29) | `docs = retriever.retrieve(query, top_k=config.rag_top_k)` |
| 配置项 | [app/config.py:40](app/config.py#L40) | `rag_mode: Literal["basic", "enhanced"] = "basic"` |
| 环境变量 | [.env:21](.env#L21) | `RAG_MODE=basic` |
| 依赖声明 | [pyproject.toml:19](pyproject.toml#L19) | `pymilvus>=2.4.6` |
| Git 提交 | `0de5b65` | `feat: Phase 1 - 实现 RAG 可插拔检索接口` |
