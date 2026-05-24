# BM25 增量更新策略（附录）

## 1. 功能和目的

本文档为 BM25 稀疏向量在文档增量更新场景下的技术方案探索。当知识库文档新增或删除时，BM25 的全局统计量（IDF、文档频率）会发生变化，导致已入库的稀疏向量与当前模型不一致。

**重要说明**：当前项目实际实现采用了 Milvus 2.5 内置的 `FunctionType.BM25`（方案四），BM25 统计由 Milvus 服务端自动维护，Python 侧无需处理 refit 问题。本文档中的方案一至三和方案五为备选设计参考，适用于无法使用 Milvus 内置 BM25 或使用 Python 侧 `BM25EmbeddingFunction` 的场景。

## 2. 抽象实现思路

### 问题根源

经典 BM25 的 IDF 计算依赖全局统计量：

$$\text{IDF}(t) = \log\left(\frac{N - df(t) + 0.5}{df(t) + 0.5}\right)$$

其中 `df(t)` 是包含词 `t` 的文档数，`N` 是语料总文档数。新增/删除任何文档都会改变这些值，导致已入库的稀疏向量与当前 BM25 模型不一致——必须重新 `fit()` + 重新 `encode_documents()` + 重新写入 Milvus。

### 项目特点

- **当前语料规模**：5 个 aiops-docs Markdown 文件，分割后约 20-50 个 chunk，总量极小
- **更新频率**：知识库文档属于低频更新（预计每月数次）
- **Milvus 已存 `content` 字段**：`biz_enhanced` collection 中每条记录已有完整文本，可直接从 Milvus 重建语料

### 五种方案概览

| 方案 | 适用阶段 | 推荐程度 | 关键优势 |
|------|----------|----------|----------|
| 方案一：全量重建（从 Milvus） | 当前（语料小） | ⭐⭐⭐⭐⭐ | 实现简单，利用现有 Milvus 数据，< 1s |
| 方案二：批量延迟 Refit | 语料增长后 | ⭐⭐⭐⭐ | 减少 refit 频率，配置灵活 |
| 方案三：增量 DF 近似 | 高频更新场景 | ⭐⭐ | 极低延迟，但实现复杂、有误差 |
| 方案四：Milvus 内置 BM25 | **当前实际采用** | ⭐⭐⭐⭐⭐ | 零维护成本，Milvus 自动管理 |
| 方案五：SPLADE 学习式稀疏编码 | 有 GPU 资源时 | ⭐⭐⭐ | 无 refit，但需额外模型 |

## 3. 具体实现流程

### 方案一：从 Milvus content 字段全量重建（计划中，尚未在项目中实现）

**核心思路**：每次有文档更新时，从 Milvus `biz_enhanced` collection 查询所有 `content` 字段，重建全量语料后再 refit BM25。

与 `VectorIndexService.index_single_file()` 的集成点：

```
删除旧 chunks（delete_by_source）
  → 插入新 chunks（dense + sparse 向量）
  → 调用 rebuild_bm25_from_milvus()  ← 新增步骤
  → 更新全局 bm25 单例
```

适用条件：语料 ≤ 10 万 chunk。当前项目语料极小，全量 refit 耗时 < 1 秒，可同步执行。

### 方案二：批量延迟 Refit 策略（计划中，尚未在项目中实现）

不在每次单文件更新时触发 refit，而是积累一批后统一处理：

- **子策略 A（N 次更新后触发）**：维护 `_dirty_count`，每 N 次更新后触发一次 refit
- **子策略 B（定时任务触发）**：使用 APScheduler 或 FastAPI lifespan 定时任务，每小时/每天执行一次
- **子策略 C（首次检索触发 / Lazy Refit）**：设置 `_is_dirty` 标志，在下次检索时检测并按需 refit

配置项建议：

```python
bm25_refit_strategy: Literal["immediate", "lazy", "scheduled", "manual"] = "immediate"
bm25_refit_batch_size: int = 5
```

### 方案三：增量 DF 统计近似更新（计划中，尚未在项目中实现）

维护 `df_cache`（词 → 文档频率映射）和 `N`（总文档数），文档增删时增量更新统计量：

- `add_document(tokens)`: N += 1, 更新每个 term 的 df
- `remove_document(tokens)`: N -= 1, 递减每个 term 的 df
- `idf(term)`: 使用增量更新的统计量计算

局限性：删除文档时需保存该文档的原始 token 列表；与 `BM25EmbeddingFunction` 生成的稀疏向量格式不完全兼容；长期运行后 IDF 值偏差会累积。

### 方案四：Milvus 2.5 内置全文检索（当前实际采用）

**已在项目中实现。** 通过 Milvus 2.5 的 `FunctionType.BM25`，文档插入时由 Milvus 服务端自动维护稀疏索引：

```python
# 实际实现在 app/core/milvus_client.py 中
bm25_function = Function(
    name="bm25",
    function_type=FunctionType.BM25,
    input_field_names=["content_text"],
    output_field_names=["sparse_vector"],
)
```

核心优势：
- 文档增删时 Milvus 自动更新 BM25 统计，Python 侧零维护
- 中文分词通过 `analyzer_params={"type": "chinese"}`（Jieba）配置
- 检索时直接传原始文本：`AnnSearchRequest(data=["查询文本"], param={"metric_type": "BM25"})`
- 无需序列化 BM25 模型文件

### 方案五：SPLADE 学习式稀疏编码（计划中，尚未在项目中实现）

使用基于 BERT 的 SPLADE 模型生成稀疏向量，每个文档/query 独立推理，不依赖语料统计：

- 推荐模型：`naver/efficient-splade-VI-BT-large-query` / `naver/splade-v3`
- 缺点：需要约 500MB+ 模型；推理延迟 > BM25；中文支持取决于模型预训练语料

## 4. 当前实现进度

### 已完成

- [x] 方案四（Milvus 内置 BM25）已在项目中完整实现，作为 `biz_enhanced` 集合的核心能力
- [x] Milvus BM25 Function 自动维护稀疏向量统计
- [x] 中文 Jieba 分词器配置完成

### 尚未完成（均为备选方案，当前无实现需求）

- [ ] 方案一（Python 侧全量重建）—— 计划中，尚未实现。由于方案四已消除 Python 侧 BM25 维护需求，此方案不再需要
- [ ] 方案二（批量延迟 Refit）—— 计划中，尚未实现。同上，不再需要
- [ ] 方案三（增量 DF 近似）—— 计划中，尚未实现。同上，不再需要
- [ ] 方案五（SPLADE）—— 计划中，尚未实现
- [ ] `bm25_refit_strategy` 配置项 —— 未添加（方案四不需要）

### 依赖其他模块

- 方案四依赖 Milvus 2.5+ 运行环境（已在 `vector-database.yml` 中配置）

## 5. Evidence

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| Milvus BM25 Function | [app/core/milvus_client.py:218](app/core/milvus_client.py#L218) | `FunctionType.BM25` 定义 |
| Jieba 分析器 | [app/core/milvus_client.py](app/core/milvus_client.py) | `analyzer_params={"type": "chinese"}` |
| Sparse 向量字段 | [app/core/milvus_client.py](app/core/milvus_client.py) | `SPARSE_FLOAT_VECTOR` 字段，BM25 自动填充 |
| Hybrid Search Sparse | [app/services/enhanced_vector_store_manager.py:157](app/services/enhanced_vector_store_manager.py#L157) | `AnnSearchRequest` Sparse 臂，`metric_type="BM25"` |
| pip 依赖 | [pyproject.toml:19](pyproject.toml#L19) | `pymilvus>=2.4.6` — 实际运行需 2.5+ |
| Docker Compose | [vector-database.yml](vector-database.yml) | Milvus standalone 镜像 |
| 方案一~三、五无实现 | `app/services/` | 未找到 `BM25Manager` 类或相关 refit 逻辑 |
| Git 提交 | `f1f48be` | `feat: Phase 2 - 实现 Enhanced RAG（双向量混合检索 + 可插拔精排）` |
