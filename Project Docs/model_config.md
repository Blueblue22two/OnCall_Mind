# 项目模型配置说明

本文档记录当前项目在数据集构造、RAG、回复生成、Multi-Agent、LLM-as-Judge 等流程中使用的模型，以及对应配置位置。

## 1. 模型使用总览

| 流程 | 当前模型 | 用途 | 配置位置 |
|---|---|---|---|
| 数据集文档生成 | `qwen-max` | 生成 AIOps SOP 文档候选 | `app/config.py` 的 `dashscope_model`，调用处为 `tests/evaluation/generate_docs.py` |
| 数据集问题生成 | `qwen-max` | 生成评估问题候选 | `app/config.py` 的 `dashscope_model`，调用处为 `tests/evaluation/generate_questions.py` |
| 数据集质量检查 | `text-embedding-v4` | 语义重复检测 | `tests/evaluation/validate_dataset.py` 中硬编码 |
| 文档向量化 / RAG 检索 | `text-embedding-v4` | 文档 embedding、query embedding | `app/config.py` 的 `dashscope_embedding_model`，调用处为 `app/services/vector_embedding_service.py` |
| RAG 回复生成 | `qwen-max` | 根据检索上下文生成回答 | `app/config.py` 的 `rag_model`，调用处为 `app/services/rag_agent_service.py` |
| Query Rewrite | `qwen-max` | Enhanced RAG 下可选查询改写 | `app/config.py` 的 `rag_model`，调用处为 `app/retriever/preprocessing/rewrite.py` |
| Enhanced RAG 精排 | `BAAI/bge-reranker-v2-m3` | Cross-Encoder rerank | `app/config.py` 的 `reranker_model`，调用处为 `app/retriever/reranker/cross_encoder.py` |
| RAGAs / LLM-as-Judge | `qwen3.5-plus` | RAG 评估打分 | `app/config.py` 的 `eval_judge_model`，调用处为 `tests/evaluation/evaluate_rag.py` |
| Agent Evaluation Judge | `qwen3.5-plus` | Agent 目标达成率评分 | `app/config.py` 的 `eval_judge_model`，调用处为 `tests/evaluation/evaluate_agent.py` |
| AIOps Planner / Executor / Replanner | `qwen-max` | Multi-Agent / Plan-Execute-Replan 推理 | `app/config.py` 的 `rag_model`，调用处为 `app/agent/aiops/*.py` |

## 2. 数据集构造阶段

### 2.1 SOP 文档生成

相关文件：

- `tests/evaluation/generate_docs.py`
- `app/config.py`

使用模型：

```python
model=config.dashscope_model
temperature=0.3
```

默认配置：

```python
dashscope_model: str = "qwen-max"
```

因此，生成知识库 SOP 文档候选时默认使用 `qwen-max`，温度固定为 `0.3`。

### 2.2 评估问题生成

相关文件：

- `tests/evaluation/generate_questions.py`
- `app/config.py`

使用模型：

```python
model=config.dashscope_model
temperature=0.4
```

默认模型同样是 `qwen-max`，温度固定为 `0.4`。

### 2.3 当前评估数据集本体

相关文件：

- `tests/evaluation/rag_testset.py`

该文件中的评估数据集是静态 Python 数据，不在运行时调用模型。模型只用于辅助生成候选文档或候选问题，最终数据仍需要人工审核后写入数据集文件。

### 2.4 数据集质量检查

相关文件：

- `tests/evaluation/validate_dataset.py`

语义重复检测使用：

```python
DashScopeEmbeddings(model="text-embedding-v4")
```

这里的 `text-embedding-v4` 是硬编码，没有读取 `app/config.py` 中的 `dashscope_embedding_model`。

## 3. RAG 流程

### 3.1 文档向量化与 Query Embedding

相关文件：

- `app/services/vector_embedding_service.py`
- `app/config.py`

使用模型：

```python
model=config.dashscope_embedding_model
```

默认配置：

```python
dashscope_embedding_model: str = "text-embedding-v4"
```

当前 embedding 维度固定为 `1024`。该 embedding 服务同时用于：

- 文档入库向量化
- RAG query embedding
- RAGAs embedding wrapper

### 3.2 Basic RAG

Basic RAG 当前主要使用：

- Embedding 模型：`text-embedding-v4`
- 回复生成模型：`qwen-max`

Basic 模式不使用 Cross-Encoder reranker。

相关配置：

```python
rag_mode: Literal["basic", "enhanced"] = "basic"
rag_top_k: int = 3
rag_model: str = "qwen-max"
```

### 3.3 Enhanced RAG

Enhanced RAG 涉及以下模型或检索组件：

| 环节 | 模型 / 组件 | 配置 |
|---|---|---|
| Dense 检索 | `text-embedding-v4` | `dashscope_embedding_model` |
| Sparse 检索 | Milvus BM25 | 无外部 LLM 模型 |
| Query Rewrite | `qwen-max` | `rag_model` |
| Cross-Encoder 精排 | `BAAI/bge-reranker-v2-m3` | `reranker_model` |

相关文件：

- `app/retriever/preprocessing/rewrite.py`
- `app/retriever/reranker/cross_encoder.py`
- `app/config.py`

默认配置：

```python
query_preprocessor_type = "none"
reranker_type = "cross_encoder"
reranker_model = "BAAI/bge-reranker-v2-m3"
rerank_coarse_top_k = 20
reranker_top_k = 3
```

注意：`query_preprocessor_type` 默认是 `none`，所以 query rewrite 默认不启用。

## 4. 回复生成阶段

相关文件：

- `app/services/rag_agent_service.py`
- `app/config.py`

使用模型：

```python
model=config.rag_model
temperature=0.7
```

默认配置：

```python
rag_model: str = "qwen-max"
```

因此，RAG 最终回答生成默认使用 `qwen-max`，温度固定为 `0.7`。

## 5. Multi-Agent / AIOps Agent 阶段

相关文件：

- `app/agent/aiops/planner.py`
- `app/agent/aiops/executor.py`
- `app/agent/aiops/replanner.py`
- `app/config.py`

使用模型：

```python
model=config.rag_model
temperature=0
```

默认模型仍是 `qwen-max`。

当前 Agent 的 planner、executor、replanner 与 RAG 回复生成共用 `rag_model`，但温度不同：

- RAG 回复生成：`temperature=0.7`
- Agent 规划 / 执行 / 重规划：`temperature=0`

## 6. LLM-as-Judge / RAGAs 评估

### 6.1 RAG 评估

相关文件：

- `tests/evaluation/evaluate_rag.py`
- `app/config.py`

Judge 模型：

```python
model=config.eval_judge_model
temperature=config.eval_judge_temperature
```

默认配置：

```python
eval_judge_model: str = "qwen3.5-plus"
eval_judge_temperature: float = 0.0
```

RAGAs 的 embeddings 使用 `vector_embedding_service`，因此 embedding 模型为 `text-embedding-v4`。

### 6.2 Agent 评估

相关文件：

- `tests/evaluation/evaluate_agent.py`
- `tests/evaluation/metrics/goal_accuracy.py`
- `app/config.py`

Judge 默认配置：

```python
eval_judge_model = "qwen3.5-plus"
eval_judge_temperature = 0.0
```

`tests/evaluation/evaluate_agent.py` 支持通过命令行覆盖 judge 模型：

```bash
--judge-model qwen-max
```

`tests/evaluation/metrics/goal_accuracy.py` 本身不创建模型，而是使用 `evaluate_agent.py` 传入的 judge LLM。

## 7. 主要配置入口

集中配置文件：

- `app/config.py`

关键字段：

```python
dashscope_model = "qwen-max"
dashscope_embedding_model = "text-embedding-v4"
rag_model = "qwen-max"
rag_mode = "basic"

query_preprocessor_type = "none"
reranker_type = "cross_encoder"
reranker_model = "BAAI/bge-reranker-v2-m3"

eval_judge_model = "qwen3.5-plus"
eval_judge_temperature = 0.0
```

这些字段可以通过 `.env` 覆盖，例如：

```bash
DASHSCOPE_MODEL=qwen-max
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
RAG_MODEL=qwen-max
RAG_MODE=enhanced
QUERY_PREPROCESSOR_TYPE=rewrite
RERANKER_TYPE=cross_encoder
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
EVAL_JUDGE_MODEL=qwen3.5-plus
EVAL_JUDGE_TEMPERATURE=0
```

## 8. 当前非集中配置点

以下模型或模型参数没有完全从 `app/config.py` 读取：

| 文件 | 非集中配置内容 |
|---|---|
| `tests/evaluation/validate_dataset.py` | embedding 模型硬编码为 `text-embedding-v4` |
| `tests/evaluation/generate_docs.py` | temperature 固定为 `0.3` |
| `tests/evaluation/generate_questions.py` | temperature 固定为 `0.4` |
| `app/services/rag_agent_service.py` | RAG 生成 temperature 固定为 `0.7` |
| `app/agent/aiops/planner.py` / `executor.py` / `replanner.py` | Agent 推理 temperature 固定为 `0` |
| `app/retriever/preprocessing/rewrite.py` | Query rewrite temperature 固定为 `0` |
| `app/services/vector_embedding_service.py` | embedding 维度固定为 `1024` |

## 9. 对后续对比实验的影响

开始对比实验时需要特别注意：

1. `rag_model` 同时影响 RAG 回复生成、Query Rewrite 和 AIOps Agent 推理。
2. `eval_judge_model` 是独立的 Judge 模型，不应在 RAG 检索对比实验中随意改变。
3. 建议固定 `eval_judge_model=qwen3.5-plus`，只改变 `RAG_MODE`、`QUERY_PREPROCESSOR_TYPE`、`RERANKER_TYPE` 等实验变量。
4. 如果需要比较不同生成模型，应单独设计实验组，并在结果中明确记录 `rag_model`。
5. 如果需要比较不同 embedding 模型，应同步确认 embedding 维度、Milvus collection schema、已有向量数据是否需要重建。

