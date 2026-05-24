# RAG 评估增强：多指标、多格式输出与消融实验

## 1. 功能和目的

在现有 RAGAs 评估体系基础上，补充以下能力：

- **多格式输出**：同时输出 JSON 和 CSV 格式的评估报告
- **可配置 Judge Model**：评估用的 LLM Judge 不再硬编码，支持用户自行选择模型和 API
- **新增检索指标**：Hit Rate@k（至少 1 个相关文档在 top-k 中的查询占比）和 MRR（平均倒数排名），不依赖 LLM Judge，纯数学计算
- **消融实验**：支持对比不同 `chunk_size`、`top_k`、`reranker_type` 组合下的评估结果，量化每个参数对检索质量的贡献

该模块与现有 [06-ragas-evaluation.md](06-ragas-evaluation.md) 中的评估体系是继承和扩展关系——在已有 4 个 RAGAs 指标基础上新增指标和实验能力，不替代现有评估逻辑。

## 2. 抽象实现思路

### 整体架构

```
tests/evaluation/
├── evaluate_rag.py          # 主评估脚本（增强：多格式输出 + 可配置 Judge + 新指标）
├── rag_testset.py           # 评估数据集（当前 59 条，含 relevant_docs / category / edge_case）
├── compare_reports.py       # 对比报告（已有）
├── run_ablation.py          # 新增：消融实验脚本
└── metrics/
    ├── hit_rate.py           # 新增：Hit Rate@k 计算
    └── mrr.py                # 新增：MRR 计算
```

### Hit Rate@k

$$HitRate@k = \frac{| \{q \in Q : \text{至少1个相关文档在top-k中} \} |}{|Q|}$$

不需要 LLM Judge。对每个 question，检查 `retriever.retrieve(question, top_k=k)` 返回的文档中是否包含至少一个 `relevant_docs` 中标注的文档。按 question 维度统计命中率。

### MRR (Mean Reciprocal Rank)

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

其中 `rank_i` 是第一个相关文档在检索结果中的排名位置（1-indexed）。如果没有相关文档被检索到，该 question 的贡献为 0。

### Judge Model 可配置

在 [app/config.py](app/config.py) 新增配置项：

```python
eval_judge_model: str = "qwen-max"           # Judge LLM 模型名
eval_judge_api_base: str = ""                # Judge API 地址（空则复用 DashScope）
eval_judge_api_key: str = ""                 # Judge API Key（空则复用 DashScope）
```

`_build_llm_wrapper()` 读取这些配置而非硬编码 `ChatTongyi`。

### 消融实验

`run_ablation.py` 的工作方式：
1. 定义参数网格（当前实现实际覆盖 `rag_mode`、`RAG_TOP_K`、`QUERY_PREPROCESSOR_TYPE`、`RERANKER_TYPE` 及其组合）
2. 对每个参数组合，通过子进程注入环境变量后调用 `evaluate_rag.py`
3. 汇总所有结果到 CSV + JSON

## 3. 具体实现流程

### Step 1：增强评估数据集

文件：[tests/evaluation/rag_testset.py](tests/evaluation/rag_testset.py)

为每个评估条目增加 `relevant_docs` 字段，标注哪些源文档与当前问题相关：

```python
{
    "question": "CPU使用率超过80%时应该如何排查？",
    "ground_truths": ["...", "..."],
    "relevant_docs": ["aiops-docs/cpu_high_usage.md"],  # 新增
}
```

这一步需要人工标注——对于当前 59 个问题，每个问题若干文档中哪些是相关的。这是 Hit Rate 和 MRR 计算的前提。

### Step 2：实现 Hit Rate 和 MRR 计算函数

新增 `tests/evaluation/metrics/hit_rate.py`：

```python
def compute_hit_rate(retrieved_docs: list[list[Document]],
                     relevant_docs_map: list[list[str]],
                     k: int) -> float:
    """计算 Hit Rate@k"""
    hits = 0
    for retrieved, relevant in zip(retrieved_docs, relevant_docs_map):
        retrieved_sources = {doc.metadata.get("_source", "") for doc in retrieved[:k]}
        if retrieved_sources & set(relevant):
            hits += 1
    return hits / len(retrieved_docs)
```

新增 `tests/evaluation/metrics/mrr.py`：

```python
def compute_mrr(retrieved_docs: list[list[Document]],
                relevant_docs_map: list[list[str]]) -> float:
    """计算 MRR"""
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_docs, relevant_docs_map):
        for rank, doc in enumerate(retrieved, start=1):
            if doc.metadata.get("_source", "") in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

### Step 3：增强 evaluate_rag.py

在 [tests/evaluation/evaluate_rag.py](tests/evaluation/evaluate_rag.py) 中：

1. `_build_llm_wrapper()` 改为读取 `config.eval_judge_model/api_base/api_key`，支持外部 Judge API
2. `run_evaluation()` 增加 CSV 输出：`scores.to_pandas().to_csv(csv_path)`
3. `run_evaluation()` 增加 `--output-format json,csv` CLI 参数
4. 在 RAGAs 评估完成后，额外计算 Hit Rate@k (k=3,5,10) 和 MRR，输出到同一份报告

### Step 4：实现消融实验脚本

新增 `tests/evaluation/run_ablation.py`：

```python
ABLATION_GRID = {
    "chunk_size": [400, 800, 1600],
    "top_k": [3, 5, 10],
    "reranker_type": ["none", "cross_encoder"],
}

def run_ablation(output_path: str):
    results = []
    for chunk_size, top_k, reranker_type in itertools.product(...):
        # 修改环境变量
        os.environ["CHUNK_MAX_SIZE"] = str(chunk_size)
        os.environ["RAG_TOP_K"] = str(top_k)
        os.environ["RERANKER_TYPE"] = reranker_type

        # 重新加载 config 和 retriever
        scores = run_evaluation(output_path=None)  # 返回内存结果
        results.append({
            "chunk_size": chunk_size,
            "top_k": top_k,
            "reranker_type": reranker_type,
            **scores,
        })

    # 输出汇总 CSV
    pd.DataFrame(results).to_csv(output_path)
```

注意：消融实验需要重新初始化 Milvus 连接和检索器（`@lru_cache` 需要失效），或通过子进程隔离每次实验。

### Step 5：添加配置项

在 [app/config.py](app/config.py) 中新增：

```python
# 评估配置
eval_judge_model: str = "qwen-max"
eval_judge_api_base: str = ""
eval_judge_api_key: str = ""
```

对应 `.env`：
```env
EVAL_JUDGE_MODEL=qwen-max
EVAL_JUDGE_API_BASE=
EVAL_JUDGE_API_KEY=
```

## 4. 当前实现进度

### 已完成

- [x] 4 个 RAGAs 指标（context_precision, context_recall, faithfulness, answer_relevancy）
- [x] JSON 格式输出
- [x] Basic vs Enhanced 对比报告

### 已完成（2026-05-22 实施）

- [x] **CSV 格式输出** — `--output-format json,csv,both` CLI 参数 + `_save_csv()` 函数
- [x] **Judge Model 可配置** — `eval_judge_api_base` + `eval_judge_api_key` 支持外部 Judge API
- [x] **Hit Rate@k 指标** — `tests/evaluation/metrics/hit_rate.py`，支持 k=3,5,10
- [x] **MRR 指标** — `tests/evaluation/metrics/mrr.py`，不依赖 LLM Judge
- [x] **`relevant_docs` 字段** — EvalSample 新增字段，25 道题已全部标注
- [x] **消融实验脚本** — `tests/evaluation/run_ablation.py`，子进程隔离，当前 10 个参数组合（围绕 `RAG_MODE` / `RAG_TOP_K` / `QUERY_PREPROCESSOR_TYPE` / `RERANKER_TYPE` 展开）
- [x] **`eval_judge_*` 配置项** — `eval_judge_model`, `eval_judge_temperature`, `eval_judge_api_base`, `eval_judge_api_key` 全部就位

### 尚未实现

- [ ] chunk_size 消融实验（需重新入库文档，不适合脚本自动化）
- [ ] 消融实验与 CI/CD 集成

## 5. Evidence

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| 评估脚本（增强） | [tests/evaluation/evaluate_rag.py](tests/evaluation/evaluate_rag.py) | 多格式输出 + Judge API + Hit Rate/MRR 集成 |
| CSV 输出 | [tests/evaluation/evaluate_rag.py:157-170](tests/evaluation/evaluate_rag.py#L157) | `_save_csv()` 平铺结果导出 |
| Judge API 配置 | [tests/evaluation/evaluate_rag.py:86-100](tests/evaluation/evaluate_rag.py#L86) | `_build_llm_wrapper()` 读取 `eval_judge_api_base/key` |
| 非 LLM 指标集成 | [tests/evaluation/evaluate_rag.py:285-294](tests/evaluation/evaluate_rag.py#L285) | Hit Rate@k + MRR 计算入口 |
| Hit Rate@k | [tests/evaluation/metrics/hit_rate.py](tests/evaluation/metrics/hit_rate.py) | `compute_hit_rate()` + `compute_hit_rate_multi_k()` |
| MRR | [tests/evaluation/metrics/mrr.py](tests/evaluation/metrics/mrr.py) | `compute_mrr()` |
| relevant_docs 标注 | [tests/evaluation/rag_testset.py:30-48](tests/evaluation/rag_testset.py#L30) | `EvalSample.relevant_docs` 字段，25 题已标注 |
| Judge 配置项 | [app/config.py:64-68](app/config.py#L64) | `eval_judge_model/temperature/api_base/api_key` |
| 消融实验脚本 | [tests/evaluation/run_ablation.py](tests/evaluation/run_ablation.py) | 子进程隔离，10 个参数组合 |
| 消融网格 | [tests/evaluation/run_ablation.py:46-120](tests/evaluation/run_ablation.py#L46) | `ABLATION_COMBINATIONS` |
| 对比报告 | [tests/evaluation/compare_reports.py](tests/evaluation/compare_reports.py) | 验证阈值：Basic ≥ 0.70, Enhanced Δ ≥ +0.10 |
