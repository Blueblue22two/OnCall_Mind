# RAGAs 评估体系

## 1. 功能和目的

利用 RAGAs（Retrieval Augmented Generation Assessment）框架，对 Basic RAG 和 Enhanced RAG 两种检索模式进行量化对比评估，验证增强检索各阶段的实际收益（查询预处理、混合检索、精排）。

该模块解决的核心需求：
- 量化不同 RAG 策略的检索质量差异（`context_precision`、`context_recall`）
- 量化生成质量（`faithfulness`、`answer_relevancy`）
- 为后续调优（RRF k 值、coarse_top_k、reranker 模型选择）提供数据依据

与整体系统的关系：
- 依赖可插拔检索接口（`get_rag_retriever()` 工厂），通过 `RAG_MODE` 环境变量切换评估目标
- 与生产系统隔离，使用独立 `session_id="eval"`
- 评估结果指导 Enhanced RAG 各阶段的参数调优

## 2. 抽象实现思路

### 评估指标

| 指标 | 评估维度 | 需要的数据 | 优先级 |
|------|---------|-----------|--------|
| `context_precision` | 检索精准率 | query + contexts + ground_truth（reference） | 高 |
| `context_recall` | 检索召回率 | query + contexts + ground_truth（reference） | 高 |
| `faithfulness` | 生成忠实度 | query + answer + contexts | 中 |
| `answer_relevancy` | 回答相关性 | query + answer | 中 |

本项目重点关注检索质量（`context_precision` + `context_recall`），因为这是 Basic 和 Enhanced 两种 RAG 系统最直接的差异所在。

### 评估架构

```
tests/evaluation/
├── __init__.py
├── rag_testset.py           # 标准评估数据集（25 个问答对）
├── evaluate_rag.py          # 主评估脚本
└── compare_reports.py       # Basic vs Enhanced 对比报告
```

### 核心评估流程

```
加载 RAG Testset（25 个问答对）
  ↓
初始化 Retriever（通过 get_rag_retriever() 工厂）
  ↓
对每条测试数据：
  1. 调用 retriever.retrieve(question) → contexts
  2. 构建 RAGAs 评估数据 {question, contexts, ground_truth}
  ↓
构建 HuggingFace Dataset
  ↓
执行 RAGAs evaluate()（使用 ChatTongyi 作为 LLM Judge）
  ↓
输出评估分数 + JSON 报告
```

### LLM Judge 配置

使用项目现有的 `ChatTongyi`（DashScope）经 RAGAs 包装后作为评估裁判：

```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ragas_llm = LangchainLLMWrapper(ChatTongyi(model=config.rag_model))
ragas_embeddings = LangchainEmbeddingsWrapper(vector_embedding_service)
```

### 对比报告

`compare_reports.py` 加载 Basic 和 Enhanced 的评估结果，生成对比表格：

| 指标 | Basic RAG | Enhanced RAG | Delta |
|------|-----------|-------------|-------|
| context_precision | 0.72 | 0.85 | +0.13 |
| context_recall | 0.68 | 0.82 | +0.14 |
| faithfulness | 0.88 | 0.91 | +0.03 |
| answer_relevancy | 0.79 | 0.83 | +0.04 |

验证标准：Basic 基线 ≥ 0.70，Enhanced 在 context_precision 和 context_recall 上提升 ≥ +0.10。

## 3. 具体实现流程

### Step 1：构建评估数据集

文件：[tests/evaluation/rag_testset.py](tests/evaluation/rag_testset.py)

基于 `aiops-docs/` 中 5 个知识库文档（cpu_high_usage, disk_high_usage, memory_high_usage, service_unavailable, slow_response），手工构建了 25 个评估问题，每个文档 5 个问题。问题覆盖：
- 精确关键词查询（如 "HighCPUUsage 告警的触发条件"）
- 口语化查询（如 "CPU 占用太高了怎么办"）
- 跨文档综合查询

每个条目包含 `question`、`ground_truth`（参考答案）和 `reference_docs`（参考文档来源）。

### Step 2：实现评估脚本

文件：[tests/evaluation/evaluate_rag.py](tests/evaluation/evaluate_rag.py)

核心函数：
- `_build_rag_pipeline()`：通过工厂创建检索器
- `_retrieve_contexts()`：对每个问题检索上下文文档
- `_build_ragas_dataset()`：构建 HuggingFace Dataset（question + contexts + ground_truth）
- `_build_llm_wrapper()`：构建 RAGAs LLM 和 Embeddings 包装器
- `run_evaluation()`：主流程入口，执行评估并打印/导出结果

### Step 3：实现对比报告

文件：[tests/evaluation/compare_reports.py](tests/evaluation/compare_reports.py)

加载两个评估 JSON 报告，生成终端表格对比，并验证：
- Basic 基线 `context_precision` 和 `context_recall` ≥ 0.70
- Enhanced 相比 Basic 在 `context_precision` 和 `context_recall` 上提升 ≥ +0.10

### Step 4：运行评估

```bash
# 评估 Basic RAG
RAG_MODE=basic python -m tests.evaluation.evaluate_rag

# 评估 Enhanced RAG
RAG_MODE=enhanced python -m tests.evaluation.evaluate_rag

# 生成对比报告
python -m tests.evaluation.compare_reports
```

### 已解决：answer 字段缺失 ✅

~~当前评估数据构建链路中未生成 `answer` 字段~~（已于 2026-05-22 修复）。

通过 `--with-generation` 参数启用 Phase 2 后，脚本调用 `RagAgentService.query(session_id="eval_{i}")` 为每个问题异步生成 answer，然后独立评估 `faithfulness` 和 `answer_relevancy`。不传 `--with-generation` 时仅执行 Phase 1 检索评估，跳过生成指标。

## 4. 当前实现进度

### 已完成

- [x] 评估数据集构建（25 个问答对，覆盖 5 个知识库文档）
- [x] 评估脚本 `evaluate_rag.py` 实现完成
- [x] 支持通过 `RAG_MODE` 环境变量切换评估目标（basic / enhanced）
- [x] RAGAs LLM Judge 配置完成（`LangchainLLMWrapper(ChatTongyi(...))`）
- [x] RAGAs Embeddings 包装器配置完成
- [x] 上下文检索流程实现完成
- [x] HuggingFace Dataset 构建完成
- [x] 四项指标（context_precision, context_recall, faithfulness, answer_relevancy）配置
- [x] 对比报告 `compare_reports.py` 实现完成
- [x] Basic 基线验证标准和 Enhanced 提升阈值定义

### 设计改进（已完成 2026-05-22）

以下六项改进已实施，覆盖 answer 生成、数据契约、Judge 配置独立和逐题明细：

- [x] **6.1 检索/生成评估分阶段** — Phase 1 检索评估（context_precision + context_recall）始终执行；Phase 2 生成评估（faithfulness + answer_relevancy）通过 `--with-generation` 启用
- [x] **6.1 answer 自动生成** — `_generate_answers()` 调用 `RagAgentService.query()` 异步生成回答（`session_id="eval_{i}"`），单条失败不阻塞
- [x] **6.2 数据契约强化** — `EvalSample` dataclass 明确必填/可选字段；`validate_testset()` 评估前校验；`DATASET_VERSION` 版本号
- [x] **6.2 问题分类** — 每道题标注 `category`（`exact_keyword` / `colloquial` / `cross_doc`），当前评估集已额外引入 `edge_case` 样本用于边界测试，但校验器仍需同步放行，详见 `09-evaluation-dataset-expansion.md`
- [x] **6.3 Judge 配置独立** — `eval_judge_model` + `eval_judge_temperature` 与线上 RAG 模型解耦；输出附带 judge 元数据
- [x] **6.4 逐题明细 + 分类统计** — `per_question` 数组记录每题上下文数/answer 状态；`category_stats` 按分类聚合

### 部分完成

- [ ] plans.md 中提到的 `TestsetGenerator` 自动生成测试问题未实现（当前仅为手工构建）

### 尚未完成

- [ ] 评估流程与 CI/CD 集成
- [ ] Stage 拆分消融实验（改写收益/混合检索收益/精排收益），需独立消融脚本

### 依赖其他模块

- 依赖 `RagAgentService` 完成 answer 生成（用于 faithfulness 和 answer_relevancy 指标）
- 依赖 DashScope API（LLM Judge 调用大量消耗 API 配额）
- `ragas>=0.2.0` 和 `datasets>=2.0.0` 已在 [pyproject.toml](pyproject.toml) 中声明

## 5. Evidence

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| 评估数据集 | [tests/evaluation/rag_testset.py](tests/evaluation/rag_testset.py) | 当前 59 个 EvalSample，含 `category` 分类与 `edge_case` 边界样本 |
| 数据契约 | [tests/evaluation/rag_testset.py:30-53](tests/evaluation/rag_testset.py#L30) | `EvalSample` dataclass + `validate_testset()` |
| 数据集版本 | [tests/evaluation/rag_testset.py:26](tests/evaluation/rag_testset.py#L26) | `DATASET_VERSION = "1.0.0"` |
| 问题分类 | [tests/evaluation/rag_testset.py:73](tests/evaluation/rag_testset.py#L73) | category 字段: exact_keyword/colloquial/cross_doc/edge_case（其中校验器仍需与数据集同步） |
| 评估脚本 | [tests/evaluation/evaluate_rag.py](tests/evaluation/evaluate_rag.py) | `run_evaluation()` 两阶段主流程 |
| 两阶段拆分 | [tests/evaluation/evaluate_rag.py:189-191](tests/evaluation/evaluate_rag.py#L189) | Phase 1 检索评估 + Phase 2 生成评估（可选） |
| answer 生成 | [tests/evaluation/evaluate_rag.py:73-103](tests/evaluation/evaluate_rag.py#L73) | `_generate_answers()` 异步调用 RagAgentService |
| 上下文检索 | [tests/evaluation/evaluate_rag.py:44-50](tests/evaluation/evaluate_rag.py#L44) | `_retrieve_contexts()` |
| LLM Judge | [tests/evaluation/evaluate_rag.py:56-71](tests/evaluation/evaluate_rag.py#L56) | 使用 `eval_judge_model` + `eval_judge_temperature` |
| 逐题明细 | [tests/evaluation/evaluate_rag.py:208-215](tests/evaluation/evaluate_rag.py#L208) | `per_question` 数组记录上下文数/answer 状态 |
| 分类统计 | [tests/evaluation/evaluate_rag.py:106-126](tests/evaluation/evaluate_rag.py#L106) | `_compute_category_stats()` 按分类聚合 |
| 失败样本 | [tests/evaluation/evaluate_rag.py:230-235](tests/evaluation/evaluate_rag.py#L230) | `failed_samples` 记录 answer 为空的样本 |
| Judge 配置 | [app/config.py:65-67](app/config.py#L65) | `eval_judge_model` + `eval_judge_temperature` |
| 对比报告 | [tests/evaluation/compare_reports.py](tests/evaluation/compare_reports.py) | 兼容新格式，检索/生成指标分组 |
| 验证阈值 | [tests/evaluation/compare_reports.py:28-29](tests/evaluation/compare_reports.py#L28) | Basic ≥ 0.70, Enhanced Δ ≥ +0.10 |
| 依赖声明 | [pyproject.toml:37-38](pyproject.toml#L37) | `ragas>=0.2.0`, `datasets>=2.0.0` |
| Git 提交 | `7a5013f` | `feat: Phase 3 - 实现 RAGAs 评估体系` |

## 6. 设计问题与改进（状态：✅ 已实施 2026-05-22）

### 6.1 需要把检索评估和生成评估拆开 ✅

**原问题**：当前脚本把四项指标放在同一个流程里，但 answer 缺失导致 faithfulness 和 answer_relevancy 不稳定。

**已实施方案**：
- 评估流程拆为 Phase 1（检索评估）和 Phase 2（生成评估），通过 `--with-generation` 参数控制是否执行 Phase 2
- Phase 1 始终执行：构建 `retrieval_dataset`（question + contexts + ground_truth），评估 `context_precision` + `context_recall`
- Phase 2 可选执行：调用 `_generate_answers()` 通过 `RagAgentService.query()` 异步生成 answer，构建 `generation_dataset`（增加 answer 字段），评估 `faithfulness` + `answer_relevancy`
- Phase 2 失败不阻塞 Phase 1：如果所有 answer 生成均失败，跳过生成指标并明确标记 `generation_metrics: null`
- 输出 JSON 中分组为 `retrieval_metrics` 和 `generation_metrics`，避免误读

### 6.2 评估数据契约需要更严格 ✅

**原问题**：`rag_testset.py` 只有 `question` 和 `ground_truths`，缺少版本号和校验。

**已实施方案**：
- 定义 `EvalSample` dataclass，明确必填字段（`question`, `ground_truths`, `category`）和可选字段（`reference_docs`）
- 新增 `validate_testset()` 函数，评估前校验：样本非空、question 非空、ground_truths 非空、category 合法；当前校验器有效值仍需补入 `edge_case`，否则包含边界题的数据集会在启动前报错
- 校验失败直接 `sys.exit(1)` 并逐条报告错误
- 新增 `DATASET_VERSION = “1.0.0”`，修改测试集内容后递增
- `ground_truths` 拼接规则固定在 `get_eval_dataset()` 的 `”\n”.join()` 中，加注释说明

### 6.3 Judge 配置应当可替换且可复现 ✅

**原问题**：Judge 硬编码 `ChatTongyi(rag_model)`，与线上模型混用，无法追溯。

**已实施方案**：
- 在 `app/config.py` 中新增 `eval_judge_model`（默认 `”qwen-max”`）和 `eval_judge_temperature`（默认 `0.0`），与线上 RAG 模型解耦
- `_build_llm_wrapper()` 使用 `config.eval_judge_model` 而非 `config.rag_model`
- 输出 JSON 中新增 `judge` 元数据：`model`、`temperature`
- 新增 `failed_samples` 列表，记录 answer 为空或生成失败的样本（含 index、question、reason）
- 对比报告中展示双方的 Judge 配置信息

### 6.4 评估结果需要更适合调参 ✅

**原问题**：只输出整体分数，无逐题明细和分组统计。

**已实施方案**：
- 新增 `per_question` 数组，每条记录包含：`index`、`question`、`category`、`contexts_count`、`answer_generated`
- 每道题标注 `category`（`exact_keyword` / `colloquial` / `cross_doc`），并为边界样本单独使用 `edge_case` 标签；后续若要把它纳入正式统计，需同步更新校验与分组逻辑
- 新增 `_compute_category_stats()` 按分类聚合：各分类题目数、平均上下文数、answer 生成成功数
- 检索指标和生成指标在报告中分组展示，避免混淆
- 失败样本沉淀：`failed_samples` 记录所有 answer 为空的样本，可作为后续回归集

**未实施（需独立消融脚本）**：
- Stage 拆分收益（改写收益/混合检索收益/精排收益）：这需要多轮对比实验（none vs rewrite, dense-only vs hybrid, no-rerank vs rerank），属于实验方法论而非代码改动
