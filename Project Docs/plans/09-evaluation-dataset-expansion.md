# 评估数据集与知识库文档扩充

## 1. 功能和目的

当前评估数据集仅有 5 个 aiops-docs 知识库文档和 25 个手工评估问题，规模偏小，可能无法充分反映检索系统在不同场景下的表现差异。本模块旨在扩充：

- **知识库文档数量**：从 5 个扩展到 10-15 个，覆盖更多运维场景
- **评估问题数量**：从 25 个扩展到 50-75 个（每文档 5 个问题）
- **问题多样性**：增加跨文档综合查询、边界情况、噪声查询等类型

该模块是 [08-rag-evaluation-enhancement.md](08-rag-evaluation-enhancement.md) 和 [06-ragas-evaluation.md](06-ragas-evaluation.md) 的数据基础——更丰富的评估数据集使所有评估指标（Hit Rate、MRR、RAGAs 四指标）更具统计意义。

## 2. 抽象实现思路

### 整体策略

采用"LLM 生成 + 人工审核"的半自动模式：

```
LLM 生成候选文档/问题
    ↓
人工审核、修正、标注 relevant_docs
    ↓
加入 aiops-docs/ 和 rag_testset.py
    ↓
重新入库到 Milvus（make upload）
    ↓
运行完整评估验证新数据质量
```

### 文档生成

为每个新文档提供标准 SOP 模板（参考现有 5 个文档的格式），让 LLM 按照模板生成内容：

```markdown
# {场景名称}

## 告警信息
- 告警名称: {AlertName}
- 严重级别: {severity}
- 触发条件: {condition}

## 排查步骤
1. {step_1}
2. {step_2}
...

## 常见原因
- {cause_1}
- {cause_2}
...

## 紧急处理
- {action_1}
- {action_2}
```

LLM 生成候选文档后，人工审核术语准确性、流程合理性，修正后加入 `aiops-docs/`。

### 问题生成

对每个文档，LLM 生成 5-10 个候选问题，覆盖以下类型：
- **精确匹配型**（如 "HighCPUUsage 告警的触发条件是什么？"）
- **口语化改写型**（如 "CPU 飙高怎么办？"）
- **跨文档综合型**（如 "哪些告警会导致服务不可用？"）
- **噪声/边界型**（如 "磁盘满了会影响 CPU 吗？"）

人工审核时标注 `relevant_docs`（哪些源文档与此问题相关）和 `ground_truths`（期望的答案要点）。

### 可选：RAGAs TestsetGenerator

RAGAs 提供了 `TestsetGenerator` 可从 LangChain Documents 自动生成评估问题：

```python
from ragas.testset.generator import TestsetGenerator
generator = TestsetGenerator.from_langchain(llm, embeddings)
testset = generator.generate_with_langchain_docs(docs, test_size=50)
```

但自动生成的问题质量依赖原始文档质量和 LLM 能力，建议作为初始种子而非最终数据集。生成后需要人工筛查去重和修正。

## 3. 具体实现流程

### Step 1：定义新文档场景

在现有 5 个文档（cpu、disk、memory、service_unavailable、slow_response）基础上，建议新增以下运维场景：

| 文件 | 场景 | 说明 |
|------|------|------|
| `network_high_latency.md` | 网络延迟过高 | 网络延迟排查、丢包检测 |
| `database_connection_pool_exhaustion.md` | 数据库连接池耗尽 | DB 连接池满、超时排查 |
| `container_oom_killed.md` | 容器 OOM 被杀 | K8s Pod OOMKilled 排查 |
| `api_error_rate_spike.md` | API 错误率飙升 | 接口 5xx 错误率突增 |
| `certificate_expiry.md` | 证书过期 | TLS 证书即将过期/已过期 |
| `message_queue_backlog.md` | 消息队列积压 | Kafka/RocketMQ 消费积压 |
| `cache_avalanche.md` | 缓存雪崩 | Redis 大量 key 同时过期 |

这不是固定列表——实际扩充时根据项目覆盖的运维领域调整。

### Step 2：LLM 生成文档内容

使用 `ChatQwen`（temperature=0.3，有一定多样性），传入现有 5 个文档作为 few-shot 示例，生成新文档。Prompt 结构：

```
你是一个运维专家。请参考以下示例文档的格式和风格，撰写一个新的运维排查 SOP 文档。

## 已有的参考文档示例
{existing_docs_as_few_shot}

## 新文档要求
场景：{scene_name}
告警严重级别：{severity}
风格：与示例文档保持一致，使用中文，包含具体的工具名称和命令
```

生成结果保存到 `aiops-docs/`，人工审核后提交。

### Step 3：LLM 生成评估问题

对每个文档（新老文档都生成），用 LLM 生成候选问题：

```
请为以下运维知识库文档生成 8 个不同角度和风格的检索评测问题：

## 文档内容
{doc_content}

## 问题类型要求
1-2: 精确关键词匹配型（如"X告警的触发条件"）
3-4: 口语化改写型（如"X出问题了怎么办"）
5-6: 操作步骤型（如"排查X的第一步是什么"）
7: 跨知识综合型（需要理解文档的多个部分）
8: 边界/噪声型（可能与文档部分相关但不完全匹配）

输出格式：每行一个问题，不要编号。
```

### Step 4：人工标注和验证

对每个生成的问题：
1. 标注 `relevant_docs`：哪些源文档与该问题相关
2. 标注 `ground_truths`：正确答案应包含的要点（3-5 个要点）
3. 过滤低质量问题：重复、过于简单、答案不在文档中的问题

### Step 5：集成到评估体系

将审核后的问题加入 [tests/evaluation/rag_testset.py](tests/evaluation/rag_testset.py) 的 `EVAL_DATASET`，确保 `question`、`ground_truths`、`relevant_docs` 三个字段完整。

将新文档通过 `make upload` 或 `POST /api/upload` 入库到 Milvus。

运行完整评估验证数据质量：
```bash
RAG_MODE=enhanced python -m tests.evaluation.evaluate_rag
```

## 4. 当前实现进度

### 已完成

- [x] 5 个 aiops-docs 基础文档（cpu, disk, memory, service_unavailable, slow_response）
- [x] 25 个手工评估问题 + `relevant_docs` 标注（08 工作中完成）
- [x] 问题类型分类（exact_keyword / colloquial / cross_doc / edge_case）

### 已完成（2026-05-22 实施）

- [x] **场景 Taxonomy 定义** — 5 大类 12 个场景: 资源告警(3)、服务可用性(2)、依赖故障(3)、链路异常(2)、容量/配置(2)
- [x] **工具能力盘点** — 9 个可用工具 (CLS + Monitor + Built-in) + 5 个不可用工具清单，作为文档生成的约束条件
- [x] **标注规范文档化** — `rag_testset.py` 文件头增加完整的标注规范: relevant_docs 判定标准、ground_truths 粒度、审核流程（生成→初审→复审→冻结）、质量控制规则
- [x] **LLM 文档生成脚本** — `tests/evaluation/generate_docs.py`，few-shot 模式，CLI 支持 --scene 筛选和 --dry-run 预览
- [x] **LLM 问题生成脚本** — `tests/evaluation/generate_questions.py`，4 种问题类型，CLI 支持 --doc 多文档和 --exclude 排除
- [x] **数据集质量检查脚本** — `tests/evaluation/validate_dataset.py`，4 项检查: embedding 去重 + 覆盖率分析 + 交叉引用验证 + 能力缺口报告
- [x] **`relevant_docs` 标注** — 已在 08 工作中完成，25 题全部标注
- [x] **新运维场景文档生成** — 7 篇新文档生成完毕（generate_docs.py），经 sub-agent 审查和人工修复后入库 Milvus
- [x] **评估问题扩充** — 34 题经 LLM 生成 + 审查 + 修复后导入 rag_testset.py，评估集从 25 题扩充到 59 题
- [x] **候选问题审查与导入工具** — `tests/evaluation/import_questions.py`，支持 --status 过滤、--dry-run 预览、自动版本号递增
- [x] **12 篇文档全部入库 Milvus** — make upload 完成，场景覆盖 12/12 (100%)
- [x] **`edge_case` 问题类型** — 从 0 题补到 6 题，占比 10.2%；但当前 `validate_testset()` 仍未将其纳入合法分类，属于“数据已扩充、校验尚未同步”的半完成状态

### 尚未实现

- [ ] RAGAs `TestsetGenerator` 集成 — 计划中，作为可选项（非主线）
- [ ] 数据集扩充后的回归基线建立 — 需运行完整评估（`evaluate_rag.py --with-generation`）并保存基线
- [ ] `api_error_rate_spike.md` 和 `message_queue_backlog.md` 各补 1 题达到 5 题底线（当前各 4 题）
- [ ] `cross_doc` 问题占比 3.4%（仅 2 题），偏低，建议补充到至少 10%
- [ ] `edge_case` 样本已写入评估集，但 `validate_testset()` 仍未放行该分类；在校验器同步更新前，完整评估会在启动阶段报错

### 依赖其他模块

- 回归基线依赖完整评估流程（需 Milvus + MCP + FastAPI 全部运行）
- 当前所有文档和问题已就位，可直接运行评估

## 5. Evidence

| 证据类型 | 位置 | 说明 |
|----------|------|------|
| 当前文档目录 | [aiops-docs/](aiops-docs/) | 12 个 .md 文件，场景覆盖 12/12 (100%) |
| 新增 7 篇文档 | `api_error_rate_spike.md`, `cache_avalanche.md`, `certificate_expiry.md`, `container_oom_killed.md`, `database_connection_pool_exhaustion.md`, `message_queue_backlog.md`, `network_high_latency.md` | 依赖故障(3) + 链路异常(2) + 容量/配置(2) |
| 当前评估集 | [tests/evaluation/rag_testset.py](tests/evaluation/rag_testset.py) | 59 条问答对，v1.1.1，含 `relevant_docs` + `category`；其中 `edge_case` 已进入数据集，但尚未被校验器接受 |
| 新增 34 题 | [tests/evaluation/rag_testset.py batch import](tests/evaluation/rag_testset.py#L442) | 覆盖全部 12 篇文档，exact=16, colloquial=35, cross_doc=2, edge_case=6；该分布目前与校验器的合法分类集合不一致 |
| 标注规范 | [tests/evaluation/rag_testset.py 文件头 docstring](tests/evaluation/rag_testset.py#L1-L100) | relevant_docs 判定标准、ground_truths 粒度、审核流程、质量控制 |
| 场景 Taxonomy | [tests/evaluation/generate_docs.py:60-130](tests/evaluation/generate_docs.py#L60) | `SCENARIO_TAXONOMY`: 5 大类 12 个场景，全部已覆盖 |
| 工具能力清单 | [tests/evaluation/generate_docs.py:135-155](tests/evaluation/generate_docs.py#L135) | `TOOL_CAPABILITIES`: 9 个可用 + 5 个不可用 |
| 文档生成脚本 | [tests/evaluation/generate_docs.py](tests/evaluation/generate_docs.py) | few-shot LLM 生成，7 篇文档已生成 + 修复 + 入库 |
| 问题生成脚本 | [tests/evaluation/generate_questions.py](tests/evaluation/generate_questions.py) | 4 种问题类型，支持 --doc 多文档和 --exclude 排除 |
| 候选问题导入脚本 | [tests/evaluation/import_questions.py](tests/evaluation/import_questions.py) | 审核后导入，支持 --status 过滤、--dry-run、自动版本递增 |
| 数据集质量检查 | [tests/evaluation/validate_dataset.py](tests/evaluation/validate_dataset.py) | 4 项检查: 去重 + 覆盖率 + 交叉引用 + 缺口报告 |
| 候选问题 JSON | `reports/candidate_questions_20260522_205717.json` | 47 题候选池，34 approved + 13 rejected |
| 数据契约 | [tests/evaluation/rag_testset.py:34-50](tests/evaluation/rag_testset.py#L34) | `EvalSample` dataclass: question, ground_truths, relevant_docs, category |
| 上传接口 | [app/api/file.py](app/api/file.py) | `POST /api/upload` 支持新文档入库 |

## 6. 设计问题与改进（状态: ✅ 已实施 2026-05-22）

### 6.1 扩充目标要先转成场景覆盖目标 ✅

当前文档把扩充目标写成”从 5 篇扩到 10-15 篇”，这对数量管理有帮助，但对质量控制还不够。更合理的方式，是先定义系统应该覆盖哪些运维场景，再决定每类场景需要多少文档和多少问题。

已实施方案:
- 在 `generate_docs.py` 中定义了 `SCENARIO_TAXONOMY`: 5 大类 12 个场景，每个场景标注了已覆盖/待生成状态。
- `validate_dataset.py` 的”检查 4: 能力缺口报告”会在每次运行 `python -m tests.evaluation.validate_dataset` 时自动对比 taxonomy 和实际文档，输出缺口清单。
- 每个场景的 pending 条目包含 scene_key、alert_name、severity、trigger、description，作为”最低可回答能力”的约束条件。
- 扩充以场景为单位: 先补充缺口场景的文档，再为其生成评估问题，最后运行完整评估验证。

### 6.2 文档生成要和现有工具能力对齐 ✅

如果新文档大量引入项目当前没有工具支持的诊断方法，评估会失真，因为模型即使知道答案，也无法通过真实链路验证出来。

已实施方案:
- 在 `generate_docs.py` 中定义了 `TOOL_CAPABILITIES`: available (9 个工具: CLS 日志查询 + Monitor 监控 + 内置工具) 和 unavailable (5 个: kubectl, DB 直连, Redis CLI, 网络诊断, SSH)。
- 文档生成的 System Prompt 中嵌入了可用/不可用工具清单，约束 LLM 只使用系统能支撑的工具。
- 对于依赖不可用工具的步骤（如 kubectl 操作容器），要求写入”## 未来能力”独立章节，不混入主排查步骤。
- 如果后续 MCP 工具扩展，只需更新 `TOOL_CAPABILITIES` 字典即可。

### 6.3 标注规范需要更加明确 ✅

`ground_truths` 和 `relevant_docs` 是评估集最关键的数据，但如果没有统一标注规则，后面很容易出现同义判断不一致、跨文档归属不一致的问题。

已实施方案:
- 在 `rag_testset.py` 文件头 docstring 中增加了完整的”标注规范（Annotation Guidelines）”章节，包含:
  - relevant_docs 判定标准: 直接相关（必填）、部分相关（选填）、背景相关（不标注），含判定原则和示例
  - ground_truths 粒度标准: 3-5 要点 × 20-60 字/条，覆盖”是什么→为什么→怎么办”，含好的示例和反例
  - 审核流程: 生成→初审→复审→冻结三阶段，冻结后修改需更新 DATASET_VERSION
  - 质量控制规则: embedding 去重（阈值 0.85）、覆盖率底线、回归基线
- `generate_questions.py` 的 prompt 中内嵌了上述粒度要求，确保 LLM 生成的候选问题基本达标。

### 6.4 扩充要考虑去重、冲突和回归成本 ✅

如果只追求规模，评估集会很快变成同义题堆叠，最后无法告诉我们系统是真的变好，还是只是题目变多了。

已实施方案:
- `validate_dataset.py` 的”检查 1: 去重检测”实现了基于 embedding 余弦相似度的语义重复检测（阈值可配置，默认 0.85），同时也有精确文本匹配的快速检测。
- “检查 2: 覆盖率分析”按文档和 category 维度统计问题分布，自动标记问题数不足 5 的文档和占比低于 10% 的 category。
- “检查 3: 交叉引用验证”确保 `relevant_docs` 引用的文件名在 `aiops-docs/` 中真实存在。
- “检查 4: 能力缺口报告”对比场景 taxonomy 和实际文档，量化覆盖率和缺失场景。
- 回归基线: 建议在每次数据集扩充后运行 `evaluate_rag.py` 并保存结果到 `reports/`，通过 `compare_reports.py` 对比前后差异。
- 目前还存在一个需要同步修正的分类口径问题：`edge_case` 已在数据集里使用，但 `validate_testset()` 仍只接受基础三类标签。只要这个不一致不修正，完整评估就会被启动校验拦住。
