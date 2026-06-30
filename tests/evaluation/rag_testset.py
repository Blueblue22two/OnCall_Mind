"""RAG 评估数据集

手工构建 + LLM 辅助生成的 Q&A 数据集，用于 RAGAs 评估。
覆盖多篇 aiops-docs 文档，按场景 taxonomy 组织。

数据契约（EvalSample）：
  必填: question, ground_truths, relevant_docs
  可选: category, reference_docs

问题分类（category）：
  - exact_keyword : 精确关键词/技术术语查询
  - colloquial    : 口语化查询
  - cross_doc     : 跨文档综合查询
  - edge_case     : 边界/反事实/关联影响查询

relevant_docs 说明：
  存储与当前问题相关的源文档文件名列表（如 ["cpu_high_usage.md"]）。
  用于 Hit Rate@k 和 MRR 计算时，与检索结果的 doc.metadata["_file_name"] 做交集匹配。
  跨文档问题可包含多个文件名。

ground_truth 拼接规则：
  每个样本的 ground_truths 是 list[str]，在构建 Dataset 时通过 "\n" 连接为
  单个字符串。RAGAs 期望 ground_truth 为字符串形式。

-------------------------------------------------------------------------------
标注规范（Annotation Guidelines）
===============================================================================

1. relevant_docs 判定标准
...............................................................................

  直接相关（必填）:
    问题的答案直接来源于该文档。该文档包含了回答问题所需的核心信息。
    示例: Q="CPU飙高怎么排查" → relevant_docs=["cpu_high_usage.md"]

  部分相关（选填，建议标注）:
    问题涉及该文档的部分内容，但主要答案在其他文档中。
    示例: Q="哪些告警会导致服务不可用" → 可能涉及 cpu/disk/memory/service_unavailable
          此时标注所有涉及的文档

  背景相关（不标注）:
    文档提到了该概念，但不是答案的直接来源。避免过度标注导致检索
    评估失真。

  判定原则:
    - 优先标注答案的"直接来源"文档
    - 跨文档问题必须标注所有直接相关的文档（2-3 个）
    - 如有歧义，在 question 上方加注释说明判定理由

2. ground_truths 粒度标准
...............................................................................

  数量: 每个问题 3-5 个要点
  长度: 每个要点 1-2 句话（20-60 字）
  内容: 覆盖"是什么 → 为什么 → 怎么办"
    - 是什么: 问题的现象或原因
    - 为什么: 产生该现象的原因
    - 怎么办: 具体的处理步骤或命令

  示例（好的 ground_truth）:
    Q: "CPU飙高可能是代码里写了死循环吗？"
    ground_truths: [
      "可能是死循环或无限递归导致。这是常见原因之一。",
      "表现为某个线程CPU使用率持续在100%左右。",
      "应该使用 jstack 或 gdb 等工具抓取线程堆栈，分析处于 RUNNABLE 状态的线程",
    ]

  反例（太简单/太冗长）:
    BAD: ["会。"]                                          # 太短，无实质内容
    BAD: ["首先登录服务器，然后执行 top 命令..." (200字)]    # 太长，应拆成多个要点

3. 标注审核流程
...............................................................................

  生成阶段 (Generation):
    LLM 按文档和问题类型模板生成候选问题 + 初步 ground_truths + relevant_docs

  初审 (Initial Review):
    人工检查: 问题是否可答、relevant_docs 是否准确、ground_truths 是否达标
    标记: 通过 / 需修改 / 淘汰

  复审 (Re-review):
    抽查初审通过的问题，确认标注一致性（不同人标注同一问题的结果应接近）

  冻结 (Freeze):
    确认后加入 EVALUATION_DATASET，递增 DATASET_VERSION
    冻结后修改问题需更新版本号并在提交信息中说明变更原因

4. 质量控制规则
...............................................................................

  去重:
    - 相似问题（意思相同仅措辞不同）只保留一个
    - 使用 embedding 余弦相似度检测，阈值 > 0.85 的视为重复

  覆盖率:
    - 每个文档至少 5 个问题
    - 每个问题类型（exact_keyword/colloquial/cross_doc）至少占总数的 10%
    - 新增数据前先运行 validate_testset() 确认增量覆盖了什么能力缺口

  回归基线:
    - 每次扩充数据集后，运行完整评估并记录基线分数
    - 如果原有问题的 Hit Rate 下降 > 5%，检查是否新文档干扰了检索排序
"""

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from datasets import Dataset

# ---------------------------------------------------------------------------
# 数据集版本号 — 修改测试集内容后递增
# ---------------------------------------------------------------------------
DATASET_VERSION = "1.4.0"

# ---------------------------------------------------------------------------
# 数据集划分配置
# ---------------------------------------------------------------------------
DEFAULT_SPLIT_RATIOS = {
    "train": 0.6,
    "dev": 0.2,
    "test": 0.2,
}
SPLIT_SEED = 42


@dataclass
class EvalSample:
    """单条评估样本的数据契约

    Attributes:
        question:                 用户查询文本（必填）
        ground_truths:            期望参考答案要点列表（必填），3-5 个要点
        relevant_docs:            相关源文档文件名列表（必填），如 ["cpu_high_usage.md"]
        category:                 问题分类标签，用于分组统计
        reference_docs:           参考文档来源（可选）
        gen_expected_facts:       生成评估：答案必须包含的关键事实（可选，若空则 fallback 到 ground_truths）
        gen_forbidden_content:    生成评估：答案不应包含的内容（可选，用于幻觉检测）
        gen_min_length:           生成评估：最小期望答案长度（字符数），0=不检查
        relevant_sections:        相关章节标识，格式为 ``file.md::H2标题``
        fact_sources:             每条 ground truth 对应的章节标识列表
        split_hint:               可选固定 split；新增定向样本不得扰动冻结 test
    """

    question: str
    ground_truths: List[str]
    relevant_docs: List[str] = field(default_factory=list)
    category: str = "exact_keyword"
    reference_docs: List[str] = field(default_factory=list)
    gen_expected_facts: List[str] = field(default_factory=list)
    gen_forbidden_content: List[str] = field(default_factory=list)
    gen_min_length: int = 0
    relevant_sections: List[str] = field(default_factory=list)
    fact_sources: List[List[str]] = field(default_factory=list)
    split_hint: Optional[str] = None


def validate_testset(samples: List[EvalSample]) -> List[str]:
    """评估前校验数据集完整性和一致性

    Returns:
        校验错误信息列表，空列表表示通过
    """
    errors: List[str] = []

    if not samples:
        errors.append("数据集为空，至少需要一条样本")
        return errors

    valid_categories = {"exact_keyword", "colloquial", "cross_doc", "edge_case"}

    for i, s in enumerate(samples):
        prefix = f"[样本 {i}]"

        if not s.question or not s.question.strip():
            errors.append(f"{prefix} question 为空或仅含空白字符")

        if not s.ground_truths:
            errors.append(f"{prefix} ground_truths 为空列表")
        else:
            for j, gt in enumerate(s.ground_truths):
                if not gt or not gt.strip():
                    errors.append(f"{prefix} ground_truths[{j}] 为空或仅含空白字符")

        if not s.relevant_docs:
            errors.append(f"{prefix} relevant_docs 为空列表，需标注至少一个相关文档")

        if s.category not in valid_categories:
            errors.append(
                f"{prefix} category='{s.category}' 不在有效值 {valid_categories} 中"
            )

        if s.fact_sources and len(s.fact_sources) != len(s.ground_truths):
            errors.append(
                f"{prefix} fact_sources 数量({len(s.fact_sources)})必须等于 "
                f"ground_truths 数量({len(s.ground_truths)})"
            )
        if not s.fact_sources:
            errors.append(f"{prefix} fact_sources 为空，v1.4.0 要求每条事实可追溯")
        for section in s.relevant_sections:
            if "::" not in section:
                errors.append(f"{prefix} relevant_sections 格式错误: {section}")
        for sources in s.fact_sources:
            for section in sources:
                if "::" not in section:
                    errors.append(f"{prefix} fact_sources 格式错误: {section}")
        if s.split_hint not in {None, "train", "dev", "test"}:
            errors.append(f"{prefix} split_hint 无效: {s.split_hint}")

    return errors


def split_dataset(
    samples: Optional[List[EvalSample]] = None,
    ratios: Optional[Dict[str, float]] = None,
    seed: int = SPLIT_SEED,
    stratify_by: str = "category",
) -> Tuple[List[EvalSample], List[EvalSample], List[EvalSample]]:
    """将评估数据集划分为 train / dev / test 三组，支持分层采样。

    划分策略：
      - 按 category 分层，确保每个 split 中各类别比例与原始分布一致
      - 使用固定随机种子（SPLIT_SEED=42），保证划分可复现
      - 样本数不足时（如某个 category 仅 1 条），该样本优先放入 train

    Args:
        samples: 评估样本列表（默认：EVALUATION_DATASET）。
        ratios: 划分比例（默认：train=0.6, dev=0.2, test=0.2）。
        seed: 随机种子。
        stratify_by: 分层字段名（默认按 category）。

    Returns:
        (train_set, dev_set, test_set) 三个 EvalSample 列表。

    Example:
        >>> train, dev, test = split_dataset()
        >>> len(train), len(dev), len(test)
        (28, 9, 10)
        >>> # 消融实验使用 dev，最终报告使用 test
    """
    if samples is None:
        samples = EVALUATION_DATASET
    if ratios is None:
        ratios = DEFAULT_SPLIT_RATIOS

    total = len(samples)
    if total < 5:
        # 样本太少，不划分
        return list(samples), [], []

    rng = random.Random(seed)

    # 固定 split 的新增样本不参与随机划分，避免扰动已冻结 test membership。
    from collections import defaultdict

    groups: Dict[str, List[EvalSample]] = defaultdict(list)
    pinned: Dict[str, List[EvalSample]] = {"train": [], "dev": [], "test": []}
    for s in samples:
        if s.split_hint:
            pinned[s.split_hint].append(s)
            continue
        key = getattr(s, stratify_by, "unknown")
        groups[key].append(s)

    train_set: List[EvalSample] = []
    dev_set: List[EvalSample] = []
    test_set: List[EvalSample] = []

    for key, group in groups.items():
        rng.shuffle(group)
        n = len(group)

        # 计算每组的 train/dev/test 数量，至少 train 分 1 条
        n_train = max(1, round(n * ratios.get("train", 0.6)))
        n_dev = max(0, round(n * ratios.get("dev", 0.2)))
        n_test = n - n_train - n_dev
        if n_test < 0:
            n_dev = max(0, n - n_train)
            n_test = 0

        train_set.extend(group[:n_train])
        dev_set.extend(group[n_train : n_train + n_dev])
        test_set.extend(group[n_train + n_dev :])

    # 各组内 shuffle 一次，打乱分组聚集
    rng.shuffle(train_set)
    rng.shuffle(dev_set)
    rng.shuffle(test_set)

    train_set.extend(pinned["train"])
    dev_set.extend(pinned["dev"])
    test_set.extend(pinned["test"])

    return train_set, dev_set, test_set


def get_split_dataset(
    split: str = "test",
    samples: Optional[List[EvalSample]] = None,
) -> "Dataset":
    """获取指定划分的 HuggingFace Dataset。

    Args:
        split: "train" / "dev" / "test"。
        samples: 可选，自定义样本列表（默认使用 EVALUATION_DATASET 并自动划分）。

    Returns:
        datasets.Dataset
    """
    train, dev, test = split_dataset(samples)
    split_map = {"train": train, "dev": dev, "test": test}

    if split not in split_map:
        raise ValueError(f"split 必须是 'train'/'dev'/'test'，收到: {split}")

    target = split_map[split]
    if not target:
        raise ValueError(f"split='{split}' 为空，请检查数据集大小和划分比例")

    return _build_dataset_from_samples(target)


def _build_dataset_from_samples(samples: List[EvalSample]) -> "Dataset":
    """将 EvalSample 列表转换为 HuggingFace Dataset。"""
    from datasets import Dataset

    questions = []
    ground_truths = []
    categories = []
    relevant_docs_list = []
    gen_expected_facts_list = []
    gen_forbidden_content_list = []
    gen_min_length_list = []
    relevant_sections_list = []
    fact_sources_list = []

    for s in samples:
        questions.append(s.question)
        ground_truths.append("\n".join(s.ground_truths))
        categories.append(s.category)
        relevant_docs_list.append(s.relevant_docs)
        gen_expected_facts_list.append(
            s.gen_expected_facts if s.gen_expected_facts else s.ground_truths
        )
        gen_forbidden_content_list.append(s.gen_forbidden_content)
        gen_min_length_list.append(s.gen_min_length)
        relevant_sections_list.append(s.relevant_sections)
        fact_sources_list.append(s.fact_sources)

    return Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "category": categories,
        "relevant_docs": relevant_docs_list,
        "gen_expected_facts": gen_expected_facts_list,
        "gen_forbidden_content": gen_forbidden_content_list,
        "gen_min_length": gen_min_length_list,
        "relevant_sections": relevant_sections_list,
        "fact_sources": fact_sources_list,
    })


# ---------------------------------------------------------------------------
# 评估数据集
# ---------------------------------------------------------------------------

EVALUATION_DATASET: List[EvalSample] = [
    # ---------------------------------------------
    # CPU 使用率过高告警处理方案 (cpu_high_usage.md)
    # ---------------------------------------------
    EvalSample(
        question="CPU 告警后，我要怎么查是哪个进程在吃 CPU？",
        ground_truths=[
            "使用 top -c 命令按 CPU 使用率排序",
            "使用 ps -eo pid,ppid,cmd,%cpu,%mem --sort=-%cpu | head -10 获取 Top 10 CPU 进程",
            "使用 pidstat 1 5 获取进程的实时 CPU 统计信息",
        ],
        relevant_docs=["cpu_high_usage.md"],
        category="colloquial",
        gen_expected_facts=[
            "使用 top -c 按 CPU 使用率排序查看进程",
            "使用 ps 命令获取 Top 10 CPU 进程",
            "使用 pidstat 获取进程实时 CPU 统计",
        ],
    ),
    EvalSample(
        question="CPU飙高可能是代码里写了死循环吗？",
        ground_truths=[
            "可能是死循环或无限递归导致。这是常见原因之一。",
            "表现为某个线程CPU使用率持续在100%左右。",
            "应该使用 jstack 或 gdb 等工具抓取线程堆栈，分析处于 RUNNABLE 状态的线程",
        ],
        relevant_docs=["cpu_high_usage.md"],
        category="colloquial",
        gen_expected_facts=[
            'CPU 飙高可能是由代码中的死循环或无限递归导致的，这是常见原因之一。',
            '死循环导致的 CPU 飙高通常表现为某个线程的 CPU 使用率持续在 100% 左右。',
            '可以使用 jstack 或 gdb 工具抓取线程堆栈来定位高 CPU 占用问题。',
            '定位时需要分析处于 RUNNABLE 状态的线程以确认是否存在死循环。',
        ],
    ),
    EvalSample(
        question="遇到CPU100%怎么紧急处理？限流还是扩容？",
        ground_truths=[
            "如果是流量突增导致，如果影响核心链路且无法自动扩容，应立即开启限流降级",
            "同时申请紧急扩容增加实例数",
            "如果是死循环导致的，且影响核心链路，应当立刻重启相关实例",
        ],
        relevant_docs=["cpu_high_usage.md"],
        category="colloquial",
        gen_expected_facts=[
            '流量突增导致 CPU100% 且影响核心链路时，若无法自动扩容应立即开启限流降级。',
            '死循环导致 CPU100% 且影响核心链路时，应当立刻重启相关实例。',
            '处理 CPU100% 紧急故障时，应同时申请紧急扩容以增加实例数。',
            'CPU100% 紧急处理策略需根据原因是流量突增还是死循环来决定。',
        ],
    ),
    EvalSample(
        question="数据库查询慢会拖慢应用服务器的CPU吗？",
        ground_truths=[
            "会。数据库查询慢会导致应用层大量线程阻塞，上下文切换频繁，从而导致 CPU 使用率升高。",
            "处理方法是通知 DBA 排查慢查询，或在应用侧紧急降级非核心查询",
        ],
        relevant_docs=["cpu_high_usage.md"],
        category="exact_keyword",
        gen_expected_facts=[
            "数据库查询慢会导致应用线程阻塞和上下文切换频繁",
            "上下文切换频繁会使 CPU 使用率升高",
            "应通知 DBA 排查慢查询",
            "可在应用侧紧急降级非核心查询",
        ],
    ),
    EvalSample(
        question="排查 CPU 问题时怎么看应用日志有没有报错？",
        ground_truths=[
            "在应用日志中搜索 ERROR 或 Exception 关键字",
            "特别关注 OutOfMemoryError、TimeoutException 等异常",
            "还要关注是否存在大量重复的错误日志",
        ],
        relevant_docs=["cpu_high_usage.md"],
        category="colloquial",
        gen_expected_facts=[
            '应用日志中包含 ERROR 或 Exception 关键字通常表示存在报错信息。',
            'OutOfMemoryError 是排查 CPU 问题时需要重点关注的内存相关异常类型。',
            'TimeoutException 是排查 CPU 问题时需要重点关注的超时相关异常类型。',
            '存在大量重复的错误日志是应用日志报错的重要特征之一。',
        ],
    ),

    # ---------------------------------------------
    # 磁盘使用率过高告警处理方案 (disk_high_usage.md)
    # ---------------------------------------------
    EvalSample(
        question="怎么看哪个文件夹把磁盘占满了？",
        ground_truths=[
            "执行 df -h 查看各分区使用情况，找出使用率超过告警阈值的磁盘分区",
            "进入目标分区，使用 du -sh * | sort -hr 找出占用空间最大的目录",
            "可以使用 find /path -type f -size +500M 查找大于500MB的大文件",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="colloquial",
        gen_expected_facts=[
            "使用 df -h 查看各分区使用情况",
            "使用 du -sh * | sort -hr 找出占用空间最大的目录",
            "使用 find 命令查找大于 500MB 的大文件",
        ],
    ),
    EvalSample(
        question="日志把磁盘写满了，可以直接清空吗？怎么操作？",
        ground_truths=[
            "不要直接使用 rm 删除正在被程序写入的日志文件",
            "应使用 echo '' > application.log 或 > application.log 清空文件内容，以保留文件句柄",
            "可以直接删除 N 天前的旧日志文件：find /var/log -type f -mtime +7 -name '*.log' -delete",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="colloquial",
        gen_expected_facts=[
            '正在被程序写入的日志文件不能直接使用 rm 命令删除，否则会导致磁盘空间无法立即释放。',
            "清空正在写入的日志文件应使用重定向方式如 echo '' > file，以确保保留文件句柄不被释放。",
            '历史旧日志文件可以直接删除，find 命令支持通过 -mtime 参数按修改时间筛选文件。',
            '清理旧日志的典型策略是删除特定天数前的文件，例如超过 7 天的日志文件。',
        ],
    ),
    EvalSample(
        question="磁盘高是不是因为 Docker 镜像太多？怎么清理？",
        ground_truths=[
            "是的，Docker镜像和无用容器积累会占用大量磁盘空间。",
            "如果是 Docker 占用，可以执行 docker system prune -a --volumes 清理无用数据",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="colloquial",
        gen_expected_facts=[
            'Docker 镜像和无用容器的积累会占用大量磁盘空间。',
            '清理操作主要针对的是 Docker 系统中的无用数据。',
            'docker system prune -a --volumes 是清理 Docker 无用数据的命令。',
            '该命令中的 --volumes 参数表示清理范围包含数据卷。',
            '执行清理操作的前提是确认磁盘占用由 Docker 引起。',
        ],
    ),
    EvalSample(
        question="磁盘告警紧急处理的30分钟内措施是什么？",
        ground_truths=[
            "压缩旧的大文件：gzip old_file.log",
            "排查并清理临时文件目录 /tmp 或 /var/tmp",
            "如果系统存在大文件传输或导入，暂停这些非紧急批处理任务",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="exact_keyword",
        gen_expected_facts=[
            "压缩旧的大文件，使用 gzip 命令",
            "排查并清理 /tmp 或 /var/tmp 临时文件目录",
            "暂停非紧急的大文件传输或导入批处理任务",
        ],
    ),
    EvalSample(
        question="如果磁盘 inode 被占满了怎么排查？",
        ground_truths=[
            "使用 df -i 命令查看 inode 的使用情况",
            "如果 inode 满了（即使磁盘空间还有剩余），通常是因为有大量小文件",
            "需要查找包含大量小文件的目录并清理",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="exact_keyword",
        gen_expected_facts=[
            "使用 df -i 命令查看 inode 使用情况",
            "inode 耗尽通常是因为大量小文件",
            "即使磁盘空间有剩余，inode 满也会导致无法创建新文件",
            "需要查找并清理包含大量小文件的目录",
        ],
    ),

    # ---------------------------------------------
    # 内存使用率过高告警处理方案 (memory_high_usage.md)
    # ---------------------------------------------
    EvalSample(
        question="内存满了怎么抓堆栈分析？是打 dump 吗？",
        ground_truths=[
            "是的，使用 jmap -dump:live,format=b,file=heap.bin <PID> 生成堆转储文件",
            "然后使用 MAT (Memory Analyzer Tool) 或 JProfiler 进行离线分析",
            "通过分析大对象或实例数最多的类来定位内存泄漏",
        ],
        relevant_docs=["memory_high_usage.md"],
        category="colloquial",
        gen_expected_facts=[
            "使用 jmap -dump:live,format=b,file=heap.bin <PID> 生成堆转储文件",
            "使用 MAT 或 JProfiler 进行离线分析",
            "通过分析大对象或实例数最多的类定位内存泄漏",
        ],
    ),
    EvalSample(
        question="OOM是不是因为缓存配置不对？",
        ground_truths=[
            "可能是缓存配置不当导致的。",
            "如果本地缓存（如 Guava Cache、Caffeine）未设置合理的过期时间或最大容量，会导致缓存无限增长",
            "或者一次性从数据库加载了过大的缓存预热数据",
        ],
        relevant_docs=["memory_high_usage.md"],
        category="exact_keyword",
        gen_expected_facts=[
            "缓存配置不当是 OOM 的可能原因之一",
            "本地缓存（Guava Cache、Caffeine）未设置过期时间或最大容量会导致内存无限增长",
            "一次性加载过大的缓存预热数据也会导致 OOM",
        ],
    ),
    EvalSample(
        question="大文件处理会导致内存高吗？",
        ground_truths=[
            "是的，一次性读取过大的文件（如 CSV、Excel）到内存中会导致对象激增",
            "或者在内存中进行了大批量的集合操作",
            "建议优化为流式处理或分页处理",
        ],
        relevant_docs=["memory_high_usage.md"],
        category="colloquial",
        gen_expected_facts=[
            '大文件处理会导致内存高，主要原因是一次性读取过大文件导致对象激增。',
            '在内存中进行大批量的集合操作也是导致内存占用过高的原因之一。',
            '流式处理或分页处理是优化大文件处理内存问题的可行方案。',
        ],
    ),
    EvalSample(
        question="内存告警 5 分钟内我该做啥操作？",
        ground_truths=[
            "如果影响核心链路，应当立刻隔离异常节点（摘除流量），避免雪崩",
            "在节点挂掉或重启前，务必抓取现场：执行 jstat 或保留 OOM 时自动生成的 heap dump",
            "如果是严重内存泄漏且无法快速修复，执行应用重启",
        ],
        relevant_docs=["memory_high_usage.md"],
        category="colloquial",
        gen_expected_facts=[
            '若内存告警影响核心链路，应当立刻隔离异常节点并摘除流量，以避免雪崩效应。',
            '节点挂掉或重启前抓取现场是必要步骤，支持通过 jstat 或 OOM 时自动生成的 heap dump 实现。',
            '针对严重内存泄漏且无法快速修复的场景，应采取执行应用重启的应对措施。',
        ],
    ),
    EvalSample(
        question="怎么看 JVM 的内存使用详情？",
        ground_truths=[
            "使用 jstat -gcutil <PID> 1000 查看实时 GC 和堆内存使用比例",
            "使用 jmap -heap <PID> 查看堆内存的配置和使用详情",
            "如果频繁出现 Full GC，说明老年代内存不足或存在内存泄漏",
        ],
        relevant_docs=["memory_high_usage.md"],
        category="exact_keyword",
        gen_expected_facts=[
            "使用 jstat -gcutil <PID> 1000 查看实时 GC 和堆内存使用比例",
            "使用 jmap -heap <PID> 查看堆内存配置和使用详情",
            "频繁 Full GC 说明老年代内存不足或存在内存泄漏",
        ],
    ),

    # ---------------------------------------------
    # 服务不可用告警处理方案 (service_unavailable.md)
    # ---------------------------------------------
    EvalSample(
        question="接口不通了，服务挂了怎么看进程还在不在？",
        ground_truths=[
            "使用 ps -ef | grep <应用名> 检查服务进程是否存在",
            "使用 netstat -tlnp | grep <端口号> 检查监听端口是否正常",
            "如果进程存在但端口不通，可能是线程池耗尽或死锁",
        ],
        relevant_docs=["service_unavailable.md"],
        category="colloquial",
        gen_expected_facts=[
            "使用 ps -ef | grep 检查服务进程是否存在",
            "使用 netstat -tlnp | grep 检查监听端口是否正常",
            "进程存在但端口不通可能是线程池耗尽或死锁",
        ],
    ),
    EvalSample(
        question="服务不可用可能是因为数据库连不上吗？",
        ground_truths=[
            "是的。数据库连接失败、连接池满、数据库本身宕机等都会导致服务不可用。",
            "表现为大量请求阻塞，应用日志中出现 SQLTimeoutException 或 Connection refused 错误",
        ],
        relevant_docs=["service_unavailable.md"],
        category="colloquial",
        gen_expected_facts=[
            '数据库连接问题是导致服务不可用的常见原因之一。',
            '具体原因包括数据库连接失败、连接池满或数据库本身宕机。',
            '数据库故障导致的服务不可用通常表现为大量请求阻塞。',
            '应用日志中会出现 SQLTimeoutException 或 Connection refused 错误信息。',
        ],
    ),
    EvalSample(
        question="出现服务不可用告警，1 分钟内必须要干嘛？",
        ground_truths=[
            "确认告警真实性，访问服务的健康检查接口 (/health)",
            "如果有备用集群或跨机房容灾，立即通知运维进行流量切换",
            "如果是新版本发布导致的，立即执行版本回滚",
        ],
        relevant_docs=["service_unavailable.md"],
        category="exact_keyword",
        gen_expected_facts=[
            '访问健康检查接口 (/health) 是确认服务不可用告警真实性的必要步骤。',
            '存在备用集群或跨机房容灾时，流量切换是处理服务不可用的标准流程。',
            '针对新版本发布导致的服务不可用，版本回滚是必须执行的恢复措施。',
        ],
    ),
    EvalSample(
        question="怎么确认是不是外部依赖服务挂了导致的不可用？",
        ground_truths=[
            "检查调用链监控或日志，看是否有大量调用外部依赖超时的报错",
            "如果有熔断器（如 Sentinel, Hystrix），检查是否已经触发熔断",
            "手动 curl 或 ping 外部依赖服务的端点进行连通性测试",
        ],
        relevant_docs=["service_unavailable.md"],
        category="colloquial",
        gen_expected_facts=[
            '调用链监控或日志中出现大量外部依赖调用超时报错是服务异常的关键迹象。',
            '熔断器组件如 Sentinel 或 Hystrix 触发熔断表明外部依赖服务可能已不可用。',
            '外部依赖服务的端点连通性状态可以通过 curl 或 ping 命令进行测试验证。',
        ],
    ),
    EvalSample(
        question="服务不可用事件结束后需要复盘吗？",
        ground_truths=[
            "必须复盘。在故障恢复后 24 小时内组织相关人员复盘",
            "分析根本原因，输出故障报告",
            "制定改进项（如加强监控、优化重试策略等）并录入系统跟踪解决",
        ],
        relevant_docs=["service_unavailable.md"],
        category="cross_doc",
        gen_expected_facts=[
            '服务不可用事件结束后必须进行复盘，这是故障恢复后的必要环节。',
            '复盘会议需要在故障恢复后的 24 小时内组织相关人员开展。',
            '复盘过程中需要分析根本原因并输出正式的故障报告。',
            '必须制定改进项并录入系统进行跟踪解决，以实现问题闭环管理。',
        ],
    ),

    # ---------------------------------------------
    # 服务响应时间过长告警处理方案 (slow_response.md)
    # ---------------------------------------------
    EvalSample(
        question="RT 升高、接口慢，怎么查是不是慢SQL导致的？",
        ground_truths=[
            "登录数据库管理平台或监控大盘，查看慢 SQL 统计",
            "或者在应用日志中搜索 Slow query 或耗时较长的 SQL 记录",
            "对疑似慢 SQL 执行 EXPLAIN 查看执行计划，确认是否命中索引或发生了全表扫描",
        ],
        relevant_docs=["slow_response.md"],
        category="colloquial",
        gen_expected_facts=[
            "登录数据库管理平台或监控大盘查看慢 SQL 统计",
            "在应用日志中搜索 Slow query 或耗时较长的 SQL 记录",
            "对疑似慢 SQL 执行 EXPLAIN 查看执行计划",
            "确认是否命中索引或发生了全表扫描",
        ],
    ),
    EvalSample(
        question="缓存击穿会导致接口变慢吗？",
        ground_truths=[
            "会。缓存穿透、击穿或雪崩会导致大量请求直接打到数据库",
            "数据库负载急剧上升，从而导致整体响应变慢",
            "监控表现为缓存命中率断崖式下跌，数据库 QPS 异常突增",
        ],
        relevant_docs=["slow_response.md"],
        category="exact_keyword",
        gen_expected_facts=[
            '缓存击穿会导致大量请求绕过缓存层直接访问数据库。',
            '数据库负载急剧上升会导致整体接口响应时间显著变慢。',
            '故障期间监控指标表现为缓存命中率出现断崖式下跌。',
            '故障期间监控指标表现为数据库 QPS 出现异常突增。',
        ],
    ),
    EvalSample(
        question="接口RT高，代码可能有啥问题？",
        ground_truths=[
            "代码中可能存在复杂的循环计算",
            "频繁的同步 IO 操作（如文件读写）",
            "或者在循环中调用外部 RPC 或查询数据库（N+1 查询问题）",
        ],
        relevant_docs=["slow_response.md"],
        category="colloquial",
        gen_expected_facts=[
            '接口 RT 高可能是由于代码逻辑中存在复杂的循环计算导致处理耗时增加。',
            '代码中执行频繁的同步 IO 操作，例如文件读写，会阻塞线程导致接口响应变慢。',
            '在循环结构中调用外部 RPC 或查询数据库会引发 N+1 查询问题，从而增加接口 RT。',
        ],
    ),
    EvalSample(
        question="响应慢的问题，30 分钟内怎么处理？",
        ground_truths=[
            "如果是慢 SQL，紧急给涉及的表添加缺失的索引",
            "如果是下游依赖慢，且非核心链路，紧急开启或调低降级开关，熔断弱依赖",
            "如果是缓存失效，修复缓存逻辑并进行缓存预热",
        ],
        relevant_docs=["slow_response.md"],
        category="exact_keyword",
        gen_expected_facts=[
            '针对慢 SQL 导致的响应慢问题，紧急处理措施包括给涉及的表添加缺失的索引。',
            '针对下游依赖慢且非核心链路问题，紧急措施包括开启降级开关或熔断弱依赖。',
            '针对缓存失效导致的响应慢问题，紧急处理措施包括修复缓存逻辑并进行缓存预热。',
        ],
    ),
    EvalSample(
        question="怎样预防接口变慢？",
        ground_truths=[
            "梳理所有外部依赖，配置合理的超时时间和重试机制",
            "对核心接口实施严格的限流和降级策略",
            "所有上线的新 SQL 必须经过 DBA 审核，确保执行计划正确",
        ],
        relevant_docs=["slow_response.md"],
        category="cross_doc",
        gen_expected_facts=[
            '预防接口变慢需要梳理所有外部依赖，并配置合理的超时时间和重试机制。',
            '核心接口需要实施严格的限流和降级策略以防止性能下降。',
            '所有上线的新 SQL 必须经过 DBA 审核，确保执行计划正确。',
        ],
    ),
    EvalSample(
        question="CPU 和内存同时告警时，怎么判断是代码问题还是流量突增？",
        ground_truths=[
            "CPU 文档建议检查是否存在单进程接近100%的死循环、重复错误堆栈，或多个进程随流量均匀升高",
            "内存文档建议观察内存是否持续缓慢上升、Full GC 后无法释放，或是否随请求量突然升高",
            "如果是代码问题，应保留日志和堆转储后回滚或修复；如果是流量突增，应扩容并启用限流保护",
        ],
        relevant_docs=["cpu_high_usage.md", "memory_high_usage.md"],
        category="cross_doc",
        gen_expected_facts=[
            "代码问题特征：单进程 CPU 接近 100%（死循环）、内存持续缓慢上升且 Full GC 后无法释放",
            "流量突增特征：多个进程 CPU 随流量均匀升高、内存随请求量突然升高",
            "代码问题应保留日志和堆转储后回滚或修复",
            "流量突增应扩容并启用限流保护",
        ],
    ),
    EvalSample(
        question="服务不可用同时 API 5xx 飙升，第一轮排查怎么做？",
        ground_truths=[
            "服务不可用文档要求先确认健康检查失败或错误率超过50%，查询最近15分钟 ERROR/FATAL/status:500 日志和系统事件",
            "API 错误率文档要求按最近30分钟检索 api-gateway-logs 中 level:ERROR 或 status:5xx，并分析错误类型、路径和堆栈",
            "两类文档都强调检查依赖服务、配置变更、网络连接和发布回滚，先止损再定位根因",
        ],
        relevant_docs=["service_unavailable.md", "api_error_rate_spike.md"],
        category="cross_doc",
        gen_expected_facts=[
            "先确认健康检查失败或错误率超过 50%",
            "查询最近 15-30 分钟的 ERROR/FATAL/status:500 日志",
            "检索 api-gateway-logs 中 level:ERROR 或 status:5xx 的日志",
            "检查依赖服务、配置变更、网络连接和发布回滚",
            "先止损再定位根因",
        ],
    ),
    EvalSample(
        question="网络延迟高导致接口变慢时，要同时看哪些指标和日志？",
        ground_truths=[
            "网络延迟文档建议查询 network-metrics 中 latency > 500 的服务对，并从应用日志确认 RPC 超时和受影响调用",
            "慢响应文档建议查询 response_time > 3000 或 slow_query 日志，同时检查数据库慢查询和系统资源使用情况",
            "需要结合 CPU、内存、网络延迟、下游超时和慢 SQL 判断瓶颈，必要时降级非核心调用并调整超时配置",
        ],
        relevant_docs=["network_high_latency.md", "slow_response.md"],
        category="cross_doc",
        gen_expected_facts=[
            "查询 network-metrics 中 latency > 500 的服务对",
            "从应用日志确认 RPC 超时和受影响调用",
            "检查 response_time > 3000 或 slow_query 日志及数据库慢查询",
            "结合 CPU、内存、网络延迟、下游超时和慢 SQL 综合判断瓶颈",
            "必要时降级非核心调用并调整超时配置",
        ],
    ),
    EvalSample(
        question="数据库连接池满了以后接口响应慢，应该怎么联动排查？",
        ground_truths=[
            "连接池耗尽文档要求检查活跃连接数、空闲连接数、等待队列长度，以及 database_connection_error 或 connection_timeout 日志",
            "慢响应文档提示慢 SQL、数据库 CPU 高和连接池接近满载都会导致 P99 响应时间超过阈值",
            "处理上应先释放或扩容连接、启用限流，再优化慢查询、添加索引或调整连接池配置",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md", "slow_response.md"],
        category="cross_doc",
        gen_expected_facts=[
            "检查活跃连接数、空闲连接数、等待队列长度",
            "查询 database_connection_error 或 connection_timeout 日志",
            "慢 SQL、数据库 CPU 高和连接池满载都会导致 P99 响应超阈值",
            "先释放或扩容连接、启用限流",
            "再优化慢查询、添加索引或调整连接池配置",
        ],
    ),
    EvalSample(
        question="缓存雪崩发展成服务不可用时，应该先恢复缓存还是先做降级？",
        ground_truths=[
            "缓存雪崩文档说明命中率骤降和数据库 QPS 飙升会让服务响应变慢，需立即预热热点数据并优化缓存策略",
            "服务不可用文档强调先确认故障、启动应急响应，并对非关键依赖启用熔断、降级或返回缓存数据",
            "应并行止损：对外降级或切流保护核心链路，同时恢复缓存、限制数据库压力并检查应用错误日志",
        ],
        relevant_docs=["cache_avalanche.md", "service_unavailable.md"],
        category="cross_doc",
        gen_expected_facts=[
            "应并行止损而非串行等待",
            "对外降级或切流保护核心链路",
            "同时恢复缓存、预热热点数据",
            "限制数据库压力并检查应用错误日志",
        ],
    ),

    # -------------------------------------------------------
    # 批次: candidate_questions_20260522_205717.json（导入于 2026-05-22 21:17）
    # -------------------------------------------------------
    EvalSample(
        question="APIErrorRateSpike 告警的触发条件是什么？",
        ground_truths=[
            "API 5xx 错误率持续3分钟超过5%",
            "告警级别为紧急",
            "触发后可能导致用户请求失败和业务中断",
            "应通过 search_log 查询日志确认是上游依赖、代码缺陷、配置错误还是流量峰值导致",
        ],
        relevant_docs=["api_error_rate_spike.md"],
        category="exact_keyword",
        gen_expected_facts=[
            'APIErrorRateSpike 告警触发条件是 API 5xx 错误率持续 3 分钟超过 5%。',
            '该告警的级别被定义为紧急，表明需要高优先级处理。',
            '触发该告警可能导致用户请求失败和业务中断等严重后果。',
            'search_log 工具可用于查询日志以确认上游依赖、代码缺陷等故障原因。',
        ],
    ),
    EvalSample(
        question="接口返回大量 5xx 错误怎么办？",
        ground_truths=[
            "确定告警发生的时间范围，用于后续日志查询",
            "搜索服务对应的日志主题并检查日志主题信息",
            "按时间范围和查询条件检索日志，分析错误日志",
        ],
        relevant_docs=["api_error_rate_spike.md"],
        category="colloquial",
        gen_expected_facts=[
            '处理 5xx 错误时，确定告警发生的时间范围是进行后续日志查询的必要前提。',
            '搜索服务对应的日志主题是检查日志信息并进行问题排查的关键依据。',
            '基于时间范围和查询条件检索日志是分析错误日志的基础。',
        ],
    ),
    EvalSample(
        question="怎么处理 API 错误率突然升高的问题？",
        ground_truths=[
            "获取当前时间以确定告警发生的时间范围",
            "使用 search_topic_by_service_name 工具搜索服务对应的日志主题",
            "按时间范围和查询条件检索日志，分析错误日志",
        ],
        relevant_docs=["api_error_rate_spike.md"],
        category="colloquial",
        gen_expected_facts=[
            '确定告警发生的时间范围需要通过获取当前时间来实现。',
            'search_topic_by_service_name 工具的主要功能是搜索服务对应的日志主题。',
            '错误日志分析依赖于按特定时间范围和查询条件检索到的日志数据。',
            '检索和分析服务日志是定位 API 错误率突然升高原因的关键步骤。',
        ],
    ),
    EvalSample(
        question="API 错误率从正常水平突然升高，一般要先怀疑哪些方向？",
        ground_truths=[
            "上游依赖故障：日志中有上游服务调用失败记录",
            "代码缺陷：特定代码路径频繁抛出异常",
            "配置错误：最近有配置变更，日志中有配置加载错误",
            "流量峰值：请求量突然激增，响应时间变长但无明显错误",
            "网络问题：无法访问服务，网络连接超时",
        ],
        relevant_docs=["api_error_rate_spike.md"],
        category="colloquial",
        gen_expected_facts=[
            '上游依赖故障是导致 API 错误率升高的方向之一，日志中会出现上游服务调用失败记录。',
            '代码缺陷可能导致错误率升高，表现为特定代码路径频繁抛出异常。',
            '配置错误是常见原因，通常伴随最近的配置变更或配置加载错误日志。',
            '流量峰值引起请求量激增时，响应时间会变长，可能导致错误率升高。',
            '网络问题如连接超时或服务无法访问也是导致 API 错误率升高的方向之一。',
        ],
    ),
    EvalSample(
        question="API 5xx 持续升高时，日志里要重点看哪些信息？",
        ground_truths=[
            "应先确定告警发生的时间范围，再检索 api-gateway-logs 中 level:ERROR 或 status:5xx 的日志",
            "需要从错误日志中提取错误类型、错误频率、请求路径、请求参数和错误堆栈",
            "结合日志判断是否是上游依赖故障、代码缺陷、配置错误、流量峰值或网络问题导致",
        ],
        relevant_docs=["api_error_rate_spike.md"],
        category="colloquial",
        gen_expected_facts=[
            '排查 API 5xx 升高时需检索 api-gateway-logs 中 level 为 ERROR 或 status 为 5xx 的日志。',
            '错误日志分析需提取错误类型、频率、请求路径、请求参数和错误堆栈等关键信息。',
            '5xx 错误原因可能包括上游依赖故障、代码缺陷、配置错误、流量峰值或网络问题。',
        ],
    ),
    EvalSample(
        question="CacheAvalanche 告警的触发条件是什么？",
        ground_truths=[
            "缓存命中率骤降30%以上且数据库QPS飙升3倍",
            "大量缓存 key 同时过期或热点 key 失效，请求穿透到数据库",
            "触发后数据库压力剧增，可能引发服务雪崩",
            "需通过 search_log 查询缓存命中率日志和数据库 QPS 确认根因",
        ],
        relevant_docs=["cache_avalanche.md"],
        category="exact_keyword",
        gen_expected_facts=[
            'CacheAvalanche 告警触发条件为缓存命中率骤降 30% 以上且数据库 QPS 飙升 3 倍。',
            '触发原因通常是大量缓存 key 同时过期或热点 key 失效，导致请求穿透到数据库。',
            '告警触发后数据库压力会剧增，这种情况可能引发服务雪崩。',
            '缓存命中率日志和数据库 QPS 数据可通过 search_log 工具进行查询确认。',
        ],
    ),
    EvalSample(
        question="缓存突然失效了怎么办？",
        ground_truths=[
            "立即预热缓存，重启后立即预热热点数据",
            "优化缓存策略，设置合理的过期时间",
            "增加缓存容量，考虑扩容缓存",
        ],
        relevant_docs=["cache_avalanche.md"],
        category="colloquial",
        gen_expected_facts=[
            '缓存失效后的恢复措施包括在系统重启后立即对热点数据进行预热。',
            '优化缓存策略是防止缓存失效的重要手段，需设置合理的过期时间。',
            '增加缓存容量或考虑扩容缓存是应对缓存失效问题的有效解决方案。',
        ],
    ),
    EvalSample(
        question="怎么排查缓存雪崩问题？",
        ground_truths=[
            "获取当前时间，确定告警发生的时间范围",
            "查询缓存服务日志主题，使用 search_topic_by_service_name 工具",
            "按时间检索缓存命中率日志，使用 search_log 工具",
        ],
        relevant_docs=["cache_avalanche.md"],
        category="colloquial",
        gen_expected_facts=[
            '排查缓存雪崩问题时，必须确定告警发生的具体时间范围。',
            'search_topic_by_service_name 工具可用于查询缓存服务日志主题。',
            'search_log 工具可用于按时间检索缓存命中率日志。',
            '获取当前时间是确定告警发生时间范围的基础步骤。',
        ],
    ),
    EvalSample(
        question="缓存命中率低会影响什么？",
        ground_truths=[
            "数据库压力剧增",
            "服务响应变慢",
            "可能引发系统雪崩效应",
            "用户体验下降",
        ],
        relevant_docs=["cache_avalanche.md"],
        category="colloquial",
        gen_expected_facts=[
            '缓存命中率低会导致大量请求直达数据库，造成数据库压力剧增。',
            '缓存未命中会增加请求处理耗时，导致整体服务响应变慢。',
            '严重情况下，低缓存命中率可能引发连锁反应，导致系统雪崩效应。',
            '服务响应延迟增加会直接导致最终用户的体验下降。',
        ],
    ),
    EvalSample(
        question="网络链路抖动会不会让缓存表现得像失效了？",
        ground_truths=[
            "网络故障可能导致缓存服务器不可达",
            "网络延迟高会导致缓存命中率骤降",
            "日志中会有网络超时错误",
        ],
        relevant_docs=["cache_avalanche.md", "network_high_latency.md"],
        category="edge_case",
        gen_expected_facts=[
            '网络链路抖动或故障可能导致应用无法连接到缓存服务器，使其表现为不可达。',
            '较高的网络延迟会导致缓存请求超时，进而造成缓存命中率急剧下降。',
            '排查时可在系统日志中发现网络连接超时或请求超时的错误记录。',
            '网络问题引起的缓存访问异常在现象上与缓存服务本身失效非常相似。',
        ],
    ),
    EvalSample(
        question="CertificateExpiry 告警的触发条件是什么？",
        ground_truths=[
            "TLS 证书距离过期不足7天",
            "告警级别为紧急",
            "触发后服务间 TLS 握手将失败，用户浏览器显示安全警告",
            "需立即续签证书、更新服务器配置并重启相关服务",
        ],
        relevant_docs=["certificate_expiry.md"],
        category="exact_keyword",
        gen_expected_facts=[
            'CertificateExpiry 告警在 TLS 证书距离过期时间不足 7 天时触发。',
            'CertificateExpiry 告警的严重级别被定义为紧急级别。',
            '告警触发后会导致服务间 TLS 握手失败及用户浏览器显示安全警告。',
            '解决该告警需要续签证书、更新服务器配置并重启相关服务。',
        ],
    ),
    EvalSample(
        question="search_log 工具在查询系统日志时需要哪些参数？",
        ground_truths=[
            "地域: ap-guangzhou",
            "日志主题: system-metrics",
            "时间范围: 最近30分钟",
            "查询条件: event:certificate_expiry OR level:ERROR",
        ],
        relevant_docs=["certificate_expiry.md"],
        category="exact_keyword",
        gen_expected_facts=[
            'search_log 工具查询系统日志时，地域参数必须设置为 ap-guangzhou。',
            '日志主题参数需要指定为 system-metrics 才能正确查询系统日志。',
            'search_log 工具查询系统日志时，时间范围参数必须限定在最近 30 分钟内。',
            '查询条件需包含 event:certificate_expiry 或 level:ERROR 的组合。',
        ],
    ),
    EvalSample(
        question="SSL 证书快到期了怎么办？",
        ground_truths=[
            "立即使用证书管理工具（如Let's Encrypt）续签证书。",
            "更新证书文件到服务器并重启相关服务。",
            "设置证书到期前30天的提醒，定期检查证书有效期。",
        ],
        relevant_docs=["certificate_expiry.md"],
        category="colloquial",
        gen_expected_facts=[
            "SSL 证书临近到期时，需要使用证书管理工具如 Let's Encrypt 进行续签操作。",
            '新证书文件更新到服务器后，必须重启相关服务才能完成证书替换并生效。',
            '建议设置证书到期前 30 天的提醒告警，以便预留充足时间处理证书续签。',
            '定期检查证书有效期是防止 SSL 证书过期导致服务不可用的必要运维措施。',
        ],
    ),
    EvalSample(
        question="网站访问时浏览器显示安全警告怎么处理？",
        ground_truths=[
            "检查TLS/SSL证书是否即将过期或已过期。",
            "如果证书过期，立即续签证书并更新到服务器。",
            "验证新证书是否正确安装，并测试SSL握手是否成功。",
        ],
        relevant_docs=["certificate_expiry.md"],
        category="colloquial",
        gen_expected_facts=[
            '浏览器显示安全警告通常是由 TLS/SSL 证书即将过期或已过期引起的。',
            '证书过期时必须立即续签证书并更新到服务器才能恢复访问。',
            '新证书部署后的必要步骤是验证证书是否正确安装。',
            'SSL 握手成功是确认证书安装有效且连接安全的技术指标。',
        ],
    ),
    EvalSample(
        question="应用出现 SSL 握手失败怎么排查？",
        ground_truths=[
            "查询系统日志和应用日志，确认证书的有效期和状态。",
            "检查证书路径和配置文件，确保证书链完整。",
            "如果证书过期，立即续签并更新到服务器。",
        ],
        relevant_docs=["certificate_expiry.md"],
        category="colloquial",
        gen_expected_facts=[
            '系统日志和应用日志中包含证书有效期和状态的关键排查信息。',
            '证书有效期过期或状态异常是导致 SSL 握手失败的常见原因。',
            '证书路径和配置文件的正确性直接影响证书链的完整性。',
            '证书过期后的标准处理流程是续签并更新到服务器。',
        ],
    ),
    EvalSample(
        question="网络延迟高会影响 SSL 证书吗？",
        ground_truths=[
            "网络延迟高不会直接影响SSL证书的有效性。",
            "但高延迟可能会导致SSL握手过程变慢，影响用户体验。",
            "建议检查网络状况，优化网络配置。",
        ],
        relevant_docs=["certificate_expiry.md", "network_high_latency.md"],
        category="edge_case",
        gen_expected_facts=[
            "网络延迟高不会直接影响 SSL 证书的有效性",
            "高延迟可能导致 SSL 握手过程变慢",
            "应检查网络状况并优化网络配置",
        ],
    ),
    EvalSample(
        question="ContainerOOMKilled 告警的触发条件是什么？",
        ground_truths=[
            "容器被 OOM Killer 终止（exit code 137）",
            "K8s Pod 因超出内存 limit 被 OOMKilled",
            "触发后容器重启，服务短暂不可用，可能导致数据丢失",
            "需通过 query_memory_metrics 和容器日志确认内存使用情况和 OOM 根因",
        ],
        relevant_docs=["container_oom_killed.md"],
        category="exact_keyword",
        gen_expected_facts=[
            'ContainerOOMKilled 告警表示容器被 OOM Killer 终止，进程退出码为 137。',
            '触发条件是 K8s Pod 内存使用量超出配置的 memory limit 限制。',
            '告警触发后容器会重启，可能造成服务短暂不可用及数据丢失。',
            '内存使用情况和 OOM 根因可通过 query_memory_metrics 及容器日志确认。',
        ],
    ),
    EvalSample(
        question="query_memory_metrics 工具需要传哪些参数？",
        ground_truths=[
            "地域: ap-guangzhou",
            "时间范围: 最近30分钟",
            "查询条件: 内存使用率超过90%",
        ],
        relevant_docs=["container_oom_killed.md"],
        category="exact_keyword",
        gen_expected_facts=[
            'query_memory_metrics 工具调用时必须指定地域参数，如 ap-guangzhou。',
            '工具参数中需要包含时间范围，示例中要求为最近 30 分钟。',
            '工具需要传入查询条件，例如内存使用率超过 90% 的阈值。',
        ],
    ),
    EvalSample(
        question="如果容器突然挂了，怎么排查是不是因为内存问题？",
        ground_truths=[
            "获取当前时间，确定告警发生的时间范围",
            "查询系统监控日志，检查内存使用率是否超过90%",
            "查询容器日志，查找 level:ERROR 或 level:WARN 或 event:OOM 的记录",
        ],
        relevant_docs=["container_oom_killed.md"],
        category="colloquial",
        gen_expected_facts=[
            '告警发生的时间范围是查询系统监控日志和容器日志的必要索引条件。',
            '系统监控日志中内存使用率超过 90% 是确认容器是否因内存问题挂掉的关键指标。',
            '容器日志中出现 level:ERROR、level:WARN 或 event:OOM 记录通常意味着发生了内存异常。',
        ],
    ),
    EvalSample(
        question="应用突然崩溃了，怎么知道是不是内存不够用了？",
        ground_truths=[
            "查看应用日志中是否有 OutOfMemoryError 或 OOM 相关错误记录",
            "使用 query_memory_metrics 查看内存使用趋势，确认是否存在持续上升或突增",
            "检查应用进程的退出码，exit code 137 表示被 OOM Killer 终止",
        ],
        relevant_docs=["container_oom_killed.md"],
        category="colloquial",
        gen_expected_facts=[
            '应用日志中出现 OutOfMemoryError 或 OOM 相关错误记录表明可能发生内存溢出。',
            '内存使用趋势存在持续上升或突增现象是判断内存不足的重要依据。',
            'query_memory_metrics 工具可用于查看内存使用趋势以确认是否存在异常增长。',
            '应用进程退出码为 137 表示该进程被系统 OOM Killer 强制终止。',
        ],
    ),
    EvalSample(
        question="数据库连接池耗尽会影响容器内存吗？",
        ground_truths=[
            "数据库连接池耗尽通常不会直接导致容器内存问题",
            "但连接等待队列堆积会间接增加应用线程数和内存占用",
            "需同时检查数据库连接池状态和容器内存使用情况，交叉确认因果关系",
        ],
        relevant_docs=["container_oom_killed.md", "database_connection_pool_exhaustion.md"],
        category="edge_case",
        gen_expected_facts=[
            "数据库连接池耗尽通常不会直接导致容器内存问题",
            "连接等待队列堆积会间接增加应用线程数和内存占用",
            "需同时检查连接池状态和容器内存使用情况交叉确认",
        ],
    ),
    EvalSample(
        question="DatabaseConnectionPoolExhaustion 告警的触发条件是什么？",
        ground_truths=[
            "数据库连接池活跃连接数持续5分钟超过90%",
            "告警级别为严重",
            "触发后新请求无法获取数据库连接，导致请求超时增加",
            "需通过 search_log 查询连接池日志，分析活跃连接数和等待队列长度",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md"],
        category="exact_keyword",
        gen_expected_facts=[
            'DatabaseConnectionPoolExhaustion 告警在活跃连接数持续 5 分钟超过 90% 时触发。',
            'DatabaseConnectionPoolExhaustion 告警的级别为严重。',
            '告警触发后新请求无法获取数据库连接，导致请求超时增加。',
            'search_log 工具可用于查询连接池日志，分析活跃连接数和等待队列长度。',
        ],
    ),
    EvalSample(
        question="遇到数据库连接池耗尽怎么办？",
        ground_truths=[
            "获取当前时间，确定告警发生的时间范围",
            "查询系统日志和数据库连接池日志，分析连接池状态",
            "查询应用日志，分析请求日志和数据库性能指标",
            "根据常见原因分析并采取相应处理方案",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md"],
        category="colloquial",
        gen_expected_facts=[
            '告警发生的具体时间范围是排查数据库连接池耗尽问题的关键信息。',
            '系统日志和数据库连接池日志中包含连接池状态的关键数据。',
            '应用日志和数据库性能指标是分析请求日志和性能问题的重要依据。',
            '数据库连接池耗尽问题存在常见原因及对应的处理方案。',
        ],
    ),
    EvalSample(
        question="数据库连接池满了怎么排查？",
        ground_truths=[
            "获取当前时间，确定告警发生的时间范围",
            "使用 search_topic_by_service_name 查询系统日志",
            "检查数据库连接池日志，提取连接池状态信息",
            "查询应用日志，分析请求量变化和错误堆栈信息",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md"],
        category="colloquial",
        gen_expected_facts=[
            '排查数据库连接池问题时，需要获取当前时间并确定告警发生的具体时间范围。',
            '系统日志可以通过 search_topic_by_service_name 方法进行查询和检索。',
            '数据库连接池日志中包含连接池状态信息，检查该日志可确认连接池状态。',
            '应用日志中包含请求量变化数据和错误堆栈信息，可用于分析故障原因。',
        ],
    ),
    EvalSample(
        question="连接池满了会导致什么问题？",
        ground_truths=[
            "新请求无法获取数据库连接",
            "请求超时增加",
            "服务响应变慢",
            "可能触发雪崩效应",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md"],
        category="colloquial",
        gen_expected_facts=[
            '连接池满载时，新的业务请求无法从池中获取到可用的数据库连接资源。',
            '应用请求因等待连接资源而导致超时次数显著增加。',
            '服务整体响应时间变慢，导致用户体验和系统性能下降。',
            '严重情况下连接池耗尽可能引发级联故障，触发系统雪崩效应。',
        ],
    ),
    EvalSample(
        question="网络延迟高会影响数据库连接池吗？",
        ground_truths=[
            "网络延迟高可能导致数据库连接不稳定",
            "日志中可能会有网络超时错误",
            "建议将数据库与应用部署在同一可用区减少网络跳数",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md", "network_high_latency.md"],
        category="edge_case",
        gen_expected_facts=[
            '网络延迟高可能导致数据库连接不稳定，从而影响连接池。',
            '高网络延迟引发的连接问题通常会在日志中体现为网络超时错误。',
            '将数据库与应用部署在同一可用区可以减少网络跳数。',
        ],
    ),
    EvalSample(
        question="消息积压了怎么办？",
        ground_truths=[
            "增加消费者实例数量",
            "启用生产者限流机制",
            "优化消费者处理逻辑",
        ],
        relevant_docs=["message_queue_backlog.md"],
        category="colloquial",
        gen_expected_facts=[
            '解决消息积压问题时，增加消费者实例数量可以提升消费端的并行处理能力。',
            '启用生产者限流机制能够降低消息生产速率，防止积压情况进一步恶化。',
            '优化消费者处理逻辑可以减少单条消息处理耗时，提高消费端的整体效率。',
        ],
    ),
    EvalSample(
        question="怎么检查消息队列的日志？",
        ground_truths=[
            "使用 search_topic_by_service_name 工具查询日志主题",
            "使用 get_topic_info_by_name 工具获取具体的日志主题信息",
            "使用 search_log 工具检索相关日志",
        ],
        relevant_docs=["message_queue_backlog.md"],
        category="colloquial",
        gen_expected_facts=[
            'search_topic_by_service_name 工具支持根据服务名称查询消息队列的日志主题。',
            'get_topic_info_by_name 工具用于获取指定日志主题的详细配置信息。',
            'search_log 工具是检索消息队列相关日志内容的必要手段。',
        ],
    ),
    EvalSample(
        question="消息队列积压后，如何分析消费者的CPU和内存使用情况？",
        ground_truths=[
            "使用 query_cpu_metrics 和 query_memory_metrics 工具查询消费者服务的资源使用情况",
            "参数包括服务名（消费者所在的服务名）和时间范围（最近1小时）",
            "如果 CPU 或内存使用率超过 80%，说明消费者可能存在资源瓶颈，需扩容或优化代码",
        ],
        relevant_docs=["message_queue_backlog.md"],
        category="colloquial",
        gen_expected_facts=[
            '需使用 query_cpu_metrics 和 query_memory_metrics 查询消费者资源。',
            '查询工具的参数需包含消费者服务名及最近 1 小时的时间范围。',
            'CPU 或内存使用率超过 80% 意味着消费者可能存在资源瓶颈。',
            '资源瓶颈的解决措施通常包括服务扩容或代码优化。',
        ],
    ),
    EvalSample(
        question="磁盘满了会影响消息队列吗？",
        ground_truths=[
            "消息队列 Broker 需要磁盘存储消息，磁盘满会直接导致消息写入失败",
            "磁盘 IO 繁忙也会拖慢消息写入和消费速度，加剧积压",
            "应使用 search_log 查询消息队列日志中是否有磁盘空间不足或 IO 超时的错误",
        ],
        relevant_docs=["message_queue_backlog.md"],
        category="edge_case",
        gen_expected_facts=[
            "消息队列 Broker 需要磁盘存储消息",
            "磁盘满会直接导致消息写入失败",
            "磁盘 IO 繁忙会拖慢消息写入和消费速度",
            "应查询消息队列日志中是否有磁盘空间不足或 IO 超时错误",
        ],
    ),
    EvalSample(
        question="MessageQueueBacklog 告警的触发条件是什么？",
        ground_truths=[
            "MessageQueueBacklog 的告警级别为严重",
            "触发条件是 Kafka 或 RocketMQ 消费延迟持续10分钟超过10000条",
            "消息积压会导致消费延迟增加、业务延迟，并可能让下游服务响应变慢或失败",
        ],
        relevant_docs=["message_queue_backlog.md"],
        category="exact_keyword",
        gen_expected_facts=[
            'MessageQueueBacklog 告警的级别被定义为严重级别，属于高优先级告警。',
            '该告警适用的消息队列系统包括 Kafka 或 RocketMQ 两种中间件。',
            '触发条件是消费延迟持续 10 分钟超过 10000 条消息积压。',
            '消息积压会导致消费延迟增加以及整体业务处理延迟现象。',
            '消息积压可能让下游服务响应变慢或出现请求失败的情况。',
        ],
    ),
    EvalSample(
        question="证书快到期但还没过期时，会不会已经影响接口调用？",
        ground_truths=[
            "证书未过期时通常不会直接阻断接口调用，但握手和校验过程可能变慢",
            "如果证书链不完整或客户端校验策略更严格，仍可能出现 TLS 握手失败",
            "应尽快续签并更新证书，避免到期后影响服务可用性",
        ],
        relevant_docs=["certificate_expiry.md", "slow_response.md"],
        category="edge_case",
        gen_expected_facts=[
            "证书未过期时通常不会直接阻断接口调用",
            "握手和校验过程可能变慢",
            "证书链不完整或客户端严格校验可能导致 TLS 握手失败",
            "应尽快续签并更新证书",
        ],
    ),
    EvalSample(
        question="NetworkHighLatency 告警的触发条件是什么？",
        ground_truths=[
            "服务间网络延迟 P99 持续5分钟超过500ms",
            "告警级别为警告",
            "触发后服务间调用可能超时，请求堆积，用户体验下降",
            "应通过 search_log 查询延迟日志，确认受影响的服务对，通知网络团队排查链路",
        ],
        relevant_docs=["network_high_latency.md"],
        category="exact_keyword",
        gen_expected_facts=[
            'NetworkHighLatency 告警触发条件是服务间网络延迟 P99 持续 5 分钟超过 500ms。',
            'NetworkHighLatency 告警的告警级别配置为警告级别。',
            '告警触发后可能导致服务间调用超时、请求堆积及用户体验下降。',
            '可通过 search_log 查询延迟日志来确认受影响的服务对。',
        ],
    ),
    EvalSample(
        question="网络延迟过高时，怎么判断是链路问题还是服务自身问题？",
        ground_truths=[
            "使用 search_log 查询 network-metrics 日志主题，确认延迟最高的服务对",
            "通过应用日志中的 RPC 调用耗时记录确认受影响的服务间调用",
            "使用 query_cpu_metrics 和 query_memory_metrics 排除服务器资源瓶颈",
            "联系网络团队检查跨地域链路的带宽和延迟",
        ],
        relevant_docs=["network_high_latency.md"],
        category="colloquial",
        gen_expected_facts=[
            'network-metrics 日志主题记录了服务对延迟数据，可用于确认延迟最高的服务对。',
            '应用日志中的 RPC 调用耗时记录能反映受影响的服务间调用情况。',
            '服务器资源瓶颈是潜在原因之一，需通过 CPU 和内存指标进行排除。',
            '跨地域链路的带宽和延迟状况需由网络团队进行检查确认。',
        ],
    ),
    EvalSample(
        question="服务调用变慢了，怎么定位问题？",
        ground_truths=[
            "获取当前时间并确定告警发生的时间范围",
            "使用 search_log 查询系统日志和应用日志，分析网络延迟和错误记录",
            "使用 query_cpu_metrics 和 query_memory_metrics 确认是否因服务器资源瓶颈导致",
        ],
        relevant_docs=["network_high_latency.md"],
        category="colloquial",
        gen_expected_facts=[
            '定位服务调用变慢问题时，首先需要确定告警发生的具体时间范围。',
            'search_log 工具可用于查询系统日志和应用日志以分析网络延迟和错误记录。',
            '服务器资源瓶颈可能导致服务变慢，可通过 query_cpu_metrics 和 query_memory_metrics 确认。',
        ],
    ),
    EvalSample(
        question="网络延迟升高后，通常先拖慢哪些业务环节？",
        ground_truths=[
            "可能导致服务间调用超时。",
            "请求堆积，用户体验下降。",
            "可能触发雪崩效应。",
        ],
        relevant_docs=["network_high_latency.md"],
        category="colloquial",
        gen_expected_facts=[
            '网络延迟升高会导致服务间调用耗时增加，可能引发调用超时。',
            '网络延迟升高会造成请求在系统中堆积，无法及时得到处理。',
            '请求处理变慢和堆积会直接导致最终用户体验下降。',
            '网络延迟引发的连锁反应严重时可能触发系统的雪崩效应。',
        ],
    ),
    EvalSample(
        question="数据库连接池耗尽会让外部请求的延迟表现出什么特征？",
        ground_truths=[
            "数据库连接池耗尽可能导致应用线程阻塞等待连接，间接增加请求处理延迟",
            "大量阻塞线程可能占满应用线程池，导致新的网络请求无法被处理",
            "需通过应用日志确认连接等待时长，同时检查网络延迟是否由其他因素引起",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md", "network_high_latency.md"],
        category="edge_case",
        gen_expected_facts=[
            "连接池耗尽导致应用线程阻塞等待连接，间接增加请求延迟",
            "大量阻塞线程可能占满应用线程池",
            "线程池满会导致新的网络请求无法被处理",
            "需通过应用日志确认连接等待时长",
        ],
    ),
    # -----------------------------------------------------------------------
    # Dataset v1.3.0 additions: cross-doc and edge-case focused samples
    # -----------------------------------------------------------------------
    EvalSample(
        question="CPU 高、响应慢但错误率没明显上升时，怎么区分是流量峰值还是代码性能问题？",
        ground_truths=[
            "流量峰值通常表现为请求量明显增加、多个进程 CPU 均匀升高、响应时间变长但无明显错误。",
            "代码性能问题通常表现为特定代码路径执行慢、日志中有性能警告或热点方法，但不一定伴随外部依赖异常。",
            "如果是流量峰值，应优先扩容、限流并观察扩容后的 CPU 使用率。",
            "如果是代码性能问题，应使用 APM、火焰图或日志定位慢方法，优化循环、递归和对象创建。",
        ],
        relevant_docs=["cpu_high_usage.md", "slow_response.md"],
        category="cross_doc",
        gen_expected_facts=[
            '流量峰值通常表现为请求量明显增加且多个进程 CPU 均匀升高。',
            '流量峰值场景下响应时间变长，但错误率通常无明显上升。',
            '代码性能问题通常表现为特定代码路径执行慢或日志中出现热点方法。',
            '代码性能问题不一定伴随外部依赖异常，CPU 升高通常不如流量峰值均匀。',
            'APM 和火焰图是定位代码性能问题中慢方法的有效工具。',
        ],
    ),
    EvalSample(
        question="缓存命中率突然下降后，为什么数据库连接池也可能被打满？",
        ground_truths=[
            "缓存失效或缓存穿透会导致数据库查询量激增，数据库 QPS 明显升高。",
            "数据库访问增加会推高连接池活跃连接数，严重时新请求无法获取连接。",
            "应同时关注缓存命中率、数据库 QPS、连接池活跃连接数和应用响应时间。",
            "处理上需要预热热点缓存、优化缓存过期策略，并根据连接池压力调整限流或连接池配置。",
        ],
        relevant_docs=["cache_avalanche.md", "database_connection_pool_exhaustion.md", "slow_response.md"],
        category="cross_doc",
        gen_expected_facts=[
            '缓存失效或缓存穿透会导致数据库查询量激增，进而造成数据库 QPS 明显升高。',
            '数据库访问增加会推高连接池活跃连接数，严重时导致新请求无法获取连接。',
            '关联监控指标包含缓存命中率、数据库 QPS、连接池活跃连接数和应用响应时间。',
            '该问题的处理策略涉及预热热点缓存、优化缓存过期策略，以及根据连接池压力调整限流配置。',
        ],
    ),
    EvalSample(
        question="消息队列积压同时消费者机器 CPU 和内存都高，应该怎样判断瓶颈？",
        ground_truths=[
            "消息积压可能来自消费者处理能力不足、生产者流量突增、消费者配置不当或系统资源不足。",
            "CPU 或内存使用率高说明消费者实例可能存在资源瓶颈，会拖慢消费速度。",
            "如果内存随运行时间持续上升且 Full GC 后无法释放，应考虑内存泄漏。",
            "如果 CPU 均匀升高且请求或消息量同步增加，应优先考虑扩容消费者实例和调整消费者线程数。",
        ],
        relevant_docs=["message_queue_backlog.md", "cpu_high_usage.md", "memory_high_usage.md"],
        category="cross_doc",
        gen_expected_facts=[
            '消息积压可能源于消费者处理能力不足、生产者流量突增、消费者配置不当或系统资源不足。',
            '消费者机器 CPU 或内存使用率高说明实例可能存在资源瓶颈，会拖慢消费速度。',
            '若内存随运行时间持续上升且 Full GC 后无法释放，应考虑存在内存泄漏问题。',
            'CPU 均匀升高且请求或消息量同步增加的现象，通常对应消费者实例不足或线程数配置不当的问题。',
        ],
    ),
    EvalSample(
        question="磁盘空间满会怎样把服务不可用和消息积压串起来？",
        ground_truths=[
            "磁盘空间满属于资源耗尽，可能导致服务无法写日志、写临时文件或正常启动，从而触发服务不可用。",
            "消息队列 Broker 依赖磁盘存储消息，磁盘满或磁盘 IO 繁忙会导致消息写入失败或消费变慢。",
            "服务不可用文档将磁盘空间满列为资源耗尽的一类，需要清理日志、临时文件或扩容资源。",
            "消息积压场景下应关注 Broker 磁盘空间和 IO 状态，避免写入失败继续扩大影响。",
        ],
        relevant_docs=["disk_high_usage.md", "service_unavailable.md", "message_queue_backlog.md"],
        category="cross_doc",
        gen_expected_facts=[
            '磁盘空间满属于资源耗尽，会导致服务无法写日志、写临时文件或正常启动，从而触发服务不可用。',
            '消息队列 Broker 依赖磁盘存储消息，磁盘满或磁盘 IO 繁忙会导致消息写入失败或消费变慢。',
            '磁盘空间满作为资源耗尽的一种，是同时导致服务不可用和消息积压现象的共同潜在根因。',
            '消息积压场景下写入失败会继续扩大影响，Broker 磁盘空间和 IO 状态是防止影响扩大的关键因素。',
        ],
    ),
    EvalSample(
        question="TLS 证书异常导致接口 5xx 增多时，应该从哪些证据判断是证书问题而不是普通网络抖动？",
        ground_truths=[
            "证书问题常见证据包括证书到期、证书链不完整、域名不匹配、证书被吊销或 SSL 握手失败日志。",
            "API 错误率飙升时应分析错误类型、请求路径、错误频率和错误堆栈，确认是否集中在 TLS 握手或安全校验。",
            "普通网络问题更常见连接超时、负载均衡异常或 DNS 解析失败等表现。",
            "如果错误集中在 HTTPS 调用并伴随证书有效期或证书配置异常，应优先续签、更新证书并验证 SSL 握手。",
        ],
        relevant_docs=["certificate_expiry.md", "api_error_rate_spike.md", "network_high_latency.md"],
        category="cross_doc",
        gen_expected_facts=[
            '证书问题常见证据包括证书到期、证书链不完整、域名不匹配、证书被吊销或 SSL 握手失败日志。',
            '证书问题引发的错误通常集中在 TLS 握手或安全校验环节，区别于普通业务逻辑错误。',
            '普通网络问题更常见连接超时、负载均衡异常或 DNS 解析失败等表现，与证书错误特征不同。',
            '错误集中在 HTTPS 调用并伴随证书有效期或配置异常时，是证书问题而非网络抖动的关键特征。',
        ],
    ),
    EvalSample(
        question="数据库慢查询为什么可能同时触发 CPU 高、响应慢和连接池耗尽？",
        ground_truths=[
            "数据库慢查询会让特定 SQL 执行时间变长，数据库 CPU 可能升高，连接池也会接近满载。",
            "CPU 高文档指出数据库查询慢可能表现为应用 CPU 高、慢查询记录和连接池占用高。",
            "响应慢文档将数据库慢查询列为常见原因，需找出最慢 SQL、检查索引并查看执行计划。",
            "连接池耗尽会导致新请求无法获取连接，使请求超时增加并进一步拖慢服务响应。",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md", "cpu_high_usage.md", "slow_response.md"],
        category="cross_doc",
        gen_expected_facts=[
            '数据库慢查询会导致特定 SQL 执行时间变长，进而引起数据库或应用 CPU 使用率升高。',
            '慢查询导致数据库连接占用时间增加，使连接池占用接近满载甚至耗尽。',
            '数据库慢查询被列为服务响应慢的常见原因之一，会直接增加请求处理时间。',
            '连接池耗尽会导致新请求无法获取连接，使请求超时增加并进一步拖慢服务响应。',
        ],
    ),
    EvalSample(
        question="跨可用区网络延迟升高时，为什么数据库连接和 MQ 消费都会受影响？",
        ground_truths=[
            "跨地域或跨可用区网络延迟会让特定服务间调用耗时升高，并在应用日志中出现 RPC 超时。",
            "数据库连接在跨可用区部署时会增加网络跳数，连接不稳定或超时可能加重连接池压力。",
            "消息队列消费者若跨地域或连接远端 Broker，会因为拉取延迟和带宽限制导致消费速度下降。",
            "优化方向包括就近部署数据库、消费者和 Broker，减少跨地域网络跳数并检查链路带宽。",
        ],
        relevant_docs=["network_high_latency.md", "database_connection_pool_exhaustion.md", "message_queue_backlog.md"],
        category="cross_doc",
        gen_expected_facts=[
            '跨地域或跨可用区网络延迟升高会导致服务间调用耗时增加，并在应用日志中出现 RPC 超时。',
            '数据库连接跨可用区部署会增加网络跳数，连接不稳定或超时可能加重数据库连接池的压力。',
            '消息队列消费者若连接远端 Broker，会因为拉取延迟和带宽限制导致消费速度下降。',
            '跨地域网络跳数和链路带宽是影响数据库连接和消息队列消费性能的关键网络因素。',
        ],
    ),
    EvalSample(
        question="容器 OOM 之后服务短暂不可用，和普通内存高告警相比要关注哪些额外信息？",
        ground_truths=[
            "容器 OOMKilled 表示容器被 OOM Killer 终止，K8s Pod 可能因超出 memory limit 而重启。",
            "普通内存高更强调内存泄漏、流量突增、缓存配置不当或 JVM 参数配置问题。",
            "容器 OOM 需要关注容器日志、exit code 137、memory limit 和重启带来的服务短暂不可用。",
            "两类场景都应在重启前尽量保留内存现场，并检查 GC、OOM 错误和内存使用趋势。",
        ],
        relevant_docs=["container_oom_killed.md", "memory_high_usage.md", "service_unavailable.md"],
        category="cross_doc",
        gen_expected_facts=[
            '容器 OOMKilled 表示容器因超出 memory limit 被终止，通常会导致 K8s Pod 重启。',
            '普通内存高告警通常源于内存泄漏、流量突增、缓存配置不当或 JVM 参数配置问题。',
            '容器 OOM 需额外关注 exit code 137、memory limit 配置及重启导致的服务短暂不可用。',
            '两类场景均需检查 GC 日志、OOM 错误信息和内存使用趋势以辅助定位问题根因。',
        ],
    ),
    EvalSample(
        question="缓存雪崩后 API 错误率上升，什么时候应该先降级而不是只扩容？",
        ground_truths=[
            "缓存雪崩会让缓存命中率骤降、数据库 QPS 激增，并可能让应用响应时间变长。",
            "API 错误率飙升如果来自上游依赖故障或数据库压力，应启用熔断、返回默认值或缓存数据。",
            "只扩容应用实例不能直接解决数据库或缓存层被打穿的问题，可能继续放大下游压力。",
            "应先通过限流、降级和缓存预热保护核心业务，再结合容量情况扩容缓存或应用实例。",
        ],
        relevant_docs=["cache_avalanche.md", "api_error_rate_spike.md", "slow_response.md"],
        category="cross_doc",
        gen_expected_facts=[
            '缓存雪崩会导致缓存命中率骤降和数据库 QPS 激增，从而引起应用响应时间变长。',
            '仅扩容应用实例无法解决数据库或缓存层被打穿的问题，反而可能继续放大下游压力。',
            'API 错误率飙升若来自上游依赖故障或数据库压力，属于需启用熔断或返回默认值的降级场景。',
            '保护核心业务的正确顺序是先通过限流和降级，然后再结合容量情况扩容缓存或应用实例。',
        ],
    ),
    EvalSample(
        question="服务完全不可用但监控只看到高内存和磁盘满，先判断哪些资源耗尽路径？",
        ground_truths=[
            "服务不可用中的资源耗尽包括磁盘空间满、文件描述符耗尽、端口占用和内存不足导致 OOM。",
            "内存高可能来自内存泄漏、流量突增、缓存配置不当、大文件处理或 JVM 参数不合理。",
            "磁盘高可能来自日志文件过大、临时文件堆积、数据文件增长、备份文件或 Docker 资源占用。",
            "应优先判断是否存在 OOM、磁盘无法写入或启动失败日志，并通过清理、扩容或重启恢复服务。",
        ],
        relevant_docs=["service_unavailable.md", "memory_high_usage.md", "disk_high_usage.md"],
        category="cross_doc",
        gen_expected_facts=[
            '服务不可用中的资源耗尽包括磁盘空间满、文件描述符耗尽、端口占用和内存不足导致 OOM。',
            '内存高可能来自内存泄漏、流量突增、缓存配置不当、大文件处理或 JVM 参数不合理。',
            '磁盘高可能来自日志文件过大、临时文件堆积、数据文件增长、备份文件或 Docker 资源占用。',
            'OOM、磁盘无法写入或启动失败日志是判断资源耗尽路径的关键指标。',
        ],
    ),
    EvalSample(
        question="CPU 已经恢复到 60% 以下，但响应时间还是高，可以说明 CPU 告警已经彻底解决了吗？",
        ground_truths=[
            "不能只凭 CPU 降到正常水平判断问题彻底解决，CPU 文档还要求检查应用响应时间是否恢复正常。",
            "响应时间持续偏高可能来自数据库慢查询、外部 API 超时、缓存失效、系统资源不足或网络问题。",
            "需要确认无新的错误日志产生，并持续观察至少 30 分钟确保问题不再复现。",
            "如果响应慢仍存在，应继续沿慢查询、外部依赖、缓存命中率和网络延迟方向排查。",
        ],
        relevant_docs=["cpu_high_usage.md", "slow_response.md"],
        category="edge_case",
        gen_expected_facts=[
            '仅凭 CPU 使用率恢复到 60% 以下不能判断告警解决，必须确认应用响应时间也恢复正常。',
            '响应时间偏高可能源于数据库慢查询、外部 API 超时、缓存失效、系统资源不足或网络问题。',
            '确认问题彻底解决需要无新的错误日志产生，并持续观察至少 30 分钟确保问题不再复现。',
        ],
    ),
    EvalSample(
        question="缓存命中率低但数据库 QPS 没升高，还能直接判定缓存雪崩吗？",
        ground_truths=[
            "不能直接判定缓存雪崩，因为缓存雪崩典型特征包括缓存命中率骤降和数据库 QPS 激增。",
            "如果数据库 QPS 没升高，可能是流量较低、请求被限流、命中的是其他缓存层或业务路径没有访问数据库。",
            "仍应检查缓存 miss 日志、热点 key 状态、缓存容量和过期策略是否异常。",
            "判断缓存雪崩需要结合应用响应时间、数据库 QPS 和缓存命中率变化共同确认。",
        ],
        relevant_docs=["cache_avalanche.md"],
        category="edge_case",
        gen_expected_facts=[
            '缓存雪崩的典型特征包括缓存命中率骤降和数据库 QPS 激增，二者需同时出现。',
            '仅出现缓存命中率低但数据库 QPS 未升高时，不能直接判定为发生了缓存雪崩。',
            '数据库 QPS 未升高可能由流量较低、请求被限流、命中其他缓存层或未访问数据库导致。',
            '判断缓存雪崩需要结合应用响应时间、数据库 QPS 和缓存命中率变化共同确认。',
        ],
    ),
    EvalSample(
        question="证书还有 5 天才过期，但日志已经出现 SSL 握手失败，是否可以忽略到期告警？",
        ground_truths=[
            "不能忽略，因为证书到期未续只是证书异常的一类原因，证书链不完整、路径错误或域名不匹配也会导致握手失败。",
            "证书配置错误可能表现为证书文件路径错误、证书链不完整或证书格式不正确。",
            "证书不匹配会导致证书域名与实际访问域名不一致，并出现 SSL 握手失败。",
            "应检查证书有效期、证书链、绑定域名和应用配置，并在修复后测试 SSL 握手是否成功。",
        ],
        relevant_docs=["certificate_expiry.md"],
        category="edge_case",
        gen_expected_facts=[
            '不能忽略到期告警，因为证书到期未续只是导致 SSL 握手失败的其中一类原因。',
            '证书链不完整、证书文件路径错误或域名不匹配同样会导致 SSL 握手失败。',
            '证书配置错误可能表现为证书链不完整、证书文件路径错误或证书格式不正确。',
            '证书域名与实际访问域名不一致会导致证书不匹配并引发 SSL 握手失败。',
            '修复问题后需要测试 SSL 握手是否成功以验证配置的有效性。',
        ],
    ),
    EvalSample(
        question="Pod 被 OOMKilled 后自动重启成功，还需要继续分析吗？",
        ground_truths=[
            "需要继续分析，因为 OOMKilled 说明容器曾因内存限制被终止，可能造成服务短暂不可用或数据丢失。",
            "应查看容器日志和系统监控日志，确认内存使用是否持续上升、突增或超过 memory limit。",
            "如果是内存泄漏，应在重启前尽量保留堆转储或日志现场，定位大量对象引用的代码。",
            "还应检查运行时内存参数和缓存配置，避免重启后再次 OOM。",
        ],
        relevant_docs=["container_oom_killed.md", "memory_high_usage.md"],
        category="edge_case",
        gen_expected_facts=[
            'OOMKilled 表明容器曾因内存限制被终止，可能造成服务不可用，需要继续分析。',
            '容器日志和系统监控日志可反映内存使用是否持续上升、突增或超过 memory limit。',
            '内存泄漏时，重启前保留堆转储或日志现场有助于定位大量对象引用的代码。',
            '检查运行时内存参数和缓存配置有助于避免重启后再次发生 OOM。',
        ],
    ),
    EvalSample(
        question="消息积压下降到正常水平，但消费者仍有大量超时日志，这算恢复了吗？",
        ground_truths=[
            "不能算完全恢复，消息队列验证要求消费延迟降到正常水平，同时消费者处理速率恢复正常。",
            "消费者处理超时或异常是消费者处理能力不足的典型特征，可能再次导致积压。",
            "应继续检查消费者代码、线程数、批次大小和系统资源使用情况。",
            "还需要观察应用日志无新的错误日志，并持续监控 30 分钟确保稳定。",
        ],
        relevant_docs=["message_queue_backlog.md"],
        category="edge_case",
        gen_expected_facts=[
            '消息积压下降但消费者有超时日志不能算完全恢复，需同时满足消费延迟和处理速率正常。',
            '消费者处理超时或异常是处理能力不足的典型特征，存在再次导致消息积压的风险。',
            '消费者处理能力问题可能涉及代码逻辑、线程数、批次大小和系统资源使用情况。',
            '系统恢复验证需要观察应用日志无新错误，并持续监控 30 分钟确保运行稳定。',
        ],
    ),
    EvalSample(
        question="网络 P99 延迟高，但 CPU 和内存都正常，能排除应用层问题吗？",
        ground_truths=[
            "不能完全排除应用层问题，网络延迟文档指出应用层协议问题也会造成连接超时或连接重置。",
            "CPU 和内存正常更符合跨地域或跨可用区网络延迟的特征，但仍需查看应用日志中的 RPC 调用耗时。",
            "应检查是否存在连接池大小或超时时间配置不合理、长耗时请求未异步处理、循环调用或 N+1 查询。",
            "需要结合服务间调用日志和正常时段耗时对比，才能区分链路问题和应用层协议问题。",
        ],
        relevant_docs=["network_high_latency.md", "slow_response.md"],
        category="edge_case",
        gen_expected_facts=[
            '不能完全排除应用层问题，应用层协议问题也会导致连接超时或重置。',
            'CPU 和内存正常更符合跨地域或跨可用区网络延迟的特征。',
            '连接池配置不合理、长耗时请求未异步处理或 N+1 查询可能引发延迟。',
            '服务间调用日志和正常时段耗时对比可用于区分链路问题和应用层协议问题。',
        ],
    ),
    EvalSample(
        question="磁盘使用率高是备份文件导致的，可以只删除备份不改策略吗？",
        ground_truths=[
            "只删除备份只能短期释放空间，备份文件占用空间的根因还包括历史备份过多、未压缩和未转移到其他存储。",
            "应优化备份策略，只保留最近 N 天备份、压缩备份文件，并将备份转移到对象存储或专用存储。",
            "如果不调整策略，备份目录可能再次占满磁盘并触发告警。",
            "删除前应确认备份保留要求，避免误删仍需保留的恢复点。",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="edge_case",
        gen_expected_facts=[
            '只删除备份文件只能短期释放空间，无法解决历史备份过多等根因。',
            '如果不调整备份策略，备份目录可能再次占满磁盘并触发告警。',
            '删除备份前应确认备份保留要求，避免误删仍需保留的恢复点。',
            '备份策略优化包括保留最近 N 天备份、压缩文件及转移到对象存储。',
        ],
    ),
    EvalSample(
        question="API 5xx 只集中在一个请求路径，是不是一定说明上游依赖故障？",
        ground_truths=[
            "不一定，上游依赖故障确实可能表现为特定请求路径错误率高和上游调用失败。",
            "代码缺陷也可能导致特定代码路径频繁抛出异常，错误堆栈会指向特定方法。",
            "配置错误或网络问题也可能让某些路径集中失败，需要结合配置变更、连接超时和错误堆栈判断。",
            "应从错误类型、错误频率、请求路径、请求参数和错误堆栈综合定位。",
        ],
        relevant_docs=["api_error_rate_spike.md"],
        category="edge_case",
        gen_expected_facts=[
            'API 5xx 集中在一个请求路径不一定说明是上游依赖故障，存在其他可能性。',
            '代码缺陷也可能导致特定代码路径频繁抛出异常，错误堆栈会指向特定方法。',
            '配置错误或网络问题也可能让某些路径集中失败，常伴随配置变更和连接超时现象。',
            '问题定位依赖错误类型、错误频率、请求路径、请求参数和错误堆栈的综合信息。',
        ],
    ),
    EvalSample(
        question="连接池活跃连接数高但数据库 CPU 很低，是否还可能是数据库性能问题？",
        ground_truths=[
            "数据库性能问题通常伴随数据库 CPU 或内存使用率高、慢查询日志大量记录和连接池接近满载。",
            "如果数据库 CPU 很低，应同时考虑连接泄漏、连接池配置不当或网络连接不稳定。",
            "连接泄漏会表现为空闲连接数持续减少和大量连接超时记录。",
            "配置不当可能是最大连接数过低或连接超时时间过短，需要验证连接池参数是否合理。",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md"],
        category="edge_case",
        gen_expected_facts=[
            '数据库性能问题通常伴随数据库 CPU 或内存使用率高、慢查询日志大量记录和连接池接近满载。',
            '数据库 CPU 很低时，活跃连接数高可能由连接泄漏、连接池配置不当或网络连接不稳定引起。',
            '连接泄漏会表现为空闲连接数持续减少和大量连接超时记录。',
            '连接池配置不当可能是最大连接数过低或连接超时时间过短。',
        ],
    ),
    EvalSample(
        question="内存突然升高但 GC 能回收大部分内存，还需要按内存泄漏处理吗？",
        ground_truths=[
            "不应优先按内存泄漏处理，因为流量突增导致对象激增的特征包括内存突然升高且 GC 能回收大部分内存。",
            "内存泄漏更典型的表现是内存持续缓慢上升、Full GC 后无法释放，并且运行时间越长占用越高。",
            "这种情况应结合请求量或流量增长判断，优先考虑扩容、限流和优化缓存策略。",
            "如果内存后续仍持续上升或出现 OOM，再进一步 dump 内存快照分析泄漏。",
        ],
        relevant_docs=["memory_high_usage.md"],
        category="edge_case",
        gen_expected_facts=[
            '内存突然升高但 GC 能回收大部分内存是流量突增导致对象激增的特征，不应优先按内存泄漏处理。',
            '内存泄漏的典型表现是内存持续缓慢上升，Full GC 后无法释放，且运行时间越长占用越高。',
            '针对流量突增导致的内存升高，解决策略应优先考虑扩容、限流和优化缓存，而非内存泄漏排查。',
            '只有当内存后续仍持续上升或出现 OOM 时，才需要进一步 dump 内存快照分析泄漏。',
        ],
    ),
    EvalSample(
        question="HighCPUUsage 告警里，死循环和流量突增的关键区别是什么？",
        ground_truths=[
            "死循环或无限递归通常表现为单个进程 CPU 占用接近 100%。",
            "死循环场景下应用日志中可能出现大量重复的错误堆栈，内存使用也可能同步增长。",
            "流量突增通常表现为多个进程 CPU 使用率均匀升高，请求量明显增加。",
            "流量突增时响应时间会变长但可能没有明显错误，处理上优先扩容和限流。",
        ],
        relevant_docs=["cpu_high_usage.md"],
        category="exact_keyword",
        gen_expected_facts=[
            '死循环或无限递归通常表现为单个进程 CPU 占用接近 100%。',
            '死循环场景下应用日志中可能出现大量重复的错误堆栈，内存使用也可能同步增长。',
            '流量突增通常表现为多个进程 CPU 使用率均匀升高，请求量明显增加。',
            '流量突增时系统响应时间会变长，但可能没有明显错误日志产生。',
        ],
    ),
    EvalSample(
        question="SlowResponse 中外部 API 调用超时的处理方案有哪些？",
        ground_truths=[
            "应设置合理的 HTTP 客户端超时时间，避免无限等待，并分别设置连接超时和读取超时。",
            "应实施降级策略，包括启用熔断、返回默认值或缓存数据。",
            "非关键的外部调用可以改为异步处理，减少对主流程响应时间的影响。",
            "还可以并行调用多个 API、使用批量接口减少调用次数，并增加本地缓存。",
        ],
        relevant_docs=["slow_response.md"],
        category="exact_keyword",
        gen_expected_facts=[
            '应设置合理的 HTTP 客户端超时时间，包括连接超时和读取超时，避免无限等待。',
            '应实施降级策略，包括启用熔断、返回默认值或使用缓存数据来保障服务可用性。',
            '非关键的外部调用可以改为异步处理，以减少对主流程响应时间的影响。',
            '可以通过并行调用多个 API 或使用批量接口来减少调用次数，并增加本地缓存。',
        ],
    ),
    EvalSample(
        question="DiskHighUsage 中 Docker 资源占用的典型表现和处理办法是什么？",
        ground_truths=[
            "Docker 资源占用通常表现为 Docker 占用大量磁盘空间、大量未使用镜像、停止容器未清理或容器日志过大。",
            "处理时可以清理未使用镜像、停止的容器、未使用卷和所有未使用资源。",
            "还应限制容器日志，配置日志驱动、限制日志文件大小并设置日志轮转。",
            "长期应优化镜像，例如使用多阶段构建、减小镜像体积并定期清理旧镜像。",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="exact_keyword",
        gen_expected_facts=[
            'DiskHighUsage 中 Docker 资源占用典型表现为大量未使用镜像、停止容器未清理或容器日志过大。',
            '处理办法包括清理未使用镜像、停止的容器、未使用卷和所有未使用资源。',
            '应配置日志驱动、限制日志文件大小并设置日志轮转来控制容器日志大小。',
            '长期优化可使用多阶段构建减小镜像体积，并定期清理旧镜像。',
        ],
    ),
    EvalSample(
        question="CacheAvalanche 里热点 key 失效应该怎么处理？",
        ground_truths=[
            "热点 key 失效通常表现为某个热点 key 突然失效、数据库特定查询激增、该 key 的 miss 日志大量出现。",
            "可以临时增加该 key 的缓存时间，并手动刷新缓存。",
            "应对热点 key 设置更长的过期时间，使用分布式锁防止并发更新。",
            "还可以实现本地缓存作为后备，并考虑使用多级缓存。",
        ],
        relevant_docs=["cache_avalanche.md"],
        category="exact_keyword",
        gen_expected_facts=[
            '热点 key 失效现象包括特定 key 突然失效、数据库查询激增及 miss 日志大量出现。',
            '临时处理方案包括增加该 key 的缓存时间以及手动刷新缓存。',
            '预防策略涉及设置更长的过期时间，并使用分布式锁防止并发更新。',
            '架构层面可实现本地缓存作为后备，或采用多级缓存方案。',
        ],
    ),
    EvalSample(
        question="NetworkHighLatency 中 DNS 解析异常有哪些特征？",
        ground_truths=[
            "DNS 解析异常的特征包括 DNS 解析时间长、日志中有 DNS 解析失败记录。",
            "该问题也可能表现为特定域名访问慢。",
            "处理时需要检查 DNS 服务器配置是否正确，并检查 DNS 缓存设置是否合理。",
            "优化方式包括配置多个 DNS 服务器作为备用，并在应用层增加 DNS 解析结果缓存。",
        ],
        relevant_docs=["network_high_latency.md"],
        category="exact_keyword",
        gen_expected_facts=[
            'DNS 解析异常的特征之一包括 DNS 解析时间长。',
            'DNS 解析异常的特征包括日志中有 DNS 解析失败记录。',
            'DNS 解析异常也可能表现为特定域名访问速度慢。',
            '处理该问题需要检查 DNS 服务器配置及缓存设置是否合理。',
        ],
    ),
    EvalSample(
        question="线上突然一堆 cache miss，数据库也变慢了，先怎么止血？",
        ground_truths=[
            "这种现象可能是缓存雪崩或缓存穿透，典型影响是缓存命中率下降、数据库查询量激增和响应时间变慢。",
            "应先启用限流，保护数据库和核心业务不被进一步压垮。",
            "应尽快预热热点数据、手动刷新缓存，并优化缓存过期时间。",
            "如果错误率已经升高，应结合熔断或降级策略返回默认值或缓存数据。",
        ],
        relevant_docs=["cache_avalanche.md", "slow_response.md", "api_error_rate_spike.md"],
        category="colloquial",
        gen_expected_facts=[
            '缓存雪崩或缓存穿透现象会导致缓存命中率下降、数据库查询量激增和响应时间变慢。',
            '发生此类故障时应先启用限流，保护数据库和核心业务不被进一步压垮。',
            '恢复阶段需预热热点数据、手动刷新缓存，并优化缓存过期时间。',
            '错误率升高时应结合熔断或降级策略，返回默认值或缓存数据。',
        ],
    ),
    EvalSample(
        question="服务启动不了，日志里有配置加载错误，我该优先看什么？",
        ground_truths=[
            "配置错误的典型特征包括最近有配置变更、日志中有配置加载错误、环境变量缺失或启动参数错误。",
            "应优先回滚到上一个正确配置，并检查配置文件语法和环境变量。",
            "修复时需要对比正确配置文件，修正错误配置项并重新加载配置。",
            "如果服务不可用，应在 5 分钟内快速判断能否修复，不能快速修复则执行回滚。",
        ],
        relevant_docs=["service_unavailable.md"],
        category="colloquial",
        gen_expected_facts=[
            '配置错误的典型特征包括最近有配置变更、日志中有配置加载错误、环境变量缺失或启动参数错误。',
            '配置加载错误的处理优先级是回滚到上一个正确配置，并检查配置文件语法和环境变量。',
            '配置修复流程包括对比正确配置文件、修正错误配置项并重新加载配置。',
            '服务不可用时应在 5 分钟内快速判断能否修复，不能快速修复则执行回滚。',
        ],
    ),
    EvalSample(
        question="消费者 lag 越来越大，但生产者只是正常发消息，可能是哪边的问题？",
        ground_truths=[
            "如果生产者没有明显流量突增，应优先怀疑消费者处理能力不足、消费者配置不当或系统资源不足。",
            "消费者处理能力不足表现为实例数量不足、单个消费者处理速度慢、处理超时或异常。",
            "消费者配置不当可能是线程数过少、批次大小不合理或消费者组配置不合理。",
            "应检查消费者处理速率、CPU/内存使用率、线程数和批次大小，并按需扩容消费者实例。",
        ],
        relevant_docs=["message_queue_backlog.md"],
        category="colloquial",
        gen_expected_facts=[
            '若生产者无流量突增，消费者 lag 增大通常源于消费者处理能力不足、配置不当或系统资源不足。',
            '消费者处理能力不足表现为实例数量不足、单个处理速度慢、处理超时或出现异常。',
            '消费者配置不当可能由线程数过少、批次大小不合理或消费者组配置不合理导致。',
            '消费者处理速率下降或 CPU 内存使用率过高是系统资源不足或处理能力瓶颈的直接体现。',
        ],
    ),
    EvalSample(
        question="接口慢得离谱但没有报错，通常要看哪些方向？",
        ground_truths=[
            "响应慢但没有明显错误可能来自数据库慢查询、外部 API 调用超时、代码性能问题、缓存失效或系统资源不足。",
            "数据库慢查询需要查看慢查询日志、数据库 CPU、连接池状态和 SQL 执行计划。",
            "外部 API 或网络问题需要关注第三方调用记录、网络超时错误和下游服务响应情况。",
            "系统资源不足时应检查 CPU、内存、磁盘 IO、网络带宽和系统负载。",
        ],
        relevant_docs=["slow_response.md", "database_connection_pool_exhaustion.md", "network_high_latency.md"],
        category="colloquial",
        gen_expected_facts=[
            '接口响应慢但无报错通常由数据库慢查询、外部 API 调用超时、代码性能问题、缓存失效或系统资源不足导致。',
            '数据库慢查询关联的关键指标包括慢查询日志、数据库 CPU、连接池状态和 SQL 执行计划。',
            '外部 API 或网络问题涉及的关键信息包括第三方调用记录、网络超时错误和下游服务响应情况。',
            '系统资源不足涉及的检查指标包括 CPU、内存、磁盘 IO、网络带宽和系统负载。',
        ],
    ),
    EvalSample(
        question="磁盘爆了以后，怎么快速判断是日志、临时文件还是 Docker 占用？",
        ground_truths=[
            "日志文件过大通常表现为 /var/log 占用大量空间、应用日志持续增长、没有日志轮转或日志级别为 DEBUG。",
            "临时文件堆积通常表现为 /tmp 或文件上传临时目录占用大量空间，大量临时文件未清理。",
            "Docker 占用通常表现为未使用镜像、停止容器、未使用卷或容器日志占用大量空间。",
            "快速处理可查找最大文件和目录，并分别清理日志、临时文件或 Docker 未使用资源。",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="colloquial",
        gen_expected_facts=[
            '日志文件过大通常表现为/var/log 目录占用大量空间，应用日志持续增长且未配置日志轮转或级别为 DEBUG。',
            '临时文件堆积通常表现为/tmp 或文件上传临时目录占用大量空间，存在大量未清理的临时文件。',
            'Docker 占用通常由未使用镜像、停止容器、未使用卷或容器日志占用大量空间导致。',
            '磁盘空间排查可通过查找最大文件和目录，区分日志、临时文件或 Docker 未使用资源。',
        ],
    ),
    # v1.4.0 定向补充：固定加入 dev，扩充跨文档覆盖评估且不改变冻结 test。
    EvalSample(
        question="API 5xx 和网络延迟同时升高，怎么串联排查？",
        ground_truths=[
            "先在同一时间窗口核对 API 错误日志、错误类型和受影响接口。",
            "再检查网络延迟、超时、DNS 和下游服务响应，判断网络是否为 5xx 的诱因。",
            "同时检查 CPU 和内存，排除资源瓶颈造成的共同症状。",
        ],
        relevant_docs=["api_error_rate_spike.md", "network_high_latency.md"],
        category="cross_doc",
        relevant_sections=[
            "api_error_rate_spike.md::排查步骤",
            "network_high_latency.md::排查步骤",
        ],
        fact_sources=[
            ["api_error_rate_spike.md::排查步骤"],
            ["network_high_latency.md::排查步骤"],
            [
                "api_error_rate_spike.md::排查步骤",
                "network_high_latency.md::排查步骤",
            ],
        ],
        split_hint="dev",
    ),
    EvalSample(
        question="缓存雪崩后连接池也耗尽、接口越来越慢，排查链路是什么？",
        ground_truths=[
            "先确认缓存命中率下降与数据库查询激增是否发生在同一时间窗口。",
            "检查数据库连接池活跃、空闲和等待连接数，并排查慢查询或连接泄漏。",
            "结合接口耗时定位缓存失效、数据库瓶颈与慢响应之间的因果链。",
        ],
        relevant_docs=[
            "cache_avalanche.md",
            "database_connection_pool_exhaustion.md",
            "slow_response.md",
        ],
        category="cross_doc",
        relevant_sections=[
            "cache_avalanche.md::排查步骤",
            "database_connection_pool_exhaustion.md::排查步骤",
            "slow_response.md::排查步骤",
        ],
        fact_sources=[
            ["cache_avalanche.md::排查步骤"],
            ["database_connection_pool_exhaustion.md::排查步骤"],
            ["slow_response.md::排查步骤"],
        ],
        split_hint="dev",
    ),
    EvalSample(
        question="Pod OOMKilled 后服务不可用，要把哪些证据串起来？",
        ground_truths=[
            "核对容器退出原因、重启次数和 OOMKilled 时间。",
            "检查进程或容器内存趋势、限制配置和是否存在内存泄漏。",
            "将 OOM 时间与服务健康检查、依赖状态和不可用日志对齐。",
        ],
        relevant_docs=[
            "container_oom_killed.md",
            "memory_high_usage.md",
            "service_unavailable.md",
        ],
        category="cross_doc",
        relevant_sections=[
            "container_oom_killed.md::排查步骤",
            "memory_high_usage.md::排查步骤",
            "service_unavailable.md::排查步骤",
        ],
        fact_sources=[
            ["container_oom_killed.md::排查步骤"],
            ["memory_high_usage.md::排查步骤"],
            ["service_unavailable.md::排查步骤"],
        ],
        split_hint="dev",
    ),
    EvalSample(
        question="消息积压时 CPU 和内存也报警，如何判断是消费者还是资源瓶颈？",
        ground_truths=[
            "先检查消费者处理速率、实例数、线程数和批次大小。",
            "对齐 CPU 高占用进程与消息消费服务，判断计算资源是否限制消费能力。",
            "检查内存增长、OOM 或 GC 情况，确认是否因内存压力拖慢消费者。",
        ],
        relevant_docs=[
            "message_queue_backlog.md",
            "cpu_high_usage.md",
            "memory_high_usage.md",
        ],
        category="cross_doc",
        relevant_sections=[
            "message_queue_backlog.md::排查步骤",
            "cpu_high_usage.md::排查步骤",
            "memory_high_usage.md::排查步骤",
        ],
        fact_sources=[
            ["message_queue_backlog.md::排查步骤"],
            ["cpu_high_usage.md::排查步骤"],
            ["memory_high_usage.md::排查步骤"],
        ],
        split_hint="dev",
    ),
    EvalSample(
        question="接口变慢且数据库连接等待和网络延迟同时升高，先查哪条链路？",
        ground_truths=[
            "先用接口耗时和慢请求日志确定受影响范围与时间窗口。",
            "检查连接池等待、活跃连接和数据库性能，判断是否为数据库侧瓶颈。",
            "检查网络超时、DNS 和下游响应，并与接口慢请求时间对齐。",
        ],
        relevant_docs=[
            "slow_response.md",
            "database_connection_pool_exhaustion.md",
            "network_high_latency.md",
        ],
        category="cross_doc",
        relevant_sections=[
            "slow_response.md::排查步骤",
            "database_connection_pool_exhaustion.md::排查步骤",
            "network_high_latency.md::排查步骤",
        ],
        fact_sources=[
            ["slow_response.md::排查步骤"],
            ["database_connection_pool_exhaustion.md::排查步骤"],
            ["network_high_latency.md::排查步骤"],
        ],
        split_hint="dev",
    ),
    EvalSample(
        question="日志写满磁盘后服务启动失败，如何从磁盘一路排到服务恢复？",
        ground_truths=[
            "先确认磁盘和 inode 使用率，并定位持续增长的日志文件。",
            "安全清理或轮转日志释放空间，避免直接删除仍被进程占用的文件。",
            "随后检查服务启动日志、配置和依赖状态，重启后验证健康检查。",
        ],
        relevant_docs=["disk_high_usage.md", "service_unavailable.md"],
        category="cross_doc",
        relevant_sections=[
            "disk_high_usage.md::排查步骤",
            "disk_high_usage.md::紧急处理措施",
            "service_unavailable.md::排查步骤",
            "service_unavailable.md::验证步骤",
        ],
        fact_sources=[
            ["disk_high_usage.md::排查步骤"],
            ["disk_high_usage.md::紧急处理措施"],
            [
                "service_unavailable.md::排查步骤",
                "service_unavailable.md::验证步骤",
            ],
        ],
        split_hint="dev",
    ),
    EvalSample(
        question="流量突增后 API 报错、缓存 miss 和连接池等待同时出现，如何联合止血？",
        ground_truths=[
            "先限流或降级，降低错误请求对核心业务和下游数据库的冲击。",
            "预热热点缓存并调整过期策略，减少 cache miss 导致的数据库压力。",
            "检查连接池等待与慢查询，必要时临时扩容并持续验证错误率和延迟。",
        ],
        relevant_docs=[
            "api_error_rate_spike.md",
            "cache_avalanche.md",
            "database_connection_pool_exhaustion.md",
        ],
        category="cross_doc",
        relevant_sections=[
            "api_error_rate_spike.md::紧急处理措施",
            "cache_avalanche.md::紧急处理措施",
            "database_connection_pool_exhaustion.md::紧急处理措施",
        ],
        fact_sources=[
            ["api_error_rate_spike.md::紧急处理措施"],
            ["cache_avalanche.md::紧急处理措施"],
            ["database_connection_pool_exhaustion.md::紧急处理措施"],
        ],
        split_hint="dev",
    ),
]


def _section_shingles(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    chinese = "".join(char for char in normalized if "\u4e00" <= char <= "\u9fff")
    shingles = {chinese[i : i + 2] for i in range(max(0, len(chinese) - 1))}
    shingles.update(re.findall(r"[a-z][a-z0-9_./-]+", normalized))
    return shingles


def _load_sop_sections() -> Dict[str, Dict[str, str]]:
    """Load the versioned SOP corpus used to produce v1.4.0 fact-source labels."""
    docs_dir = Path(__file__).resolve().parents[2] / "aiops-docs"
    result: Dict[str, Dict[str, str]] = {}
    for path in docs_dir.glob("*.md"):
        content = path.read_text(encoding="utf-8")
        matches = list(re.finditer(r"^##\s+(.+?)\s*$", content, flags=re.MULTILINE))
        sections: Dict[str, str] = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            sections[match.group(1).strip()] = content[match.start() : end]
        result[path.name] = sections
    return result


def _apply_corpus_fact_source_audit() -> None:
    """Map every unlabeled fact to the best-matching H2 section in its source SOP."""
    corpus = _load_sop_sections()
    for sample in EVALUATION_DATASET:
        if sample.fact_sources:
            continue
        mapped_facts: List[List[str]] = []
        for fact in sample.ground_truths:
            fact_terms = _section_shingles(fact)
            candidates: List[Tuple[float, str]] = []
            for doc_name in sample.relevant_docs:
                for section_name, section_text in corpus.get(doc_name, {}).items():
                    section_terms = _section_shingles(section_text)
                    overlap = len(fact_terms & section_terms) / max(1, len(fact_terms))
                    candidates.append((overlap, f"{doc_name}::{section_name}"))
            if not candidates:
                raise ValueError(f"无法为事实定位 SOP section: {sample.question} / {fact}")
            mapped_facts.append([max(candidates, key=lambda item: item[0])[1]])
        sample.fact_sources = mapped_facts
        sample.relevant_sections = list(
            dict.fromkeys(section for sources in mapped_facts for section in sources)
        )


# dev 的高风险/跨文档标签经过人工复核；其余标签由固定 v1.4.0 SOP 语料匹配生成。
# 问题文本作为人工覆盖的稳定主键。
_REVIEWED_SECTION_LABELS: Dict[str, List[str]] = {
    "NetworkHighLatency 告警的触发条件是什么？": ["network_high_latency.md::告警名称"],
    "缓存命中率低但数据库 QPS 没升高，还能直接判定缓存雪崩吗？": [
        "cache_avalanche.md::常见原因分析"
    ],
    "遇到数据库连接池耗尽怎么办？": [
        "database_connection_pool_exhaustion.md::排查步骤",
        "database_connection_pool_exhaustion.md::紧急处理措施",
    ],
    "怎么处理 API 错误率突然升高的问题？": ["api_error_rate_spike.md::排查步骤"],
    "DatabaseConnectionPoolExhaustion 告警的触发条件是什么？": [
        "database_connection_pool_exhaustion.md::告警名称"
    ],
    "磁盘空间满会怎样把服务不可用和消息积压串起来？": [
        "disk_high_usage.md::常见原因分析",
        "service_unavailable.md::常见原因分析",
        "message_queue_backlog.md::常见原因分析",
    ],
    "日志把磁盘写满了，可以直接清空吗？怎么操作？": [
        "disk_high_usage.md::常见原因分析",
        "disk_high_usage.md::常用命令",
    ],
    "网络延迟过高时，怎么判断是链路问题还是服务自身问题？": [
        "network_high_latency.md::排查步骤",
        "network_high_latency.md::常见原因分析",
    ],
    "遇到CPU100%怎么紧急处理？限流还是扩容？": ["cpu_high_usage.md::紧急处理措施"],
    "服务完全不可用但监控只看到高内存和磁盘满，先判断哪些资源耗尽路径？": [
        "service_unavailable.md::常见原因分析",
        "memory_high_usage.md::排查步骤",
        "disk_high_usage.md::排查步骤",
    ],
    "数据库连接池满了以后接口响应慢，应该怎么联动排查？": [
        "database_connection_pool_exhaustion.md::排查步骤",
        "slow_response.md::常见原因分析",
    ],
    "CPU 告警后，我要怎么查是哪个进程在吃 CPU？": ["cpu_high_usage.md::排查步骤"],
    "OOM是不是因为缓存配置不对？": ["memory_high_usage.md::常见原因分析"],
    "接口慢得离谱但没有报错，通常要看哪些方向？": [
        "slow_response.md::常见原因分析",
        "database_connection_pool_exhaustion.md::常见原因分析",
        "network_high_latency.md::常见原因分析",
    ],
    "磁盘满了会影响消息队列吗？": ["message_queue_backlog.md::常见原因分析"],
    "消费者 lag 越来越大，但生产者只是正常发消息，可能是哪边的问题？": [
        "message_queue_backlog.md::常见原因分析"
    ],
    "Pod 被 OOMKilled 后自动重启成功，还需要继续分析吗？": [
        "container_oom_killed.md::验证步骤",
        "memory_high_usage.md::常见原因分析",
    ],
    "缓存击穿会导致接口变慢吗？": ["slow_response.md::常见原因分析"],
}


def _apply_reviewed_fact_sources() -> None:
    """Attach reviewed section evidence to every fact in reviewed samples."""
    for sample in EVALUATION_DATASET:
        sections = _REVIEWED_SECTION_LABELS.get(sample.question, [])
        if not sections:
            continue
        sample.relevant_sections = list(sections)
        sample.fact_sources = [list(sections) for _ in sample.ground_truths]


_apply_corpus_fact_source_audit()
_apply_reviewed_fact_sources()


def get_eval_dataset():
    """将 EvalSample 列表转换为 RAGAs 评估所需的 Dataset 格式

    RAGAs 期望字段：
      - question      (str)
      - ground_truth  (str) — 多个 ground_truths 通过 "\n" 拼接为单个字符串

    Returns:
        datasets.Dataset: 包含 question, ground_truth, category, relevant_docs,
                           gen_expected_facts, gen_forbidden_content, gen_min_length 的 Dataset
    """
    from datasets import Dataset

    questions = []
    ground_truths = []
    categories = []
    relevant_docs_list = []
    gen_expected_facts_list = []
    gen_forbidden_content_list = []
    gen_min_length_list = []
    relevant_sections_list = []
    fact_sources_list = []

    for s in EVALUATION_DATASET:
        questions.append(s.question)
        ground_truths.append("\n".join(s.ground_truths))
        categories.append(s.category)
        relevant_docs_list.append(s.relevant_docs)
        # 生成评估字段：若未标注 gen_expected_facts 则 fallback 到 ground_truths
        gen_expected_facts_list.append(
            s.gen_expected_facts if s.gen_expected_facts else s.ground_truths
        )
        gen_forbidden_content_list.append(s.gen_forbidden_content)
        gen_min_length_list.append(s.gen_min_length)
        relevant_sections_list.append(s.relevant_sections)
        fact_sources_list.append(s.fact_sources)

    return Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "category": categories,
        "relevant_docs": relevant_docs_list,
        "gen_expected_facts": gen_expected_facts_list,
        "gen_forbidden_content": gen_forbidden_content_list,
        "gen_min_length": gen_min_length_list,
        "relevant_sections": relevant_sections_list,
        "fact_sources": fact_sources_list,
    })
