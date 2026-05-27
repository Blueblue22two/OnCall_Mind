"""评估数据集质量检查脚本

对 rag_testset.py 中的 EVALUATION_DATASET 执行高级质量检查：

  1. 去重检测 — 基于 embedding 余弦相似度检测语义重复问题
  2. 覆盖率分析 — 每文档问题数、category 分布、场景 taxonomy 覆盖
  3. 交叉引用检查 — relevant_docs 引用的文件是否存在
  4. 缺口报告 — 对比场景 taxonomy，指出未覆盖的能力域

用法（在项目根目录执行）:

  python -m tests.evaluation.validate_dataset                     # 完整检查
  python -m tests.evaluation.validate_dataset --similarity 0.85   # 自定义相似度阈值
  python -m tests.evaluation.validate_dataset --no-embedding      # 跳过 embedding（快速模式）
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 场景 Taxonomy（与 generate_docs.py 保持一致）
# ---------------------------------------------------------------------------
SCENARIO_TAXONOMY = {
    "resource": {
        "label": "资源告警",
        "docs": ["cpu_high_usage.md", "disk_high_usage.md", "memory_high_usage.md"],
    },
    "availability": {
        "label": "服务可用性",
        "docs": ["service_unavailable.md", "slow_response.md"],
    },
    "dependency": {
        "label": "依赖故障",
        "docs": [
            "database_connection_pool_exhaustion.md",
            "message_queue_backlog.md",
            "cache_avalanche.md",
        ],
    },
    "connectivity": {
        "label": "链路异常",
        "docs": ["network_high_latency.md", "api_error_rate_spike.md"],
    },
    "capacity_config": {
        "label": "容量/配置",
        "docs": ["container_oom_killed.md", "certificate_expiry.md"],
    },
}


def _load_aiops_docs(docs_dir: str = "aiops-docs") -> set[str]:
    """加载知识库文档文件名集合"""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return set()
    return {f.name for f in docs_path.glob("*.md")}


def _load_dataset():
    """加载评估数据集"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.evaluation.rag_testset import EVALUATION_DATASET

    return EVALUATION_DATASET


# ---------------------------------------------------------------------------
# 检查 1: 去重检测
# ---------------------------------------------------------------------------
def _compute_embedding_similarities(
    questions: list[str],
    threshold: float = 0.85,
) -> list[Tuple[int, int, float]]:
    """基于 embedding 计算语义相似度，返回高于阈值的相似对"""
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        print("  [跳过] embedding 去重需要 dashscope + scikit-learn")
        return []

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        # 尝试从 config 加载
        try:
            from app.config import config
            api_key = config.dashscope_api_key
        except Exception:
            pass
    if not api_key:
        print("  [跳过] 未配置 DASHSCOPE_API_KEY")
        return []

    print(f"  正在计算 {len(questions)} 个问题的 embedding...")
    embedder = DashScopeEmbeddings(model="text-embedding-v4", dashscope_api_key=api_key)
    embeddings = embedder.embed_documents(questions)
    emb_matrix = np.array(embeddings)

    sim_matrix = cosine_similarity(emb_matrix)

    # 找到阈值以上的非自身对 (i < j，避免重复)
    duplicates = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            if sim_matrix[i][j] >= threshold:
                duplicates.append((i, j, float(sim_matrix[i][j])))

    return duplicates


def _check_duplicates(dataset, threshold: float = 0.85, use_embedding: bool = True):
    """去重检测"""
    print("\n" + "=" * 70)
    print("  检查 1: 去重检测")
    print("=" * 70)

    questions = [s.question for s in dataset]
    print(f"  总问题数: {len(questions)}")

    # 文本完全重复检测（快速）
    seen = {}
    exact_dups = []
    for i, q in enumerate(questions):
        q_lower = q.lower().strip()
        if q_lower in seen:
            exact_dups.append((seen[q_lower], i))
        else:
            seen[q_lower] = i

    if exact_dups:
        print(f"\n  ⚠️  完全重复 ({len(exact_dups)} 对):")
        for i, j in exact_dups:
            print(f"    [{i}] ≈ [{j}]: {questions[i][:80]}...")
    else:
        print(f"  ✅ 无完全重复问题")

    # Embedding 语义相似度检测
    if use_embedding:
        semantic_dups = _compute_embedding_similarities(questions, threshold)
        if semantic_dups:
            print(f"\n  ⚠️  语义相似 > {threshold} ({len(semantic_dups)} 对):")
            semantic_dups.sort(key=lambda x: -x[2])
            for i, j, sim in semantic_dups[:10]:  # 最多显示 10 对
                print(f"    [{i}] ≈ [{j}] sim={sim:.3f}")
                print(f"      Q{i}: {questions[i][:80]}...")
                print(f"      Q{j}: {questions[j][:80]}...")
            if len(semantic_dups) > 10:
                print(f"    ... 及其他 {len(semantic_dups) - 10} 对")
        else:
            print(f"  ✅ 无语义重复 (阈值={threshold})")

    return len(exact_dups) + (len(semantic_dups) if use_embedding else 0)


# ---------------------------------------------------------------------------
# 检查 2: 覆盖率分析
# ---------------------------------------------------------------------------
def _check_coverage(dataset, existing_docs: set[str]):
    """覆盖率分析"""
    print("\n" + "=" * 70)
    print("  检查 2: 覆盖率分析")
    print("=" * 70)

    # 每文档问题数
    doc_question_count: Counter = Counter()
    for s in dataset:
        for doc in s.relevant_docs:
            doc_question_count[doc] += 1

    print(f"\n  文档 → 问题数:")
    for doc in sorted(doc_question_count):
        bar = "█" * min(doc_question_count[doc], 30)
        print(f"    {doc:<45} {doc_question_count[doc]:>3} {bar}")

    # 未被任何问题引用的文档
    referenced_docs = set(doc_question_count.keys())
    unreferenced = existing_docs - referenced_docs
    if unreferenced:
        print(f"\n  ⚠️  未被任何问题引用的文档:")
        for doc in sorted(unreferenced):
            print(f"    - {doc}")
    else:
        print(f"\n  ✅ 所有文档至少被 1 个问题引用")

    # 问题不足 5 个的文档
    under_covered = {doc: count for doc, count in doc_question_count.items() if count < 5}
    if under_covered:
        print(f"\n  ⚠️  问题数不足 5 的文档:")
        for doc, count in sorted(under_covered.items()):
            print(f"    - {doc}: {count} 个问题")
    else:
        print(f"\n  ✅ 所有文档至少有 5 个问题")

    # Category 分布
    cat_counts = Counter(s.category for s in dataset)
    total = len(dataset)
    print(f"\n  Category 分布 (目标: 每种 ≥ 10%):")
    for cat in ["exact_keyword", "colloquial", "cross_doc", "edge_case"]:
        count = cat_counts.get(cat, 0)
        pct = count / total * 100 if total > 0 else 0
        flag = "✅" if pct >= 10 else "⚠️"
        bar = "█" * int(pct / 2)
        print(f"    {flag} {cat:<20} {count:>3} ({pct:>5.1f}%) {bar}")

    # 场景 Taxonomy 覆盖
    print(f"\n  场景 Taxonomy 覆盖:")
    doc_set = set(doc_question_count.keys())
    for cat_key, cat_info in SCENARIO_TAXONOMY.items():
        cat_docs = set(cat_info["docs"])
        covered = cat_docs & doc_set
        missing = cat_docs - doc_set
        pct = len(covered) / len(cat_docs) * 100 if cat_docs else 100
        if missing:
            print(f"    ⚠️  {cat_info['label']}: {len(covered)}/{len(cat_docs)} 文档覆盖 ({pct:.0f}%)")
            for m in sorted(missing):
                print(f"        - 缺失: {m}")
        else:
            print(f"    ✅ {cat_info['label']}: {len(covered)}/{len(cat_docs)} 文档完全覆盖")


# ---------------------------------------------------------------------------
# 检查 3: 交叉引用验证
# ---------------------------------------------------------------------------
def _check_cross_references(dataset, existing_docs: set[str]):
    """验证 relevant_docs 引用的文件是否存在"""
    print("\n" + "=" * 70)
    print("  检查 3: 交叉引用验证")
    print("=" * 70)

    issues = []
    for i, s in enumerate(dataset):
        for doc in s.relevant_docs:
            if doc not in existing_docs:
                issues.append((i, doc))

    if issues:
        print(f"\n  ⚠️  无效引用 ({len(issues)} 处):")
        for i, doc in issues:
            print(f"    [{i}] 引用了不存在的文件: {doc}")
    else:
        print(f"\n  ✅ 所有 relevant_docs 引用有效 ({len(existing_docs)} 个可用文档)")

    return len(issues)


# ---------------------------------------------------------------------------
# 检查 4: 缺口报告
# ---------------------------------------------------------------------------
def _check_gaps(existing_docs: set[str]):
    """对比场景 taxonomy，报告缺口"""
    print("\n" + "=" * 70)
    print("  检查 4: 能力缺口报告")
    print("=" * 70)

    total_docs = sum(len(info["docs"]) for info in SCENARIO_TAXONOMY.values())
    covered = 0
    gaps = []

    for cat_key, cat_info in SCENARIO_TAXONOMY.items():
        cat_docs = set(cat_info["docs"])
        existing_in_cat = cat_docs & existing_docs
        missing_in_cat = cat_docs - existing_docs
        covered += len(existing_in_cat)

        if missing_in_cat:
            gaps.append((cat_info["label"], list(missing_in_cat)))

    pct = covered / total_docs * 100 if total_docs > 0 else 0
    print(f"\n  整体覆盖: {covered}/{total_docs} ({pct:.0f}%)")

    if gaps:
        print(f"\n  待补充的场景:")
        for label, docs in gaps:
            print(f"    [{label}]")
            for doc in docs:
                print(f"      - {doc}")
    else:
        print(f"\n  ✅ 所有场景已完全覆盖")

    print(f"\n  当前已有文档 ({len(existing_docs)}):")
    for doc in sorted(existing_docs):
        # 找到对应的分类
        cat_label = "未分类"
        for cat_key, cat_info in SCENARIO_TAXONOMY.items():
            if doc in cat_info["docs"]:
                cat_label = cat_info["label"]
                break
        if doc not in [d for info in SCENARIO_TAXONOMY.values() for d in info["docs"]]:
            cat_label = "已有文档（超出 taxonomy）"
        print(f"    - {doc:<45} [{cat_label}]")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def validate_dataset(
    similarity_threshold: float = 0.85,
    use_embedding: bool = True,
):
    """执行完整的数据集质量检查

    Returns:
        int: 发现的问题总数（0 = 全部通过）
    """
    print("=" * 70)
    print("  评估数据集质量检查")
    print("=" * 70)

    existing_docs = _load_aiops_docs()
    print(f"\n知识库文档: {len(existing_docs)} 篇")

    try:
        dataset = _load_dataset()
    except Exception as e:
        print(f"\n❌ 加载数据集失败: {e}")
        return 1

    print(f"评估问题: {len(dataset)} 条")

    issues = 0
    issues += _check_duplicates(dataset, similarity_threshold, use_embedding)
    _check_coverage(dataset, existing_docs)
    issues += _check_cross_references(dataset, existing_docs)
    _check_gaps(existing_docs)

    # 汇总
    print("\n" + "=" * 70)
    if issues == 0:
        print("  ✅ 所有检查通过")
    else:
        print(f"  ⚠️  发现 {issues} 个问题，请检查上述输出")
    print("=" * 70)
    print()

    return min(issues, 255)  # exit code 限制在 0-255


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估数据集质量检查")
    parser.add_argument(
        "--similarity", "-s", type=float, default=0.85,
        help="语义相似度阈值（默认: 0.85）",
    )
    parser.add_argument(
        "--no-embedding", action="store_true",
        help="跳过 embedding 去重检测（快速模式）",
    )
    args = parser.parse_args()

    sys.exit(validate_dataset(
        similarity_threshold=args.similarity,
        use_embedding=not args.no_embedding,
    ))
