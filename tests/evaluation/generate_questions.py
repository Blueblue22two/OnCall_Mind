"""LLM 辅助评估问题生成脚本

读取知识库文档，按 4 种问题类型生成候选评估问题，输出 JSON 供人工审核。

用法（在项目根目录执行）:

  python -m tests.evaluation.generate_questions                           # 为所有文档生成
  python -m tests.evaluation.generate_questions --doc cpu_high_usage.md   # 指定文档
  python -m tests.evaluation.generate_questions --category colloquial     # 只生成某一类
  python -m tests.evaluation.generate_questions --dry-run                 # 预览 prompt
  python -m tests.evaluation.generate_questions --output candidates.json  # 指定输出

工作流程:
  1. LLM 读取每篇文档，生成 6-8 个候选问题（4 种类型混合）
  2. LLM 为每个问题生成初步 ground_truths 和 relevant_docs
  3. 输出到 JSON 文件，人工审核后导入 rag_testset.py

问题类型:
  - exact_keyword : 精确关键词/技术术语查询（如"X 告警的触发条件是什么？"）
  - colloquial    : 口语化改写（如"X 出问题了怎么办？"）
  - cross_doc     : 跨文档综合查询（如"哪些告警会导致服务不可用？"）
  - edge_case     : 边界/噪声查询（部分相关但不完全匹配）
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是运维领域的评估数据集标注专家。你的任务是为知识库文档生成检索评估问题。

## 问题类型要求

请为每篇文档生成 6-8 个问题，覆盖以下 4 种类型：

1. **exact_keyword（精确关键词型）** — 1-2 个
   - 直接提及文档中的技术术语、告警名、工具名
   - 示例: "HighCPUUsage 告警的触发条件是什么？"
   - 示例: "query_cpu_metrics 工具需要传哪些参数？"

2. **colloquial（口语化改写型）** — 2-3 个
   - 用日常/运维口语表达技术问题，不出现精确术语
   - 示例: "CPU 飙高怎么办？"
   - 示例: "接口变慢了该从哪查起？"

3. **cross_doc（跨文档综合型）** — 1 个
   - 需要综合多篇文档的知识才能完整回答
   - 如果只有单篇文档，基于文档中提到的"相关告警"生成
   - 示例: "哪些告警会导致服务不可用？"

4. **edge_case（边界/噪声型）** — 1 个
   - 问题与文档部分相关但非直接对应
   - 术语有歧义，或问题范围比文档覆盖的更广
   - 示例: "磁盘满了会影响 CPU 吗？"

## 输出格式

严格返回 JSON 数组，每个元素包含:

{
  "question": "问题文本（字符串）",
  "category": "exact_keyword | colloquial | cross_doc | edge_case",
  "ground_truths": ["答案要点1", "答案要点2", ...],   // 3-5 个要点，每个 1-2 句话
  "relevant_docs": ["文件名.md"]                       // 答案来源文档文件名
}

## ground_truths 粒度要求
- 每个要点 1-2 句话（20-60 字）
- 覆盖"是什么 → 为什么 → 怎么办"
- 包含具体的工具名、命令或参数

只返回 JSON 数组，不要任何额外的文字说明。"""


def _build_user_prompt(doc_name: str, doc_content: str, all_doc_names: list[str]) -> str:
    """构建单文档的问题生成 prompt"""
    doc_list = "\n".join(f"  - {n}" for n in sorted(all_doc_names))
    return f"""## 目标文档

文件名: {doc_name}

文档内容:
---
{doc_content}
---

## 知识库全部文档列表（用于生成 cross_doc 问题时参考）

{doc_list}

请为 {doc_name} 生成 6-8 个候选评估问题，覆盖 4 种类型。
注意：生成的 relevant_docs 字段只能使用上述文档列表中实际存在的文件名。"""


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------
def _get_llm():
    """延迟初始化 LLM"""
    from langchain_community.chat_models.tongyi import ChatTongyi

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.config import config

    kwargs = {
        "model": config.dashscope_model,
        "temperature": 0.4,
        "dashscope_api_key": config.dashscope_api_key,
    }
    api_base = os.environ.get("DASHSCOPE_API_BASE", "")
    if api_base:
        kwargs["dashscope_api_base"] = api_base
    return ChatTongyi(**kwargs)


def _generate_for_doc(
    doc_path: Path,
    all_doc_names: list[str],
    dry_run: bool = False,
) -> list[dict] | None:
    """为单个文档生成候选问题"""
    doc_name = doc_path.name
    doc_content = doc_path.read_text(encoding="utf-8")

    if dry_run:
        print(f"\n  DRY RUN — 文档: {doc_name} ({len(doc_content)} chars)")
        print(f"  Prompt 长度: ~{len(_build_user_prompt(doc_name, doc_content, all_doc_names))} chars")
        return None

    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(doc_name, doc_content, all_doc_names)),
    ]
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    # 提取 JSON（LLM 可能在前后加了 ```json ... ``` 或其他文字）
    return _parse_json_response(raw, doc_name)


def _parse_json_response(raw: str, doc_name: str) -> list[dict] | None:
    """从 LLM 响应中提取 JSON 数组"""
    # 尝试直接解析
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 代码块
    import re

    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 [ 到最后一个 ]
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    print(f"  警告: {doc_name} 的响应无法解析为 JSON，原始响应前 200 字符: {raw[:200]}")
    return None


# ---------------------------------------------------------------------------
# 后处理验证
# ---------------------------------------------------------------------------
def _validate_questions(questions: list[dict], doc_name: str, all_doc_names: set[str]) -> list[dict]:
    """验证并清洗生成的候选问题"""
    valid = []
    seen_questions = set()

    for i, q in enumerate(questions):
        prefix = f"  [{doc_name}#{i}]"

        # 必填字段检查
        if not q.get("question") or not isinstance(q["question"], str):
            print(f"{prefix} 缺少 question 字段，跳过")
            continue
        if not q.get("ground_truths") or not isinstance(q["ground_truths"], list):
            print(f"{prefix} 缺少 ground_truths 字段，跳过")
            continue
        if not q.get("relevant_docs") or not isinstance(q["relevant_docs"], list):
            print(f"{prefix} 缺少 relevant_docs 字段，跳过")
            continue

        # 去重（同一文档内）
        q_text = q["question"].strip()
        if q_text in seen_questions:
            print(f"{prefix} 问题重复: {q_text[:50]}...，跳过")
            continue
        seen_questions.add(q_text)

        # category 规范化
        valid_categories = {"exact_keyword", "colloquial", "cross_doc", "edge_case"}
        if q.get("category", "") not in valid_categories:
            q["category"] = "exact_keyword"  # 默认

        # relevant_docs 文件名合法性检查
        q["relevant_docs"] = [
            d for d in q["relevant_docs"]
            if d in all_doc_names or d.endswith(".md")
        ]

        # ground_truths 清洗（去空）
        q["ground_truths"] = [g.strip() for g in q["ground_truths"] if g and g.strip()]

        if len(q["ground_truths"]) < 2:
            print(f"{prefix} ground_truths 过滤后不足 2 条，跳过")
            continue

        valid.append(q)

    return valid


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def generate_questions(
    docs_dir: str = "aiops-docs",
    doc_filter: Optional[list[str]] = None,
    exclude_docs: Optional[list[str]] = None,
    category_filter: Optional[str] = None,
    output_path: Optional[str] = None,
    dry_run: bool = False,
):
    """为知识库文档生成候选评估问题

    Args:
        docs_dir: 文档目录
        doc_filter: 可选，只处理指定文档列表（如 ["cpu_high_usage.md"]）
        exclude_docs: 可选，排除指定文档列表（如跳过已有问题的文档）
        category_filter: 可选，只输出指定类型的问题
        output_path: JSON 输出路径
        dry_run: 仅预览 prompt，不调用 LLM
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"错误: 文档目录不存在: {docs_dir}")
        sys.exit(1)

    # 收集目标文档
    md_files = sorted(docs_path.glob("*.md"))
    # 排除 _generated 后缀的待审核文件
    md_files = [f for f in md_files if "_generated" not in f.name]

    if doc_filter:
        filter_set = set(doc_filter)
        md_files = [f for f in md_files if f.name in filter_set]
        if not md_files:
            print(f"错误: 未找到匹配的文档: {doc_filter}")
            sys.exit(1)

    if exclude_docs:
        exclude_set = set(exclude_docs)
        before = len(md_files)
        md_files = [f for f in md_files if f.name not in exclude_set]
        print(f"排除 {before - len(md_files)} 篇已指定排除的文档")

    all_doc_names = [f.name for f in md_files]
    print(f"目标文档: {len(md_files)} 篇")
    print(f"问题类型: {category_filter or '全部'}")
    print()

    all_questions = []
    for i, doc_path in enumerate(md_files, 1):
        print(f"[{i}/{len(md_files)}] {doc_path.name}")
        questions = _generate_for_doc(doc_path, all_doc_names, dry_run)

        if dry_run:
            continue

        if questions is None:
            print(f"  跳过（生成失败）")
            continue

        valid = _validate_questions(questions, doc_path.name, set(all_doc_names))
        all_questions.extend(valid)

        # 统计类型分布
        cat_dist = {}
        for q in valid:
            cat_dist[q["category"]] = cat_dist.get(q["category"], 0) + 1
        cat_summary = ", ".join(f"{k}={v}" for k, v in sorted(cat_dist.items()))
        print(f"  生成 {len(valid)} 个候选问题 ({cat_summary})")

    if dry_run:
        print("\nDRY RUN 完成，未调用 LLM。")
        return

    # 按 category 过滤
    if category_filter:
        all_questions = [q for q in all_questions if q.get("category") == category_filter]
        print(f"\n按 category={category_filter} 过滤后: {len(all_questions)} 个问题")

    # 汇总统计
    print(f"\n总计生成: {len(all_questions)} 个候选问题")
    cat_counts = {}
    for q in all_questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    print(f"类型分布: {json.dumps(cat_counts, ensure_ascii=False)}")

    doc_counts = {}
    for q in all_questions:
        for d in q.get("relevant_docs", []):
            doc_counts[d] = doc_counts.get(d, 0) + 1
    print(f"文档覆盖: {json.dumps(doc_counts, ensure_ascii=False)}")

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_path or f"reports/candidate_questions_{ts}.json"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "_metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_questions": len(all_questions),
            "category_distribution": cat_counts,
            "doc_coverage": doc_counts,
            "status": "PENDING_REVIEW",
        },
        "questions": all_questions,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n候选问题已保存: {out_path}")
    print("下一步: 人工审核后导入 rag_testset.py 的 EVALUATION_DATASET")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 辅助评估问题生成")
    parser.add_argument(
        "--doc", "-d", type=str, nargs="*", default=None,
        help="只处理指定文档，可多个（如 -d cpu_high_usage.md disk_high_usage.md）",
    )
    parser.add_argument(
        "--exclude", "-x", type=str, nargs="*", default=None,
        help="排除指定文档（如 -x cpu_high_usage.md 跳过已有问题的文档）",
    )
    parser.add_argument(
        "--category", "-c", type=str, default=None,
        choices=["exact_keyword", "colloquial", "cross_doc", "edge_case"],
        help="只输出指定类型的问题",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="JSON 输出路径（默认: reports/candidate_questions_{timestamp}.json）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览 prompt，不调用 LLM",
    )
    parser.add_argument(
        "--docs-dir", type=str, default="aiops-docs",
        help="文档目录（默认: aiops-docs）",
    )
    args = parser.parse_args()

    generate_questions(
        docs_dir=args.docs_dir,
        doc_filter=args.doc,
        exclude_docs=args.exclude,
        category_filter=args.category,
        output_path=args.output,
        dry_run=args.dry_run,
    )
