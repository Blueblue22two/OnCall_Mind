"""将审核通过的候选问题导入 rag_testset.py

用法:
  # 预览（不写入，只打印将要导入的问题）
  python -m tests.evaluation.import_questions --dry-run

  # 导入全部候选问题
  python -m tests.evaluation.import_questions

  # 只导入指定 status 的问题
  python -m tests.evaluation.import_questions --status approved

  # 指定输入文件
  python -m tests.evaluation.import_questions --input reports/candidate_questions_20260522_205717.json

审核流程:
  1. 打开 candidate_questions JSON，审核每个问题
  2. 通过的问题: 设置 "status": "approved"
  3. 需修改的问题: 直接修改 JSON 中的 question/ground_truths/relevant_docs，设置 "status": "approved"
  4. 淘汰的问题: 设置 "status": "rejected"（或直接删除）
  5. 运行此脚本导入 approved 的问题
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _format_eval_sample(q: dict) -> str:
    """将单个问题格式化为 EvalSample Python 代码"""
    question = json.dumps(q["question"], ensure_ascii=False)
    ground_truths = ",\n".join(
        f'            {json.dumps(g, ensure_ascii=False)}' for g in q["ground_truths"]
    )
    relevant_docs = json.dumps(q.get("relevant_docs", []), ensure_ascii=False)
    category = q.get("category", "exact_keyword")

    return f'''    EvalSample(
        question={question},
        ground_truths=[
{ground_truths},
        ],
        relevant_docs={relevant_docs},
        category="{category}",
    ),'''


def _bump_version(content: str) -> str:
    """递增 DATASET_VERSION 的补丁版本号"""
    match = re.search(r'DATASET_VERSION\s*=\s*"(\d+)\.(\d+)\.(\d+)"', content)
    if match:
        major, minor, patch = int(match[1]), int(match[2]), int(match[3])
        new_version = f'{major}.{minor}.{patch + 1}'
        return content.replace(match.group(0), f'DATASET_VERSION = "{new_version}"')
    return content


def _find_insertion_point(lines: list[str]) -> int:
    """找到 EVALUATION_DATASET 列表的关闭 `]` 行号（0-indexed）"""
    in_dataset = False
    bracket_depth = 0
    for i, line in enumerate(lines):
        if "EVALUATION_DATASET" in line and "[" in line:
            in_dataset = True
            bracket_depth = 1
            continue
        if in_dataset:
            bracket_depth += line.count("[") - line.count("]")
            if bracket_depth == 0:
                return i  # 这是 `]` 行
    return -1


def import_questions(
    input_path: str,
    status_filter: Optional[str] = None,
    dry_run: bool = False,
):
    """导入审核通过的候选问题

    Args:
        input_path: 候选问题 JSON 文件路径
        status_filter: 可选，只导入指定 status 的问题（如 "approved"）
        dry_run: 仅预览，不实际写入
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"错误: 文件不存在: {input_path}")
        sys.exit(1)

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    if not questions:
        print("错误: JSON 中没有 questions 数据")
        sys.exit(1)

    # 过滤
    if status_filter:
        questions = [q for q in questions if q.get("status") == status_filter]
        print(f"按 status='{status_filter}' 过滤: {len(questions)} 个问题")
    else:
        # 排除明确 rejected 的
        questions = [q for q in questions if q.get("status") != "rejected"]
        print(f"排除 rejected 后: {len(questions)} 个问题")

    if not questions:
        print("没有需要导入的问题")
        return

    # 格式化
    new_entries = []
    for q in questions:
        # 清洗: 去除幻觉的文档引用（不存在于 aiops-docs/ 中的文件名）
        q["relevant_docs"] = [
            d for d in q.get("relevant_docs", [])
            if d.endswith(".md") and not d.startswith("high")  # 粗略过滤明显的幻觉
        ]

        code = _format_eval_sample(q)
        new_entries.append(code)

    # 添加批次标记
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    source = input_file.name
    header = f'''
    # -------------------------------------------------------
    # 批次: {source}（导入于 {ts}）
    # -------------------------------------------------------
'''

    if dry_run:
        print(f"\n{'='*60}")
        print(f"  DRY RUN — 将导入 {len(new_entries)} 个问题")
        print(f"{'='*60}")
        print(header)
        for entry in new_entries:
            print(entry)
        return

    # 读取 rag_testset.py
    testset_path = Path(__file__).resolve().parent / "rag_testset.py"
    content = testset_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # 找到插入点
    insert_at = _find_insertion_point(lines)
    if insert_at < 0:
        print("错误: 无法找到 EVALUATION_DATASET 的结束位置")
        sys.exit(1)

    # 插入新条目（在 `]` 行之前）
    indent = "    "  # EvalSample 缩进 4 空格
    insert_lines = [header.rstrip()] + new_entries
    for entry_line in insert_lines:
        lines.insert(insert_at, entry_line)
        insert_at += 1

    new_content = "\n".join(lines)

    # 递增版本号
    new_content = _bump_version(new_content)

    # 写回
    testset_path.write_text(new_content, encoding="utf-8")
    print(f"\n✅ 已导入 {len(new_entries)} 个问题到 {testset_path}")
    print(f"   来源: {input_path}")
    print(f"   下一步: 运行 python -m tests.evaluation.validate_dataset 验证")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导入审核通过的候选问题到 rag_testset.py")
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="候选问题 JSON 文件路径（默认: 自动查找 reports/ 下最新的 candidate_questions_*.json）",
    )
    parser.add_argument(
        "--status", "-s", type=str, default=None,
        help="只导入指定 status 的问题（如 --status approved）",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="仅预览，不实际写入",
    )
    args = parser.parse_args()

    # 自动查找最新的候选文件
    input_path = args.input
    if not input_path:
        reports_dir = Path("reports")
        candidates = sorted(
            reports_dir.glob("candidate_questions_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("错误: 未找到候选问题文件。请用 --input 指定路径")
            sys.exit(1)
        input_path = str(candidates[0])
        print(f"自动选择: {input_path}")

    import_questions(
        input_path=input_path,
        status_filter=args.status,
        dry_run=args.dry_run,
    )
