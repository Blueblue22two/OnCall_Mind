"""将 backfill_facts.json 中的 gen_expected_facts 回填到 rag_testset.py

读取审核通过的 backfill JSON，按 index 匹配 EVALUATION_DATASET 中的样本，
将 gen_expected_facts 写入 rag_testset.py 对应位置。

用法（在项目根目录执行）：

  # 预览变更（不修改文件）
  python -m tests.evaluation.apply_backfill reports/backfill_facts.json --dry-run

  # 应用变更
  python -m tests.evaluation.apply_backfill reports/backfill_facts.json

  # 同时递增版本号
  python -m tests.evaluation.apply_backfill reports/backfill_facts.json --bump-version
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


def _parse_backfill_json(path: Path) -> dict[int, list[str]]:
    """解析回填 JSON，返回 {index: gen_expected_facts} 映射"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: dict[int, list[str]] = {}
    for sample in data.get("samples", []):
        idx = sample["index"]
        facts = sample.get("gen_expected_facts", [])
        if facts:
            mapping[idx] = facts
    return mapping


def _find_eval_sample_boundaries(lines: list[str]) -> list[tuple[int, int]]:
    """找出文件中每个 EvalSample( 的起止行号（0-indexed）。

    策略：从 'EvalSample(' 开始，到对应的 '),' 结束（匹配括号层级）。
    """
    boundaries = []
    in_sample = False
    start = 0
    depth = 0

    for i, line in enumerate(lines):
        if not in_sample and "EvalSample(" in line:
            in_sample = True
            start = i
            depth = line.count("(") - line.count(")")

            # 单行 EvalSample(...)
            if ")," in line or line.rstrip().endswith("),"):
                # 需要确认括号闭合
                if depth <= 0:
                    boundaries.append((start, i))
                    in_sample = False
                    depth = 0
            continue

        if in_sample:
            depth += line.count("(") - line.count(")")

            # 找到闭合
            stripped = line.rstrip()
            if depth <= 0 and (stripped.endswith("),") or stripped.endswith(")")):
                boundaries.append((start, i))
                in_sample = False
                depth = 0

    return boundaries


def _apply_backfill_to_file(
    lines: list[str],
    boundaries: list[tuple[int, int]],
    mapping: dict[int, list[str]],
    existing_dataset: list,
) -> tuple[list[str], int]:
    """将 gen_expected_facts 写入文件中对应 EvalSample 的位置。

    策略：在每个 EvalSample 块的 gen_min_length 行之后插入 gen_expected_facts。
    如果 gen_expected_facts 已存在，则替换。
    """
    updated = list(lines)
    applied = 0

    # 倒序处理，保证行号不偏移
    for sample_idx, (start, end) in reversed(list(enumerate(boundaries))):
        if sample_idx not in mapping:
            continue

        facts = mapping[sample_idx]
        facts_str = _format_facts_list(facts)

        block_lines = updated[start : end + 1]

        # 查找 gen_expected_facts 是否已存在
        gen_ef_line = None
        gen_ef_start = None
        gen_ef_end = None
        in_gen_ef = False
        gen_ef_depth = 0

        for local_i, line in enumerate(block_lines):
            abs_i = start + local_i

            if "gen_expected_facts=" in line and not in_gen_ef:
                gen_ef_line = abs_i
                gen_ef_start = abs_i

                # 检查是否值在同一行
                rest = line.split("gen_expected_facts=", 1)[1]
                if rest.strip().startswith("[") and "]" in rest and not rest.strip().startswith("[]"):
                    # 单行完整列表，替换
                    indent = line[: len(line) - len(line.lstrip())]
                    new_line = line.split("gen_expected_facts=")[0] + f"gen_expected_facts={facts_str},"
                    # 保持原有缩进
                    updated[abs_i] = new_line
                    applied += 1
                    break
                elif rest.strip() == "[]" or rest.strip().startswith("field"):
                    # 空列表，替换此行
                    indent = line[: len(line) - len(line.lstrip())]
                    new_line = line.split("gen_expected_facts=")[0] + f"gen_expected_facts={facts_str},"
                    updated[abs_i] = new_line
                    applied += 1
                    break
                else:
                    # 多行列表
                    in_gen_ef = True
                    gen_ef_depth = rest.count("[") - rest.count("]")
                    continue

            if in_gen_ef:
                gen_ef_depth += line.count("[") - line.count("]")
                if gen_ef_depth <= 0:
                    gen_ef_end = abs_i
                    # 替换 gen_ef_start 到 gen_ef_end 的行
                    indent = updated[gen_ef_start][: len(updated[gen_ef_start]) - len(updated[gen_ef_start].lstrip())]
                    updated[gen_ef_start] = f"{indent}gen_expected_facts={facts_str},"
                    # 删除中间行
                    for del_i in range(gen_ef_end, gen_ef_start, -1):
                        del updated[del_i]
                    applied += 1
                    break

        if gen_ef_line is not None:
            continue  # 已处理

        # gen_expected_facts 不存在，需要插入
        # 找到 gen_min_length 行，在其后插入
        insert_after = None
        for local_i, line in enumerate(reversed(block_lines)):
            abs_i = end - local_i
            if "gen_min_length=" in line:
                insert_after = abs_i
                break

        if insert_after is None:
            # 没有 gen_min_length，在最后一个 ) 之前插入
            insert_after = end - 1 if end > start else start

        indent = updated[insert_after][: len(updated[insert_after]) - len(updated[insert_after].lstrip())]
        updated.insert(insert_after + 1, f"{indent}gen_expected_facts={facts_str},\n")
        applied += 1

    return updated, applied


def _format_facts_list(facts: list[str]) -> str:
    """将事实列表格式化为 Python 代码字符串"""
    if len(facts) <= 2:
        # 短列表，放一行
        items = ", ".join(repr(f) for f in facts)
        return f"[{items}]"
    else:
        # 长列表，多行缩进
        items = ",\n            ".join(repr(f) for f in facts)
        return f"[\n            {items},\n        ]"


def apply_backfill(
    backfill_path: str,
    dry_run: bool = False,
    bump_version: bool = False,
):
    """主入口：读取回填 JSON，应用到 rag_testset.py

    Args:
        backfill_path: backfill JSON 文件路径
        dry_run: 仅预览变更，不写文件
        bump_version: 自动递增 DATASET_VERSION 的 patch 号
    """
    backfill_file = Path(backfill_path)
    if not backfill_file.exists():
        print(f"错误: 回填文件不存在: {backfill_path}")
        sys.exit(1)

    # 加载数据
    mapping = _parse_backfill_json(backfill_file)
    print(f"加载回填数据: {len(mapping)} 条 gen_expected_facts")

    # 定位 rag_testset.py
    testset_path = Path(__file__).resolve().parent / "rag_testset.py"

    with open(testset_path, "r", encoding="utf-8") as f:
        original = f.read()
        lines = original.splitlines(keepends=True)

    # 找出所有 EvalSample 的起止行
    boundaries = _find_eval_sample_boundaries(lines)
    print(f"找到 {len(boundaries)} 个 EvalSample 定义")

    # 匹配
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.evaluation.rag_testset import EVALUATION_DATASET

    if len(boundaries) != len(EVALUATION_DATASET):
        print(f"⚠️  边界检测数量 ({len(boundaries)}) 与 EVALUATION_DATASET 长度 "
              f"({len(EVALUATION_DATASET)}) 不一致，可能无法完全匹配")

    # 验证 mapping 中的 index 范围
    max_idx = len(EVALUATION_DATASET) - 1
    invalid = [i for i in mapping if i < 0 or i > max_idx]
    if invalid:
        print(f"⚠️  回填数据中有 {len(invalid)} 个无效 index（超出 0-{max_idx}），将跳过")
        for i in invalid:
            del mapping[i]

    # 应用变更
    updated_lines, applied = _apply_backfill_to_file(lines, boundaries, mapping, EVALUATION_DATASET)

    # 版本递增
    if bump_version:
        for i, line in enumerate(updated_lines):
            if line.startswith("DATASET_VERSION = "):
                old_ver = line.split('"')[1]
                parts = old_ver.split(".")
                parts[-1] = str(int(parts[-1]) + 1)
                new_ver = ".".join(parts)
                updated_lines[i] = f'DATASET_VERSION = "{new_ver}"\n'
                print(f"版本号: {old_ver} → {new_ver}")
                break

    new_content = "".join(updated_lines)

    if dry_run:
        # 显示 diff
        import difflib

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile="rag_testset.py (original)",
            tofile="rag_testset.py (updated)",
        )
        diff_text = "".join(diff)
        if diff_text:
            print("\n变更预览:\n")
            # 只显示前 100 行
            diff_lines = diff_text.splitlines(keepends=True)
            for line in diff_lines[:100]:
                print(line, end="")
            if len(diff_lines) > 100:
                print(f"\n... 及其他 {len(diff_lines) - 100} 行")
        else:
            print("\n无变更（所有样本可能已有 gen_expected_facts）")
    else:
        with open(testset_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"\n已写入 rag_testset.py: {applied} 条样本更新 gen_expected_facts")

        # 验证
        print("验证导入结果...")
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import importlib
        import tests.evaluation.rag_testset as rts

        importlib.reload(rts)
        has = sum(1 for s in rts.EVALUATION_DATASET if s.gen_expected_facts)
        print(f"  gen_expected_facts 覆盖: {has}/{len(rts.EVALUATION_DATASET)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 backfill JSON 应用到 rag_testset.py")
    parser.add_argument(
        "backfill_json",
        type=str,
        help="回填 JSON 文件路径（如 reports/backfill_facts.json）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览变更，不修改文件",
    )
    parser.add_argument(
        "--bump-version", action="store_true",
        help="自动递增 DATASET_VERSION patch 号",
    )
    args = parser.parse_args()

    apply_backfill(
        backfill_path=args.backfill_json,
        dry_run=args.dry_run,
        bump_version=args.bump_version,
    )
