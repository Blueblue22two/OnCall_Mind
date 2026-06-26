"""回填 gen_expected_facts 标注脚本

对 EVALUATION_DATASET 中缺少 gen_expected_facts 的样本，使用 LLM 从已有
ground_truths 中提炼关键事实，输出可直接用于更新 rag_testset.py 的 JSON。

gen_expected_facts 与 ground_truths 的区别：
  - ground_truths: 参考答案要点（3-5 条，覆盖"是什么→为什么→怎么办"）
  - gen_expected_facts: 答案必须包含的关键事实（3-5 条，仅保留可验证的事实性陈述，
    排除工具操作步骤、概括性描述）

用法（在项目根目录执行）：

  # 生成回填数据（不修改 rag_testset.py）
  python -m tests.evaluation.backfill_expected_facts

  # 预览模式（只看哪些样本缺少 gen_expected_facts）
  python -m tests.evaluation.backfill_expected_facts --dry-run

  # 指定输出路径
  python -m tests.evaluation.backfill_expected_facts --output reports/backfill_facts.json

输出 JSON 格式：
  {
    "_metadata": {...},
    "samples": [
      {"index": 5, "question": "...", "gen_expected_facts": ["事实1", "事实2", ...]},
      ...
    ]
  }
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是运维领域的评估数据集标注专家。你的任务是为已有的 RAG 评估问题提炼"必须包含的关键事实"（gen_expected_facts）。

## gen_expected_facts 的定义

gen_expected_facts 是评估 RAG 答案完整性时使用的检查清单。它表示：
  - 一个合格的 RAG 答案必须覆盖的最小事实集合
  - 只包含可验证的事实性陈述（能明确判断"答案说了/没说"）
  - 排除纯工具操作步骤（如"使用 top 命令查看"）、排除概括性描述

## 提炼规则

1. 从 ground_truths 中提取 3-5 条核心事实
2. 每条 1-2 句话（20-60 字），使用陈述句
3. 优先提取"是什么"和"为什么"的事实，减少"怎么办"的操作步骤
4. 如果 ground_truths 只有操作步骤（如"使用 X 工具查询 Y"），将其改写为事实陈述（如"X 工具可以查询 Y"）
5. 保留具体的阈值数字、告警名、技术术语

## 输出格式

严格返回 JSON 对象：
{
  "gen_expected_facts": ["事实1", "事实2", "事实3"]
}

只返回 JSON 对象，不要任何额外的文字说明。"""


def _build_user_prompt(question: str, ground_truths: list[str]) -> str:
    """构建单样本的事实提炼 prompt"""
    gt_text = "\n".join(f"  [{i+1}] {gt}" for i, gt in enumerate(ground_truths))
    return f"""## 用户问题
{question}

## 参考答案要点（ground_truths）
{gt_text}

请从上述参考答案中提炼 3-5 条必须包含的关键事实（gen_expected_facts）。"""


# ---------------------------------------------------------------------------
# LLM 调用（使用 ChatOpenAI + OpenAI 兼容端点，与 evaluate_rag.py 一致）
# ---------------------------------------------------------------------------
def _get_llm():
    """延迟初始化 LLM（使用 eval_judge_model，OpenAI 兼容端点）"""
    from langchain_openai import ChatOpenAI

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.config import config

    judge_api_base = config.eval_judge_api_base or config.dashscope_api_base
    judge_api_key = config.eval_judge_api_key or config.dashscope_api_key

    return ChatOpenAI(
        model=config.eval_judge_model,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=judge_api_base,
    )


def _parse_json_response(raw: str) -> Optional[dict]:
    """从 LLM 响应中提取 JSON 对象"""
    import re

    # 尝试直接解析
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 提取 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # 找到第一个 { 到最后一个 }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def backfill_expected_facts(
    output_path: Optional[str] = None,
    dry_run: bool = False,
    resume: bool = False,
):
    """为缺少 gen_expected_facts 的样本生成回填数据

    支持断点续传：使用 --resume 时，从已有输出文件中加载已完成样本，
    跳过已处理的 index，仅处理剩余样本。运行中每成功一条立即写入磁盘。

    Args:
        output_path: JSON 输出路径（也是断点续传的检查点文件）
        dry_run: 仅列出缺失样本，不调用 LLM
        resume: 从已有输出文件恢复，跳过已完成的样本
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.evaluation.rag_testset import EVALUATION_DATASET, DATASET_VERSION

    # 确定输出路径
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_path or f"reports/backfill_facts_{ts}.json"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 找出缺少 gen_expected_facts 的样本
    all_missing = [
        (i, s) for i, s in enumerate(EVALUATION_DATASET)
        if not s.gen_expected_facts
    ]

    print(f"数据集版本: {DATASET_VERSION}")
    print(f"总样本数:   {len(EVALUATION_DATASET)}")
    print(f"已有标注:   {len(EVALUATION_DATASET) - len(all_missing)}")
    print(f"缺少标注:   {len(all_missing)}")

    if not all_missing:
        print("✅ 所有样本均已标注 gen_expected_facts，无需回填")
        return

    # 断点续传：加载已完成的样本，从剩余的开始
    completed_indices: set = set()
    existing_results: list = []

    if resume and out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            existing_results = prev.get("samples", [])
            completed_indices = {s["index"] for s in existing_results}
            print(f"📋 断点续传: 已从 {out_path} 加载 {len(completed_indices)} 个已完成样本")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  无法解析已有输出文件，将从头开始: {e}")

    # 过滤出待处理的样本
    missing = [(i, s) for i, s in all_missing if i not in completed_indices]
    if not missing:
        print(f"✅ 所有 {len(all_missing)} 个样本已处理完成，无需继续")
        return

    print(f"待处理:   {len(missing)} (已完成: {len(completed_indices)})")
    print(f"输出文件: {out_path}")
    print()

    if dry_run:
        print("DRY RUN — 以下样本缺少 gen_expected_facts:")
        print()
        from collections import Counter

        cat_dist = Counter(s.category for _, s in missing)
        print("  按类别:")
        for cat in ["exact_keyword", "colloquial", "cross_doc", "edge_case"]:
            print(f"    {cat}: {cat_dist.get(cat, 0)}")
        return

    # 初始化 LLM
    print("初始化 LLM...")
    llm = _get_llm()
    from langchain_core.messages import HumanMessage, SystemMessage

    results = list(existing_results)  # 从已有结果开始
    success = len(completed_indices)
    failed = 0

    for idx, (orig_idx, sample) in enumerate(missing, 1):
        total = len(missing) + len(completed_indices)
        current = idx + len(completed_indices)
        print(f"[{current}/{total}] 处理: {sample.question[:60]}...")

        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=_build_user_prompt(sample.question, sample.ground_truths)),
            ]
            response = llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)

            parsed = _parse_json_response(raw)
            if parsed and "gen_expected_facts" in parsed:
                facts = parsed["gen_expected_facts"]
                if isinstance(facts, list) and len(facts) >= 2:
                    results.append({
                        "index": orig_idx,
                        "question": sample.question[:80],
                        "gen_expected_facts": [f.strip() for f in facts if f.strip()],
                    })
                    success += 1
                    print(f"  ✓ 生成 {len(facts)} 条事实")

                    # 每成功一条立即写入磁盘（增量保存，防止中断丢失进度）
                    _save_checkpoint(
                        out_path, DATASET_VERSION, len(all_missing),
                        success, failed, results,
                    )
                    continue

            print(f"  ✗ 解析失败，原始响应: {raw[:100]}...")
            failed += 1

        except Exception as e:
            print(f"  ✗ LLM 调用失败: {e}")
            failed += 1
            # 失败也保存检查点，记录当前进度
            _save_checkpoint(
                out_path, DATASET_VERSION, len(all_missing),
                success, failed, results,
            )

    print(f"\n完成: 成功={success}, 失败={failed} (本次新增成功={success - len(completed_indices)})")

    if not results:
        print("没有成功生成任何事实，退出")
        return

    # 最终保存
    _save_checkpoint(
        out_path, DATASET_VERSION, len(all_missing),
        success, failed, results, final=True,
    )

    print(f"\n回填数据已保存: {out_path}")
    print(f"共 {len(results)} 条样本，请人工审核后更新 rag_testset.py")
    if failed > 0:
        print(f"⚠️  {failed} 条失败，可重新运行相同命令（--resume）重试失败的样本")
        print(f"   注意：当前脚本不会自动重试失败样本，需手动处理")


def _save_checkpoint(
    out_path: Path,
    dataset_version: str,
    total_missing: int,
    success: int,
    failed: int,
    results: list,
    final: bool = False,
):
    """保存增量检查点（每成功一条样本写入一次磁盘）"""
    status = "PENDING_REVIEW" if not final else "PENDING_REVIEW"
    payload = {
        "_metadata": {
            "generated_at": datetime.now().isoformat(),
            "dataset_version": dataset_version,
            "total_missing": total_missing,
            "generated": success,
            "failed": failed,
            "in_progress": not final,
            "judge_model": "qwen3.5-plus",
            "judge_temperature": 0.0,
            "status": status,
            "usage": (
                "人工审核后，将 gen_expected_facts 逐条粘贴到 "
                "rag_testset.py 对应样本的 gen_expected_facts 字段中"
            ),
        },
        "samples": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回填 gen_expected_facts 标注")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="JSON 输出路径（默认: reports/backfill_facts_{timestamp}.json）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅列出缺失样本，不调用 LLM",
    )
    parser.add_argument(
        "--resume", "-r", action="store_true",
        help="从已有输出文件恢复，跳过已完成的样本（需指定 --output）",
    )
    args = parser.parse_args()

    if args.resume and not args.output:
        parser.error("--resume 需要配合 --output 指定已有的检查点文件路径")

    backfill_expected_facts(
        output_path=args.output,
        dry_run=args.dry_run,
        resume=args.resume,
    )
