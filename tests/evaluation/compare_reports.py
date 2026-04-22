"""basic vs enhanced RAG 对比报告生成脚本

用法：

  # 先分别运行两次评估，生成报告文件
  RAG_MODE=basic    python -m tests.evaluation.evaluate_rag --output reports/basic.json
  RAG_MODE=enhanced python -m tests.evaluation.evaluate_rag --output reports/enhanced.json

  # 再生成对比报告
  python -m tests.evaluation.compare_reports \\
    --basic    reports/basic.json \\
    --enhanced reports/enhanced.json

  # 指定输出文件
  python -m tests.evaluation.compare_reports \\
    --basic    reports/basic.json \\
    --enhanced reports/enhanced.json \\
    --output   reports/comparison.json

输出：
  - 终端打印对比表格
  - 保存对比结果到 JSON
"""

import argparse
import json
from pathlib import Path


METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

# Enhanced 模式相对 Basic 的期望提升阈值
IMPROVEMENT_THRESHOLD = 0.10   # context_precision 和 context_recall 要求提升 ≥ 0.10
PASS_THRESHOLD = 0.70          # Basic 基线要求 ≥ 0.70


def load_report(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"报告文件不存在: {path}\n"
            f"请先运行: python -m tests.evaluation.evaluate_rag --output {path}"
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def print_comparison(basic: dict, enhanced: dict) -> dict:
    """打印对比表格并返回对比结果 dict"""
    basic_m = basic["metrics"]
    enhanced_m = enhanced["metrics"]

    print()
    print("=" * 72)
    print("  RAGAs 评估对比报告")
    print("=" * 72)
    print(f"  Basic 模式    — rag_mode={basic['rag_mode']}, "
          f"preprocessor={basic['query_preprocessor_type']}, "
          f"reranker={basic['reranker_type']}")
    print(f"  Enhanced 模式 — rag_mode={enhanced['rag_mode']}, "
          f"preprocessor={enhanced['query_preprocessor_type']}, "
          f"reranker={enhanced['reranker_type']}")
    print(f"  评估时间      — Basic: {basic['evaluated_at'][:19]}, "
          f"Enhanced: {enhanced['evaluated_at'][:19]}")
    print(f"  问题数量      — {basic['num_questions']} 条")
    print("-" * 72)

    header = f"  {'指标':<22} {'Basic':>10} {'Enhanced':>10} {'提升':>10}  {'达标':>6}"
    print(header)
    print("-" * 72)

    comparison = {}
    for metric in METRICS:
        b_val = basic_m.get(metric, 0.0)
        e_val = enhanced_m.get(metric, 0.0)
        delta = e_val - b_val

        # 判断是否达标
        if metric in ("context_precision", "context_recall"):
            basic_pass = b_val >= PASS_THRESHOLD
            improve_pass = delta >= IMPROVEMENT_THRESHOLD
            status = ("✅" if basic_pass else "❌") + ("↑" if improve_pass else " ")
        else:
            basic_pass = None
            improve_pass = None
            status = "  "

        print(
            f"  {metric:<22} {b_val:>10.4f} {e_val:>10.4f} "
            f"{delta:>+10.4f}  {status:>6}"
        )
        comparison[metric] = {
            "basic": b_val,
            "enhanced": e_val,
            "delta": delta,
            "basic_pass_threshold": basic_pass,
            "improvement_pass_threshold": improve_pass,
        }

    print("-" * 72)

    # 综合判断
    cp_ok = comparison["context_precision"]["improvement_pass_threshold"]
    cr_ok = comparison["context_recall"]["improvement_pass_threshold"]
    b_cp_ok = comparison["context_precision"]["basic_pass_threshold"]
    b_cr_ok = comparison["context_recall"]["basic_pass_threshold"]

    print()
    print("  综合评定：")
    print(f"    Basic 基线达标（≥ {PASS_THRESHOLD}）   :"
          f"  context_precision {'✅' if b_cp_ok else '❌'}  "
          f"context_recall {'✅' if b_cr_ok else '❌'}")
    print(f"    Enhanced 提升达标（≥ +{IMPROVEMENT_THRESHOLD}）:"
          f"  context_precision {'✅' if cp_ok else '❌'}  "
          f"context_recall {'✅' if cr_ok else '❌'}")
    overall = b_cp_ok and b_cr_ok and cp_ok and cr_ok
    print(f"    总体结论           :  {'✅ 达到预期目标' if overall else '❌ 未达预期目标，需继续调优'}")
    print("=" * 72)
    print()

    return comparison


def run_compare(basic_path: str, enhanced_path: str, output_path: str | None = None):
    basic = load_report(basic_path)
    enhanced = load_report(enhanced_path)

    comparison = print_comparison(basic, enhanced)

    result = {
        "basic": {
            "rag_mode": basic["rag_mode"],
            "query_preprocessor_type": basic["query_preprocessor_type"],
            "reranker_type": basic["reranker_type"],
            "evaluated_at": basic["evaluated_at"],
            "metrics": basic["metrics"],
        },
        "enhanced": {
            "rag_mode": enhanced["rag_mode"],
            "query_preprocessor_type": enhanced["query_preprocessor_type"],
            "reranker_type": enhanced["reranker_type"],
            "evaluated_at": enhanced["evaluated_at"],
            "metrics": enhanced["metrics"],
        },
        "comparison": comparison,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"对比报告已保存: {out}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic vs Enhanced RAG 对比报告")
    parser.add_argument(
        "--basic", "-b",
        type=str,
        default="reports/basic.json",
        help="Basic 模式评估结果 JSON（默认: reports/basic.json）"
    )
    parser.add_argument(
        "--enhanced", "-e",
        type=str,
        default="reports/enhanced.json",
        help="Enhanced 模式评估结果 JSON（默认: reports/enhanced.json）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="对比结果输出 JSON（可选）"
    )
    args = parser.parse_args()

    run_compare(args.basic, args.enhanced, args.output)
