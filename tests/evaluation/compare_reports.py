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

# 检索指标名称
RETRIEVAL_METRICS = ["context_precision", "context_recall"]
# 生成指标名称
GENERATION_METRICS = ["faithfulness", "answer_relevancy"]

IMPROVEMENT_THRESHOLD = 0.10
PASS_THRESHOLD = 0.70


def _get_metrics(report: dict) -> dict:
    """兼容新旧格式：新格式使用 retrieval_metrics/generation_metrics，旧格式使用 flat metrics"""
    if "retrieval_metrics" in report:
        return report["retrieval_metrics"]
    if "metrics" in report:
        return report["metrics"]
    return {}


def _get_gen_metrics(report: dict) -> dict | None:
    """获取生成指标（可能为 None 表示未执行 Phase 2）"""
    return report.get("generation_metrics") if "generation_metrics" in report else None


def _get_judge_info(report: dict) -> str:
    """提取 Judge 元数据用于展示"""
    judge = report.get("judge", {})
    if judge:
        return f"{judge.get('model', '?')} (T={judge.get('temperature', '?')})"
    return "(无 Judge 元数据)"


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
    basic_metrics = _get_metrics(basic)
    enhanced_metrics = _get_metrics(enhanced)
    basic_gen = _get_gen_metrics(basic)
    enhanced_gen = _get_gen_metrics(enhanced)

    print()
    print("=" * 72)
    print("  RAGAs 评估对比报告")
    print("=" * 72)
    print(f"  Basic 模式    — rag_mode={basic['rag_mode']}, "
          f"preprocessor={basic.get('query_preprocessor_type', '?')}, "
          f"reranker={basic.get('reranker_type', '?')}")
    print(f"  Enhanced 模式 — rag_mode={enhanced['rag_mode']}, "
          f"preprocessor={enhanced.get('query_preprocessor_type', '?')}, "
          f"reranker={enhanced.get('reranker_type', '?')}")
    print(f"  Basic Judge   : {_get_judge_info(basic)}")
    print(f"  Enhanced Judge: {_get_judge_info(enhanced)}")
    if basic.get("dataset_version"):
        print(f"  数据集版本    : {basic['dataset_version']}")
    print(f"  评估时间      — Basic: {basic['evaluated_at'][:19]}, "
          f"Enhanced: {enhanced['evaluated_at'][:19]}")
    print(f"  问题数量      — {basic['num_questions']} 条")

    comparison = {}

    # --- 检索指标对比 ---
    print("-" * 72)
    print("  [检索指标]")
    print("-" * 72)
    header = f"  {'指标':<22} {'Basic':>10} {'Enhanced':>10} {'提升':>10}  {'达标':>6}"
    print(header)
    print("-" * 72)

    for metric in RETRIEVAL_METRICS:
        b_val = basic_metrics.get(metric, 0.0)
        e_val = enhanced_metrics.get(metric, 0.0)
        delta = e_val - b_val

        basic_pass = b_val >= PASS_THRESHOLD
        improve_pass = delta >= IMPROVEMENT_THRESHOLD
        status = ("✅" if basic_pass else "❌") + ("↑" if improve_pass else " ")

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

    # --- 生成指标对比 ---
    has_gen = basic_gen is not None and enhanced_gen is not None
    gen_skipped = False

    if basic_gen or enhanced_gen:
        print("-" * 72)
        print("  [生成指标]")
        if basic_gen and basic_gen.get("answers_generated", 0) > 0:
            print(f"  Basic    answer 生成: {basic_gen.get('answers_generated', 0)}"
                  f" (+{basic_gen.get('answers_failed', 0)} 失败)")
        if enhanced_gen and enhanced_gen.get("answers_generated", 0) > 0:
            print(f"  Enhanced answer 生成: {enhanced_gen.get('answers_generated', 0)}"
                  f" (+{enhanced_gen.get('answers_failed', 0)} 失败)")
        print("-" * 72)

        for metric in GENERATION_METRICS:
            b_val = basic_gen.get(metric) if basic_gen else None
            e_val = enhanced_gen.get(metric) if enhanced_gen else None

            if b_val is None or e_val is None:
                print(f"  {metric:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
                comparison[metric] = {
                    "basic": None,
                    "enhanced": None,
                    "delta": None,
                }
                gen_skipped = True
            else:
                delta = e_val - b_val
                print(
                    f"  {metric:<22} {b_val:>10.4f} {e_val:>10.4f} "
                    f"{delta:>+10.4f}"
                )
                comparison[metric] = {
                    "basic": b_val,
                    "enhanced": e_val,
                    "delta": delta,
                }
    else:
        gen_skipped = True

    # --- 分类统计对比 ---
    basic_cat = basic.get("category_stats")
    enhanced_cat = enhanced.get("category_stats")
    if basic_cat and enhanced_cat:
        print("-" * 72)
        print("  [分类统计 — 平均检索上下文数]")
        print("-" * 72)
        cat_header = f"  {'分类':<18} {'Basic':>10} {'Enhanced':>10} {'增加':>10}"
        print(cat_header)
        print("-" * 72)
        for cat in sorted(basic_cat.keys()):
            b_avg = basic_cat[cat].get("avg_contexts_count", 0)
            e_avg = enhanced_cat.get(cat, {}).get("avg_contexts_count", 0)
            delta = e_avg - b_avg
            print(f"  {cat:<18} {b_avg:>10.1f} {e_avg:>10.1f} {delta:>+10.1f}")

    # --- 综合评定 ---
    print("-" * 72)
    print()
    print("  综合评定：")

    cp_ok = comparison.get("context_precision", {}).get("improvement_pass_threshold", False)
    cr_ok = comparison.get("context_recall", {}).get("improvement_pass_threshold", False)
    b_cp_ok = comparison.get("context_precision", {}).get("basic_pass_threshold", False)
    b_cr_ok = comparison.get("context_recall", {}).get("basic_pass_threshold", False)

    print(f"    Basic 基线达标（≥ {PASS_THRESHOLD}）   :"
          f"  context_precision {'✅' if b_cp_ok else '❌'}  "
          f"context_recall {'✅' if b_cr_ok else '❌'}")
    print(f"    Enhanced 提升达标（≥ +{IMPROVEMENT_THRESHOLD}）:"
          f"  context_precision {'✅' if cp_ok else '❌'}  "
          f"context_recall {'✅' if cr_ok else '❌'}")

    if gen_skipped:
        print(f"    生成指标           :  ⚠ 未执行 Phase 2（使用 --with-generation 启动）")
    elif has_gen:
        print(f"    生成指标           :  已评估（含 faithfulness + answer_relevancy）")

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
            "query_preprocessor_type": basic.get("query_preprocessor_type", ""),
            "reranker_type": basic.get("reranker_type", ""),
            "evaluated_at": basic["evaluated_at"],
            "dataset_version": basic.get("dataset_version", ""),
            "judge": basic.get("judge", {}),
            "retrieval_metrics": _get_metrics(basic),
            "generation_metrics": _get_gen_metrics(basic),
        },
        "enhanced": {
            "rag_mode": enhanced["rag_mode"],
            "query_preprocessor_type": enhanced.get("query_preprocessor_type", ""),
            "reranker_type": enhanced.get("reranker_type", ""),
            "evaluated_at": enhanced["evaluated_at"],
            "dataset_version": enhanced.get("dataset_version", ""),
            "judge": enhanced.get("judge", {}),
            "retrieval_metrics": _get_metrics(enhanced),
            "generation_metrics": _get_gen_metrics(enhanced),
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
