"""basic vs enhanced RAG 对比报告生成脚本

用法：

  # 先分别运行两次评估，生成报告文件
  RAG_MODE=basic    python -m tests.evaluation.evaluate_rag --output reports/basic.json
  RAG_MODE=enhanced python -m tests.evaluation.evaluate_rag --output reports/enhanced.json

  # 再生成对比报告
  python -m tests.evaluation.compare_reports \\
    --basic    reports/basic.json \\
    --enhanced reports/enhanced.json

  # 也支持对比生成评估报告（evaluate_generation.py 输出）
  python -m tests.evaluation.compare_reports \\
    --basic    reports/gen_basic.json \\
    --enhanced reports/gen_enhanced.json

  # 指定输出文件
  python -m tests.evaluation.compare_reports \\
    --basic    reports/basic.json \\
    --enhanced reports/enhanced.json \\
    --output   reports/comparison.json

输出：
  - 终端打印对比表格（检索 + 生成 + 分类统计）
  - 保存对比结果到 JSON
"""

import argparse
import json
from pathlib import Path

# 检索指标名称
RETRIEVAL_METRICS = ["context_precision", "context_recall"]
# 生成指标名称（from evaluate_rag.py）
GENERATION_METRICS = ["faithfulness", "answer_relevancy", "answer_correctness"]
# 自定义生成指标名称（from evaluate_generation.py）
CUSTOM_GEN_METRICS = [
    "answer_completeness",
    "hallucination_score",
    "semantic_similarity",
    "avg_latency_ms",
    "success_rate",
]

IMPROVEMENT_THRESHOLD = 0.10
PASS_THRESHOLD = 0.70

# 生成指标达标阈值
GEN_PASS_THRESHOLD = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "answer_correctness": 0.65,
    "answer_completeness": 1.0,       # 0-2 标度，1.0 = 平均覆盖 ≥50%
    "hallucination_score": 1.5,       # 0-2 标度，1.5 = 轻微幻觉以下
    "semantic_similarity": 0.50,
    "success_rate": 0.90,
}


def _get_metrics(report: dict) -> dict:
    """兼容新旧格式：新格式使用 retrieval_metrics/generation_metrics，旧格式使用 flat metrics"""
    if "retrieval_metrics" in report:
        return report["retrieval_metrics"]
    if "metrics" in report:
        return report["metrics"]
    return {}


def _get_gen_metrics(report: dict) -> dict | None:
    """获取生成指标（兼容 evaluate_rag.py 和 evaluate_generation.py 两种报告格式）

    evaluate_rag.py 使用 "generation_metrics" 键，
    evaluate_generation.py 使用 "aggregate_metrics" 键。
    """
    if "generation_metrics" in report:
        return report["generation_metrics"]
    if "aggregate_metrics" in report:
        return report["aggregate_metrics"]
    return None


def _is_gen_report(report: dict) -> bool:
    """判断是否为生成评估报告（evaluate_generation.py 输出）"""
    return "aggregate_metrics" in report and "retrieval_metrics" not in report


def _get_config_info(report: dict) -> dict:
    """提取配置信息（兼容两种格式）"""
    if "config" in report:
        return report["config"]
    return {
        "rag_mode": report.get("rag_mode", "?"),
        "query_preprocessor_type": report.get("query_preprocessor_type", "?"),
        "reranker_type": report.get("reranker_type", "?"),
    }


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
    basic_is_gen = _is_gen_report(basic)
    enhanced_is_gen = _is_gen_report(enhanced)
    basic_config = _get_config_info(basic)
    enhanced_config = _get_config_info(enhanced)

    report_type = "RAG 生成质量" if (basic_is_gen and enhanced_is_gen) else "RAGAs"

    print()
    print("=" * 78)
    print(f"  {report_type} 评估对比报告")
    print("=" * 78)
    print(f"  Basic 模式    — rag_mode={basic_config.get('rag_mode', '?')}, "
          f"preprocessor={basic_config.get('query_preprocessor_type', '?')}, "
          f"reranker={basic_config.get('reranker_type', '?')}")
    print(f"  Enhanced 模式 — rag_mode={enhanced_config.get('rag_mode', '?')}, "
          f"preprocessor={enhanced_config.get('query_preprocessor_type', '?')}, "
          f"reranker={enhanced_config.get('reranker_type', '?')}")
    if basic_config.get("generation_model"):
        print(f"  Basic    生成模型: {basic_config['generation_model']}")
        print(f"  Enhanced 生成模型: {enhanced_config.get('generation_model', '?')}")
    print(f"  Basic Judge   : {_get_judge_info(basic)}")
    print(f"  Enhanced Judge: {_get_judge_info(enhanced)}")
    if basic.get("dataset_version"):
        print(f"  数据集版本    : {basic['dataset_version']}")
    print(f"  评估时间      — Basic: {basic['evaluated_at'][:19]}, "
          f"Enhanced: {enhanced['evaluated_at'][:19]}")

    # 样本数（兼容两种格式）
    basic_num = basic.get("num_questions") or basic_gen.get("num_samples", "?") if basic_gen else "?"
    enhanced_num = enhanced.get("num_questions") or enhanced_gen.get("num_samples", "?") if enhanced_gen else "?"
    print(f"  样本数        — Basic: {basic_num}, Enhanced: {enhanced_num}")

    comparison = {}

    has_retrieval = bool(basic_metrics and enhanced_metrics)
    has_gen = basic_gen is not None and enhanced_gen is not None

    # --- 检索指标对比（仅当两个报告都含检索指标时显示） ---
    if has_retrieval:
        print("-" * 78)
        print("  [检索指标]")
        print("-" * 78)
        header = f"  {'指标':<22} {'Basic':>10} {'Enhanced':>10} {'提升':>10}  {'达标':>6}"
        print(header)
        print("-" * 78)

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
    gen_skipped = False

    if has_gen:
        print("-" * 78)
        print("  [生成指标]")

        # 显示生成统计（兼容两种格式）
        if basic_gen.get("answers_generated"):
            print(f"  Basic    answer 生成: {basic_gen.get('answers_generated', 0)}"
                  f" (+{basic_gen.get('answers_failed', 0)} 失败)")
        elif basic_gen.get("num_samples"):
            print(f"  Basic    样本数: {basic_gen['num_samples']}"
                  f" (成功={basic_gen.get('num_success', '?')}, 失败={basic_gen.get('num_failed', '?')})")

        if enhanced_gen.get("answers_generated"):
            print(f"  Enhanced answer 生成: {enhanced_gen.get('answers_generated', 0)}"
                  f" (+{enhanced_gen.get('answers_failed', 0)} 失败)")
        elif enhanced_gen.get("num_samples"):
            print(f"  Enhanced 样本数: {enhanced_gen['num_samples']}"
                  f" (成功={enhanced_gen.get('num_success', '?')}, 失败={enhanced_gen.get('num_failed', '?')})")

        print("-" * 78)
        gen_header = f"  {'指标':<24} {'Basic':>10} {'Enhanced':>10} {'提升':>10}  {'达标':>6}"
        print(gen_header)
        print("-" * 78)

        # 合并所有生成指标（RAGAs + 自定义）
        all_gen_metrics = GENERATION_METRICS + CUSTOM_GEN_METRICS

        for metric in all_gen_metrics:
            b_val = basic_gen.get(metric) if basic_gen else None
            e_val = enhanced_gen.get(metric) if enhanced_gen else None

            if b_val is None and e_val is None:
                continue  # 两份报告都没有该指标，跳过

            if b_val is None or e_val is None:
                b_str = f"{b_val:.4f}" if isinstance(b_val, (int, float)) else "N/A"
                e_str = f"{e_val:.4f}" if isinstance(e_val, (int, float)) else "N/A"
                print(f"  {metric:<24} {b_str:>10} {e_str:>10} {'N/A':>10}")
                comparison[metric] = {
                    "basic": b_val,
                    "enhanced": e_val,
                    "delta": None,
                    "basic_pass_threshold": None,
                }
            else:
                delta = e_val - b_val
                threshold = GEN_PASS_THRESHOLD.get(metric)
                b_pass = b_val >= threshold if threshold is not None else None
                e_pass = e_val >= threshold if threshold is not None else None

                status_parts = []
                if b_pass is True:
                    status_parts.append("✅")
                elif b_pass is False:
                    status_parts.append("❌")
                if delta >= IMPROVEMENT_THRESHOLD:
                    status_parts.append("↑")
                status = "".join(status_parts) if status_parts else " "

                print(
                    f"  {metric:<24} {b_val:>10.4f} {e_val:>10.4f} "
                    f"{delta:>+10.4f}  {status:>6}"
                )
                comparison[metric] = {
                    "basic": b_val,
                    "enhanced": e_val,
                    "delta": delta,
                    "basic_pass_threshold": b_pass,
                    "threshold": threshold,
                }
    else:
        gen_skipped = True

    # --- 分类统计对比 ---
    basic_cat = basic.get("category_stats") or basic.get("category_breakdown")
    enhanced_cat = enhanced.get("category_stats") or enhanced.get("category_breakdown")
    if basic_cat and enhanced_cat:
        print("-" * 78)
        print("  [分类统计]")
        print("-" * 78)

        # 检测分类统计的字段类型
        sample_cat = next(iter(basic_cat.values()), {})
        is_gen_breakdown = "semantic_similarity" in sample_cat or "faithfulness" in sample_cat

        if is_gen_breakdown:
            # 生成评估的分类统计：显示 count + 关键指标
            cat_header = f"  {'分类':<18} {'数量':>6} {'语义相似度':>12} {'完整性':>8}"
            print(cat_header)
            print("-" * 78)
            for cat in sorted(basic_cat.keys()):
                b_info = basic_cat[cat]
                e_info = enhanced_cat.get(cat, {})
                print(
                    f"  {cat:<18} {b_info.get('count', 0):>6} "
                    f"{b_info.get('semantic_similarity', 0):>11.4f}→{e_info.get('semantic_similarity', 0):.4f} "
                    f"{b_info.get('answer_completeness', 0):>8.2f}"
                )
        else:
            # 检索评估的分类统计：显示 avg_contexts_count
            cat_header = f"  {'分类':<18} {'Basic':>10} {'Enhanced':>10} {'增加':>10}"
            print(cat_header)
            print("-" * 78)
            for cat in sorted(basic_cat.keys()):
                b_avg = basic_cat[cat].get("avg_contexts_count", 0)
                e_avg = enhanced_cat.get(cat, {}).get("avg_contexts_count", 0)
                delta = e_avg - b_avg
                print(f"  {cat:<18} {b_avg:>10.1f} {e_avg:>10.1f} {delta:>+10.1f}")

    # --- 低质量样本对比 ---
    basic_lq = basic.get("low_quality_samples")
    enhanced_lq = enhanced.get("low_quality_samples")
    if basic_lq is not None and enhanced_lq is not None:
        print("-" * 78)
        print("  [低质量样本]")
        print(f"    Basic:    {len(basic_lq)} 条低质量样本")
        print(f"    Enhanced: {len(enhanced_lq)} 条低质量样本")
        if len(basic_lq) > len(enhanced_lq):
            print(f"    ✅ Enhanced 减少了 {len(basic_lq) - len(enhanced_lq)} 条低质量样本")
        elif len(enhanced_lq) > len(basic_lq):
            print(f"    ⚠ Enhanced 增加了 {len(enhanced_lq) - len(basic_lq)} 条低质量样本")

    # --- 综合评定 ---
    print("-" * 78)
    print()
    print("  综合评定：")

    # 检索指标评定
    if has_retrieval:
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

    # 生成指标评定
    if has_gen and not gen_skipped:
        gen_pass_count = 0
        gen_total_count = 0
        for metric, info in comparison.items():
            if metric in RETRIEVAL_METRICS:
                continue
            if info.get("basic_pass_threshold") is not None:
                gen_total_count += 1
                if info["basic_pass_threshold"]:
                    gen_pass_count += 1
        if gen_total_count > 0:
            print(f"    生成指标达标        :  {gen_pass_count}/{gen_total_count} 项"
                  f" ({'✅' if gen_pass_count >= gen_total_count * 0.6 else '❌'})")

    if gen_skipped:
        print(f"    生成指标           :  ⚠ 未执行（使用 --with-generation 启动）")

    # 总体结论
    overall = True
    if has_retrieval:
        overall = overall and all([
            comparison.get("context_precision", {}).get("improvement_pass_threshold", False),
            comparison.get("context_recall", {}).get("improvement_pass_threshold", False),
        ])
    print(f"    总体结论           :  {'✅ 达到预期目标' if overall else '❌ 未达预期目标，需继续调优'}")
    print("=" * 78)
    print()

    return comparison


def run_compare(basic_path: str, enhanced_path: str, output_path: str | None = None):
    basic = load_report(basic_path)
    enhanced = load_report(enhanced_path)

    comparison = print_comparison(basic, enhanced)

    # 提取配置信息（兼容两种格式）
    basic_config = _get_config_info(basic)
    enhanced_config = _get_config_info(enhanced)

    result = {
        "basic": {
            **basic_config,
            "evaluated_at": basic["evaluated_at"],
            "dataset_version": basic.get("dataset_version", ""),
            "judge": basic.get("judge", {}),
            "retrieval_metrics": _get_metrics(basic) if not _is_gen_report(basic) else None,
            "generation_metrics": _get_gen_metrics(basic),
        },
        "enhanced": {
            **enhanced_config,
            "evaluated_at": enhanced["evaluated_at"],
            "dataset_version": enhanced.get("dataset_version", ""),
            "judge": enhanced.get("judge", {}),
            "retrieval_metrics": _get_metrics(enhanced) if not _is_gen_report(enhanced) else None,
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
