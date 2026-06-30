"""Check the quality/latency admission gate for a v4 RAG dev report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    metrics = candidate.get("retrieval_metrics", {})
    cp = float(metrics.get("context_precision", 0.0))
    cr = float(metrics.get("context_recall", 0.0))
    if cp < 0.70:
        failures.append(f"context_precision={cp:.4f} < 0.70")
    if cr < 0.65:
        failures.append(f"context_recall={cr:.4f} < 0.65")

    cross_doc = candidate.get("category_stats", {}).get("cross_doc", {})
    cross_doc_cr = cross_doc.get("context_recall")
    if cross_doc_cr is None or float(cross_doc_cr) < 0.60:
        failures.append(f"cross_doc context_recall={cross_doc_cr} < 0.60")

    base_categories = baseline.get("category_stats", {})
    for category, base_values in base_categories.items():
        base_cr = base_values.get("context_recall")
        candidate_cr = candidate.get("category_stats", {}).get(category, {}).get(
            "context_recall"
        )
        if base_cr is None or candidate_cr is None:
            continue
        drop = float(base_cr) - float(candidate_cr)
        if drop > 0.05:
            failures.append(f"{category} context_recall drop={drop:.4f} > 0.05")

    candidate_p95 = (
        candidate.get("retrieval_latency_ms", {}).get("total_time_ms", {}).get("p95")
    )
    baseline_p95 = (
        baseline.get("retrieval_latency_ms", {}).get("total_time_ms", {}).get("p95")
    )
    if candidate_p95 is None or baseline_p95 is None:
        failures.append("total latency P95 missing")
    elif float(candidate_p95) > float(baseline_p95) * 1.20:
        failures.append(
            f"total latency P95={candidate_p95}ms > baseline {baseline_p95}ms × 1.20"
        )
    return failures


def evaluate_final_gate(
    candidate: dict[str, Any], generation: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    metrics = candidate.get("retrieval_metrics", {})
    for name in ("context_precision", "context_recall"):
        value = float(metrics.get(name, 0.0))
        if value < 0.70:
            failures.append(f"{name}={value:.4f} < 0.70")
    generation_metrics = generation.get("aggregate_metrics", {})
    for name in ("faithfulness", "answer_completeness", "hallucination_score"):
        if generation_metrics.get(name) is None:
            failures.append(f"generation metric missing: {name}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--generation")
    parser.add_argument("--stage", choices=["dev", "final"], default="dev")
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    if args.stage == "dev":
        if not args.baseline:
            parser.error("--baseline is required for --stage dev")
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        failures = evaluate_gate(candidate, baseline)
    else:
        if not args.generation:
            parser.error("--generation is required for --stage final")
        generation = json.loads(Path(args.generation).read_text(encoding="utf-8"))
        failures = evaluate_final_gate(candidate, generation)
    if failures:
        print("RAG v4 gate: FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"RAG v4 {args.stage} gate: PASSED")


if __name__ == "__main__":
    main()
