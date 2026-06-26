"""RAG 消融实验脚本

对不同的 RAG 参数组合进行评估，量化每个参数对检索质量的贡献。

用法（在项目根目录执行，需 Milvus 已运行且包含数据）：

  python -m tests.evaluation.run_ablation --output reports/ablation.csv

工作原理：
  1. 定义参数组合（rag_mode × top_k × reranker_type × query_preprocessor_type
     × coarse_top_k × model）
  2. 每个组合通过子进程运行 evaluate_rag.py，注入环境变量
  3. 收集所有 JSON 结果，汇总为对比 CSV

消融维度：
  - rag_mode: basic / enhanced
  - top_k: 3 / 5 / 10
  - reranker_type: none / cross_encoder
  - query_preprocessor_type: none / rewrite
  - coarse_top_k: 10 / 20（enhanced 模式候选池大小）
  - generation_model: qwen-max / qwen-plus（生成模型对比）

局限：
  - chunk_size 不纳入消融（需重新入库文档，不适合脚本自动化）
  - embedding 模型不纳入消融（需重新入库文档）
  - 每个组合独立启动子进程，含完整 Milvus 连接 + 模型加载，耗时长
  - 消融实验的总时间 ≈ 组合数 × 单次评估时间（约 15+ 组合 × 2-5分钟）

输出文件：
  - ablation CSV：每个组合一行，列包含参数 + 各项指标
  - ablation JSON：完整汇总结果，含每次评估的原始 scores
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# 消融参数网格
#
# 每项是一个 dict，key 为环境变量名，value 为覆盖值。
# 不设置的环境变量继承当前 shell 的值。
# ---------------------------------------------------------------------------
ABLATION_COMBINATIONS = [
    # --- basic 模式基线 ---
    {
        "label": "basic (baseline k=3)",
        "RAG_MODE": "basic",
        "RAG_TOP_K": "3",
    },
    {
        "label": "basic k=5",
        "RAG_MODE": "basic",
        "RAG_TOP_K": "5",
    },
    {
        "label": "basic k=10",
        "RAG_MODE": "basic",
        "RAG_TOP_K": "10",
    },
    # --- enhanced: dense+sparse, no reranker, no preprocessor ---
    {
        "label": "enhanced (hybrid, no-rerank) k=3",
        "RAG_MODE": "enhanced",
        "QUERY_PREPROCESSOR_TYPE": "none",
        "RERANKER_TYPE": "none",
        "RERANK_COARSE_TOP_K": "20",
        "RERANKER_TOP_K": "3",
    },
    {
        "label": "enhanced (hybrid, no-rerank) k=5",
        "RAG_MODE": "enhanced",
        "QUERY_PREPROCESSOR_TYPE": "none",
        "RERANKER_TYPE": "none",
        "RERANK_COARSE_TOP_K": "20",
        "RERANKER_TOP_K": "5",
    },
    # --- enhanced: cross_encoder reranker, no preprocessor ---
    {
        "label": "enhanced (cross_encoder) k=3",
        "RAG_MODE": "enhanced",
        "QUERY_PREPROCESSOR_TYPE": "none",
        "RERANKER_TYPE": "cross_encoder",
        "RERANK_COARSE_TOP_K": "20",
        "RERANKER_TOP_K": "3",
    },
    {
        "label": "enhanced (cross_encoder) k=5",
        "RAG_MODE": "enhanced",
        "QUERY_PREPROCESSOR_TYPE": "none",
        "RERANKER_TYPE": "cross_encoder",
        "RERANK_COARSE_TOP_K": "20",
        "RERANKER_TOP_K": "5",
    },
    {
        "label": "enhanced (cross_encoder) k=10",
        "RAG_MODE": "enhanced",
        "QUERY_PREPROCESSOR_TYPE": "none",
        "RERANKER_TYPE": "cross_encoder",
        "RERANK_COARSE_TOP_K": "20",
        "RERANKER_TOP_K": "10",
    },
    # --- enhanced: cross_encoder + query rewrite ---
    {
        "label": "enhanced (cross_encoder+rewrite) k=3",
        "RAG_MODE": "enhanced",
        "QUERY_PREPROCESSOR_TYPE": "rewrite",
        "RERANKER_TYPE": "cross_encoder",
        "RERANK_COARSE_TOP_K": "20",
        "RERANKER_TOP_K": "3",
    },
    {
        "label": "enhanced (cross_encoder+rewrite) k=5",
        "RAG_MODE": "enhanced",
        "QUERY_PREPROCESSOR_TYPE": "rewrite",
        "RERANKER_TYPE": "cross_encoder",
        "RERANK_COARSE_TOP_K": "20",
        "RERANKER_TOP_K": "5",
    },
    # --- coarse_top_k 消融：候选池大小对精排的影响 ---
    {
        "label": "enhanced (cross_encoder, coarse=10) k=5",
        "RAG_MODE": "enhanced",
        "QUERY_PREPROCESSOR_TYPE": "none",
        "RERANKER_TYPE": "cross_encoder",
        "RERANK_COARSE_TOP_K": "10",
        "RERANKER_TOP_K": "5",
    },
    {
        "label": "enhanced (cross_encoder, coarse=30) k=5",
        "RAG_MODE": "enhanced",
        "QUERY_PREPROCESSOR_TYPE": "none",
        "RERANKER_TYPE": "cross_encoder",
        "RERANK_COARSE_TOP_K": "30",
        "RERANKER_TOP_K": "5",
    },
    # --- 生成模型消融：对比不同 LLM 对 generation 指标的影响 ---
    {
        "label": "enhanced (cross_encoder) k=5, model=qwen-plus",
        "RAG_MODE": "enhanced",
        "QUERY_PREPROCESSOR_TYPE": "none",
        "RERANKER_TYPE": "cross_encoder",
        "RERANK_COARSE_TOP_K": "20",
        "RERANKER_TOP_K": "5",
        "RAG_MODEL": "qwen-plus",
    },
    {
        "label": "enhanced (cross_encoder) k=5, model=qwen-max",
        "RAG_MODE": "enhanced",
        "QUERY_PREPROCESSOR_TYPE": "none",
        "RERANKER_TYPE": "cross_encoder",
        "RERANK_COARSE_TOP_K": "20",
        "RERANKER_TOP_K": "5",
        "RAG_MODEL": "qwen-max",
    },
]


def _run_one_ablation(combo: dict, idx: int, total: int) -> dict | None:
    """通过子进程运行一次评估

    子进程隔离确保每次评估的配置干净，避免 LRU cache 污染。
    """
    label = combo.pop("label", f"combo_{idx}")
    env = os.environ.copy()
    env.update(combo)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        logger.info(f"[{idx}/{total}] 开始: {label}  env={combo}")

        result = subprocess.run(
            [
                sys.executable, "-m", "tests.evaluation.evaluate_rag",
                "--output", tmp_path,
                "--output-format", "json",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟超时
        )

        if result.returncode != 0:
            logger.error(f"[{idx}/{total}] {label} 失败 (rc={result.returncode})")
            logger.error(f"  stderr: {result.stderr[-500:]}")
            return None

        with open(tmp_path, encoding="utf-8") as f:
            scores = json.load(f)

        scores["_ablation_label"] = label
        scores["_ablation_params"] = combo
        logger.info(f"[{idx}/{total}] {label} 完成: "
                    f"cp={scores['retrieval_metrics']['context_precision']:.3f}, "
                    f"cr={scores['retrieval_metrics']['context_recall']:.3f}")
        return scores

    except subprocess.TimeoutExpired:
        logger.error(f"[{idx}/{total}] {label} 超时（>10min）")
        return None
    except Exception as e:
        logger.error(f"[{idx}/{total}] {label} 失败: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_ablation(output_path: str | None = None):
    """执行完整消融实验并汇总结果

    Args:
        output_path: 汇总 CSV 输出路径（默认: reports/ablation_{timestamp}.csv）
    """
    total = len(ABLATION_COMBINATIONS)
    logger.info("=" * 60)
    logger.info(f"消融实验开始: {total} 个参数组合")
    logger.info("警告: 消融实验耗时较长，请确保 Milvus 已运行且包含向量数据")
    logger.info("=" * 60)

    all_scores = []
    for idx, combo in enumerate(ABLATION_COMBINATIONS, 1):
        # 复制 combo 避免 label 被 pop 修改原列表
        scores = _run_one_ablation(dict(combo), idx, total)
        if scores:
            all_scores.append(scores)

    if not all_scores:
        logger.error("所有消融实验均失败")
        sys.exit(1)

    # 汇总为 CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path or f"reports/ablation_{ts}.csv"
    json_path = Path(csv_path).with_suffix(".json")

    try:
        import pandas as pd

        rows = []
        for s in all_scores:
            row = {
                "label": s["_ablation_label"],
                "rag_mode": s["rag_mode"],
                "query_preprocessor_type": s["query_preprocessor_type"],
                "reranker_type": s["reranker_type"],
                "top_k": s["top_k"],
                "context_precision": s["retrieval_metrics"]["context_precision"],
                "context_recall": s["retrieval_metrics"]["context_recall"],
                "context_relevancy": s["retrieval_metrics"].get("context_relevancy"),
                "context_entity_recall": s["retrieval_metrics"].get("context_entity_recall"),
            }
            # 非 LLM 指标
            nlm = s.get("non_llm_metrics", {})
            for k, v in nlm.items():
                row[k] = v
            # 生成指标（可能为 null）
            gen = s.get("generation_metrics")
            if gen:
                for k in ("faithfulness", "answer_relevancy"):
                    row[f"gen_{k}"] = gen.get(k)
            rows.append(row)

        df = pd.DataFrame(rows)
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"消融汇总 CSV 已保存: {csv_path}")

    except ImportError:
        logger.warning("pandas 未安装，跳过汇总 CSV。请运行: pip install pandas")

    # 保存完整 JSON
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)
    logger.info(f"消融完整 JSON 已保存: {json_path}")

    # 打印摘要
    _print_ablation_summary(all_scores)
    logger.info("=" * 60)
    logger.info("消融实验完成")


def _print_ablation_summary(all_scores: list[dict]):
    """打印消融实验摘要表格"""
    print()
    print("=" * 90)
    print("  消融实验摘要")
    print("=" * 90)
    header = (
        f"  {'组合':<38} {'cp':>8} {'cr':>8} {'crel':>8} {'cer':>8} {'hit@3':>8} {'mrr':>8}"
    )
    print(header)
    print("-" * 90)

    for s in all_scores:
        label = s["_ablation_label"][:36]
        cp = s["retrieval_metrics"]["context_precision"]
        cr = s["retrieval_metrics"]["context_recall"]
        crel = s["retrieval_metrics"].get("context_relevancy", 0)
        cer = s["retrieval_metrics"].get("context_entity_recall", 0)
        nlm = s.get("non_llm_metrics", {})
        h3 = nlm.get("hit_rate@3", 0)
        mr = nlm.get("mrr", 0)
        print(f"  {label:<38} {cp:>8.4f} {cr:>8.4f} {crel:>8.4f} {cer:>8.4f} {h3:>8.4f} {mr:>8.4f}")

    # 找出最优组合
    best = max(all_scores, key=lambda s: (
        s["retrieval_metrics"]["context_precision"] + s["retrieval_metrics"]["context_recall"]
    ))
    print("-" * 90)
    print(f"  最优组合 (cp+cr): {best['_ablation_label'][:50]}")
    print(f"    cp={best['retrieval_metrics']['context_precision']:.4f}  "
          f"cr={best['retrieval_metrics']['context_recall']:.4f}")
    print("=" * 90)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 消融实验")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="汇总 CSV 输出路径（默认: reports/ablation_{timestamp}.csv）",
    )
    parser.add_argument(
        "--combos", "-c",
        type=str,
        default=None,
        help="可选：逗号分隔的组合索引（从 0 开始），用于只跑部分组合。如: --combos 0,3,5",
    )
    args = parser.parse_args()

    if args.combos:
        indices = [int(i.strip()) for i in args.combos.split(",")]
        ABLATION_COMBINATIONS = [ABLATION_COMBINATIONS[i] for i in indices]
        logger.info(f"仅运行 {len(ABLATION_COMBINATIONS)} 个选定组合: {args.combos}")

    run_ablation(output_path=args.output)
