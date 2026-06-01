"""RAGAs 评估脚本

用法（在项目根目录执行）：

  # 评估 basic 模式（仅检索指标，跳过生成评估）
  RAG_MODE=basic python -m tests.evaluation.evaluate_rag

  # 评估 enhanced 模式（需要 Milvus 已包含 biz_enhanced 数据）
  RAG_MODE=enhanced \\
  QUERY_PREPROCESSOR_TYPE=rewrite \\
  RERANKER_TYPE=cross_encoder \\
  python -m tests.evaluation.evaluate_rag

  # 包含生成评估（faithfulness + answer_relevancy，需 Agent 生成 answer）
  RAG_MODE=basic python -m tests.evaluation.evaluate_rag --with-generation

  # 完整生成评估（含 answer_correctness）
  RAG_MODE=basic python -m tests.evaluation.evaluate_rag --with-generation --generation-metrics full

  # 指定输出路径
  RAG_MODE=basic python -m tests.evaluation.evaluate_rag --output reports/basic.json

评估指标（两阶段）：
  Phase 1 - 检索评估：context_precision, context_recall
  Phase 2 - 生成评估：faithfulness, answer_relevancy（需要 --with-generation）
            --generation-metrics full 时额外包含 answer_correctness

目标基线（basic 模式）：context_precision ≥ 0.70, context_recall ≥ 0.70
Enhanced 模式目标：      context_precision ≥ 0.80, context_recall ≥ 0.80
"""

import argparse
import asyncio
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger


def _coerce_metric_score(value: Any, metric_name: str) -> float:
    """Convert RAGAs metric output to a stable aggregate float.

    RAGAs versions differ in whether ``result[metric]`` returns an aggregate
    score or per-sample scores. With ``raise_exceptions=False``, failed jobs can
    also leave None/NaN values in the per-sample list.
    """
    if isinstance(value, (list, tuple)):
        scores = []
        skipped = 0
        for item in value:
            try:
                score = float(item)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if math.isfinite(score):
                scores.append(score)
            else:
                skipped += 1

        if not scores:
            logger.warning(f"{metric_name} 没有可用分数，返回 0.0")
            return 0.0

        if skipped:
            logger.warning(
                f"{metric_name} 有 {skipped} 个无效/超时分数被忽略，"
                f"使用 {len(scores)} 个有效分数取平均"
            )
        return round(sum(scores) / len(scores), 4)

    try:
        score = float(value)
    except (TypeError, ValueError):
        logger.warning(f"{metric_name} 返回值无法转换为 float: {type(value).__name__}，返回 0.0")
        return 0.0

    if not math.isfinite(score):
        logger.warning(f"{metric_name} 返回 NaN/Inf，返回 0.0")
        return 0.0

    return round(score, 4)


def _build_rag_pipeline():
    """构建 RAG 检索 pipeline"""
    from app.retriever.factory import get_rag_retriever
    from app.config import config

    retriever = get_rag_retriever()
    logger.info(f"RAG pipeline 初始化完成: mode={config.rag_mode}")
    return retriever


def _retrieve_contexts(retriever, question: str, top_k: int = 3) -> list[str]:
    """使用 RAG 检索器获取上下文列表（仅文本内容）"""
    try:
        docs = retriever.retrieve(question, top_k=top_k)
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"检索失败: question='{question[:40]}', error={e}")
        return []


def _retrieve_docs(retriever, questions: list, top_k: int = 3):
    """批量检索，返回上下文字符串和文档元数据（_file_name）

    Returns:
        tuple: (all_contexts: list[list[str]], all_file_names: list[list[str]])
    """
    all_contexts = []
    all_file_names = []

    for i, question in enumerate(questions, 1):
        try:
            docs = retriever.retrieve(question, top_k=top_k)
            contexts = [doc.page_content for doc in docs]
            file_names = [doc.metadata.get("_file_name", "") for doc in docs]
        except Exception as e:
            logger.error(f"检索失败: question='{question[:40]}', error={e}")
            contexts = []
            file_names = []

        all_contexts.append(contexts if contexts else [""])
        all_file_names.append(file_names if file_names else [])
        if i % 10 == 0 or i == len(questions):
            logger.debug(f"[{i}/{len(questions)}] '{question[:40]}...' → {len(contexts) if contexts else 0} 段上下文")

    return all_contexts, all_file_names


def _build_llm_wrapper():
    """构建 RAGAs 需要的 LLM 和 Embeddings 包装器

    Judge 使用独立的 eval_judge_* 配置，与线上 RAG 模型解耦，确保评估可复现。
    支持通过 eval_judge_api_base / eval_judge_api_key 指定外部 Judge API，
    为空时复用 DashScope 配置。
    """
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from app.config import config
        from app.services.vector_embedding_service import vector_embedding_service

        judge_api_base = config.eval_judge_api_base or config.dashscope_api_base
        judge_api_key = config.eval_judge_api_key or config.dashscope_api_key

        llm = ChatOpenAI(
            model=config.eval_judge_model,
            temperature=config.eval_judge_temperature,
            api_key=judge_api_key,
            base_url=judge_api_base,
        )
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(vector_embedding_service)
        return ragas_llm, ragas_embeddings

    except ImportError as e:
        logger.error(f"RAGAs 依赖未安装: {e}")
        logger.error("请运行: pip install 'ragas>=0.2.0' 'datasets>=2.0.0'")
        sys.exit(1)


async def _generate_answers(
    questions: list,
    contexts_list: list[list[str]],
) -> list[str]:
    """Phase 2: 通过 RagAgentService 为每个问题生成回答

    每个问题使用独立的 session_id="eval_{i}" 确保互不干扰。
    单条失败不阻塞整体流程，失败的问题对应 answer 为空字符串。
    """
    from app.services.rag_agent_service import rag_agent_service

    answers: list[str] = []
    total = len(questions)

    logger.info(f"开始为 {total} 个问题生成 answer（通过 RAG Agent）...")

    for i, (question, contexts) in enumerate(zip(questions, contexts_list)):
        try:
            answer = await rag_agent_service.query(
                question, session_id=f"eval_{i}"
            )
            answers.append(answer if answer else "")
            status = "✓" if answer else "✗(empty)"
        except Exception as e:
            logger.error(f"[{i+1}/{total}] answer 生成失败: '{question[:40]}...' -> {e}")
            answers.append("")

        if (i + 1) % 5 == 0 or i == total - 1:
            logger.info(
                f"[{i+1}/{total}] answer 生成进度: "
                f"成功={sum(1 for a in answers if a)} 失败={sum(1 for a in answers if not a)}"
            )

    return answers


def _compute_category_stats(
    categories: list,
    per_question: list[dict],
) -> dict:
    """按问题分类聚合统计（contexts 数量等元数据）"""
    from collections import defaultdict

    stats: dict = {}
    cat_groups = defaultdict(list)

    for pq in per_question:
        cat_groups[pq["category"]].append(pq)

    for cat, items in sorted(cat_groups.items()):
        stats[cat] = {
            "count": len(items),
            "avg_contexts_count": round(
                sum(it["contexts_count"] for it in items) / len(items), 1
            ),
            "answer_generated": sum(1 for it in items if it.get("answer_generated")),
        }

    return stats


def _flatten_scores(scores: dict) -> dict:
    """将嵌套的评估结果平铺为适合 CSV 的单层 dict"""
    flat = {
        "rag_mode": scores["rag_mode"],
        "query_preprocessor_type": scores["query_preprocessor_type"],
        "reranker_type": scores["reranker_type"],
        "top_k": scores["top_k"],
        "evaluated_at": scores["evaluated_at"],
        "num_questions": scores["num_questions"],
        "dataset_version": scores["dataset_version"],
        "judge_model": scores["judge"].get("model", ""),
        "judge_temperature": scores["judge"].get("temperature", ""),
    }
    # 检索指标
    for k, v in scores["retrieval_metrics"].items():
        flat[k] = v
    # 非 LLM 指标
    for k, v in scores.get("non_llm_metrics", {}).items():
        flat[k] = v
    # 生成指标
    gen = scores.get("generation_metrics")
    if gen:
        for k, v in gen.items():
            if isinstance(v, (int, float)):
                flat[f"gen_{k}"] = v
    return flat


def _save_csv(scores: dict, csv_path: str):
    """将评估结果保存为 CSV 文件"""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas 未安装，跳过 CSV 输出。请运行: pip install pandas")
        return

    flat = _flatten_scores(scores)
    df = pd.DataFrame([flat])
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"CSV 结果已保存: {csv_path}")


def run_evaluation(
    output_path: Optional[str] = None,
    with_generation: bool = False,
    output_format: str = "json",
    generation_metrics_mode: str = "minimal",
) -> dict:
    """执行分阶段 RAGAs 评估

    阶段划分：
      Phase 1（检索评估）— 始终执行
        检索 contexts → 评估 context_precision + context_recall
      Phase 2（生成评估）— 仅在 --with-generation 时执行
        Agent 生成 answer → 评估 faithfulness + answer_relevancy
        --generation-metrics full 时额外评估 answer_correctness

    Args:
        output_path: 可选，结果输出 JSON 文件路径
        with_generation: 是否执行 Phase 2 生成评估
        output_format: 输出格式 ("json", "csv", "both")
        generation_metrics_mode: 生成指标范围 ("minimal" | "full")

    Returns:
        dict: 包含分组指标、逐题明细、分类统计的完整评估结果
    """
    from app.config import config
    from tests.evaluation.rag_testset import (
        DATASET_VERSION,
        EVALUATION_DATASET,
        get_eval_dataset,
        validate_testset,
    )

    # 0. 校验数据集
    errors = validate_testset(EVALUATION_DATASET)
    if errors:
        logger.error(f"数据集校验失败（{len(errors)} 条错误）:")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)
    logger.info(f"数据集校验通过: {len(EVALUATION_DATASET)} 条样本, version={DATASET_VERSION}")

    try:
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        )
    except ImportError as e:
        logger.error(f"RAGAs 依赖未安装: {e}")
        logger.error("请运行: pip install 'ragas>=0.2.0' 'datasets>=2.0.0'")
        sys.exit(1)

    # 确定 effective top_k（与 knowledge_tool.py 保持一致）
    if config.rag_mode == "enhanced":
        effective_top_k = config.reranker_top_k
    else:
        effective_top_k = config.rag_top_k

    logger.info("=" * 60)
    logger.info("RAGAs 评估开始")
    logger.info(f"  RAG 模式:       {config.rag_mode}")
    logger.info(f"  预处理方式:     {config.query_preprocessor_type}")
    logger.info(f"  精排器:         {config.reranker_type}")
    logger.info(f"  top_k:          {effective_top_k}")
    logger.info(f"  Judge 模型:     {config.eval_judge_model}")
    logger.info(f"  Judge 温度:     {config.eval_judge_temperature}")
    gen_label = "关闭（仅检索指标）"
    if with_generation:
        gen_label = f"开启（{'完整' if generation_metrics_mode == 'full' else '基础'}指标）"
    logger.info(f"  生成评估:       {gen_label}")
    logger.info(f"  数据集版本:     {DATASET_VERSION}")
    logger.info("=" * 60)

    # 1. 加载 RAG pipeline
    retriever = _build_rag_pipeline()

    # 2. 加载测试集
    testset = get_eval_dataset()
    questions = testset["question"]
    ground_truths = testset["ground_truth"]
    categories = testset["category"]
    logger.info(f"测试集加载完成: {len(testset)} 条")

    # 3. Phase 1: 检索上下文（含文档元数据，用于 Hit Rate/MRR）
    logger.info("--- Phase 1: 检索评估 ---")
    all_contexts, all_file_names = _retrieve_docs(retriever, list(questions), top_k=effective_top_k)

    per_question = []
    for i, question in enumerate(questions):
        per_question.append({
            "index": i,
            "question": question,
            "category": categories[i],
            "contexts_count": len(all_contexts[i]) if all_contexts[i] != [""] else 0,
        })

    # 构建检索评估 Dataset
    from datasets import Dataset

    retrieval_dataset = Dataset.from_dict({
        "question": list(questions),
        "contexts": all_contexts,
        "ground_truth": list(ground_truths),
    })
    logger.info("Phase 1 检索完成，准备评估...")

    # 构建 LLM Judge
    ragas_llm, ragas_embeddings = _build_llm_wrapper()

    # 执行检索指标评估（RAGAs LLM Judge 指标）
    retrieval_result = evaluate(
        dataset=retrieval_dataset,
        metrics=[context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )

    retrieval_metrics = {
        "context_precision": _coerce_metric_score(
            retrieval_result["context_precision"],
            "context_precision",
        ),
        "context_recall": _coerce_metric_score(
            retrieval_result["context_recall"],
            "context_recall",
        ),
    }

    # 计算非 LLM 检索指标（Hit Rate@k + MRR，基于 relevant_docs 标注）
    relevant_docs_map = testset["relevant_docs"]
    from tests.evaluation.metrics import compute_hit_rate_multi_k, compute_mrr

    hit_rates = compute_hit_rate_multi_k(all_file_names, relevant_docs_map, ks=(3, 5, 10))
    mrr = compute_mrr(all_file_names, relevant_docs_map)

    non_llm_metrics = {**hit_rates, "mrr": round(mrr, 4)}

    logger.info(f"  context_precision : {retrieval_metrics['context_precision']:.4f}  (目标 ≥ 0.70)")
    logger.info(f"  context_recall    : {retrieval_metrics['context_recall']:.4f}  (目标 ≥ 0.70)")
    logger.info(f"  [非 LLM 指标]")
    for k, v in non_llm_metrics.items():
        logger.info(f"    {k}: {v:.4f}")

    # 4. Phase 2: 生成评估（可选）
    generation_metrics = {}
    failed_samples = []

    if with_generation:
        logger.info("--- Phase 2: 生成评估 ---")

        answers = asyncio.run(_generate_answers(questions, all_contexts))
        answers_generated = sum(1 for a in answers if a)
        answers_failed = sum(1 for a in answers if not a)

        # 更新 per_question
        for i, answer in enumerate(answers):
            per_question[i]["answer_generated"] = bool(answer and answer.strip())
            if not answer or not answer.strip():
                failed_samples.append({
                    "index": i,
                    "question": questions[i],
                    "reason": "answer 为空",
                })

        if answers_generated == 0:
            logger.warning("所有 answer 生成均失败，跳过 Phase 2 生成评估")
            generation_metrics = {
                "faithfulness": None,
                "answer_relevancy": None,
                "answers_generated": 0,
                "answers_failed": len(questions),
            }
        else:
            generation_dataset = Dataset.from_dict({
                "question": list(questions),
                "contexts": all_contexts,
                "ground_truth": list(ground_truths),
                "answer": [a if a else " " for a in answers],
            })

            try:
                gen_metrics_list = [faithfulness, answer_relevancy]
                if generation_metrics_mode == "full":
                    gen_metrics_list.append(answer_correctness)

                gen_result = evaluate(
                    dataset=generation_dataset,
                    metrics=gen_metrics_list,
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    raise_exceptions=False,
                )
                generation_metrics = {
                    "faithfulness": _coerce_metric_score(
                        gen_result["faithfulness"],
                        "faithfulness",
                    ),
                    "answer_relevancy": _coerce_metric_score(
                        gen_result["answer_relevancy"],
                        "answer_relevancy",
                    ),
                    "answers_generated": answers_generated,
                    "answers_failed": answers_failed,
                }
                if generation_metrics_mode == "full":
                    generation_metrics["answer_correctness"] = _coerce_metric_score(
                        gen_result["answer_correctness"],
                        "answer_correctness",
                    )
                logger.info(f"  faithfulness      : {generation_metrics['faithfulness']:.4f}")
                logger.info(f"  answer_relevancy  : {generation_metrics['answer_relevancy']:.4f}")
                if generation_metrics_mode == "full":
                    logger.info(f"  answer_correctness: {generation_metrics['answer_correctness']:.4f}")
            except Exception as e:
                logger.error(f"Phase 2 生成评估失败: {e}")
                generation_metrics = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "answers_generated": answers_generated,
                    "answers_failed": answers_failed,
                    "error": str(e),
                }

    # 5. 分类统计
    category_stats = _compute_category_stats(categories, per_question)

    # 6. 组装结果
    judge_meta = {
        "model": config.eval_judge_model,
        "temperature": config.eval_judge_temperature,
    }
    if config.eval_judge_api_base:
        judge_meta["api_base"] = config.eval_judge_api_base

    scores = {
        "rag_mode": config.rag_mode,
        "query_preprocessor_type": config.query_preprocessor_type,
        "reranker_type": config.reranker_type,
        "top_k": effective_top_k,
        "evaluated_at": datetime.now().isoformat(),
        "num_questions": len(questions),
        "dataset_version": DATASET_VERSION,
        "judge": judge_meta,
        "retrieval_metrics": retrieval_metrics,
        "non_llm_metrics": non_llm_metrics,
        "generation_metrics": generation_metrics if generation_metrics else None,
        "category_stats": category_stats,
        "per_question": per_question,
        "failed_samples": failed_samples if failed_samples else [],
    }

    # 7. 打印摘要
    logger.info("=" * 60)
    logger.info("评估结果摘要")
    logger.info(f"  [检索指标]")
    logger.info(f"    context_precision : {retrieval_metrics['context_precision']:.4f}  (目标 ≥ 0.70)")
    logger.info(f"    context_recall    : {retrieval_metrics['context_recall']:.4f}  (目标 ≥ 0.70)")

    retrieval_passed = (
        retrieval_metrics["context_precision"] >= 0.70
        and retrieval_metrics["context_recall"] >= 0.70
    )
    logger.info(f"    检索达标（≥ 0.70）: {'✅ 是' if retrieval_passed else '❌ 否'}")

    if generation_metrics and generation_metrics.get("faithfulness") is not None:
        logger.info(f"  [生成指标]")
        logger.info(f"    faithfulness      : {generation_metrics['faithfulness']:.4f}")
        logger.info(f"    answer_relevancy  : {generation_metrics['answer_relevancy']:.4f}")
        if generation_metrics.get("answer_correctness") is not None:
            logger.info(f"    answer_correctness: {generation_metrics['answer_correctness']:.4f}")
        logger.info(f"    answer 生成: {generation_metrics['answers_generated']}/{len(questions)}"
                    f" (+{generation_metrics['answers_failed']} 失败)")

    if failed_samples:
        logger.warning(f"  失败样本: {len(failed_samples)} 条")

    logger.info(f"  [分类统计]")
    for cat, info in category_stats.items():
        logger.info(f"    {cat}: {info['count']} 题, 平均 contexts={info['avg_contexts_count']}")

    logger.info("=" * 60)

    # 8. 保存结果（JSON / CSV / both）
    gen_suffix = "_full" if with_generation else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path:
        json_path = Path(output_path) if output_format in ("json", "both") else None
        csv_path = Path(output_path).with_suffix(".csv") if output_format in ("csv", "both") else None
    else:
        default_stem = Path(f"reports/eval_{config.rag_mode}{gen_suffix}_{ts}")
        json_path = default_stem.with_suffix(".json") if output_format in ("json", "both") else None
        csv_path = default_stem.with_suffix(".csv") if output_format in ("csv", "both") else None

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 结果已保存: {json_path}")

    if csv_path:
        _save_csv(scores, str(csv_path))

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAs 两阶段评估脚本")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径（默认：reports/eval_{mode}_{timestamp}.json）",
    )
    parser.add_argument(
        "--output-format", "-f",
        type=str,
        choices=["json", "csv", "both"],
        default="json",
        help="输出格式 (default: json)",
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        default=False,
        help="启用 Phase 2 生成评估（Agent 生成 answer，评估 faithfulness + answer_relevancy）",
    )
    parser.add_argument(
        "--generation-metrics",
        type=str,
        choices=["minimal", "full"],
        default="minimal",
        dest="generation_metrics_mode",
        help="生成指标范围: minimal=faithfulness+answer_relevancy, full=+answer_correctness (default: minimal)",
    )
    args = parser.parse_args()

    run_evaluation(
        output_path=args.output,
        with_generation=args.with_generation,
        output_format=args.output_format,
        generation_metrics_mode=args.generation_metrics_mode,
    )
