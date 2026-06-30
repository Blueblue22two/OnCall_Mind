"""RAG 生成质量独立评估脚本

专注于评估 RAG Agent 的生成质量（答案质量），与检索评估（evaluate_rag.py）互补。

评估指标：
  RAGAs 指标（LLM Judge）:
    - faithfulness       : 答案是否忠实于检索上下文（无幻觉）
    - answer_relevancy   : 答案是否与问题相关
    - answer_correctness : 答案是否与 ground_truth 事实一致

  自定义 LLM Judge 指标:
    - answer_completeness : 答案是否覆盖所有期望关键事实（0/1/2 评分）
    - hallucination_score : 答案是否包含检索上下文之外的事实（0/1/2 评分）

  非 LLM 指标:
    - semantic_similarity : 答案与 ground_truth 的语义相似度（embedding cosine）
    - response_latency_ms : 生成耗时（毫秒）

用法（在项目根目录执行）：

  # 评估 basic 模式的生成质量
  RAG_MODE=basic python -m tests.evaluation.evaluate_generation

  # 评估 enhanced 模式的生成质量
  RAG_MODE=enhanced python -m tests.evaluation.evaluate_generation

  # 快速验证（仅评估前 5 题）
  python -m tests.evaluation.evaluate_generation --sample-size 5

  # 跳过 LLM Judge 指标（仅计算 semantic_similarity + latency）
  python -m tests.evaluation.evaluate_generation --skip-llm-judge

  # 指定输出路径和格式
  python -m tests.evaluation.evaluate_generation --output reports/gen_basic.json --format both
"""

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Utility: 处理 RAGAs 版本输出不一致
# ---------------------------------------------------------------------------


def _coerce_metric_score(value: Any, metric_name: str) -> float:
    """Convert RAGAs metric output to a stable aggregate float.

    RAGAs versions differ in whether ``result[metric]`` returns an aggregate
    score or per-sample scores.
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


# ---------------------------------------------------------------------------
# Judge LLM 构建
# ---------------------------------------------------------------------------


def _build_ragas_llm_and_embeddings():
    """构建 RAGAs 需要的 LLM 和 Embeddings 包装器。

    Judge 使用独立的 eval_judge_* 配置，与线上 RAG 模型解耦。
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


def _build_custom_judge_llm():
    """构建自定义指标（completeness, hallucination）需要的 LangChain LLM。

    使用 ChatQwen（OpenAI 兼容模式），支持 ainvoke 异步调用。
    """
    try:
        from langchain_qwq import ChatQwen
        from app.config import config

        api_key = config.eval_judge_api_key or config.dashscope_api_key
        api_base = config.eval_judge_api_base or config.dashscope_api_base

        llm = ChatQwen(
            model=config.eval_judge_model,
            temperature=config.eval_judge_temperature,
            api_key=api_key,
            api_base=api_base,
        )
        return llm

    except ImportError as e:
        logger.error(f"依赖未安装: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Answer 生成
# ---------------------------------------------------------------------------


async def _generate_single_answer(
    index: int,
    question: str,
    total: int,
) -> dict[str, Any]:
    """为单个问题生成答案，记录 answer、检索上下文和耗时。

    Args:
        index: 问题序号。
        question: 问题文本。
        total: 总问题数。

    Returns:
        dict: {"answer": str, "contexts": list[str], "latency_ms": float, "error": str|None}
    """
    from app.services.rag_agent_service import rag_agent_service
    from app.retriever.factory import get_rag_retriever
    from app.config import config

    # 确定 effective top_k（与 knowledge_tool.py 保持一致）
    if config.rag_mode == "enhanced":
        effective_top_k = config.reranker_top_k
    else:
        effective_top_k = config.rag_top_k

    start = time.perf_counter()
    try:
        # 先获取检索上下文（用于后续幻觉检测）
        retriever = get_rag_retriever()
        try:
            docs = retriever.retrieve(question, top_k=effective_top_k)
            contexts = [doc.page_content for doc in docs]
        except Exception:
            contexts = []

        # 调用 RAG Agent 生成答案
        answer = await rag_agent_service.query(
            question, session_id=f"gen_eval_{index}"
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if answer:
            logger.info(f"[{index+1}/{total}] ✓ '{question[:50]}...' → {len(answer)} chars, {elapsed_ms:.0f}ms")
        else:
            logger.warning(f"[{index+1}/{total}] ✗ '{question[:50]}...' → 空回答")

        return {
            "answer": answer if answer else "",
            "contexts": contexts,
            "latency_ms": round(elapsed_ms, 1),
            "error": None if answer else "Agent 返回空回答",
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(f"[{index+1}/{total}] 生成失败: '{question[:50]}...' -> {e}")
        return {
            "answer": "",
            "contexts": [],
            "latency_ms": round(elapsed_ms, 1),
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 单题评分
# ---------------------------------------------------------------------------


async def _score_single_sample(
    index: int,
    question: str,
    ground_truth: str,
    gen_expected_facts: list[str],
    answer: str,
    contexts: list[str],
    latency_ms: float,
    has_error: bool,
    ragas_llm,
    ragas_embeddings,
    custom_judge_llm,
    skip_llm_judge: bool,
    skip_custom_judge: bool,
    embedding_service,
) -> dict[str, Any]:
    """对单个样本的生成结果进行多维度评分。

    Returns:
        dict: 包含所有指标的评分结果。
    """
    result: dict[str, Any] = {
        "index": index,
        "question": question[:100],
        "answer_length": len(answer) if answer else 0,
        "latency_ms": latency_ms,
        "error": has_error,
    }

    if has_error or not answer or not answer.strip():
        # 生成失败，所有指标置零
        result.update({
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "answer_correctness": 0.0,
            "answer_completeness": 0.0,
            "hallucination_score": 0.0,
            "semantic_similarity": 0.0,
        })
        return result

    # --- RAGAs 指标（LLM Judge） ---
    if not skip_llm_judge:
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
            from datasets import Dataset

            single_ds = Dataset.from_dict({
                "question": [question],
                "contexts": [contexts if contexts else [" "]],
                "ground_truth": [ground_truth],
                "answer": [answer],
            })

            ragas_result = evaluate(
                dataset=single_ds,
                metrics=[faithfulness, answer_relevancy, answer_correctness],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False,
            )

            result["faithfulness"] = _coerce_metric_score(
                ragas_result["faithfulness"], "faithfulness"
            )
            result["answer_relevancy"] = _coerce_metric_score(
                ragas_result["answer_relevancy"], "answer_relevancy"
            )
            result["answer_correctness"] = _coerce_metric_score(
                ragas_result["answer_correctness"], "answer_correctness"
            )
        except Exception as e:
            logger.error(f"[{index+1}] RAGAs 评分失败: {e}")
            result.update({
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "answer_correctness": 0.0,
            })
    else:
        result.update({
            "faithfulness": None,
            "answer_relevancy": None,
            "answer_correctness": None,
        })

    # --- 自定义 LLM Judge 指标 ---
    if not skip_custom_judge and custom_judge_llm is not None:
        # Answer Completeness
        try:
            from tests.evaluation.metrics import compute_answer_completeness

            completeness = await compute_answer_completeness(
                question=question,
                expected_facts=gen_expected_facts if gen_expected_facts else [ground_truth],
                agent_answer=answer,
                judge_llm=custom_judge_llm,
                num_trials=3,
            )
            result["answer_completeness"] = completeness["score"]
            result["completeness_detail"] = {
                "covered_facts": completeness.get("covered_facts", []),
                "missed_facts": completeness.get("missed_facts", []),
                "reason": completeness.get("reason", ""),
            }
        except Exception as e:
            logger.error(f"[{index+1}] Answer Completeness 评分失败: {e}")
            result["answer_completeness"] = 0.0

        # Hallucination Score
        try:
            from tests.evaluation.metrics import compute_hallucination_score

            hallucination = await compute_hallucination_score(
                question=question,
                agent_answer=answer,
                retrieved_contexts=contexts,
                judge_llm=custom_judge_llm,
                num_trials=1,
            )
            result["hallucination_score"] = hallucination["score"]
            result["hallucination_detail"] = {
                "hallucinated_claims": hallucination.get("hallucinated_claims", []),
                "reason": hallucination.get("reason", ""),
            }
        except Exception as e:
            logger.error(f"[{index+1}] Hallucination 评分失败: {e}")
            result["hallucination_score"] = 0.0
    else:
        result.update({
            "answer_completeness": None,
            "hallucination_score": None,
        })

    # --- 非 LLM 指标：语义相似度 ---
    try:
        from tests.evaluation.metrics import compute_semantic_similarity

        result["semantic_similarity"] = compute_semantic_similarity(
            answer=answer,
            ground_truth=ground_truth,
            embedding_service=embedding_service,
        )
    except Exception as e:
        logger.error(f"[{index+1}] Semantic Similarity 计算失败: {e}")
        result["semantic_similarity"] = 0.0

    return result


# ---------------------------------------------------------------------------
# 主评估函数
# ---------------------------------------------------------------------------


async def run_generation_evaluation(
    output_path: Optional[str] = None,
    output_format: str = "json",
    sample_size: Optional[int] = None,
    skip_llm_judge: bool = False,
    skip_custom_judge: bool = False,
    split: str = "all",
) -> dict[str, Any]:
    """执行完整的 RAG 生成质量评估。

    Args:
        output_path: 输出文件路径（默认：reports/gen_{mode}_{timestamp}.json）。
        output_format: 输出格式 ("json", "csv", "both")。
        sample_size: 可选，仅评估前 N 条（0 表示全部，调试用）。
        skip_llm_judge: 跳过 RAGAs LLM Judge 指标（faithfulness 等）。
        skip_custom_judge: 跳过自定义 LLM Judge 指标（completeness 等）。
        split: 数据集划分（train/dev/test/all）。

    Returns:
        dict: 完整评估结果。
    """
    from app.config import config
    from app.services.vector_embedding_service import vector_embedding_service
    from tests.evaluation.rag_testset import (
        DATASET_VERSION,
        EVALUATION_DATASET,
        _build_dataset_from_samples,
        split_dataset,
        validate_testset,
    )

    # 0. 校验数据集
    errors = validate_testset(EVALUATION_DATASET)
    if errors:
        logger.error(f"数据集校验失败（{len(errors)} 条错误）:")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)

    if split == "all":
        dataset_samples = list(EVALUATION_DATASET)
    else:
        train, dev, test = split_dataset(EVALUATION_DATASET)
        dataset_samples = {"train": train, "dev": dev, "test": test}[split]
        logger.info(f"数据集划分: split='{split}', samples={len(dataset_samples)}")

    # 在 split 之后应用 sample_size，避免调试采样改变冻结集合语义。
    if sample_size and sample_size > 0:
        dataset_samples = dataset_samples[:sample_size]
        logger.info(f"数据集采样: {len(dataset_samples)} 条")

    total = len(dataset_samples)
    logger.info(f"数据集校验通过: {total} 条样本, version={DATASET_VERSION}")

    # 确定 effective top_k
    if config.rag_mode == "enhanced":
        effective_top_k = config.reranker_top_k
    else:
        effective_top_k = config.rag_top_k

    logger.info("=" * 60)
    logger.info("RAG 生成质量评估开始")
    logger.info(f"  RAG 模式:          {config.rag_mode}")
    logger.info(f"  预处理方式:        {config.query_preprocessor_type}")
    logger.info(f"  精排器:            {config.reranker_type}")
    logger.info(f"  top_k:             {effective_top_k}")
    logger.info(f"  生成模型:          {config.rag_model}")
    logger.info(f"  Judge 模型:        {config.eval_judge_model}")
    logger.info(f"  Judge 温度:        {config.eval_judge_temperature}")
    logger.info(f"  样本数:            {total}")
    logger.info(f"  RAGAs LLM Judge:   {'跳过' if skip_llm_judge else '启用'}")
    logger.info(f"  自定义 LLM Judge:  {'跳过' if skip_custom_judge else '启用'}")
    logger.info(f"  数据集版本:        {DATASET_VERSION}")
    logger.info("=" * 60)

    # 1. 初始化 RAG Agent
    from app.services.rag_agent_service import rag_agent_service

    logger.info("初始化 RAG Agent...")
    await rag_agent_service._initialize_agent()

    # 2. 加载测试集并准备数据
    testset = _build_dataset_from_samples(dataset_samples)

    questions = testset["question"]
    ground_truths = testset["ground_truth"]
    categories = testset["category"]
    gen_expected_facts_list = testset["gen_expected_facts"]
    relevant_docs_list = testset["relevant_docs"]

    # 3. 构建 Judge LLM
    ragas_llm = None
    ragas_embeddings = None
    custom_judge_llm = None

    if not skip_llm_judge:
        ragas_llm, ragas_embeddings = _build_ragas_llm_and_embeddings()
        logger.info("RAGAs LLM Judge 初始化完成")

    if not skip_custom_judge:
        custom_judge_llm = _build_custom_judge_llm()
        logger.info("自定义 LLM Judge 初始化完成")

    # 4. Phase 1: 逐题生成回答
    logger.info("--- Phase 1: 生成回答 ---")
    gen_results = []
    for i, question in enumerate(questions):
        result = await _generate_single_answer(i, question, total)
        gen_results.append(result)

    answers = [r["answer"] for r in gen_results]
    all_contexts = [r["contexts"] for r in gen_results]
    all_latencies = [r["latency_ms"] for r in gen_results]
    all_errors = [r["error"] is not None for r in gen_results]

    success_count = sum(1 for a in answers if a)
    fail_count = total - success_count
    logger.info(f"生成完成: 成功={success_count} 失败={fail_count}")

    # 5. Phase 2: 逐题评分
    logger.info("--- Phase 2: 评分 ---")
    per_question = []

    for i in range(total):
        logger.debug(f"[{i+1}/{total}] 评分: '{questions[i][:50]}...'")
        score = await _score_single_sample(
            index=i,
            question=questions[i],
            ground_truth=ground_truths[i],
            gen_expected_facts=gen_expected_facts_list[i],
            answer=answers[i],
            contexts=all_contexts[i],
            latency_ms=all_latencies[i],
            has_error=all_errors[i],
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
            custom_judge_llm=custom_judge_llm,
            skip_llm_judge=skip_llm_judge,
            skip_custom_judge=skip_custom_judge,
            embedding_service=vector_embedding_service,
        )
        # 附加分类信息
        score["category"] = categories[i]
        score["relevant_docs"] = relevant_docs_list[i]
        per_question.append(score)

    # 6. 聚合统计
    def _safe_mean(values: list, default: float = 0.0) -> float:
        valid = [v for v in values if v is not None]
        return round(sum(valid) / len(valid), 4) if valid else default

    # RAGAs 指标聚合
    if not skip_llm_judge:
        agg_faithfulness = _safe_mean([s.get("faithfulness") for s in per_question])
        agg_relevancy = _safe_mean([s.get("answer_relevancy") for s in per_question])
        agg_correctness = _safe_mean([s.get("answer_correctness") for s in per_question])
    else:
        agg_faithfulness = None
        agg_relevancy = None
        agg_correctness = None

    # 自定义 Judge 指标聚合
    if not skip_custom_judge:
        agg_completeness = _safe_mean([s.get("answer_completeness") for s in per_question])
        agg_hallucination = _safe_mean([s.get("hallucination_score") for s in per_question])
    else:
        agg_completeness = None
        agg_hallucination = None

    # 非 LLM 指标聚合
    agg_semantic_sim = _safe_mean([s.get("semantic_similarity") for s in per_question])
    agg_latency = round(sum(all_latencies) / len(all_latencies), 1) if all_latencies else 0.0
    success_rate = round(success_count / total, 4) if total > 0 else 0.0

    aggregate_metrics: dict[str, Any] = {
        "faithfulness": agg_faithfulness,
        "answer_relevancy": agg_relevancy,
        "answer_correctness": agg_correctness,
        "answer_completeness": agg_completeness,
        "hallucination_score": agg_hallucination,
        "semantic_similarity": agg_semantic_sim,
        "avg_latency_ms": agg_latency,
        "success_rate": success_rate,
        "num_samples": total,
        "num_success": success_count,
        "num_failed": fail_count,
    }

    # 按 category 分组统计
    from collections import defaultdict

    cat_groups = defaultdict(list)
    for s in per_question:
        cat_groups[s["category"]].append(s)

    category_breakdown = {}
    for cat, items in sorted(cat_groups.items()):
        cat_metrics: dict[str, Any] = {"count": len(items)}
        if not skip_llm_judge:
            cat_metrics["faithfulness"] = _safe_mean([s.get("faithfulness") for s in items])
            cat_metrics["answer_relevancy"] = _safe_mean([s.get("answer_relevancy") for s in items])
            cat_metrics["answer_correctness"] = _safe_mean([s.get("answer_correctness") for s in items])
        if not skip_custom_judge:
            cat_metrics["answer_completeness"] = _safe_mean([s.get("answer_completeness") for s in items])
            cat_metrics["hallucination_score"] = _safe_mean([s.get("hallucination_score") for s in items])
        cat_metrics["semantic_similarity"] = _safe_mean([s.get("semantic_similarity") for s in items])
        cat_metrics["avg_latency_ms"] = _safe_mean([s.get("latency_ms") for s in items])
        category_breakdown[cat] = cat_metrics

    # 低质量样本识别（基于数据分布百分位数动态校准）
    def _calc_percentile_threshold(values: list, percentile: float = 10.0) -> float:
        """计算分位数阈值，用于低质量标记。"""
        valid = sorted([v for v in values if v is not None])
        if not valid:
            return 0.0
        idx = max(0, int(len(valid) * percentile / 100))
        return valid[idx]

    # 度量名称 → 使用的百分位数（分数越低越差的指标用 P25，越高越差的用 P75）
    PERCENTILE_CONFIG = {
        "faithfulness": 25,           # 低分为差
        "answer_relevancy": 25,
        "answer_correctness": 25,
        "answer_completeness": 25,
        "hallucination_score": 75,    # 高分为差（0=无幻觉, 2=严重幻觉）
        "semantic_similarity": 25,
    }

    # 基于实际数据分布计算阈值
    LOW_THRESHOLDS = {}
    for metric, percentile in PERCENTILE_CONFIG.items():
        all_vals = [s.get(metric) for s in per_question]
        threshold = _calc_percentile_threshold(all_vals, percentile)
        if percentile > 50:
            # 分数越高越差的指标（hallucination），阈值以上视为低质量
            LOW_THRESHOLDS[metric] = {
                "threshold": threshold,
                "comparison": "gt",
                "percentile": percentile,
            }
        else:
            LOW_THRESHOLDS[metric] = {
                "threshold": threshold,
                "comparison": "lt",
                "percentile": percentile,
            }
        logger.info(
            f"  低质量阈值 {metric}: {'>' if percentile > 50 else '<'} "
            f"{threshold:.2f} (P{percentile})"
        )

    low_quality_samples = []
    for s in per_question:
        low_flags = []
        for metric, threshold_config in LOW_THRESHOLDS.items():
            val = s.get(metric)
            if val is None:
                continue
            threshold = threshold_config["threshold"]
            if threshold_config["comparison"] == "gt" and val > threshold:
                low_flags.append(
                    f"{metric}={val:.2f}>{threshold:.2f}"
                    f"(P{threshold_config['percentile']})"
                )
            elif threshold_config["comparison"] == "lt" and val < threshold:
                low_flags.append(
                    f"{metric}={val:.2f}<{threshold:.2f}"
                    f"(P{threshold_config['percentile']})"
                )

        if low_flags:
            low_quality_samples.append({
                "index": s["index"],
                "question": s["question"],
                "category": s["category"],
                "flags": low_flags,
                "answer_preview": answers[s["index"]][:200] if answers[s["index"]] else "(empty)",
            })

    # 7. 组装结果
    judge_meta: dict[str, Any] = {
        "model": config.eval_judge_model,
        "temperature": config.eval_judge_temperature,
    }
    if config.eval_judge_api_base:
        judge_meta["api_base"] = config.eval_judge_api_base

    scores: dict[str, Any] = {
        "config": {
            "rag_mode": config.rag_mode,
            "query_preprocessor_type": config.query_preprocessor_type,
            "reranker_type": config.reranker_type,
            "top_k": effective_top_k,
            "generation_model": config.rag_model,
        },
        "evaluated_at": datetime.now().isoformat(),
        "dataset_version": DATASET_VERSION,
        "split": split,
        "judge": judge_meta,
        "aggregate_metrics": aggregate_metrics,
        "category_breakdown": category_breakdown,
        "low_quality_samples": low_quality_samples,
        "per_question": per_question,
    }

    # 8. 打印摘要
    logger.info("=" * 60)
    logger.info("RAG 生成质量评估结果摘要")
    logger.info(f"  样本数:           {total}")
    logger.info(f"  成功率:           {success_rate:.2%} ({success_count}/{total})")
    logger.info(f"  平均延迟:         {agg_latency:.0f} ms")
    logger.info("  --- RAGAs 指标 ---")
    if agg_faithfulness is not None:
        logger.info(f"  faithfulness:      {agg_faithfulness:.4f}  (目标 ≥ 0.75)")
    if agg_relevancy is not None:
        logger.info(f"  answer_relevancy:  {agg_relevancy:.4f}  (目标 ≥ 0.70)")
    if agg_correctness is not None:
        logger.info(f"  answer_correctness:{agg_correctness:.4f}  (目标 ≥ 0.65)")
    logger.info("  --- 自定义 Judge 指标 ---")
    if agg_completeness is not None:
        logger.info(f"  answer_completeness:{agg_completeness:.2f}  (目标 ≥ 1.0, 满分 2.0)")
    if agg_hallucination is not None:
        logger.info(f"  hallucination_score:{agg_hallucination:.2f}  (目标 ≥ 1.5, 满分 2.0)")
    logger.info("  --- 非 LLM 指标 ---")
    logger.info(f"  semantic_similarity:{agg_semantic_sim:.4f}")
    logger.info("  --- 分类统计 ---")
    for cat, info in category_breakdown.items():
        logger.info(f"    {cat}: {info['count']} 题")
    if low_quality_samples:
        logger.warning(f"  ⚠ 低质量样本: {len(low_quality_samples)} 条")
        for lq in low_quality_samples[:5]:
            logger.warning(f"    [{lq['index']}] {lq['category']}: {', '.join(lq['flags'])}")
    logger.info("=" * 60)

    # 9. 保存结果
    gen_suffix = "_no_judge" if skip_llm_judge else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path:
        out_path = Path(output_path)
        json_path = out_path if output_format in ("json", "both") else None
        csv_path = out_path.with_suffix(".csv") if output_format in ("csv", "both") else None
    else:
        default_stem = Path(f"reports/gen_{config.rag_mode}{gen_suffix}_{ts}")
        json_path = default_stem.with_suffix(".json") if output_format in ("json", "both") else None
        csv_path = default_stem.with_suffix(".csv") if output_format in ("csv", "both") else None

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 结果已保存: {json_path}")

    if csv_path:
        _save_gen_csv(scores, str(csv_path))

    return scores


# ---------------------------------------------------------------------------
# CSV 输出
# ---------------------------------------------------------------------------


def _flatten_gen_scores(scores: dict) -> dict:
    """将生成评估结果平铺为适合 CSV 的单层 dict。"""
    flat: dict[str, Any] = {
        "rag_mode": scores["config"]["rag_mode"],
        "query_preprocessor_type": scores["config"]["query_preprocessor_type"],
        "reranker_type": scores["config"]["reranker_type"],
        "top_k": scores["config"]["top_k"],
        "generation_model": scores["config"]["generation_model"],
        "evaluated_at": scores["evaluated_at"],
        "dataset_version": scores["dataset_version"],
        "judge_model": scores["judge"].get("model", ""),
        "judge_temperature": scores["judge"].get("temperature", ""),
    }

    metrics = scores.get("aggregate_metrics", {})
    metric_name_mapping = {
        "faithfulness": "gen_faithfulness",
        "answer_relevancy": "gen_answer_relevancy",
        "answer_correctness": "gen_answer_correctness",
        "answer_completeness": "gen_answer_completeness",
        "hallucination_score": "gen_hallucination_score",
        "semantic_similarity": "gen_semantic_similarity",
        "avg_latency_ms": "gen_avg_latency_ms",
        "success_rate": "gen_success_rate",
        "num_samples": "gen_num_samples",
        "num_success": "gen_num_success",
        "num_failed": "gen_num_failed",
    }

    for k, v in metrics.items():
        flat_name = metric_name_mapping.get(k, f"gen_{k}")
        if isinstance(v, (int, float)):
            flat[flat_name] = v

    return flat


def _save_gen_csv(scores: dict, csv_path: str):
    """将生成评估结果保存为 CSV 文件。"""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas 未安装，跳过 CSV 输出。请运行: pip install pandas")
        return

    flat = _flatten_gen_scores(scores)
    df = pd.DataFrame([flat])
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"CSV 结果已保存: {csv_path}")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 生成质量独立评估脚本")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径（默认：reports/gen_{mode}_{timestamp}.json）",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["json", "csv", "both"],
        default="json",
        dest="output_format",
        help="输出格式 (default: json)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="仅评估前 N 条样本（调试用，0=全部）",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        default=False,
        help="跳过 RAGAs LLM Judge 指标（faithfulness, answer_relevancy, answer_correctness）",
    )
    parser.add_argument(
        "--skip-custom-judge",
        action="store_true",
        default=False,
        help="跳过自定义 LLM Judge 指标（answer_completeness, hallucination_score）",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="all",
        help="数据集划分 (default: all)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_generation_evaluation(
            output_path=args.output,
            output_format=args.output_format,
            sample_size=args.sample_size,
            skip_llm_judge=args.skip_llm_judge,
            skip_custom_judge=args.skip_custom_judge,
            split=args.split,
        )
    )
