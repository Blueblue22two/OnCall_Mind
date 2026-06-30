"""RAGAs 评估脚本

用法（在项目根目录执行）：

  # 评估 basic 模式（仅检索指标，跳过生成评估）
  RAG_MODE=basic python -m tests.evaluation.evaluate_rag

  # 评估 enhanced 模式（需要 Milvus 已包含 biz_enhanced 数据）
  RAG_MODE=enhanced \\
  QUERY_PREPROCESSOR_TYPE=rewrite \\
  RERANKER_TYPE=cross_encoder \\
  python -m tests.evaluation.evaluate_rag

  # 包含生成评估（faithfulness + answer_relevancy，模型基于已检索 contexts 生成 answer）
  RAG_MODE=basic python -m tests.evaluation.evaluate_rag --with-generation

  # 完整生成评估（含 answer_correctness）
  RAG_MODE=basic python -m tests.evaluation.evaluate_rag --with-generation --generation-metrics full

  # 指定输出路径
  RAG_MODE=basic python -m tests.evaluation.evaluate_rag --output reports/basic.json

评估指标（两阶段）：
  Phase 1 - 检索评估：context_precision, context_recall, context_relevancy,
                        context_entity_recall
  Phase 2 - 生成评估：faithfulness, answer_relevancy（需要 --with-generation）
            --generation-metrics full 时额外包含 answer_correctness

目标基线（basic 模式）：context_precision ≥ 0.70, context_recall ≥ 0.70
Enhanced 模式目标：      context_precision ≥ 0.80, context_recall ≥ 0.80
"""

import argparse
import asyncio
import importlib.metadata
import json
import math
import random
import sys
import subprocess
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger


def _git_revision() -> str:
    """Return the current commit hash without failing outside a Git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _judge_cache_namespace(
    model: str,
    api_base: str,
    temperature: float,
    ragas_version: str,
) -> str:
    """Build a stable cache namespace for one Judge configuration."""
    return json.dumps(
        {
            "schema": 2,
            "model": model,
            "api_base": api_base,
            "temperature": temperature,
            "ragas_version": ragas_version,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _judge_metric_name(prompt: str) -> str:
    """Extract the RAGAS metric family so it is explicit in every cache key."""
    normalized = prompt.lower().replace("_", " ")
    aliases = (
        ("context entity recall", "context_entity_recall"),
        ("context precision", "context_precision"),
        ("context recall", "context_recall"),
        ("context relevance", "context_relevancy"),
        ("faithfulness", "faithfulness"),
        ("answer relevancy", "answer_relevancy"),
        ("answer correctness", "answer_correctness"),
    )
    for marker, name in aliases:
        if marker in normalized:
            return name
    return "unknown_metric"


def _bootstrap_confidence_interval(
    scores: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """对分数列表进行 bootstrap 重采样，返回均值及置信区间。

    Args:
        scores: 有效的逐样本分数列表。
        n_bootstrap: 重采样次数。
        confidence: 置信水平（默认 0.95）。
        seed: 随机种子，保证可复现。

    Returns:
        dict: {"mean": float, "ci_lower": float, "ci_upper": float, "n": int}
    """
    if not scores or len(scores) < 2:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": len(scores)}

    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = (1 - confidence) / 2
    ci_lower = means[int(alpha * n_bootstrap)]
    ci_upper = means[int((1 - alpha) * n_bootstrap)]
    mean = sum(scores) / n

    return {
        "mean": round(mean, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n": n,
    }


def _extract_per_sample_scores(
    ragas_result: Any,
    metric_name: str,
) -> List[Optional[float]]:
    """从 RAGAS 评估结果中提取逐样本分数。

    RAGAs 0.2.x 返回的 result[metric_name] 可能是一个列表（逐样本分数）
    或单个聚合值。此函数统一提取为列表形式。
    """
    value = _ragas_metric_value(ragas_result, metric_name)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        scores = []
        for item in value:
            try:
                s = float(item)
                scores.append(s if math.isfinite(s) else None)
            except (TypeError, ValueError):
                scores.append(None)
        return scores
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _ragas_metric_value(ragas_result: Any, metric_name: str, default: Any = None) -> Any:
    """Read a metric from RAGAS results across 0.2.x dicts and 0.4.x EvaluationResult.

    RAGAS 0.4 returns ``EvaluationResult`` which supports ``result[name]`` but
    does not implement ``dict.get``.
    """
    if hasattr(ragas_result, "get"):
        return ragas_result.get(metric_name, default)
    try:
        return ragas_result[metric_name]
    except (KeyError, TypeError, AttributeError):
        return default


def _coerce_metric_score(
    value: Any,
    metric_name: str,
    return_per_sample: bool = False,
) -> Any:
    """Convert RAGAs metric output to a stable aggregate float.

    RAGAs versions differ in whether ``result[metric]`` returns an aggregate
    score or per-sample scores. With ``raise_exceptions=False``, failed jobs can
    also leave None/NaN values in the per-sample list.

    When ``return_per_sample=True``, returns ``(aggregate: float, per_sample: list)``.
    """
    if isinstance(value, (list, tuple)):
        scores = []
        aligned_scores: list[float | None] = []
        skipped = 0
        for item in value:
            try:
                score = float(item)
            except (TypeError, ValueError):
                skipped += 1
                aligned_scores.append(None)
                continue
            if math.isfinite(score):
                scores.append(score)
                aligned_scores.append(score)
            else:
                skipped += 1
                aligned_scores.append(None)

        if not scores:
            logger.warning(f"{metric_name} 没有可用分数，返回 0.0")
            if return_per_sample:
                return 0.0, aligned_scores
            return 0.0

        if skipped:
            logger.warning(
                f"{metric_name} 有 {skipped} 个无效/超时分数被忽略，"
                f"使用 {len(scores)} 个有效分数取平均"
            )
        agg = round(sum(scores) / len(scores), 4)
        if return_per_sample:
            return agg, aligned_scores
        return agg

    try:
        score = float(value)
    except (TypeError, ValueError):
        logger.warning(f"{metric_name} 返回值无法转换为 float: {type(value).__name__}，返回 0.0")
        if return_per_sample:
            return 0.0, []
        return 0.0

    if not math.isfinite(score):
        logger.warning(f"{metric_name} 返回 NaN/Inf，返回 0.0")
        if return_per_sample:
            return 0.0, []
        return 0.0

    agg = round(score, 4)
    if return_per_sample:
        return agg, [agg]
    return agg


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
    """批量检索，返回上下文字符串、文档元数据和检索诊断信息

    Returns:
        tuple: (all_contexts: list[list[str]], all_file_names: list[list[str]],
                all_diagnostics: list[dict])
    """
    all_contexts = []
    all_file_names = []
    all_section_ids = []
    all_diagnostics = []

    for i, question in enumerate(questions, 1):
        try:
            docs = retriever.retrieve(question, top_k=top_k)
            contexts = [doc.page_content for doc in docs]
            file_names = [doc.metadata.get("_file_name", "") for doc in docs]
            section_ids = [
                f"{doc.metadata.get('_file_name', '')}::{doc.metadata.get('h2', '')}"
                for doc in docs
            ]
            # 捕获检索诊断信息（P0-1）
            diag = getattr(retriever, "last_retrieval_meta", {})
        except Exception as e:
            logger.error(f"检索失败: question='{question[:40]}', error={e}")
            contexts = []
            file_names = []
            section_ids = []
            diag = {"error": str(e), "chunk_diagnostics": None}

        all_contexts.append(contexts if contexts else [""])
        all_file_names.append(file_names if file_names else [])
        all_section_ids.append(section_ids if section_ids else [])
        all_diagnostics.append(diag)
        if i % 10 == 0 or i == len(questions):
            logger.debug(f"[{i}/{len(questions)}] '{question[:40]}...' → {len(contexts) if contexts else 0} 段上下文")

    return all_contexts, all_file_names, all_section_ids, all_diagnostics


def _truncate_contexts(contexts_list: list[list[str]], top_k: int) -> list[list[str]]:
    """将检索上下文截断到 RAGAS/生成实际使用的 top_k。"""
    truncated = []
    for contexts in contexts_list:
        if not contexts or contexts == [""]:
            truncated.append([""])
            continue
        sliced = contexts[:top_k]
        truncated.append(sliced if sliced else [""])
    return truncated


def _build_per_question_rows(
    questions: list,
    categories: list,
    contexts_list: list[list[str]],
    file_names_list: list[list[str]],
    relevant_docs_map: list[list[str]],
    section_ids_list: list[list[str]] | None = None,
    relevant_sections_map: list[list[str]] | None = None,
    retrieval_diagnostics: list[dict] | None = None,
) -> list[dict]:
    """构建逐题明细骨架，后续再填充 RAGAS 分数和生成分数。"""
    per_question = []
    for i, question in enumerate(questions):
        contexts = contexts_list[i]
        file_names = file_names_list[i] if i < len(file_names_list) else []
        relevant_docs = relevant_docs_map[i] if i < len(relevant_docs_map) else []
        sections = section_ids_list[i] if section_ids_list and i < len(section_ids_list) else []
        relevant_sections = (
            relevant_sections_map[i]
            if relevant_sections_map and i < len(relevant_sections_map)
            else []
        )
        diag = retrieval_diagnostics[i] if retrieval_diagnostics and i < len(retrieval_diagnostics) else {}
        chunk_diagnostics = deepcopy(diag.get("chunk_diagnostics"))
        if chunk_diagnostics:
            for chunk in chunk_diagnostics.get("chunks", []):
                output_rank = chunk.get("output_rank")
                chunk["selected_in_context"] = bool(
                    output_rank and output_rank <= len(contexts)
                )
        pq = {
            "index": i,
            "question": question,
            "category": categories[i],
            "contexts_count": len(contexts) if contexts != [""] else 0,
            "retrieved_docs": file_names,
            "relevant_docs": relevant_docs,
            "retrieved_sections": sections,
            "relevant_sections": relevant_sections,
            "hit": bool(set(file_names) & set(relevant_docs)),
            "retrieval_diagnostics": chunk_diagnostics,
            "retrieval_latency_ms": {
                key: diag.get(key)
                for key in (
                    "preprocessing_time_ms",
                    "hybrid_search_time_ms",
                    "reranker_time_ms",
                    "total_time_ms",
                )
                if diag.get(key) is not None
            },
        }
        per_question.append(pq)
    return per_question


def _build_llm_wrapper(judge_timeout_s: int | None = None):
    """构建 RAGAs 需要的 LLM 和 Embeddings 包装器

    Judge 使用独立的 eval_judge_* 配置，与线上 RAG 模型解耦，确保评估可复现。
    支持通过 eval_judge_api_base / eval_judge_api_key 指定外部 Judge API，
    为空时复用 DashScope 配置。

    当 eval_judge_temperature=0 时启用 Judge 缓存，相同 prompt 不重复调用 API。
    缓存使用带命名空间的 SQLite 后端，模型、API、温度或 RAGAS 版本变化时
    自动进入不同命名空间，避免跨 Judge 污染。
    """
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from langchain_community.cache import SQLiteCache
        from app.config import config
        from app.services.vector_embedding_service import vector_embedding_service

        judge_api_base = config.eval_judge_api_base or config.dashscope_api_base
        judge_api_key = config.eval_judge_api_key or config.dashscope_api_key

        timeout = judge_timeout_s or config.llm_timeout

        use_cache = config.eval_judge_temperature == 0.0
        cache = None
        if use_cache:
            try:
                ragas_version = importlib.metadata.version("ragas")
            except importlib.metadata.PackageNotFoundError:
                ragas_version = "unknown"
            namespace = _judge_cache_namespace(
                config.eval_judge_model,
                judge_api_base,
                config.eval_judge_temperature,
                ragas_version,
            )

            class _NamespacedSQLiteCache(SQLiteCache):
                def lookup(self, prompt: str, llm_string: str):
                    metric = _judge_metric_name(prompt)
                    return super().lookup(prompt, f"{namespace}||{metric}||{llm_string}")

                def update(self, prompt: str, llm_string: str, return_val) -> None:
                    from sqlalchemy.exc import IntegrityError

                    metric = _judge_metric_name(prompt)
                    try:
                        super().update(
                            prompt,
                            f"{namespace}||{metric}||{llm_string}",
                            return_val,
                        )
                    except IntegrityError:
                        # RAGAS evaluates samples concurrently. Two workers can
                        # miss the same key and race to insert an identical
                        # response; the winning row is already a valid cache hit.
                        logger.debug("Judge cache 并发写入命中已有 key，忽略重复结果")

            cache_path = Path(config.eval_judge_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache = _NamespacedSQLiteCache(database_path=str(cache_path))

        llm = ChatOpenAI(
            model=config.eval_judge_model,
            temperature=config.eval_judge_temperature,
            api_key=judge_api_key,
            base_url=judge_api_base,
            timeout=timeout,
            max_retries=config.llm_max_retries,
            cache=cache,
        )

        # RAGAS 0.4 marks LangChain wrappers as deprecated, but they remain the
        # safest bridge here because the project uses OpenAI-compatible Judge
        # config plus a custom DashScope embedding service.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LangchainLLMWrapper is deprecated.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="LangchainEmbeddingsWrapper is deprecated.*",
                category=DeprecationWarning,
            )
            ragas_llm = LangchainLLMWrapper(llm)
            ragas_embeddings = LangchainEmbeddingsWrapper(vector_embedding_service)

        cache_msg = "（已启用 Judge 缓存）" if use_cache else ""
        logger.info(f"Judge LLM 初始化完成: model={config.eval_judge_model}{cache_msg}")
        return ragas_llm, ragas_embeddings

    except ImportError as e:
        logger.error(f"RAGAs 依赖未安装: {e}")
        logger.error("请运行: pip install 'ragas>=0.2.0' 'datasets>=2.0.0'")
        sys.exit(1)


async def _generate_answers(
    questions: list,
    contexts_list: list[list[str]],
) -> list[str]:
    """Phase 2: 让生成模型严格基于 Phase 1 的检索上下文回答。

    生成评估不能再次调用 Agent 的检索工具，否则用于评分的 contexts 与模型实际
    看到的 contexts 可能不一致；加载无关 MCP 工具也会引入额外失败点。单条失败
    不阻塞整体流程，失败的问题对应 answer 为空字符串。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.core.llm_factory import create_chat_qwen

    model = create_chat_qwen(temperature=0.0, streaming=False)

    answers: list[str] = []
    total = len(questions)

    logger.info(f"开始为 {total} 个问题生成 answer（基于 Phase 1 固定 contexts）...")

    for i, (question, contexts) in enumerate(zip(questions, contexts_list)):
        try:
            context_text = "\n\n".join(
                f"[上下文 {j + 1}]\n{context}" for j, context in enumerate(contexts)
            )
            response = await model.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "你是智能运维知识库问答助手。只能依据给定上下文回答；"
                            "上下文不足时明确说明，不得补充未经上下文支持的事实。"
                            "回答应直接、完整，并保留必要的排查步骤和命令。"
                        )
                    ),
                    HumanMessage(
                        content=f"问题：\n{question}\n\n可用上下文：\n{context_text}"
                    ),
                ]
            )
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            answers.append(str(content).strip() if content else "")
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
        precision_values = [
            float(it["context_precision"])
            for it in items
            if it.get("context_precision") is not None
        ]
        recall_values = [
            float(it["context_recall"])
            for it in items
            if it.get("context_recall") is not None
        ]
        stats[cat] = {
            "count": len(items),
            "avg_contexts_count": round(
                sum(it["contexts_count"] for it in items) / len(items), 1
            ),
            "answer_generated": sum(1 for it in items if it.get("answer_generated")),
            "context_precision": round(
                sum(precision_values) / len(precision_values), 4
            )
            if precision_values
            else None,
            "context_precision_n": len(precision_values),
            "context_recall": round(sum(recall_values) / len(recall_values), 4)
            if recall_values
            else None,
            "context_recall_n": len(recall_values),
        }

    return stats


def _flatten_scores(scores: dict) -> dict:
    """将嵌套的评估结果平铺为适合 CSV 的单层 dict"""
    flat = {
        "rag_mode": scores["rag_mode"],
        "query_preprocessor_type": scores["query_preprocessor_type"],
        "reranker_type": scores["reranker_type"],
        "top_k": scores["top_k"],
        "retrieval_pool_top_k": scores.get("retrieval_pool_top_k", scores["top_k"]),
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
    retrieval_metrics_mode: str = "minimal",
    ragas_timeout_s: int = 240,
    ragas_max_workers: int = 8,
    judge_timeout_s: int = 120,
    split: str = "all",
    sample_size: Optional[int] = None,
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
        retrieval_metrics_mode: 检索指标范围 ("minimal" | "full")。
            minimal 只跑 context_precision/context_recall；
            full 额外尝试 context_entity_recall/context_relevancy。
        ragas_timeout_s: RAGAS 单任务超时（秒）
        ragas_max_workers: RAGAS 并发 worker 数，调低可减少 Judge API 拥塞
        judge_timeout_s: Judge LLM 单请求超时（秒）
        split: 数据集划分 ("train" | "dev" | "test" | "all")
        sample_size: 可选，仅使用前 N 条样本（调试用，0 或 None = 全部）

    Returns:
        dict: 包含分组指标、逐题明细、分类统计的完整评估结果
    """
    from app.config import config
    from tests.evaluation.rag_testset import (
        DATASET_VERSION,
        EVALUATION_DATASET,
        get_eval_dataset,
        validate_testset,
        split_dataset,
    )

    # 0. 校验数据集
    errors = validate_testset(EVALUATION_DATASET)
    if errors:
        logger.error(f"数据集校验失败（{len(errors)} 条错误）:")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)

    # 0a. 数据集划分
    if split != "all":
        train, dev, test = split_dataset(EVALUATION_DATASET)
        split_map = {"train": train, "dev": dev, "test": test}
        eval_samples = split_map[split]
        logger.info(
            f"数据集划分: split='{split}', "
            f"total={len(EVALUATION_DATASET)} → selected={len(eval_samples)}, "
            f"train={len(train)}, dev={len(dev)}, test={len(test)}"
        )
        if not eval_samples:
            logger.error(f"split='{split}' 为空，请检查数据集大小")
            sys.exit(1)
        # 使用划分后的样本构建 testset
        from tests.evaluation import rag_testset

        testset = rag_testset._build_dataset_from_samples(eval_samples)
    else:
        eval_samples = EVALUATION_DATASET
        testset = get_eval_dataset()

    logger.info(f"数据集校验通过: {len(eval_samples)} 条样本, version={DATASET_VERSION}")

    # 0b. 采样（调试用）
    if sample_size and sample_size > 0 and sample_size < len(eval_samples):
        eval_samples = eval_samples[:sample_size]
        from tests.evaluation import rag_testset as _rts

        testset = _rts._build_dataset_from_samples(eval_samples)
        logger.info(f"采样: {sample_size} 条（数据集共 {len(EVALUATION_DATASET)} 条）")

    try:
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
            context_entity_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        )
        from ragas.run_config import RunConfig
    except ImportError as e:
        logger.error(f"RAGAs 导入失败: {e}")
        logger.error(
            "请检查 RAGAs 版本与依赖。推荐使用 pyproject.toml 中锁定的 "
            "'ragas>=0.2.10,<0.3.0'，或使用当前脚本的 0.4.x 兼容路径。"
        )
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
    logger.info(f"  检索指标模式:   {retrieval_metrics_mode}")
    logger.info(f"  RAGAS timeout:  {ragas_timeout_s}s, max_workers={ragas_max_workers}")
    logger.info(f"  Judge timeout:  {judge_timeout_s}s")
    logger.info(f"  数据集版本:     {DATASET_VERSION}")
    logger.info("=" * 60)

    # 1. 加载 RAG pipeline
    retriever = _build_rag_pipeline()

    # 2. 提取问题列表（testset 已在上面加载）
    questions = testset["question"]
    ground_truths = testset["ground_truth"]
    categories = testset["category"]
    logger.info(f"测试集加载完成: {len(testset)} 条")

    # 3. Phase 1: 检索上下文（含文档元数据，用于 Hit Rate/MRR）
    logger.info("--- Phase 1: 检索评估 ---")
    non_llm_ks = (3, 5, 10)
    retrieval_pool_k = max(effective_top_k, max(non_llm_ks))
    (
        retrieval_pool_contexts,
        retrieval_pool_file_names,
        retrieval_pool_section_ids,
        retrieval_diagnostics,
    ) = _retrieve_docs(
        retriever,
        list(questions),
        top_k=retrieval_pool_k,
    )
    context_top_ks = [effective_top_k for _ in questions]
    if config.rag_mode == "enhanced" and config.rag_query_routing:
        from app.retriever.query_router import classify_query

        context_top_ks = [
            config.rag_cross_doc_top_k
            if classify_query(question).query_type == "cross_doc"
            else effective_top_k
            for question in questions
        ]
    all_contexts = [
        contexts[:query_top_k] if contexts != [""] else [""]
        for contexts, query_top_k in zip(retrieval_pool_contexts, context_top_ks)
    ]
    all_file_names = [
        names[:query_top_k]
        for names, query_top_k in zip(retrieval_pool_file_names, context_top_ks)
    ]
    all_section_ids = [
        sections[:query_top_k]
        for sections, query_top_k in zip(retrieval_pool_section_ids, context_top_ks)
    ]
    relevant_docs_map = testset["relevant_docs"]
    relevant_sections_map = testset["relevant_sections"]
    per_question = _build_per_question_rows(
        questions=list(questions),
        categories=list(categories),
        contexts_list=all_contexts,
        file_names_list=all_file_names,
        relevant_docs_map=relevant_docs_map,
        section_ids_list=all_section_ids,
        relevant_sections_map=relevant_sections_map,
        retrieval_diagnostics=retrieval_diagnostics,
    )

    # 构建检索评估 Dataset
    from datasets import Dataset

    retrieval_dataset = Dataset.from_dict({
        # RAGAS 0.2.x column names
        "question": list(questions),
        "contexts": all_contexts,
        "ground_truth": list(ground_truths),
        # RAGAS 0.4.x column names
        "user_input": list(questions),
        "retrieved_contexts": all_contexts,
        "reference": list(ground_truths),
    })
    logger.info("Phase 1 检索完成，准备评估...")

    # 构建 LLM Judge
    ragas_llm, ragas_embeddings = _build_llm_wrapper(judge_timeout_s=judge_timeout_s)
    ragas_run_config = RunConfig(
        timeout=ragas_timeout_s,
        max_retries=2,
        max_wait=30,
        max_workers=ragas_max_workers,
    )

    context_relevancy_metric = None
    context_relevancy_result_key = "context_relevancy"
    if retrieval_metrics_mode == "full":
        # RAGAS 0.2.x 暴露 context_relevancy；0.4.x 改为 ContextRelevance，
        # 且需要 modern InstructorLLM。该指标不可用时跳过。
        try:
            from ragas.metrics import context_relevancy as _context_relevancy

            context_relevancy_metric = _context_relevancy
        except (ImportError, AttributeError):
            try:
                from ragas.metrics.collections.context_relevance import ContextRelevance

                context_relevancy_metric = ContextRelevance(llm=ragas_llm)
                context_relevancy_result_key = "context_relevance"
            except Exception as e:
                logger.warning(f"context_relevancy/context_relevance 指标不可用，跳过: {e}")

    retrieval_metric_list = [
        context_precision,
        context_recall,
    ]
    if retrieval_metrics_mode == "full":
        retrieval_metric_list.append(context_entity_recall)
        if context_relevancy_metric is not None:
            retrieval_metric_list.append(context_relevancy_metric)

    # 执行检索指标评估（RAGAs LLM Judge 指标，含逐样本分数）
    retrieval_result = evaluate(
        dataset=retrieval_dataset,
        metrics=retrieval_metric_list,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=ragas_run_config,
        raise_exceptions=False,
    )

    # 提取聚合分数 + 逐样本分数
    cp_agg, cp_per_sample = _coerce_metric_score(
        retrieval_result["context_precision"],
        "context_precision",
        return_per_sample=True,
    )
    cr_agg, cr_per_sample = _coerce_metric_score(
        retrieval_result["context_recall"],
        "context_recall",
        return_per_sample=True,
    )
    crel_agg = 0.0
    cer_agg = 0.0
    if retrieval_metrics_mode == "full":
        crel_agg = _coerce_metric_score(
            _ragas_metric_value(
                retrieval_result,
                context_relevancy_result_key,
                _ragas_metric_value(retrieval_result, "context_relevancy", 0.0),
            ),
            "context_relevancy",
        )
        cer_agg = _coerce_metric_score(
            _ragas_metric_value(retrieval_result, "context_entity_recall", 0.0),
            "context_entity_recall",
        )

    retrieval_metrics = {
        "context_precision": cp_agg,
        "context_recall": cr_agg,
        "context_relevancy": crel_agg,
        "context_entity_recall": cer_agg,
    }

    # 为检索指标计算 bootstrap 置信区间
    retrieval_ci = {}
    for name, per_sample in [
        ("context_precision", cp_per_sample),
        ("context_recall", cr_per_sample),
    ]:
        valid_scores = [score for score in per_sample if score is not None]
        if valid_scores:
            retrieval_ci[name] = _bootstrap_confidence_interval(valid_scores)
        else:
            retrieval_ci[name] = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    # 计算非 LLM 检索指标（Hit Rate@k + MRR，基于 relevant_docs 标注）
    from tests.evaluation.metrics import (
        all_relevant_hit_at_k,
        compute_hit_rate_multi_k,
        compute_mrr,
        document_coverage_at_k,
        ndcg_at_k,
        section_hit_at_k,
    )

    hit_rates = compute_hit_rate_multi_k(
        retrieval_pool_file_names,
        relevant_docs_map,
        ks=non_llm_ks,
    )
    mrr = compute_mrr(retrieval_pool_file_names, relevant_docs_map)

    non_llm_metrics = {**hit_rates, "mrr": round(mrr, 4)}
    for k in non_llm_ks:
        non_llm_metrics[f"document_coverage@{k}"] = round(
            document_coverage_at_k(retrieval_pool_file_names, relevant_docs_map, k), 4
        )
        non_llm_metrics[f"all_relevant_hit@{k}"] = round(
            all_relevant_hit_at_k(retrieval_pool_file_names, relevant_docs_map, k), 4
        )
        non_llm_metrics[f"document_ndcg@{k}"] = round(
            ndcg_at_k(retrieval_pool_file_names, relevant_docs_map, k), 4
        )
        section_hit = section_hit_at_k(
            retrieval_pool_section_ids, relevant_sections_map, k
        )
        non_llm_metrics[f"section_hit@{k}"] = (
            round(section_hit, 4) if section_hit is not None else None
        )
        non_llm_metrics[f"section_ndcg@{k}"] = round(
            ndcg_at_k(retrieval_pool_section_ids, relevant_sections_map, k), 4
        )

    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = (len(ordered) - 1) * percentile
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return round(ordered[lower], 2)
        value = ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
        return round(value, 2)

    latency_metrics: dict[str, dict[str, float | int | None]] = {}
    for stage in (
        "preprocessing_time_ms",
        "hybrid_search_time_ms",
        "reranker_time_ms",
        "total_time_ms",
    ):
        values = [
            float(meta[stage])
            for meta in retrieval_diagnostics
            if meta.get(stage) is not None
        ]
        latency_metrics[stage] = {
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "n": len(values),
        }

    logger.info(f"  context_precision    : {retrieval_metrics['context_precision']:.4f}  (目标 ≥ 0.70)")
    logger.info(f"  context_recall       : {retrieval_metrics['context_recall']:.4f}  (目标 ≥ 0.70)")
    logger.info(f"  context_relevancy    : {retrieval_metrics['context_relevancy']:.4f}")
    logger.info(f"  context_entity_recall: {retrieval_metrics['context_entity_recall']:.4f}")
    for name, ci in retrieval_ci.items():
        if ci["n"] >= 2:
            logger.info(
                f"    {name} 95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] "
                f"(mean={ci['mean']:.4f}, n={ci['n']})"
            )
    logger.info("  [非 LLM 指标]")
    for k, v in non_llm_metrics.items():
        logger.info(f"    {k}: {v:.4f}" if v is not None else f"    {k}: N/A")
    logger.info("  [检索延迟]")
    for stage, values in latency_metrics.items():
        logger.info(f"    {stage}: p50={values['p50']}ms, p95={values['p95']}ms")

    # 将逐样本检索分数同步到 per_question。必须在 RAGAS evaluate 之后执行。
    for i in range(len(per_question)):
        if i < len(cp_per_sample):
            per_question[i]["context_precision"] = cp_per_sample[i]
        if i < len(cr_per_sample):
            per_question[i]["context_recall"] = cr_per_sample[i]

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
                # RAGAS 0.2.x column names
                "question": list(questions),
                "contexts": all_contexts,
                "ground_truth": list(ground_truths),
                "answer": [a if a else " " for a in answers],
                # RAGAS 0.4.x column names
                "user_input": list(questions),
                "retrieved_contexts": all_contexts,
                "reference": list(ground_truths),
                "response": [a if a else " " for a in answers],
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
                    run_config=ragas_run_config,
                    raise_exceptions=False,
                )

                # 提取聚合分数 + 逐样本分数
                faith_agg, faith_per_sample = _coerce_metric_score(
                    gen_result["faithfulness"],
                    "faithfulness",
                    return_per_sample=True,
                )
                relev_agg, relev_per_sample = _coerce_metric_score(
                    gen_result["answer_relevancy"],
                    "answer_relevancy",
                    return_per_sample=True,
                )

                generation_metrics = {
                    "faithfulness": faith_agg,
                    "answer_relevancy": relev_agg,
                    "answers_generated": answers_generated,
                    "answers_failed": answers_failed,
                }

                # 将逐样本生成分数同步到 per_question
                for i in range(len(per_question)):
                    if i < len(faith_per_sample):
                        per_question[i]["faithfulness"] = faith_per_sample[i]
                    if i < len(relev_per_sample):
                        per_question[i]["answer_relevancy"] = relev_per_sample[i]

                if generation_metrics_mode == "full":
                    corr_agg, corr_per_sample = _coerce_metric_score(
                        gen_result["answer_correctness"],
                        "answer_correctness",
                        return_per_sample=True,
                    )
                    generation_metrics["answer_correctness"] = corr_agg
                    for i in range(len(per_question)):
                        if i < len(corr_per_sample):
                            per_question[i]["answer_correctness"] = corr_per_sample[i]

                # 逐样本生成指标 bootstrap 置信区间
                gen_ci = {}
                for name, per_sample in [
                    ("faithfulness", faith_per_sample),
                    ("answer_relevancy", relev_per_sample),
                ]:
                    valid_scores = [score for score in per_sample if score is not None]
                    if valid_scores:
                        gen_ci[name] = _bootstrap_confidence_interval(valid_scores)
                    else:
                        gen_ci[name] = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
                if generation_metrics_mode == "full":
                    gen_ci["answer_correctness"] = _bootstrap_confidence_interval(
                        [score for score in corr_per_sample if score is not None]
                    )
                generation_metrics["confidence_intervals"] = gen_ci

                logger.info(f"  faithfulness      : {generation_metrics['faithfulness']:.4f}")
                logger.info(f"  answer_relevancy  : {generation_metrics['answer_relevancy']:.4f}")
                if generation_metrics_mode == "full":
                    logger.info(f"  answer_correctness: {generation_metrics['answer_correctness']:.4f}")
                for name, ci in gen_ci.items():
                    if ci["n"] >= 2:
                        logger.info(
                            f"    {name} 95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] "
                            f"(n={ci['n']})"
                        )
            except Exception as e:
                logger.error(f"Phase 2 生成评估失败: {e}")
                generation_metrics = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "answers_generated": answers_generated,
                    "answers_failed": answers_failed,
                    "error": str(e),
                }

    # 5. 检索→生成联合分析（当同时有检索和生成逐样本分数时）
    retrieval_gen_correlation = None
    if with_generation and cp_per_sample and generation_metrics.get("faithfulness") is not None:
        # 按 context_precision 分组分析 faithfulness
        high_retrieval = []
        low_retrieval = []
        valid_cp = sorted(score for score in cp_per_sample if score is not None)
        median_cp = valid_cp[len(valid_cp) // 2] if valid_cp else 0.5

        for pq in per_question:
            cp_val = pq.get("context_precision")
            faith_val = pq.get("faithfulness")
            if cp_val is not None and faith_val is not None:
                if cp_val >= median_cp:
                    high_retrieval.append(faith_val)
                else:
                    low_retrieval.append(faith_val)

        if high_retrieval and low_retrieval:
            high_mean = sum(high_retrieval) / len(high_retrieval)
            low_mean = sum(low_retrieval) / len(low_retrieval)
            retrieval_gen_correlation = {
                "median_context_precision": round(median_cp, 4),
                "high_retrieval_avg_faithfulness": round(high_mean, 4),
                "low_retrieval_avg_faithfulness": round(low_mean, 4),
                "faithfulness_gap": round(high_mean - low_mean, 4),
                "high_retrieval_count": len(high_retrieval),
                "low_retrieval_count": len(low_retrieval),
            }
            logger.info("  [检索→生成联合分析]")
            logger.info(f"    高检索质量组 (CP ≥ {median_cp:.2f}) faithfulness 均值: {high_mean:.4f}")
            logger.info(f"    低检索质量组 (CP < {median_cp:.2f}) faithfulness 均值: {low_mean:.4f}")
            logger.info(f"    faithfulness 差距: {high_mean - low_mean:.4f}")

    # 6. 分类统计
    category_stats = _compute_category_stats(categories, per_question)

    # 7. 组装结果
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
        "retrieval_pool_top_k": retrieval_pool_k,
        "retrieval_metrics_mode": retrieval_metrics_mode,
        "ragas_timeout_s": ragas_timeout_s,
        "ragas_max_workers": ragas_max_workers,
        "judge_timeout_s": judge_timeout_s,
        "evaluated_at": datetime.now().isoformat(),
        "num_questions": len(questions),
        "dataset_version": DATASET_VERSION,
        "judge": judge_meta,
        "retrieval_metrics": retrieval_metrics,
        "retrieval_confidence_intervals": retrieval_ci,
        "non_llm_metrics": non_llm_metrics,
        "retrieval_latency_ms": latency_metrics,
        "code_revision": _git_revision(),
        "retrieval_config": {
            "rag_top_k": config.rag_top_k,
            "reranker_top_k": config.reranker_top_k,
            "rerank_coarse_top_k": config.rerank_coarse_top_k,
            "query_preprocessor_type": config.query_preprocessor_type,
            "reranker_model": config.reranker_model,
            "reranker_model_path": config.reranker_model_path,
            "rag_diversify_by_file": config.rag_diversify_by_file,
            "rag_query_routing": config.rag_query_routing,
            "rag_cross_doc_top_k": config.rag_cross_doc_top_k,
            "rag_section_prior": config.rag_section_prior,
            "rag_diversity_score_margin": config.rag_diversity_score_margin,
            "rag_max_chunks_per_file": config.rag_max_chunks_per_file,
            "rag_parent_context": config.rag_parent_context,
            "rag_parent_context_max_tokens": config.rag_parent_context_max_tokens,
            "rag_chunk_strategy": config.rag_chunk_strategy,
            "rag_include_section_prefix": config.rag_include_section_prefix,
            "enhanced_collection_name": config.enhanced_collection_name,
            "judge_cache_path": config.eval_judge_cache_path,
        },
        "generation_metrics": generation_metrics if generation_metrics else None,
        "retrieval_gen_correlation": retrieval_gen_correlation,
        "category_stats": category_stats,
        "per_question": per_question,
        "failed_samples": failed_samples if failed_samples else [],
    }

    # 7. 打印摘要
    logger.info("=" * 60)
    logger.info("评估结果摘要")
    logger.info("  [检索指标]")
    logger.info(f"    context_precision : {retrieval_metrics['context_precision']:.4f}  (目标 ≥ 0.70)")
    logger.info(f"    context_recall    : {retrieval_metrics['context_recall']:.4f}  (目标 ≥ 0.70)")

    retrieval_passed = (
        retrieval_metrics["context_precision"] >= 0.70
        and retrieval_metrics["context_recall"] >= 0.70
    )
    logger.info(f"    检索达标（≥ 0.70）: {'✅ 是' if retrieval_passed else '❌ 否'}")

    if generation_metrics and generation_metrics.get("faithfulness") is not None:
        logger.info("  [生成指标]")
        logger.info(f"    faithfulness      : {generation_metrics['faithfulness']:.4f}")
        logger.info(f"    answer_relevancy  : {generation_metrics['answer_relevancy']:.4f}")
        if generation_metrics.get("answer_correctness") is not None:
            logger.info(f"    answer_correctness: {generation_metrics['answer_correctness']:.4f}")
        logger.info(f"    answer 生成: {generation_metrics['answers_generated']}/{len(questions)}"
                    f" (+{generation_metrics['answers_failed']} 失败)")

    if failed_samples:
        logger.warning(f"  失败样本: {len(failed_samples)} 条")

    logger.info("  [分类统计]")
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
    parser.add_argument(
        "--retrieval-metrics",
        type=str,
        choices=["minimal", "full"],
        default="minimal",
        dest="retrieval_metrics_mode",
        help=(
            "检索指标范围: minimal=context_precision+context_recall; "
            "full=额外尝试 context_entity_recall/context_relevancy (default: minimal)"
        ),
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=240,
        dest="ragas_timeout_s",
        help="RAGAS 单任务超时秒数 (default: 240)",
    )
    parser.add_argument(
        "--ragas-max-workers",
        type=int,
        default=8,
        dest="ragas_max_workers",
        help="RAGAS 并发 worker 数，调低可减少 Judge API 拥塞 (default: 8)",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=120,
        dest="judge_timeout_s",
        help="Judge LLM 单请求超时秒数 (default: 120)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test", "all"],
        default="all",
        help="数据集划分: train/dev/test/all (default: all，使用全部样本)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="仅评估前 N 条样本（调试用，默认全部）",
    )
    args = parser.parse_args()

    run_evaluation(
        output_path=args.output,
        with_generation=args.with_generation,
        output_format=args.output_format,
        generation_metrics_mode=args.generation_metrics_mode,
        retrieval_metrics_mode=args.retrieval_metrics_mode,
        ragas_timeout_s=args.ragas_timeout_s,
        ragas_max_workers=args.ragas_max_workers,
        judge_timeout_s=args.judge_timeout_s,
        split=args.split,
        sample_size=args.sample_size,
    )
