"""RAGAs 评估脚本

用法（在项目根目录执行）：

  # 评估 basic 模式
  RAG_MODE=basic python -m tests.evaluation.evaluate_rag

  # 评估 enhanced 模式（需要 Milvus 已包含 biz_enhanced 数据）
  RAG_MODE=enhanced \\
  QUERY_PREPROCESSOR_TYPE=rewrite \\
  RERANKER_TYPE=cross_encoder \\
  python -m tests.evaluation.evaluate_rag

  # 指定输出路径
  RAG_MODE=basic python -m tests.evaluation.evaluate_rag --output reports/basic.json

评估指标：
  - context_precision:  检索到的内容中有多少是相关的（精确率）
  - context_recall:     ground_truth 中有多少信息被检索到了（召回率）
  - faithfulness:       LLM 的回答是否忠实于检索到的上下文
  - answer_relevancy:   LLM 的回答是否切题

目标基线（basic 模式）：context_precision ≥ 0.70, context_recall ≥ 0.70
Enhanced 模式目标：      context_precision ≥ 0.80, context_recall ≥ 0.80
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def _build_rag_pipeline():
    """构建 RAG 检索 pipeline（懒导入，避免导入时副作用）"""
    from app.retriever.factory import get_rag_retriever
    from app.config import config

    retriever = get_rag_retriever()
    logger.info(f"RAG pipeline 初始化完成: mode={config.rag_mode}")
    return retriever


def _retrieve_contexts(retriever, question: str, top_k: int = 3) -> list[str]:
    """使用 RAG 检索器获取上下文列表"""
    try:
        docs = retriever.retrieve(question, top_k=top_k)
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"检索失败: question='{question[:40]}', error={e}")
        return []


def _build_ragas_dataset(retriever, testset, top_k: int = 3):
    """构建 RAGAs 评估所需的 Dataset（含 contexts 字段）

    RAGAs evaluate() 需要的字段：
      - question:     查询文本
      - contexts:     list[list[str]] —— 每个问题对应的检索上下文列表
      - ground_truth: 标准答案（字符串）
    """
    from datasets import Dataset

    logger.info(f"开始为 {len(testset)} 个问题检索上下文...")
    questions = testset["question"]
    ground_truths = testset["ground_truth"]

    all_contexts = []
    for i, question in enumerate(questions, 1):
        contexts = _retrieve_contexts(retriever, question, top_k=top_k)
        all_contexts.append(contexts if contexts else [""])
        logger.debug(f"[{i}/{len(questions)}] '{question[:40]}...' → {len(contexts)} 段上下文")

    return Dataset.from_dict({
        "question": list(questions),
        "contexts": all_contexts,
        "ground_truth": list(ground_truths),
    })


def _build_llm_wrapper():
    """构建 RAGAs 需要的 LLM 和 Embeddings 包装器（使用 ChatQwen）"""
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.chat_models import ChatTongyi
        from app.config import config
        from app.services.vector_embedding_service import vector_embedding_service

        llm = ChatTongyi(
            model=config.rag_model,
            temperature=0,
            dashscope_api_key=config.dashscope_api_key,
        )
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(vector_embedding_service)
        return ragas_llm, ragas_embeddings

    except ImportError as e:
        logger.error(f"RAGAs 依赖未安装: {e}")
        logger.error("请运行: pip install 'ragas>=0.2.0' 'datasets>=2.0.0'")
        sys.exit(1)


def run_evaluation(output_path: str | None = None) -> dict:
    """执行完整 RAGAs 评估

    Args:
        output_path: 可选，结果输出的 JSON 文件路径

    Returns:
        dict: 包含各项指标分数的字典
    """
    from app.config import config
    from tests.evaluation.rag_testset import get_eval_dataset

    try:
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        )
    except ImportError as e:
        logger.error(f"RAGAs 依赖未安装: {e}")
        logger.error("请运行: pip install 'ragas>=0.2.0' 'datasets>=2.0.0'")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"RAGAs 评估开始")
    logger.info(f"  RAG 模式:     {config.rag_mode}")
    logger.info(f"  预处理方式:   {config.query_preprocessor_type}")
    logger.info(f"  精排器:       {config.reranker_type}")
    logger.info(f"  top_k:        {config.rag_top_k}")
    logger.info("=" * 60)

    # 1. 加载 RAG pipeline
    retriever = _build_rag_pipeline()

    # 2. 加载测试集
    testset = get_eval_dataset()
    logger.info(f"测试集加载完成: {len(testset)} 条")

    # 3. 检索上下文（生成 contexts 字段）
    eval_dataset = _build_ragas_dataset(retriever, testset, top_k=config.rag_top_k)
    logger.info("上下文检索完成，准备调用 RAGAs 评估...")

    # 4. 构建 LLM/Embeddings 包装器
    ragas_llm, ragas_embeddings = _build_llm_wrapper()

    # 5. 执行 RAGAs 评估
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,    # 部分数据失败不中断整体评估
    )

    # 6. 整理结果
    scores = {
        "rag_mode": config.rag_mode,
        "query_preprocessor_type": config.query_preprocessor_type,
        "reranker_type": config.reranker_type,
        "top_k": config.rag_top_k,
        "evaluated_at": datetime.now().isoformat(),
        "num_questions": len(testset),
        "metrics": {
            "context_precision": float(result["context_precision"]),
            "context_recall":    float(result["context_recall"]),
            "faithfulness":      float(result["faithfulness"]),
            "answer_relevancy":  float(result["answer_relevancy"]),
        }
    }

    # 7. 打印摘要
    logger.info("=" * 60)
    logger.info("评估结果摘要")
    logger.info(f"  context_precision : {scores['metrics']['context_precision']:.4f}  (目标 ≥ 0.70)")
    logger.info(f"  context_recall    : {scores['metrics']['context_recall']:.4f}  (目标 ≥ 0.70)")
    logger.info(f"  faithfulness      : {scores['metrics']['faithfulness']:.4f}")
    logger.info(f"  answer_relevancy  : {scores['metrics']['answer_relevancy']:.4f}")

    passed = (
        scores["metrics"]["context_precision"] >= 0.70
        and scores["metrics"]["context_recall"] >= 0.70
    )
    logger.info(f"  达标（≥ 0.70）  : {'✅ 是' if passed else '❌ 否'}")
    logger.info("=" * 60)

    # 8. 保存 JSON 结果
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存: {out}")
    else:
        # 默认保存到 reports/
        default_filename = f"reports/eval_{config.rag_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out = Path(default_filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存: {out}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAs 评估脚本")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出 JSON 文件路径（默认：reports/eval_{mode}_{timestamp}.json）"
    )
    args = parser.parse_args()

    run_evaluation(output_path=args.output)
