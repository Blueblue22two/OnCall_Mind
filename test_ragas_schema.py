"""Minimal test: verify RAGAs 0.4.3 field schema + judge API with ChatOpenAI."""
import math
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from app.config import config
from app.services.vector_embedding_service import vector_embedding_service


def main():
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

    question = "CPU 告警后，我要怎么查是哪个进程在吃 CPU？"
    contexts = [
        "使用 top -c 命令按 CPU 使用率排序，定位高 CPU 进程。",
        "使用 ps -eo pid,ppid,cmd,%cpu,%mem --sort=-%cpu | head -10 获取 Top 10 CPU 进程。",
        "使用 pidstat 1 5 获取进程的实时 CPU 统计信息。",
    ]
    ground_truth = "使用 top -c 命令查看 CPU 使用率最高的进程。使用 ps 命令获取 Top 10 CPU 进程。使用 pidstat 获取实时 CPU 统计。"

    # Test 1: OLD schema
    print("=== Test 1: OLD schema (question/contexts/ground_truth) ===")
    ds_old = Dataset.from_dict({
        "question": [question],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    })
    result = evaluate(
        dataset=ds_old,
        metrics=[context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )
    print(f"  result.scores: {result.scores}")
    cp = result["context_precision"]
    cr = result["context_recall"]
    print(f"  context_precision = {cp!r}")
    print(f"  context_recall    = {cr!r}")

    # Test 2: NEW schema
    print("\n=== Test 2: NEW schema (user_input/retrieved_contexts/reference) ===")
    ds_new = Dataset.from_dict({
        "user_input": [question],
        "retrieved_contexts": [contexts],
        "reference": [ground_truth],
    })
    result2 = evaluate(
        dataset=ds_new,
        metrics=[context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )
    print(f"  result.scores: {result2.scores}")
    cp2 = result2["context_precision"]
    cr2 = result2["context_recall"]
    print(f"  context_precision = {cp2!r}")
    print(f"  context_recall    = {cr2!r}")


if __name__ == "__main__":
    main()
