from __future__ import annotations

from langchain_core.documents import Document

from app.config import config
from app.retriever.enhanced import EnhancedRAGRetriever
from app.retriever.query_router import classify_query
from app.retriever.reranker.cross_encoder import CrossEncoderReranker
from app.services.document_splitter_service import DocumentSplitterService
from tests.evaluation.evaluate_rag import (
    _coerce_metric_score,
    _judge_cache_namespace,
    _judge_metric_name,
)
from tests.evaluation.check_rag_v4_gate import evaluate_gate
from tests.evaluation.metrics.retrieval_coverage import (
    all_relevant_hit_at_k,
    document_coverage_at_k,
    ndcg_at_k,
    section_hit_at_k,
)
from tests.evaluation.rag_testset import DATASET_VERSION, EVALUATION_DATASET, split_dataset


def _doc(name: str, score: float = 0.0, **metadata) -> Document:
    return Document(
        page_content=name,
        metadata={"_file_name": name, "_rerank_adjusted_score": score, **metadata},
    )


def test_coverage_metrics_measure_all_documents() -> None:
    retrieved = [["a.md", "a.md", "b.md"], ["x.md"]]
    relevant = [["a.md", "b.md"], ["x.md", "y.md"]]
    assert document_coverage_at_k(retrieved, relevant, 3) == 0.75
    assert all_relevant_hit_at_k(retrieved, relevant, 3) == 0.5
    assert 0.0 < ndcg_at_k(retrieved, relevant, 3) <= 1.0
    assert section_hit_at_k([["a::steps"]], [["a::steps"]], 1) == 1.0
    assert section_hit_at_k([["a::steps"]], [[]], 1) is None


def test_judge_cache_namespace_isolates_model_backend_and_version() -> None:
    base = _judge_cache_namespace("judge-a", "https://a/v1", 0.0, "0.4.3")
    assert base != _judge_cache_namespace("judge-b", "https://a/v1", 0.0, "0.4.3")
    assert base != _judge_cache_namespace("judge-a", "https://b/v1", 0.0, "0.4.3")
    assert base != _judge_cache_namespace("judge-a", "https://a/v1", 0.0, "0.4.4")
    assert _judge_metric_name("Evaluate Context Precision") == "context_precision"
    assert _judge_metric_name("Evaluate faithfulness") == "faithfulness"


def test_ragas_invalid_scores_keep_sample_alignment() -> None:
    aggregate, scores = _coerce_metric_score(
        [1.0, None, float("nan"), 0.0], "context_recall", return_per_sample=True
    )
    assert aggregate == 0.5
    assert scores == [1.0, None, None, 0.0]


def test_query_router_covers_supported_intents() -> None:
    assert classify_query("NetworkHighLatency 告警条件").query_type == "exact_keyword"
    assert classify_query("怎么查哪个进程 CPU 高").query_type == "procedural"
    assert classify_query("top 怎么看 CPU").skip_rewrite is True
    assert classify_query("磁盘和消息队列如何联动排查").query_type == "cross_doc"
    assert classify_query("缓存与数据库分别有什么异常").query_type == "cross_doc"
    assert classify_query("缓存为什么会雪崩").query_type == "general"


def test_section_child_split_has_stable_parent_metadata(monkeypatch) -> None:
    splitter = DocumentSplitterService()
    markdown = """# CPU SOP
## 排查步骤
### 获取指标
查询 CPU 指标并记录时间。
### 定位进程
使用 top 和 pidstat 定位进程。
## 常见原因分析
### 死循环
线程可能持续占用 CPU。
"""
    monkeypatch.setattr(config, "rag_parent_context_max_chars", 2400)
    docs = splitter.split_markdown(
        markdown,
        "cpu.md",
        strategy="section_child",
        include_section_prefix=True,
    )
    assert docs
    assert len({doc.metadata["chunk_id"] for doc in docs}) == len(docs)
    assert all(doc.metadata["parent_id"] for doc in docs)
    assert all(doc.metadata["section_path"] for doc in docs)
    assert {doc.metadata.get("h3") for doc in docs} >= {"获取指标", "定位进程", "死循环"}
    assert all(doc.page_content.startswith("文档: cpu.md") for doc in docs)
    parent_sections: dict[str, set[str]] = {}
    for doc in docs:
        parent_sections.setdefault(doc.metadata["parent_id"], set()).add(doc.metadata["h2"])
    assert all(len(sections) == 1 for sections in parent_sections.values())


def test_diagnostics_use_rerank_order() -> None:
    first = _doc("first.md")
    second = _doc("second.md")

    class FakeReranker:
        last_scores = [(0.9, second), (0.2, first)]

    diagnostics = EnhancedRAGRetriever._build_chunk_diagnostics(
        FakeReranker(), "cross_encoder", [first, second], [second], 1, False
    )
    assert diagnostics["chunks"][0]["file_name"] == "second.md"
    assert diagnostics["chunks"][0]["rank"] == 1
    assert diagnostics["chunks"][0]["output_rank"] == 1


def test_guarded_diversity_limits_duplicate_files_with_small_score_gap() -> None:
    ranked = [
        _doc("a.md", 1.0),
        _doc("a.md", 0.9),
        _doc("a.md", 0.8),
        _doc("b.md", 0.7),
    ]
    selected = EnhancedRAGRetriever._guarded_cross_doc_diversity(
        ranked, top_k=3, max_per_file=2, score_margin=0.15
    )
    assert [doc.metadata["_file_name"] for doc in selected] == ["a.md", "a.md", "b.md"]


def test_parent_expansion_avoids_duplicate_parent_context() -> None:
    docs = [
        _doc("a.md", parent_id="p1", _parent_content="parent", chunk_id="a1"),
        _doc("a.md", parent_id="p1", _parent_content="parent", chunk_id="a2"),
    ]
    expanded = EnhancedRAGRetriever._expand_parent_context(
        docs, max_chars=100, max_tokens=100
    )
    assert expanded[0].page_content == "parent"
    assert expanded[0].metadata["context_expanded"] == "parent"
    assert expanded[1].metadata["context_expanded"] == "child"


def test_cross_encoder_scores_even_when_candidate_count_equals_top_k() -> None:
    reranker = CrossEncoderReranker("unused")

    class FakeModel:
        def predict(self, pairs, apply_softmax=False):
            assert apply_softmax is False
            return [0.1, 0.9]

    reranker._model = FakeModel()
    docs = [_doc("low.md"), _doc("high.md")]
    result = reranker.rerank("query", docs, top_k=2)
    assert [doc.metadata["_file_name"] for doc in result] == ["high.md", "low.md"]


def test_v14_dev_samples_have_reviewed_sections() -> None:
    _, dev, test = split_dataset()
    assert DATASET_VERSION == "1.4.0"
    assert len(dev) == 25
    assert sum(sample.category == "cross_doc" for sample in dev) >= 10
    assert len(test) == 21
    assert all(sample.split_hint != "dev" for sample in test)
    assert all(sample.relevant_sections for sample in dev)
    assert all(len(sample.fact_sources) == len(sample.ground_truths) for sample in dev)
    assert all(sample.relevant_sections for sample in EVALUATION_DATASET)
    assert all(
        len(sample.fact_sources) == len(sample.ground_truths)
        for sample in EVALUATION_DATASET
    )


def test_v4_gate_checks_quality_category_and_latency() -> None:
    baseline = {
        "category_stats": {"exact_keyword": {"context_recall": 0.75}},
        "retrieval_latency_ms": {"total_time_ms": {"p95": 100.0}},
    }
    candidate = {
        "retrieval_metrics": {"context_precision": 0.71, "context_recall": 0.66},
        "category_stats": {
            "cross_doc": {"context_recall": 0.61},
            "exact_keyword": {"context_recall": 0.71},
        },
        "retrieval_latency_ms": {"total_time_ms": {"p95": 119.0}},
    }
    assert evaluate_gate(candidate, baseline) == []
    candidate["retrieval_latency_ms"]["total_time_ms"]["p95"] = 121.0
    assert any("P95" in failure for failure in evaluate_gate(candidate, baseline))
