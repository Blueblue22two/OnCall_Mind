"""Regression tests for DashScope embedding request batching."""

from types import SimpleNamespace

from app.services.vector_embedding_service import DashScopeEmbeddings


def test_embed_documents_splits_requests_at_dashscope_limit() -> None:
    service = DashScopeEmbeddings(api_key="test-api-key", dimensions=2, batch_size=10)
    calls: list[list[str]] = []

    def create(**kwargs):
        batch = kwargs["input"]
        calls.append(batch)
        data = [
            SimpleNamespace(index=index, embedding=[float(text[1:]), 1.0])
            for index, text in enumerate(batch)
        ]
        return SimpleNamespace(data=data)

    service.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    vectors = service.embed_documents([f"t{index}" for index in range(23)])

    assert [len(batch) for batch in calls] == [10, 10, 3]
    assert len(vectors) == 23
    assert vectors[0] == [0.0, 1.0]
    assert vectors[-1] == [22.0, 1.0]


def test_embedding_batch_size_rejects_provider_invalid_value() -> None:
    try:
        DashScopeEmbeddings(api_key="test-api-key", batch_size=11)
    except ValueError as exc:
        assert "1~10" in str(exc)
    else:
        raise AssertionError("batch_size=11 should be rejected")
