"""Deterministic coverage and ranking metrics for RAG retrieval."""

from __future__ import annotations

import math
from collections.abc import Sequence


def document_coverage_at_k(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[Sequence[str]],
    k: int,
) -> float:
    """Return the mean fraction of relevant documents covered in top-k."""
    if not retrieved or not relevant:
        return 0.0
    scores = []
    for actual, expected in zip(retrieved, relevant):
        expected_set = set(expected)
        if not expected_set:
            continue
        scores.append(len(set(actual[:k]) & expected_set) / len(expected_set))
    return sum(scores) / len(scores) if scores else 0.0


def all_relevant_hit_at_k(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[Sequence[str]],
    k: int,
) -> float:
    """Return the share of queries whose every relevant document appears in top-k."""
    if not retrieved or not relevant:
        return 0.0
    hits = []
    for actual, expected in zip(retrieved, relevant):
        expected_set = set(expected)
        if not expected_set:
            continue
        hits.append(expected_set.issubset(set(actual[:k])))
    return sum(hits) / len(hits) if hits else 0.0


def section_hit_at_k(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[Sequence[str]],
    k: int,
) -> float | None:
    """Return section-level hit rate, ignoring samples without section labels."""
    hits = []
    for actual, expected in zip(retrieved, relevant):
        expected_set = set(expected)
        if not expected_set:
            continue
        hits.append(bool(set(actual[:k]) & expected_set))
    return (sum(hits) / len(hits)) if hits else None


def ndcg_at_k(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[Sequence[str]],
    k: int,
) -> float:
    """Compute binary-relevance nDCG@k over document or section identifiers."""
    values = []
    for actual, expected in zip(retrieved, relevant):
        expected_set = set(expected)
        if not expected_set:
            continue
        seen: set[str] = set()
        dcg = 0.0
        for rank, item in enumerate(actual[:k], 1):
            if item in expected_set and item not in seen:
                dcg += 1.0 / math.log2(rank + 1)
                seen.add(item)
        ideal_hits = min(len(expected_set), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        values.append(dcg / idcg if idcg else 0.0)
    return sum(values) / len(values) if values else 0.0
