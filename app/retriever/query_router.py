"""Zero-cost rule router for adaptive RAG retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

QueryType = Literal["exact_keyword", "procedural", "cross_doc", "general"]

_CROSS_MARKERS = ("联动", "串起来", "同时", "哪些方向", "哪些路径", "共同", "关系")
_PROCEDURAL_MARKERS = ("怎么查", "怎么处理", "怎么办", "步骤", "命令", "如何排查", "如何处理")
_DOMAIN_TERMS = (
    "CPU",
    "内存",
    "磁盘",
    "服务不可用",
    "消息队列",
    "消息积压",
    "数据库",
    "连接池",
    "网络",
    "缓存",
    "API",
    "OOM",
)
_CAMEL_CASE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]+)+\b")
_ERROR_CODE = re.compile(r"\b(?:[45]\d\d|[A-Z][A-Z0-9]+-\d+)\b")
_TOOL_NAME = re.compile(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b")
_KNOWN_TOOL = re.compile(r"\b(?:top|ps|pidstat|vmstat|iostat|sar|df|du|kubectl)\b", re.I)
_HTTP_CLASS = re.compile(r"\b[45]xx\b", re.I)


@dataclass(frozen=True)
class QueryRoute:
    query_type: QueryType
    skip_rewrite: bool = False
    target_section_types: tuple[str, ...] = ()


def classify_query(query: str) -> QueryRoute:
    """Classify a query without another model call."""
    domain_count = sum(1 for term in _DOMAIN_TERMS if term.lower() in query.lower())
    if domain_count >= 2 or any(marker in query for marker in _CROSS_MARKERS):
        return QueryRoute("cross_doc")
    if any(
        pattern.search(query)
        for pattern in (_CAMEL_CASE, _ERROR_CODE, _TOOL_NAME, _KNOWN_TOOL, _HTTP_CLASS)
    ):
        return QueryRoute("exact_keyword", skip_rewrite=True)
    if any(marker in query for marker in _PROCEDURAL_MARKERS):
        return QueryRoute(
            "procedural",
            target_section_types=("procedure", "commands"),
        )
    return QueryRoute("general")
