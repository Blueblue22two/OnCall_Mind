"""结构化 Agent Trace 持久化模块（P1-2.1）

记录 AIOps Agent 每一步的完整执行轨迹，用于后续评估和分析。

Trace 数据结构包含：
  - 计划内容与变更
  - 每次工具调用的名称、参数、耗时、返回结果
  - 失败原因、重试次数
  - Token 消耗量（从 LLM response.usage_metadata 提取）
  - 最终状态

存储后端：Redis 优先，文件回退（复用 DiagnosisStore 模式）。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from app.config import config


class TraceStore:
    """Agent Trace 持久化存储。

    Redis 优先（如果配置了 REDIS_URL），否则使用本地文件。
    """

    def __init__(self):
        self._redis_client: Any = None
        self._file_dir = Path("agent_traces")
        self._file_dir.mkdir(exist_ok=True)
        if config.redis_url:
            try:
                import redis as redis_lib

                self._redis_client = redis_lib.from_url(config.redis_url)
                # 快速验证连接可用
                self._redis_client.ping()
                logger.info(f"TraceStore 使用 Redis: {config.redis_url}")
            except Exception as e:
                logger.warning(f"TraceStore Redis 连接失败 ({e})，回退到文件存储")
                self._redis_client = None

        if not self._redis_client:
            logger.info(f"TraceStore 使用文件存储: {self._file_dir}")

    def _file_save(self, key: str, data: str, prefix: str = "trace_") -> None:
        """文件存储回退方法。"""
        file_path = self._file_dir / f"{prefix}{key.replace(':', '_')}.json"
        file_path.write_text(data)

    def _save_to_storage(self, key: str, data: str, ttl_days: int = 14, file_prefix: str = "trace_") -> str:
        """统一的存储写入，Redis 优先，连接失败时自动回退文件。

        Args:
            key: 存储 key（不含前缀）。
            data: JSON 字符串。
            ttl_days: Redis TTL 天数。
            file_prefix: 文件前缀（trace_ 用于完整 trace，partial_ 用于节点 trace）。

        Returns:
            str: 写入成功的存储后端标识 "redis" / "file"。
        """
        redis_key = f"trace:{key}"
        if self._redis_client:
            try:
                self._redis_client.setex(redis_key, 86400 * ttl_days, data)
                return "redis"
            except Exception as e:
                logger.warning(f"Redis 写入失败 ({e})，回退到文件存储")
                self._redis_client = None
        # 文件回退
        self._file_save(key, data, prefix=file_prefix)
        return "file"

    def save_trace(self, trace: dict[str, Any]) -> str:
        """保存完整的 Agent Trace。

        Redis 优先，连接失败时自动回退本地文件。

        Args:
            trace: Trace 字典，必须包含 trace_id, session_id。

        Returns:
            str: trace_id。
        """
        trace_id = trace.get("trace_id", f"trace:{int(time.time() * 1000)}")
        trace.setdefault("saved_at", datetime.now().isoformat())
        trace.setdefault("app_version", config.app_version)

        data = json.dumps(trace, ensure_ascii=False, default=str)
        backend = self._save_to_storage(trace_id, data, ttl_days=14)
        logger.debug(f"Trace 已保存 ({backend}): {trace_id}")
        return trace_id

    def save_node_trace(self, trace_id: str, node_name: str, node_data: dict[str, Any]) -> None:
        """追加（或更新）单个节点的 trace 数据。

        用于增量写入：每个节点执行完即记录，避免最后一次性写入丢失中间数据。

        Args:
            trace_id: Trace ID。
            node_name: 节点名（planner/executor/replanner）。
            node_data: 节点执行数据。
        """
        key = f"node:{trace_id}:{node_name}"
        node_data["saved_at"] = datetime.now().isoformat()
        data = json.dumps(node_data, ensure_ascii=False, default=str)
        self._save_to_storage(key, data, ttl_days=1, file_prefix="partial_")

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """获取完整 Trace。

        Args:
            trace_id: Trace ID。

        Returns:
            dict | None: Trace 数据，不存在则返回 None。
        """
        file_path = self._file_dir / f"trace_{trace_id.replace(':', '_')}.json"

        if self._redis_client:
            try:
                data = self._redis_client.get(f"trace:{trace_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis 读取失败 ({e})，尝试文件读取")

        if file_path.exists():
            return json.loads(file_path.read_text())
        return None

    def list_by_session(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """列出某个会话的所有 Trace（按时间倒序）。

        Args:
            session_id: 会话ID。
            limit: 返回记录数上限。

        Returns:
            list[dict]: Trace 列表。
        """
        records: list[dict[str, Any]] = []

        # 从文件读取（始终可用）
        for f in sorted(self._file_dir.glob("trace_*.json"), reverse=True):
            try:
                record = json.loads(f.read_text())
                if record.get("session_id") == session_id:
                    records.append(record)
                    if len(records) >= limit:
                        break
            except Exception:
                continue

        # 也从 Redis 读取（合并去重）
        if self._redis_client:
            try:
                pattern = "trace:*"
                keys = list(self._redis_client.scan_iter(match=pattern, count=100))
                seen = {r.get("trace_id") for r in records}
                for key in keys:
                    data = self._redis_client.get(key)
                    if data:
                        record = json.loads(data)
                        if record.get("session_id") == session_id and record.get("trace_id") not in seen:
                            records.append(record)
            except Exception as e:
                logger.warning(f"Redis 列出失败 ({e})，仅返回文件结果")

        records.sort(key=lambda r: r.get("timestamp", r.get("saved_at", "")), reverse=True)
        return records[:limit]


# 提取 token_usage 的工具函数
def extract_token_usage(llm_response: Any) -> dict[str, int]:
    """从 LLM response 中提取 token 用量。

    兼容 ChatQwen / ChatOpenAI 的 usage_metadata 字段。

    Args:
        llm_response: LLM 响应对象。

    Returns:
        dict: {"input": int, "output": int, "total": int}
    """
    usage = {"input": 0, "output": 0, "total": 0}
    try:
        meta = getattr(llm_response, "usage_metadata", None)
        if meta:
            usage["input"] = meta.get("input_tokens", 0)
            usage["output"] = meta.get("output_tokens", 0)
            usage["total"] = meta.get("total_tokens", 0)
    except Exception:
        pass
    return usage


# 全局单例
_trace_store: Optional[TraceStore] = None


def get_trace_store() -> TraceStore:
    """获取 TraceStore 全局单例。"""
    global _trace_store
    if _trace_store is None:
        _trace_store = TraceStore()
    return _trace_store
