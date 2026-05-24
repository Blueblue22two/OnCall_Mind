"""诊断报告持久化存储

支持 Redis 和文件双后端，Redis 优先。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from app.config import config


class DiagnosisStore:
    """诊断报告持久化存储。

    Redis 优先（如果配置了 REDIS_URL），否则使用本地文件。
    """

    def __init__(self):
        self._redis_client: Any = None
        if config.redis_url:
            try:
                import redis as redis_lib

                self._redis_client = redis_lib.from_url(config.redis_url)
                logger.info(f"DiagnosisStore 使用 Redis: {config.redis_url}")
            except Exception as e:
                logger.warning(f"Redis 连接失败，回退到文件存储: {e}")

        if not self._redis_client:
            self._file_dir = Path("diagnosis_reports")
            self._file_dir.mkdir(exist_ok=True)
            logger.info(f"DiagnosisStore 使用文件存储: {self._file_dir}")

    def save(
        self,
        session_id: str,
        input_data: str,
        plan: list[str],
        past_steps: list[tuple[Any, Any]],
        response: str,
    ) -> str:
        """保存诊断记录。

        Args:
            session_id: 会话ID。
            input_data: 用户输入。
            plan: 执行计划步骤列表。
            past_steps: 已执行的步骤列表。
            response: 最终响应。

        Returns:
            str: 记录ID。
        """
        record_id = f"diagnosis:{session_id}:{int(time.time())}"
        record = {
            "record_id": record_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "plan": plan,
            "past_steps": [[step, str(result)[:200]] for step, result in past_steps],
            "response": response[:5000],  # 截断长响应
        }

        if self._redis_client:
            self._redis_client.setex(
                record_id,
                86400 * 7,  # 7天过期
                json.dumps(record, ensure_ascii=False, default=str),
            )
        else:
            file_path = self._file_dir / f"{record_id.replace(':', '_')}.json"
            file_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str)
            )

        return record_id

    def get(self, record_id: str) -> dict[str, Any] | None:
        """获取单条诊断记录。

        Args:
            record_id: 记录ID。

        Returns:
            dict | None: 诊断记录字典，不存在则返回 None。
        """
        if self._redis_client:
            data = self._redis_client.get(record_id)
            return json.loads(data) if data else None
        else:
            file_path = self._file_dir / f"{record_id.replace(':', '_')}.json"
            if file_path.exists():
                return json.loads(file_path.read_text())
            return None

    def list_by_session(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """列出某个会话的所有诊断记录（按时间倒序）。

        Args:
            session_id: 会话ID。
            limit: 返回记录数上限（默认 20）。

        Returns:
            list[dict]: 诊断记录列表。
        """
        records: list[dict[str, Any]] = []

        if self._redis_client:
            pattern = f"diagnosis:{session_id}:*"
            keys = list(self._redis_client.scan_iter(match=pattern, count=100))
            for key in keys:
                data = self._redis_client.get(key)
                if data:
                    records.append(json.loads(data))
        else:
            prefix = f"diagnosis_{session_id}_"
            for f in sorted(self._file_dir.glob(f"{prefix}*.json"), reverse=True):
                records.append(json.loads(f.read_text()))
                if len(records) >= limit:
                    break

        records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return records[:limit]


# 全局单例
_diagnosis_store: Optional[DiagnosisStore] = None


def get_diagnosis_store() -> DiagnosisStore:
    """获取 DiagnosisStore 全局单例。"""
    global _diagnosis_store
    if _diagnosis_store is None:
        _diagnosis_store = DiagnosisStore()
    return _diagnosis_store
