"""健康检查与可观测性接口

P1-2.4: 新增 /metrics 端点暴露 Prometheus 指标。
"""

from typing import Any
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from app.config import config
from app.core.milvus_client import milvus_manager
from loguru import logger

router = APIRouter()


@router.get("/health")
async def health_check():
    
    """健康检查接口
    检查服务状态和数据库连接状态
    
    Returns:
        JSONResponse: 健康检查结果
    """
    # 检查服务基本状态
    health_data: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
        "service": config.app_name,
        "version": config.app_version,
        "status": "healthy"
    }
    
    # 检查 Milvus 连接状态
    try:
        milvus_healthy = milvus_manager.health_check()
        milvus_status: str = "connected" if milvus_healthy else "disconnected"
        milvus_message: str = "Milvus 连接正常" if milvus_healthy else "Milvus 连接异常"
        health_data["milvus"] = {
            "status": milvus_status,
            "message": milvus_message
        }
    except Exception as e:
        logger.warning(f"Milvus 健康检查失败: {e}")
        health_data["milvus"] = {
            "status": "error",
            "message": f"Milvus 检查失败: {str(e)}"
        }
    
    # 判断整体健康状态
    overall_status = "healthy"
    status_code = 200
    
    # 如果 Milvus 不可用，服务不可用
    if health_data["milvus"]["status"] != "connected":
        overall_status = "unhealthy"
        status_code = 503
        health_data["error"] = "数据库不可用"
    
    health_data["status"] = overall_status
    
    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "message": "服务运行正常" if overall_status == "healthy" else "服务不可用",
            "data": health_data
        }
    )


@router.get("/metrics")
async def metrics():
    """Prometheus 指标端点（P1-2.4）。

    暴露 LLM 调用延迟/成功率/重试次数、Agent 执行次数/错误等指标。

    Returns:
        Response: Prometheus text format。
    """
    from app.core.metrics import get_metrics_response
    return Response(
        content=get_metrics_response(),
        media_type="text/plain; charset=utf-8",
    )
