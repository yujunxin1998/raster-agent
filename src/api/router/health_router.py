"""健康检查接口。"""
from __future__ import annotations

from fastapi import APIRouter

from src.common.response import ApiResponse, success

router = APIRouter(prefix="/health")


@router.get("/", response_model=ApiResponse[dict], summary="健康检查")
async def health_check() -> ApiResponse:
    """返回服务存活状态，供负载均衡器/容器编排探活使用。"""
    return success({"status": "ok"})
