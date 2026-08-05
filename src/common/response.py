"""统一 API 响应封装。

所有 REST 接口的成功/失败响应都通过 ApiResponse 包装，前端固定按
`{code, msg, data, timestamp}` 解包（沿用 diit-agent-web 各 api/*.js 里的
`request()` 辅助函数约定），本工程的 Controller 层不允许直接返回裸 dict。
"""
from __future__ import annotations

import time
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

_SUCCESS_CODE = 200
_DEFAULT_ERROR_CODE = 500


class ApiResponse(BaseModel, Generic[T]):
    """统一响应体。

    Attributes:
        code: 业务状态码，200 表示成功，其余沿用 HTTP 状态码语义。
        msg: 状态说明，成功时固定为 "success"，失败时为可读的错误信息。
        data: 业务数据，失败时为 None。
        timestamp: 响应生成时刻的毫秒级时间戳。
    """

    code: int = _SUCCESS_CODE
    msg: str = "success"
    data: Optional[T] = None
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


def success(data: Any = None, msg: str = "success") -> ApiResponse:
    """构造一个成功响应。

    Args:
        data: 业务数据，默认为 None。
        msg: 状态说明，默认为 "success"。

    Returns:
        code=200 的 ApiResponse 实例。
    """
    return ApiResponse(code=_SUCCESS_CODE, msg=msg, data=data)


def error(code: int = _DEFAULT_ERROR_CODE, msg: str = "服务器内部错误", data: Any = None) -> ApiResponse:
    """构造一个失败响应。

    Args:
        code: 业务/HTTP 状态码，默认为 500。
        msg: 错误说明。
        data: 附加数据，通常为 None。

    Returns:
        对应 code 的 ApiResponse 实例。
    """
    return ApiResponse(code=code, msg=msg, data=data)
