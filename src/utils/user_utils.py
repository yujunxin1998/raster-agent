"""用户 ID 相关的小工具函数。"""
from __future__ import annotations

DEFAULT_USER_ID = "default"


def resolve_user_id(user_id: str | None) -> str:
    """客户端未指定 user_id 时统一回退到 DEFAULT_USER_ID。

    Args:
        user_id: 客户端传入的用户 ID，可能为空。

    Returns:
        非空的用户 ID。
    """
    return user_id or DEFAULT_USER_ID
