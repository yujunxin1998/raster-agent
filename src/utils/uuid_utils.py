"""ID 生成工具。"""
from __future__ import annotations

import uuid


def generate_uuid() -> str:
    """生成一个新的 UUID4 字符串，用于 conversation_id 等业务主键。"""
    return str(uuid.uuid4())
