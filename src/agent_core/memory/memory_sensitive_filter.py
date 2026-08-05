"""敏感信息检测：写入长期记忆前的一道防线。

原样迁移自 `src/core/memory/sensitive_filter.py`。规则覆盖：常见密钥/Token
格式、数据库连接串、邮箱+密码组合、中国大陆身份证号、银行卡号，命中任一
规则即认为内容包含敏感信息，默认拒绝写入长期记忆。
"""
from __future__ import annotations

import re

_SENSITIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),                                   # OpenAI/Anthropic 风格 API key
    re.compile(r"AKIA[0-9A-Z]{16}"),                                      # AWS Access Key
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret|password|passwd)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)(postgres|postgresql|mysql|mongodb|redis)://[^\s]+:[^\s]+@[^\s]+"),  # 带凭证的连接串
    re.compile(r"\b\d{17}[\dXx]\b"),                                      # 中国大陆身份证号（18 位）
    re.compile(r"\b\d{16,19}\b"),                                         # 银行卡号（16-19 位连续数字）
]


def contains_sensitive_info(content: str) -> bool:
    """检测内容是否包含密钥、密码、连接串、证件号等敏感信息。

    Args:
        content: 待检测的文本内容。

    Returns:
        命中任一规则返回 True。
    """
    if not content:
        return False
    return any(pattern.search(content) for pattern in _SENSITIVE_PATTERNS)
