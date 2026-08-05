"""输入清洗中间件（设计文档 4.2 节 #1）。

统一清洗用户输入，为流水线里后续所有中间件提供干净的消息——排在流水线
最外层，对应文档"输入清洗在最外层"的顺序原则。
"""
from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from loguru import logger

# 控制字符（不含 \t \n \r）视为噪音，直接剔除；连续空白折叠为单个空格。
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EXCESS_WHITESPACE_PATTERN = re.compile(r"[ \t]{2,}")


def _sanitize_text(text: str) -> str:
    """清洗单条文本：剔除控制字符，折叠多余空白，去首尾空白。

    Args:
        text: 原始文本。

    Returns:
        清洗后的文本。
    """
    cleaned = _CONTROL_CHAR_PATTERN.sub("", text)
    cleaned = _EXCESS_WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


class InputSanitizationMiddleware(AgentMiddleware):
    """在每次模型调用前，清洗最后一条 HumanMessage 的内容。"""

    async def abefore_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        """清洗 state 里最后一条 HumanMessage，按消息 id 原地替换。

        Args:
            state: 当前 Agent 状态，含 `messages` 列表。
            runtime: LangGraph 运行时（本中间件不使用）。

        Returns:
            需要更新的 state 字段（`add_messages` reducer 按消息 id 原地覆盖）；
            无需清洗（非文本内容/内容未变化）时返回 None，不产生多余的 state 更新。
        """
        messages = state.get("messages") or []
        last_human_index = next(
            (index for index in range(len(messages) - 1, -1, -1) if isinstance(messages[index], HumanMessage)),
            None,
        )
        if last_human_index is None:
            return None

        last_human = messages[last_human_index]
        if not isinstance(last_human.content, str):
            return None

        cleaned = _sanitize_text(last_human.content)
        if cleaned == last_human.content:
            return None

        logger.debug(f"[InputSanitizationMiddleware] 已清洗用户输入 message_id={last_human.id}")
        return {"messages": [HumanMessage(content=cleaned, id=last_human.id)]}
