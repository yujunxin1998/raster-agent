"""带 reasoning_content 多轮续接修复的 ChatDeepSeek 子类。

对应设计文档"参考 DeerFlow 封装 LLM 工厂"的决策：DeepSeek 的思考模型把上一轮的
思维链存在 `AIMessage.additional_kwargs["reasoning_content"]` 里，但 LangChain
序列化请求负载时默认不会把这个字段带回给 API；多轮对话里如果不回填，部分需要
`reasoning_content` 存在于每条 assistant 消息上的接口会报错/丢失推理连续性。

本类只做这一件事，逻辑简化自 DeerFlow
`packages/harness/deerflow/models/patched_deepseek.py` +
`assistant_payload_replay.py::restore_assistant_payloads`——DeerFlow 需要在多个
供应商 patch 类之间共用匹配逻辑，本仓库只有 DeepSeek 一个供应商，直接按位置
一一对应即可，不需要那套按签名匹配的通用框架。

刻意不做的事：不剥离 `tool_choice`（原项目 `LLMFactory._DeepSeekNoToolChoice` 里
无条件 `kwargs.pop("tool_choice")`，是为了绕开 supervisor 结构化路由强制单工具
选择的场景；新架构里路由是普通工具调用，不强制 `tool_choice`，这个补丁大概率
不再需要，见设计文档"关键设计决策"第 3 条）。
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek


def _restore_reasoning_content(payload_messages: list[dict], original_messages: list) -> None:
    """把 assistant 消息的 reasoning_content 从原始消息回填到序列化后的请求负载。

    按位置一一对应：LangChain 序列化请求负载时消息条数与原始消息列表一致
    （不会合并/丢弃消息），所以直接按下标配对即可，不需要额外的签名匹配。

    Args:
        payload_messages: `_get_request_payload` 序列化产出的消息 dict 列表，
            原地修改。
        original_messages: 序列化前的原始 `BaseMessage` 列表。
    """
    if len(payload_messages) != len(original_messages):
        return
    for payload_message, original_message in zip(payload_messages, original_messages, strict=True):
        if payload_message.get("role") != "assistant" or not isinstance(original_message, AIMessage):
            continue
        reasoning_content = original_message.additional_kwargs.get("reasoning_content")
        if reasoning_content is not None:
            payload_message["reasoning_content"] = reasoning_content


class PatchedChatDeepSeek(ChatDeepSeek):
    """带 reasoning_content 多轮续接修复的 ChatDeepSeek。"""

    def _get_request_payload(
        self,
        input_,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """构造请求负载后，把原始消息的 reasoning_content 回填到序列化结果里。

        Args:
            input_: 传给模型的输入（消息列表或可转换为消息列表的对象）。
            stop: 停止词列表。
            **kwargs: 透传给父类的其余参数。

        Returns:
            回填 reasoning_content 后的请求负载字典。
        """
        original_messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        _restore_reasoning_content(payload.get("messages", []), original_messages)
        return payload
