"""带思考内容解析修复的 ChatOpenAI 子类，用于 vLLM/Qwen。

vLLM 侧思考内容的字段形态实测过三种，且同一套模型部署在不同网络路径下
表现还不一样（`docker-compose.yaml` 显示 vLLM 用 `--reasoning-parser qwen3`
启动，但本工程实际配置的 `OPENAI_BASE_URL` 打的是它前面一层纯转发
middleware）：

1. 直连 vLLM 原生端口、且这次生成完整跑完（`finish_reason != "length"`）时，
   reasoning parser 已经把思考内容解析进 `delta.reasoning` / `message.reasoning`
   字段（**不是** OpenAI 生态更常见的 `reasoning_content`）。
2. 经过 8001 端口的纯转发 middleware 时，实测拿到的却是 `<think>...</think>`
   标签直接混在 `content` 字符串里，`reasoning`/`reasoning_content` 字段都不
   存在——怀疑 middleware 或其上游把两者做了转换，具体机制未知，按黑盒处理。
3. 一般 OpenAI 兼容习惯用法里的 `reasoning_content` 字段（部分其他供应商，
   或未来这套部署方式变化后可能出现）。

本类按 1→3→2 的优先级依次尝试：先信任服务端已经结构化拆好的字段
（`reasoning`/`reasoning_content`，服务端一次只会给其中一种，谁存在用谁），
都没有时才回退到从 `content` 里解析 `<think>` 标签。三条路径的输出统一写回
`additional_kwargs["reasoning_content"]`，下游 `chat_pipeline.py::
_handle_ai_message` 只需要认一个字段名，不用感知这三种服务端形态的差异。
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

_REASONING_CONTENT_KEY = "reasoning_content"
_RAW_REASONING_FIELD_CANDIDATES = ("reasoning_content", "reasoning")
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def _extract_raw_reasoning(source: dict) -> str | None:
    """按优先级从 delta/message dict 里取第一个非空的结构化思考内容字段。"""
    for field_name in _RAW_REASONING_FIELD_CANDIDATES:
        value = source.get(field_name)
        if value:
            return value
    return None


def _split_think_blocks(content: str) -> tuple[str, str]:
    """从完整的 `content` 字符串里摘出所有 `<think>...</think>` 块。

    Returns:
        `(reasoning_content, visible_content)`：思考内容拼接结果，以及去掉
        思考标签后剩余的正文。
    """
    reasoning_parts = _THINK_BLOCK_RE.findall(content)
    visible_content = _THINK_BLOCK_RE.sub("", content)
    return "".join(reasoning_parts), visible_content


class PatchedChatOpenAI(ChatOpenAI):
    """把 vLLM 等 OpenAI 兼容供应商的思考内容透传进 additional_kwargs。

    流式场景下 `<think>`/`</think>` 标签可能被拆到不同 chunk 里，因此用实例
    属性 `_think_buffer`/`_in_think_block` 跨 chunk 缓冲——本工厂函数
    （`create_chat_model()`）每次调用都现造一个新实例、不做单例缓存，一次
    流式请求内 `StreamingModelMiddleware` 只顺序调用一次 `astream()`
    （唯一的例外是 `DatasourceRoutingMiddleware` 纠正重试时的第二次完整
    调用，此时 `_reset_think_state()` 会在新一轮 `astream()` 开始前清空缓冲），
    不存在并发 stream 复用同一实例的场景。
    """

    _think_buffer: str = ""
    _in_think_block: bool = False

    def _reset_think_state(self) -> None:
        self._think_buffer = ""
        self._in_think_block = False

    def _feed_think_parser(self, text_delta: str) -> tuple[str, str]:
        """喂入一段增量文本，返回 `(reasoning_delta, visible_delta)`。

        用简单状态机处理标签跨 chunk 拆分：把新增量拼到缓冲区，只要缓冲区
        里还残留不完整的 `<think>`/`</think>` 标签前缀就继续等待，避免把半个
        标签当作正文吐出去。
        """
        self._think_buffer += text_delta
        reasoning_out: list[str] = []
        visible_out: list[str] = []

        while True:
            if not self._in_think_block:
                open_index = self._think_buffer.find(_THINK_OPEN)
                if open_index == -1:
                    # 缓冲区可能以 "<think" 这样的不完整前缀结尾，留到下次判断。
                    safe_len = len(self._think_buffer)
                    for tail in range(min(len(_THINK_OPEN), len(self._think_buffer)), 0, -1):
                        if self._think_buffer.endswith(_THINK_OPEN[:tail]):
                            safe_len = len(self._think_buffer) - tail
                            break
                    visible_out.append(self._think_buffer[:safe_len])
                    self._think_buffer = self._think_buffer[safe_len:]
                    break
                visible_out.append(self._think_buffer[:open_index])
                self._think_buffer = self._think_buffer[open_index + len(_THINK_OPEN):]
                self._in_think_block = True
            else:
                close_index = self._think_buffer.find(_THINK_CLOSE)
                if close_index == -1:
                    safe_len = len(self._think_buffer)
                    for tail in range(min(len(_THINK_CLOSE), len(self._think_buffer)), 0, -1):
                        if self._think_buffer.endswith(_THINK_CLOSE[:tail]):
                            safe_len = len(self._think_buffer) - tail
                            break
                    reasoning_out.append(self._think_buffer[:safe_len])
                    self._think_buffer = self._think_buffer[safe_len:]
                    break
                reasoning_out.append(self._think_buffer[:close_index])
                self._think_buffer = self._think_buffer[close_index + len(_THINK_CLOSE):]
                self._in_think_block = False

        return "".join(reasoning_out), "".join(visible_out)

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """流式增量：优先信任服务端结构化字段，否则解析 `<think>` 标签。"""
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices", [])
        delta = choices[0].get("delta") if choices else None
        if delta and delta.get("role") == "assistant" and not delta.get("content") and not _extract_raw_reasoning(delta):
            # 首个只带 role 的空增量：`<think>` 状态机在此重置，避免复用上一轮残留状态。
            self._reset_think_state()

        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None or not isinstance(generation_chunk.message, AIMessageChunk):
            return generation_chunk

        server_reasoning = _extract_raw_reasoning(delta or {})
        if server_reasoning:
            generation_chunk.message.additional_kwargs[_REASONING_CONTENT_KEY] = (
                generation_chunk.message.additional_kwargs.get(_REASONING_CONTENT_KEY, "") + server_reasoning
            )
            return generation_chunk

        text_delta = generation_chunk.message.content
        if isinstance(text_delta, str) and text_delta:
            reasoning_delta, visible_delta = self._feed_think_parser(text_delta)
            generation_chunk.message.content = visible_delta
            if reasoning_delta:
                generation_chunk.message.additional_kwargs[_REASONING_CONTENT_KEY] = (
                    generation_chunk.message.additional_kwargs.get(_REASONING_CONTENT_KEY, "") + reasoning_delta
                )
        return generation_chunk

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """非流式响应：优先信任服务端结构化字段，否则解析 `<think>` 标签。"""
        result = super()._create_chat_result(response, generation_info)

        response_dict = response if isinstance(response, dict) else response.model_dump()
        choices = response_dict.get("choices") or []
        for generation, choice in zip(result.generations, choices, strict=False):
            message_dict = choice.get("message") or {}
            server_reasoning = _extract_raw_reasoning(message_dict)
            if server_reasoning:
                generation.message.additional_kwargs[_REASONING_CONTENT_KEY] = server_reasoning
                continue

            content = generation.message.content
            if isinstance(content, str) and _THINK_OPEN in content:
                reasoning_content, visible_content = _split_think_blocks(content)
                if reasoning_content:
                    generation.message.additional_kwargs[_REASONING_CONTENT_KEY] = reasoning_content
                    generation.message.content = visible_content
        return result
