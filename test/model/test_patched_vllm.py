"""`PatchedChatOpenAI` 的思考内容解析单测。

覆盖三条路径（详见 `patched_vllm.py` 模块 docstring 里记录的三种实测形态）：
1. 服务端拆到独立 `reasoning_content` 字段（通用 OpenAI 兼容习惯用法）。
2. 服务端拆到独立 `reasoning` 字段（本工程实测直连 vLLM 原生端口、
   `--reasoning-parser qwen3` 生效时的真实形态）。
3. 服务端完全不拆分，思考内容以 `<think>...</think>` 标签混在 `content`
   字符串里返回（本工程实测经过 8001 端口转发 middleware 时的真实形态）——
   第一版 patch 只覆盖了路径 1，两次实测都没覆盖到，实际没有生效。
"""
from __future__ import annotations

from langchain_core.messages import AIMessageChunk

from src.agent_core.model.patched_vllm import PatchedChatOpenAI

_MODEL_KWARGS = {"model": "qwen3", "api_key": "k", "base_url": "http://localhost:8001/v1"}


def _stream_deltas(llm: PatchedChatOpenAI, deltas: list[str]) -> tuple[str, str]:
    """依次喂入若干段 content 增量，返回拼接后的 (reasoning, visible)。"""
    reasoning_acc = ""
    visible_acc = ""
    for index, delta in enumerate(deltas):
        chunk = {
            "choices": [
                {
                    "delta": {"role": "assistant", "content": delta} if index == 0 else {"content": delta},
                    "index": 0,
                }
            ],
        }
        generation_chunk = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)
        reasoning_acc += generation_chunk.message.additional_kwargs.get("reasoning_content", "")
        visible_acc += generation_chunk.message.content
    return reasoning_acc, visible_acc


def test_stream_chunk_extracts_reasoning_content() -> None:
    llm = PatchedChatOpenAI(**_MODEL_KWARGS)
    chunk = {
        "choices": [
            {"delta": {"role": "assistant", "reasoning_content": "让我想想", "content": ""}, "index": 0}
        ],
    }

    generation_chunk = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)

    assert generation_chunk.message.additional_kwargs.get("reasoning_content") == "让我想想"


def test_stream_chunk_extracts_bare_reasoning_field() -> None:
    """vLLM 原生端口 `--reasoning-parser qwen3` 生效时用的是 `reasoning`
    字段，不是 `reasoning_content`（实测确认，见模块 docstring）。"""
    llm = PatchedChatOpenAI(**_MODEL_KWARGS)
    chunk = {"choices": [{"delta": {"reasoning": "让我想想"}, "index": 0}]}

    generation_chunk = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)

    assert generation_chunk.message.additional_kwargs.get("reasoning_content") == "让我想想"


def test_non_streaming_response_extracts_bare_reasoning_field() -> None:
    llm = PatchedChatOpenAI(**_MODEL_KWARGS)
    response = {
        "id": "chatcmpl-1",
        "model": "Qwen3.5-35B-A3B",
        "choices": [
            {
                "message": {"role": "assistant", "content": "最终答案", "reasoning": "推理过程"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }

    result = llm._create_chat_result(response)

    assert result.generations[0].message.additional_kwargs.get("reasoning_content") == "推理过程"
    assert result.generations[0].message.content == "最终答案"


def test_stream_chunk_without_reasoning_content_is_untouched() -> None:
    llm = PatchedChatOpenAI(**_MODEL_KWARGS)
    chunk = {"choices": [{"delta": {"role": "assistant", "content": "答案"}, "index": 0}]}

    generation_chunk = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)

    assert "reasoning_content" not in generation_chunk.message.additional_kwargs


def test_non_streaming_response_extracts_reasoning_content() -> None:
    llm = PatchedChatOpenAI(**_MODEL_KWARGS)
    response = {
        "id": "chatcmpl-1",
        "model": "qwen3",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "最终答案",
                    "reasoning_content": "推理过程",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }

    result = llm._create_chat_result(response)

    assert result.generations[0].message.additional_kwargs.get("reasoning_content") == "推理过程"
    assert result.generations[0].message.content == "最终答案"


def test_non_streaming_response_parses_think_tag_when_reasoning_content_absent() -> None:
    """服务端没有 reasoning_content 字段，思考内容以 <think> 标签混在 content 里。"""
    llm = PatchedChatOpenAI(**_MODEL_KWARGS)
    response = {
        "id": "chatcmpl-1",
        "model": "qwen3.5-35b-a3b",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>让我想想\n分步推理</think>\n\n最终答案",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }

    result = llm._create_chat_result(response)

    assert result.generations[0].message.additional_kwargs.get("reasoning_content") == "让我想想\n分步推理"
    assert result.generations[0].message.content == "最终答案"


def test_non_streaming_response_without_think_tag_is_untouched() -> None:
    llm = PatchedChatOpenAI(**_MODEL_KWARGS)
    response = {
        "id": "chatcmpl-1",
        "model": "qwen3.5-35b-a3b",
        "choices": [{"message": {"role": "assistant", "content": "没有思考标签的普通回答"}, "finish_reason": "stop", "index": 0}],
    }

    result = llm._create_chat_result(response)

    assert "reasoning_content" not in result.generations[0].message.additional_kwargs
    assert result.generations[0].message.content == "没有思考标签的普通回答"


def test_stream_think_tag_within_single_chunk() -> None:
    """流式路径逐字符转发，不像非流式那样吞掉标签后的空白——下游按 token 拼接，
    多余的换行不影响最终展示，但强行去除反而可能丢用户期望保留的换行。"""
    llm = PatchedChatOpenAI(**_MODEL_KWARGS)

    reasoning, visible = _stream_deltas(llm, ["<think>推理内容</think>\n\n答案"])

    assert reasoning == "推理内容"
    assert visible == "\n\n答案"


def test_stream_think_tag_split_across_chunks() -> None:
    """<think>/</think> 标签本身被拆到不同 chunk 边界的场景（真实流式最常见）。"""
    llm = PatchedChatOpenAI(**_MODEL_KWARGS)

    reasoning, visible = _stream_deltas(
        llm, ["<thi", "nk>推理", "内容</th", "ink>\n\n", "最终", "答案"]
    )

    assert reasoning == "推理内容"
    assert visible == "\n\n最终答案"


def test_stream_without_think_tag_passes_through() -> None:
    llm = PatchedChatOpenAI(**_MODEL_KWARGS)

    reasoning, visible = _stream_deltas(llm, ["普通", "回答", "没有思考"])

    assert reasoning == ""
    assert visible == "普通回答没有思考"
