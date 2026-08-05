"""REST `/chat` 与 WebSocket `/ws/chat` 共用的对话轮次驱动逻辑。

对应设计文档探索阶段发现的清理机会：原项目 `chat_service.py` 里 `chat()`
（REST，内部也用 `astream_events` 驱动但不流式返回）和 `stream_chat()`（WS）
两条路径手写了大量重复逻辑（悬空工具调用清洗、事件消费、引用来源解析）。
本模块把这部分收敛成一个共享的异步生成器 `run_chat_turn()`：
    - REST 层（`chat_router.py`）消费完生成器后只读 `ChatTurnResult`，不转发事件。
    - WS 层（`chat_ws.py`）把每个 yield 的事件原样转发给客户端。

记忆注入/提取/压缩/标题生成不需要在这里手写——那些是 Lead Agent 中间件流水线
（`MemoryInjectionMiddleware`/`MemoryExtractionMiddleware`/`SummarizationMiddleware`/
`TitleMiddleware`）的职责，`build_lead_agent()` 组装时已经自动带上。
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from loguru import logger

from src.agent_core.agents.checkpointer import get_checkpointer
from src.agent_core.agents.dangling_tool_calls import sanitize_dangling_tool_calls
from src.agent_core.agents.lead_agent import build_lead_agent, resolve_recursion_limit
from src.agent_core.eval.emitter import EvalEventEmitter
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.skills import get_skill_manager
from src.agent_core.tools.tool_filter import tool_filter_registry

_TOOL_TYPE_MAP: dict[str, str] = {
    "web_search": "web_search",
    "search_knowledge_base": "knowledge_base",
    "save_memory": "memory",
    "recall_memory": "memory",
    "task": "delegate",
}

_REF_START = "<ref_json>"
_REF_END = "</ref_json>"
_REF_START_LEN = len(_REF_START)

_KNOWLEDGE_CHUNK_PATTERN = re.compile(
    r'<知识片段 \[(\d+)\] id=(\S+) title="([^"]*)" score=([\d.]+)>\n(.*?)\n?</知识片段>', re.DOTALL,
)
_WEB_SOURCE_PATTERN = re.compile(r"\d+\.\s*(.+?)\n\s*链接：(\S*)\n\s*摘要：(.*?)(?=\n\d+\.|\Z)", re.DOTALL)

_SOURCE_TEXT_TRUNCATE_CHARS = 200


@dataclass
class ChatTurnResult:
    """一次对话轮次的累计结果，供调用方在生成器耗尽后读取。"""

    ai_response_parts: list[str] = field(default_factory=list)
    thinking_parts: list[str] = field(default_factory=list)
    tool_call_records: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)

    @property
    def ai_response(self) -> str:
        return "".join(self.ai_response_parts)

    @property
    def thinking_content(self) -> Optional[str]:
        return "".join(self.thinking_parts) or None


def _push_ref_text(text: str, ref_state: dict) -> list[str]:
    """把模型输出文本送入 `<ref_json>` 状态机，返回应转发给前端的文本块列表。"""
    emitted: list[str] = []
    if not ref_state["in_ref"]:
        ref_state["pending"] += text
        while True:
            start_idx = ref_state["pending"].find(_REF_START)
            if start_idx >= 0:
                before = ref_state["pending"][:start_idx]
                after = ref_state["pending"][start_idx + _REF_START_LEN:]
                if before:
                    emitted.append(before)
                ref_state["in_ref"] = True
                ref_state["pending"] = ""
                ref_state["buf"] = after
                end_idx = ref_state["buf"].find(_REF_END)
                if end_idx >= 0:
                    _parse_ref_json(ref_state["buf"][:end_idx], ref_state)
                    ref_state["buf"] = ""
                    ref_state["in_ref"] = False
                break
            elif len(ref_state["pending"]) > _REF_START_LEN:
                flush_end = len(ref_state["pending"]) - _REF_START_LEN
                emitted.append(ref_state["pending"][:flush_end])
                ref_state["pending"] = ref_state["pending"][flush_end:]
            else:
                break
    else:
        ref_state["buf"] += text
        end_idx = ref_state["buf"].find(_REF_END)
        if end_idx >= 0:
            _parse_ref_json(ref_state["buf"][:end_idx], ref_state)
            ref_state["buf"] = ""
            ref_state["in_ref"] = False
    return emitted


def _parse_ref_json(raw: str, ref_state: dict) -> None:
    """解析 `<ref_json>` 内部 JSON，写入 `ref_state['sources']`。"""
    try:
        data = json.loads(raw.strip())
        if isinstance(data, list):
            ref_state["sources"] = data
        elif isinstance(data, dict):
            ref_state["sources"] = data.get("sources") or []
    except Exception as exc:
        logger.debug(f"[chat_pipeline] 解析 <ref_json> 失败（不影响主流程）: {exc}")


def _extract_knowledge_base_sources(text: str) -> list[dict]:
    """从 `search_knowledge_base` 工具原始输出中解析结构化来源（片段级，不跨片段合并）。"""
    seen: set[str] = set()
    sources: list[dict] = []
    for match in _KNOWLEDGE_CHUNK_PATTERN.finditer(text):
        chunk_id, name, content = match.group(2), match.group(3), match.group(5)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        sources.append({
            "id": chunk_id, "type": "file", "name": name,
            "desc": content[:_SOURCE_TEXT_TRUNCATE_CHARS].replace("\n", " "), "url": "",
        })
    return sources


def _extract_web_sources(text: str) -> list[dict]:
    """从 `web_search` 工具原始输出（【来源】编号列表）中解析结构化来源。"""
    sources: list[dict] = []
    for match in _WEB_SOURCE_PATTERN.finditer(text):
        title, url, content = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
        if not title:
            continue
        sources.append({
            "type": "web", "name": title, "url": url,
            "desc": content[:_SOURCE_TEXT_TRUNCATE_CHARS].replace("\n", " "),
        })
    return sources


def _to_json_safe(obj: Any) -> Any:
    """递归把不可 JSON 序列化的对象转为字符串，防止 `send_json` 崩溃。"""
    if isinstance(obj, dict):
        return {key: _to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(item) for item in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _is_skill(tool_name: str) -> bool:
    """判断某个工具名是否应作为"通用技能"推给前端（独立 skill_call/skill_response
    事件），而不是内置 tool_call/tool_response。`_TOOL_TYPE_MAP` 里列出的名字
    始终保留 tool_call 语义（前端已有专门样式），不受影响。
    """
    return tool_name in get_skill_manager().registry.names and tool_name not in _TOOL_TYPE_MAP


def _tool_type(tool_name: str) -> str:
    return _TOOL_TYPE_MAP.get(tool_name, "custom")


def _build_final_sources(ref_state: dict, fallback_sources: list[dict]) -> list[dict]:
    """参考文献：优先使用模型通过 `<ref_json>` 明确选取的来源；模型未输出时以
    工具提取结果兜底，保证只要发生了检索，引用就能落库、历史查看不丢失。
    """
    if not ref_state["sources"]:
        return fallback_sources

    full_lookup = {source["name"]: source for source in fallback_sources}
    final_sources = []
    for source in ref_state["sources"]:
        name = source.get("name", "")
        if not name:
            continue
        base = dict(full_lookup.get(name, {}))
        base["index"] = source.get("index") if source.get("index") is not None else base.get("index")
        base.setdefault("name", name)
        base.setdefault("type", "file")
        base.setdefault("url", "")
        final_sources.append(base)
    return final_sources


async def run_chat_turn(
    *,
    conversation_id: str,
    user_id: str,
    message: str,
    thinking: bool = False,
    datasource_id: Optional[str] = None,
    extra_tools: Optional[list[BaseTool]] = None,
    emitter: Optional[EvalEventEmitter] = None,
    result: ChatTurnResult,
) -> AsyncGenerator[dict, None]:
    """驱动一次完整对话轮次，逐个 yield WS 形状的事件。

    Args:
        conversation_id: 会话 ID（同时是 checkpointer 的 thread_id）。
        user_id: 归属用户 ID。
        message: 用户本轮发言。
        thinking: 是否开启深度思考模式。
        datasource_id: 数据源 ID，供 `DatasourceRoutingMiddleware`/
            `delegate_to_database_agent` 读取。
        extra_tools: 前端本次请求注入的工具（经 `CustomToolConverter` 转换）。
        emitter: Eval 遥测状态机，为空时跳过埋点。
        result: 调用方传入的累计结果容器，本函数在迭代过程中原地写入。

    Yields:
        WS 形状的事件字典（`start`/`token`/`thinking`/`tool_call`/`skill_call`/
        `tool_response`/`skill_response`/`reference`）。REST 层可以直接丢弃这些
        事件，只读 `result`；WS 层把每个事件原样转发给客户端。
    """
    agent = build_lead_agent(
        thinking_enabled=thinking, extra_tools=extra_tools, checkpointer=get_checkpointer(), user_id=user_id,
    )
    config = {
        "configurable": {"thread_id": conversation_id, "secrets": {}, "datasource_id": datasource_id},
        "recursion_limit": resolve_recursion_limit(thinking),
    }
    context = AgentRuntimeContext(
        conversation_id=conversation_id, user_id=user_id, thinking=thinking, datasource_id=datasource_id,
    )

    await sanitize_dangling_tool_calls(agent, config)

    ref_state: dict = {"pending": "", "in_ref": False, "buf": "", "sources": None}
    fallback_sources: list[dict] = []

    async for stream_mode, chunk in agent.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config, context=context, stream_mode=["messages", "custom"],
    ):
        if stream_mode == "custom":
            if isinstance(chunk, dict) and "tool_call_pending" in chunk:
                # 参数还没拼完时提前广播的"仅名字"标记，见该中间件的说明——
                # 只用来让前端提前弹一张 pending 卡片，不落 `result.tool_call_records`
                # （完整数据等下面 `tool_calls` 分支的正式事件到达时才落）。
                tool_event = _handle_tool_call_pending(chunk["tool_call_pending"])
                if tool_event is not None:
                    yield tool_event
                continue
            if isinstance(chunk, dict) and "tool_calls" in chunk:
                # `StreamingModelMiddleware` 在拿到完整 `final_message` 后补发的
                # 一次性标记，携带已经解析完整的 `tool_calls`——`stream_mode=
                # "messages"` 通道里的 `AIMessage.tool_calls` 是跨多个增量 chunk
                # 拼出来的半成品（参数越长，中途看到的空/野值越多，是曾经"任务
                # 卡在写文件这一步不动"的根因，见该中间件的说明），不能再信任
                # 那条通道来触发 `tool_call` 事件，只信这里。
                for tool_call in chunk["tool_calls"]:
                    tool_event = _handle_tool_call_start(tool_call, emitter, result)
                    if tool_event is not None:
                        yield tool_event
                continue
            # `StreamingModelMiddleware` 逐 token 推送的增量 `AIMessageChunk`——
            # 真正的打字机效果来源，见该模块说明。
            for text_event in _handle_ai_message(chunk, emitter, thinking, ref_state, result):
                yield text_event
            continue

        msg_chunk, _metadata = chunk

        if isinstance(msg_chunk, ToolMessage):
            tool_event = _handle_tool_message(msg_chunk, emitter, fallback_sources, result)
            if tool_event is not None:
                yield tool_event

    if ref_state["pending"]:
        # `_push_ref_text` 为了识别跨 chunk 到达的 `<ref_json>` 起始标签，
        # 总会在 `pending` 里保留最后最多 `_REF_START_LEN`（10）个字符不发出，
        # 等下一块内容到达时再判断这几个字符是不是标签的开头。如果回复在这
        # 之前就正常结束（没有引用标签），这几个字符会一直留在 `pending`
        # 里发不出去——前端看到的内容会稳定地在结尾少最后几个字（实测正好
        # 少 10 个字，与 `<ref_json>` 的长度分毫不差）。这里在流结束后把它
        # 兜底吐出去。
        result.ai_response_parts.append(ref_state["pending"])
        yield {"type": "token", "content": ref_state["pending"]}

    if ref_state["buf"]:
        # 状态机停在 <ref_json> 内部（哨兵未闭合）——安全兜底，原样吐出剩余缓冲
        result.ai_response_parts.append(ref_state["buf"])
        yield {"type": "token", "content": ref_state["buf"]}

    final_sources = _build_final_sources(ref_state, fallback_sources)
    if final_sources:
        result.references = final_sources
        yield {"type": "reference", "content": {"title": "参考文献", "sources": final_sources}}


def _handle_tool_call_pending(pending: dict) -> Optional[dict]:
    """处理 `StreamingModelMiddleware` 提前广播的"仅工具名"标记。

    Args:
        pending: `{"id": ..., "name": ...}`，工具调用参数还没流完时就已知的
            那部分（见该中间件对 `tool_call_chunks` 的说明）。

    Returns:
        要推给前端的 pending `tool_call`/`skill_call` 消息（`tool_args` 恒为
        `None`，`content.pending` 恒为 `True`，跟随后到达的正式事件区分开）；
        工具名在过滤名单里或者信息不全时返回 `None`（不埋点、不落
        `result.tool_call_records`——那是正式事件的职责，这里只是提前预告）。
    """
    tool_name, request_id = pending.get("name", ""), pending.get("id", "")
    if not tool_name or not request_id or tool_filter_registry.is_filtered(tool_name):
        return None

    is_skill = _is_skill(tool_name)
    return {
        "type": "skill_call" if is_skill else "tool_call",
        "content": {
            "tool_name": tool_name,
            "tool_args": None,
            "request_id": request_id,
            "tool_type": "skill" if is_skill else _tool_type(tool_name),
            "pending": True,
        },
    }


def _handle_tool_call_start(
    tool_call: dict, emitter: Optional[EvalEventEmitter], result: ChatTurnResult,
) -> Optional[dict]:
    """处理模型消息里的一个 `tool_call` 条目：埋点 + 构造要推给前端的 tool_call/skill_call 消息。"""
    tool_name = tool_call.get("name", "")
    tool_input = tool_call.get("args", {})
    request_id = tool_call.get("id", "")

    if emitter is not None:
        try:
            emitter.on_tool_start_sync(tool_name, request_id, tool_input)
        except Exception as exc:
            logger.debug(f"[chat_pipeline] on_tool_start 埋点失败（不影响主流程）: {exc}")

    if tool_filter_registry.is_filtered(tool_name):
        return None

    is_skill = _is_skill(tool_name)
    record = {
        "tool_name": tool_name,
        "tool_args": _to_json_safe(tool_input),
        "request_id": request_id,
        "tool_type": "skill" if is_skill else _tool_type(tool_name),
    }
    result.tool_call_records.append(record)
    return {"type": "skill_call" if is_skill else "tool_call", "content": record}


def _handle_tool_message(
    message: ToolMessage, emitter: Optional[EvalEventEmitter], fallback_sources: list[dict], result: ChatTurnResult,
) -> Optional[dict]:
    """处理一条 `ToolMessage`：埋点 + 收集兜底引用来源 + 构造 tool_response 消息。"""
    request_id = message.tool_call_id or ""
    tool_name = message.name or ""
    response_text = message.content if isinstance(message.content, str) else str(message.content or "")

    if emitter is not None:
        try:
            status = getattr(message, "status", None) or "success"
            error_msg = response_text if status == "error" else ""
            asyncio.create_task(
                emitter.on_tool_end_sync(request_id, response_text, status=status, error_msg=error_msg)
            )
        except Exception as exc:
            logger.debug(f"[chat_pipeline] on_tool_end 埋点失败（不影响主流程）: {exc}")

    if tool_name == "search_knowledge_base":
        extracted = _extract_knowledge_base_sources(response_text)
    elif tool_name == "web_search":
        extracted = _extract_web_sources(response_text)
    else:
        extracted = []
    if extracted:
        seen_names = {source["name"] for source in fallback_sources}
        for source in extracted:
            if source.get("name") and source["name"] not in seen_names:
                fallback_sources.append(source)
                seen_names.add(source["name"])

    for record in result.tool_call_records:
        if record["request_id"] == request_id:
            record["tool_response"] = response_text
            break

    if tool_filter_registry.is_filtered(tool_name):
        return None
    return {
        "type": "skill_response" if _is_skill(tool_name) else "tool_response",
        "content": {"request_id": request_id, "response": response_text},
    }


def _handle_ai_message(
    message: AIMessageChunk, emitter: Optional[EvalEventEmitter], thinking: bool, ref_state: dict, result: ChatTurnResult,
) -> list[dict]:
    """处理一条模型增量消息（`AIMessageChunk`）：做埋点，返回应推给前端的消息列表。

    LangChain 1.x `create_agent` 的模型节点内部固定用 `model.ainvoke(...)`
    （而非 `.astream(...)`）执行模型调用，且该节点以 `trace=False` 注册进图——
    `astream_events` 因此完全不会产生 `on_chat_model_stream` 事件，
    `agent.astream(..., stream_mode=["messages"])` 拿到的也只是一整条完整
    `AIMessage`（这是本模块早前"前端啥也收不到/只能整段甩文字"两个问题的
    真实根因）。真正的逐 token 增量改由 `StreamingModelMiddleware` 通过
    `runtime.stream_writer()` 推到 `stream_mode="custom"` 通道——本函数处理的
    正是这个通道里的每个 `AIMessageChunk`（调用方在 `run_chat_turn` 里对
    `stream_mode == "custom"` 的分支调用本函数；`"messages"` 分支只用来读
    最终完整消息的 `tool_calls`，不再重复处理正文，避免文字重复两遍）。

    委派工具内部子 Agent 的模型消息已经被 `sub_agent_factory.py` 的 config
    隔离处理挡在外层之外（见该模块的说明），也没有装配 `StreamingModelMiddleware`
    （只有 Lead Agent 自己装配），所以子 Agent 内部产出的 chunk 不会出现在这个
    通道里——这里收到的每个 chunk 都属于用户可见的响应。
    """
    if emitter is not None:
        try:
            usage_metadata = getattr(message, "usage_metadata", None)
            if usage_metadata:
                emitter.on_token_usage(usage_metadata.get("input_tokens", 0), usage_metadata.get("output_tokens", 0))
        except Exception as exc:
            logger.debug(f"[chat_pipeline] on_token_usage 埋点失败（不影响主流程）: {exc}")

    out: list[dict] = []

    if thinking:
        reasoning = message.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            result.thinking_parts.append(reasoning)
            out.append({"type": "thinking", "content": reasoning})

    content = message.content
    if isinstance(content, str) and content:
        for chunk_text in _push_ref_text(content, ref_state):
            result.ai_response_parts.append(chunk_text)
            if emitter is not None:
                emitter.on_agent_text(chunk_text)
            out.append({"type": "token", "content": chunk_text})
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "thinking" and item.get("thinking") and thinking:
                result.thinking_parts.append(item["thinking"])
                out.append({"type": "thinking", "content": item["thinking"]})
            elif item.get("type") == "text" and item.get("text"):
                for chunk_text in _push_ref_text(item["text"], ref_state):
                    result.ai_response_parts.append(chunk_text)
                    if emitter is not None:
                        emitter.on_agent_text(chunk_text)
                    out.append({"type": "token", "content": chunk_text})

    return out
