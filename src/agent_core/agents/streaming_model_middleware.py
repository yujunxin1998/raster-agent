"""真正执行模型调用的最内层中间件：把 `create_agent` 内置的阻塞式
`model.ainvoke()` 换成 `model.astream()`，实现逐 token 的真流式输出。

背景（`chat_pipeline.py` 模块文档也记了这件事）：LangChain 1.x `create_agent`
编译出的图里，`"model"` 节点固定调用 `await model_.ainvoke(messages)`
（`langchain/agents/factory.py::_execute_model_async`），且该节点以
`trace=False` 注册进图——不管 `astream_events` 还是 `agent.astream(...,
stream_mode=["messages"])`，模型的最终回复都只会作为"一整条"消息出现，不会有
逐 token 的增量，前端体验退化成"一次性甩一大段文字"而不是打字机效果。

`wrap_model_call` 中间件可以完全绕开框架内置的 `handler`（也就是那个调用
`ainvoke()` 的默认实现），自己执行模型调用——本类正是这么做的：手动复刻
`_get_bound_model` 里"无 response_format"分支的绑定逻辑（本工程的 Lead Agent
从不设置 `response_format`，所以其余分支永远不会走到，不需要照搬），改用
`.astream()` 逐块拉取，每块都通过 `runtime.stream_writer()` 推到
LangGraph 的 `stream_mode="custom"` 通道（`chat_pipeline.py` 消费这个通道，
逐块转发成 WS `token`/`thinking` 事件），同时在本地把所有块累加成一条完整的
`AIMessage`，作为 `ModelResponse` 返回——图的 state/checkpoint 侧完全不受影响
（拿到的还是一条完整消息，行为和内置 `ainvoke()` 路径等价）。

必须是中间件链里最内层（离真实模型调用最近）的一个：注册顺序上要排在
`DatasourceRoutingMiddleware` 之后，这样 `DatasourceRoutingMiddleware` 重试时
调用的 `handler(request)` 实际落到的就是本类的 `awrap_model_call`——两次调用
（首次 + 纠正重试）都会各自完整地流式输出一遍，这和纠正重试机制本身"两次真实
模型调用"的语义一致，不是本类引入的新行为。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.constants import CONF

from src.agent_core.middlewares.context import AgentRuntimeContext

# `create_agent` 把 "model" 节点注册为 `trace=False`（见模块 docstring），这个
# 节点的执行链路上 LangChain 的 `var_child_runnable_config` contextvar 始终是
# 空的（不是本类导致的——`langsmith` 的 `@traceable` 包装器给每层中间件调用套了
# 一层 `asyncio.create_task(..., context=copy_context())`，而 `trace=False` 的
# 节点从未在任何外层正确设置过这个 contextvar，从源头上就没有"正确值"可捕获）。
# `runtime.stream_writer` 内部靠 `get_config()[CONF][CONFIG_KEY_CHECKPOINT_NS]`
# 计算子图命名空间前缀，因此每次调用前都必须手动把这个 contextvar 设置成一个
# 够用的最小 config——本工程的委派工具调用的是完全独立的子 Agent
# （`sub_agent_factory.py::_invoke` 用的是普通 `.ainvoke()`，不是 LangGraph 的
# 子图节点机制），Lead Agent 自身的图从不产生真正的子图嵌套，因此
# `checkpoint_ns` 在这里恒为空字符串，可以放心写死。
_CHECKPOINT_NS_KEY = "checkpoint_ns"
_STREAM_WRITER_CONFIG = {CONF: {_CHECKPOINT_NS_KEY: ""}}


class StreamingModelMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """用 `.astream()` 替代 `create_agent` 内置的 `.ainvoke()`，推送真流式 token。"""

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """手动执行一次流式模型调用，逐块推送，最终汇总成一条 `AIMessage`。

        Args:
            request: 本次模型调用请求（已经过前面所有中间件的 system_prompt/
                messages 改写）。
            handler: 框架默认的 `ainvoke()` 执行器，仅在模型一个 chunk 都没
                产出时（理论上不应发生）作为兜底调用一次。

        Returns:
            `ModelResponse(result=[final_message])`，与内置 `handler` 返回的
            结构完全一致。
        """
        bound_model = (
            request.model.bind_tools(request.tools, tool_choice=request.tool_choice, **request.model_settings)
            if request.tools
            else request.model.bind(**request.model_settings)
        )

        messages = request.messages
        if request.system_message:
            messages = [request.system_message, *messages]

        stream_writer = request.runtime.stream_writer

        def _write(payload: Any) -> None:
            token = var_child_runnable_config.set(_STREAM_WRITER_CONFIG)
            try:
                stream_writer(payload)
            finally:
                var_child_runnable_config.reset(token)

        final_chunk: AIMessageChunk | None = None
        announced_tool_call_ids: set[str] = set()
        async for chunk in bound_model.astream(messages):
            _write(chunk)
            # `tool_call_chunks` 是原始增量字段（跟下面 `tool_calls` property 那种
            # "尝试解析半截 JSON"不是一回事），`id`/`name` 按 OpenAI 系流式协议
            # 只在该工具调用的第一个 chunk 里出现一次，后续几块只补 `args` 的
            # 片段——用它可以在参数还没拼完之前就提前广播"某个工具调用已经开始，
            # 名字是什么"，让前端能立刻弹出一张 pending 卡片，不用死等参数全部
            # 流完（`write_file` 的 `content` 可能要好几百个 token 才流完，
            # 期间前端如果什么反馈都没有，观感上会像是"文字说完了但卡在原地"）。
            for raw_call in chunk.tool_call_chunks or []:
                call_id, name = raw_call.get("id"), raw_call.get("name")
                if call_id and name and call_id not in announced_tool_call_ids:
                    announced_tool_call_ids.add(call_id)
                    _write({"tool_call_pending": {"id": call_id, "name": name}})
            final_chunk = chunk if final_chunk is None else final_chunk + chunk

        if final_chunk is None:
            return await handler(request)

        final_message = AIMessage(
            content=final_chunk.content,
            additional_kwargs=final_chunk.additional_kwargs,
            tool_calls=final_chunk.tool_calls,
            invalid_tool_calls=final_chunk.invalid_tool_calls,
            usage_metadata=final_chunk.usage_metadata,
            response_metadata=final_chunk.response_metadata,
            id=final_chunk.id,
        )

        if final_message.tool_calls:
            # 见 `_STREAM_WRITER_CONFIG` 上方的说明：`stream_mode="messages"` 通道
            # 里的 `AIMessageChunk` 是 LangChain 给模型 `.astream()` 自动挂的回调
            # 副产物，跟上面手动推的 "custom" 通道是两条独立链路，一样是逐块到达
            # 的增量——`tool_calls` 字段在参数较长（如本工程 write_file 的
            # content）时会跨好几个 chunk 才拼完整，中途每个 chunk 的
            # `tool_calls` 都是尝试解析半截 JSON 的产物（`name` 只在第一块出现，
            # 之后几块 `name=""`/`id=None`，`args` 在拼完之前恒为 `{}`）。
            # `chat_pipeline.py` 曾经直接读这个通道的 `tool_calls` 来生成
            # `tool_call` 事件，参数越长，推给前端的半成品事件越多，是"任务
            # 卡在写文件这一步不动"这个问题的真实根因（前端拿到
            # `request_id=None` 的野事件后状态没法收尾）。这里改成只在本函数
            # 已经拿到完整 `final_message` 之后，把解析好的 `tool_calls` 通过
            # 同一个 stream_writer 补发一次——`chat_pipeline.py` 改为只信这里
            # 发的、不再信 "messages" 通道的 `tool_calls`。
            _write({"tool_calls": final_message.tool_calls})

        return ModelResponse(result=[final_message])
