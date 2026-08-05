"""Lead Agent 组装（设计文档 4.1 节）。

取代原项目"Supervisor 多图路由"的核心改动：不再有一个专门做结构化路由决策的
LLM 调用，Lead Agent 自身就是一个普通的 `create_agent`，工具集里既有可以直接
执行的基础工具（记忆、通用/地图技能、前端 `extra_tools`），也有"委派"工具
（`delegate_to_rag_agent`/`delegate_to_web_search_agent`）——路由降级为一次普通
的工具调用决策，而不是每轮都必须做的独立 LLM 调用。

`thinking_enabled` 是运行时配置，不是另一套并行维护的 Agent 构建代码：复用同一套
委派工具，只是模型侧开启扩展推理 + 调用方在 `.ainvoke(..., config={"recursion_limit": ...})`
传入更大的步数预算。

本函数不做单例缓存，每次调用都现造（`create_agent` 构造成本是纯 Python 对象组装，
没有网络调用，可以忽略）——这样委派工具/技能工具集/`web_search_agent` 的当天日期
都能拿到最新状态，不会像原项目那样把 `{current_date}` 冻结在进程启动的那一刻。
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.agent_core.agents.datasource_routing_middleware import DatasourceRoutingMiddleware
from src.agent_core.agents.delegation_tools import (
    build_database_delegate_tool,
    build_rag_delegate_tool,
    build_web_search_delegate_tool,
)
from src.agent_core.agents.streaming_model_middleware import StreamingModelMiddleware
from src.agent_core.guardrail import get_guardrail_provider
from src.agent_core.loop import TitleModelSettings, build_middlewares
from src.agent_core.memory import get_memory_manager
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.model import create_chat_model
from src.agent_core.prompts import prompt_factory
from src.agent_core.sandbox import get_sandbox_provider
from src.agent_core.skills import get_skill_manager
from src.agent_core.tools.memory_tools import recall_memory, save_memory
from src.agent_core.workspace import get_thread_workspace_manager
from src.config.settings import get_settings
from src.storage.conversation_store import get_conversation_store

_LEAD_AGENT_PROMPT_NAME = "LEAD_AGENT"
_DEFAULT_RECURSION_LIMIT = 25
_THINKING_RECURSION_LIMIT = 40


def build_lead_agent(
    *,
    thinking_enabled: bool = False,
    extra_tools: list[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    user_id: str | None = None,
):
    """组装一个 Lead Agent。

    Args:
        thinking_enabled: 是否开启深度思考模式（更大推理预算 + 模型侧扩展推理）。
        extra_tools: 前端本次请求注入的工具（经 `CustomToolConverter` 转换），
            并入 Lead Agent 自己的基础工具集，对应原 `tool_agent` 的职责。
        checkpointer: 会话持久化 checkpointer，为空时该 Agent 不具备跨轮记忆
            （仅供单次调用/测试场景使用）。
        user_id: 归属用户 ID，用于 `TitleMiddleware` 生成标题后持久化到
            `conversation_store`；为空时标题只记日志，不落库。

    Returns:
        编译好的 Lead Agent（`CompiledStateGraph`），可直接
        `.ainvoke({"messages": [...]}, config)` 调用。
    """
    skill_manager = get_skill_manager()
    settings = get_settings()

    async def _persist_title(conversation_id: str, title: str) -> None:
        if user_id:
            await get_conversation_store().update_title(conversation_id, user_id, title)

    base_tools: list[BaseTool] = [
        save_memory,
        recall_memory,
        *skill_manager.get_tools("general"),
        *skill_manager.get_tools("tool"),
        *(extra_tools or []),
        build_rag_delegate_tool(),
        build_web_search_delegate_tool(),
        build_database_delegate_tool(),
    ]

    middlewares = [
        *build_middlewares(
            guardrail_provider=get_guardrail_provider(),
            sandbox_provider=get_sandbox_provider(),
            workspace_manager=get_thread_workspace_manager(),
            memory_manager=get_memory_manager(),
            prompt_factory=prompt_factory,
            title_model_settings=TitleModelSettings(
                model_name=settings.DEFAULT_MODEL,
                provider=settings.PROVIDER,
                api_key=settings.API_KEY,
                base_url=settings.BASE_URL,
            ),
            on_title_generated=_persist_title,
        ),
        # Lead Agent 专属业务规则，不是通用中间件流水线的一部分，追加在最后。
        DatasourceRoutingMiddleware(),
        # 必须是最内层（离真实模型调用最近）：见该模块说明。
        StreamingModelMiddleware(),
    ]

    return create_agent(
        model=create_chat_model(thinking_enabled=thinking_enabled),
        tools=base_tools,
        system_prompt=prompt_factory.get(_LEAD_AGENT_PROMPT_NAME),
        middleware=middlewares,
        context_schema=AgentRuntimeContext,
        checkpointer=checkpointer,
    )


def resolve_recursion_limit(thinking_enabled: bool) -> int:
    """按是否深度思考模式返回本次调用应使用的 `recursion_limit`。"""
    return _THINKING_RECURSION_LIMIT if thinking_enabled else _DEFAULT_RECURSION_LIMIT
