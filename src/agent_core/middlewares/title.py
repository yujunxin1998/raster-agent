"""会话标题生成中间件（设计文档 4.2 节 #10）。

文档里对应的原 `src/conversation/title_generator.py` 这次没有随基础设施
一起迁移（仓库里没有 `conversations` 表/DAO，见 README「本工程范围」），
这里给出一个自包含实现：复用 `MemoryCompressor`/`MemoryExtractor` 已经在
用的 `init_chat_model` 调用方式 + 新增的 `TITLE_GENERATION` 提示词模板生成
标题；持久化到 `conversations` 表这一步，通过可选的 `on_title_generated`
回调交给二期 conversation 层接入时再实现——本中间件不假装自己知道数据要
存进哪张表。
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.model import create_chat_model
from src.agent_core.prompts.prompt_factory import PromptFactory

_TITLE_PROMPT_NAME = "TITLE_GENERATION"
_MAX_TITLE_LENGTH = 20


class TitleMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """首轮对话结束后，异步生成一个简短的会话标题。"""

    def __init__(
        self,
        prompt_factory: PromptFactory,
        *,
        model_name: str,
        provider: str,
        api_key: str,
        base_url: str,
        on_title_generated: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ) -> None:
        """初始化中间件。

        Args:
            prompt_factory: 提示词工厂，用于渲染 `TITLE_GENERATION` 模板。
            model_name: 标题生成用的模型名称。
            provider: 模型供应商标识。
            api_key: 模型服务 API Key。
            base_url: 模型服务 Base URL。
            on_title_generated: 标题生成完成后的回调 `(conversation_id, title)`，
                用于持久化到会话存储；为空时只记 info 日志（二期 conversation
                层落地前的默认行为）。
        """
        super().__init__()
        self._prompt_factory = prompt_factory
        self._model_name = model_name
        self._provider = provider
        self._api_key = api_key
        self._base_url = base_url
        self._on_title_generated = on_title_generated

    async def aafter_agent(self, state: Any, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any] | None:
        """仅在首轮（一问一答）结束后，异步触发标题生成。

        Args:
            state: 当前 Agent 状态，含本轮结束后的完整 `messages`。
            runtime: 运行时上下文，携带 `conversation_id`。

        Returns:
            始终返回 None（本中间件不更新 state，只是 fire-and-forget 触发
            后台标题生成任务）。
        """
        context = runtime.context
        messages = state.get("messages") or []
        # 不能直接用 len(messages) 判断"是不是第一轮"——第一轮里只要触发过一次
        # 工具调用（哪怕只是 search_knowledge_base/query_database 这类直接挂载
        # 的技能，不需要好几轮），就会往消息列表里插入
        # AIMessage(tool_calls=[...])/ToolMessage 这一对，把消息数顶到 2 条以上，
        # 之前用固定阈值 2 判断会被直接跳过、标题永远生成不出来。改成数
        # HumanMessage 的条数：第一轮不管中间发生了多少次工具调用，用户消息
        # 永远只有 1 条；第二轮开始变成 2 条，自然不再触发（本中间件只想在
        # 第一轮结束后生成一次标题）。
        human_message_count = sum(1 for message in messages if isinstance(message, HumanMessage))
        if context is None or not context.conversation_id or human_message_count != 1:
            return None

        user_message = self._last_text(messages, HumanMessage)
        ai_response = self._last_text(messages, AIMessage)
        if not user_message or not ai_response:
            return None

        asyncio.create_task(self._generate(context.conversation_id, user_message, ai_response))
        return None

    async def _generate(self, conversation_id: str, user_message: str, ai_response: str) -> None:
        try:
            prompt = self._prompt_factory.render(
                _TITLE_PROMPT_NAME, user_message=user_message, ai_response=ai_response,
            )
            llm = create_chat_model(
                model=self._model_name, provider=self._provider,
                api_key=self._api_key, base_url=self._base_url,
            )
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            title = str(result.content).strip().strip('"').strip("《》")[:_MAX_TITLE_LENGTH]
        except Exception as exc:
            logger.warning(f"[TitleMiddleware] 标题生成失败，已跳过 conversation_id={conversation_id}: {exc}")
            return

        if not title:
            return

        if self._on_title_generated is not None:
            await self._on_title_generated(conversation_id, title)
        else:
            logger.info(f"[TitleMiddleware] 生成标题 conversation_id={conversation_id} title={title}")

    @staticmethod
    def _last_text(messages: list, message_type: type) -> str:
        """取消息列表里最后一条指定类型消息的文本内容。"""
        for message in reversed(messages):
            if isinstance(message, message_type) and isinstance(message.content, str):
                return message.content
        return ""
