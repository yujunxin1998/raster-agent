"""记忆机制暴露给 Agent 的两个工具：save_memory / recall_memory。

原样迁移自 `src/core/tools/memory_tools.py`。这两个工具属于"内部工具"，
调用方应当在装配 Agent 工具集时同时调用
`tool_filter_registry.register("save_memory", "recall_memory")`
（对应 main.py 里原来的 `tool_filter.register(...)` 调用点），使其不出现在
前端展示事件和持久化记录中。
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agent_core.memory import get_memory_manager
from src.common.constants import MemoryAuditAction, MemorySource
from src.config.settings import get_settings
from src.storage.memory_audit_store import get_memory_audit_store
from src.utils.user_utils import resolve_user_id

_MEMORY_FEATURE_DISABLED_MESSAGE = "记忆功能当前已关闭。"


@tool
async def save_memory(
    content: str,
    config: RunnableConfig,
    memory_type: str = "fact",
    importance: int = 5,
) -> str:
    """将对话中值得长期保留的信息存入记忆库，供未来对话使用。

    memory_type 可选：fact（客观事实）、preference（偏好习惯）、decision（重要决策）、
    instruction（用户明确要求长期遵守的规则）、correction（用户纠正过的错误）。
    importance 范围 1-10，越高越重要。
    """
    settings = get_settings()
    if not settings.MEMORY_ENABLED or not settings.MEMORY_TOOL_ENABLED:
        return _MEMORY_FEATURE_DISABLED_MESSAGE

    configurable = config.get("configurable", {})
    user_id = resolve_user_id(configurable.get("user_id"))
    conversation_id = configurable.get("thread_id")

    memory_id = await get_memory_manager().store.save(
        content=content, user_id=user_id, conversation_id=conversation_id,
        memory_type=memory_type, importance=importance, source="tool",
    )
    if not memory_id:
        return "记忆保存失败：内容可能包含敏感信息或记忆服务不可用。"
    return f"已保存记忆：{content}"


@tool
async def recall_memory(query: str, config: RunnableConfig) -> str:
    """从长期记忆库检索与当前话题相关的历史记忆。

    当需要回忆用户过去提到的信息时调用。
    """
    settings = get_settings()
    if not settings.MEMORY_ENABLED or not settings.MEMORY_TOOL_ENABLED:
        return _MEMORY_FEATURE_DISABLED_MESSAGE

    configurable = config.get("configurable", {})
    user_id = resolve_user_id(configurable.get("user_id"))
    conversation_id = configurable.get("thread_id")

    memories = await get_memory_manager().store.search(query=query, user_id=user_id)
    memories = [memory for memory in memories if memory.get("score", 1.0) >= settings.MEMORY_MIN_RECALL_SCORE]
    if not memories:
        return "记忆库中未找到相关信息。"

    audit_store = get_memory_audit_store()
    if audit_store is not None:
        await audit_store.record(
            action=MemoryAuditAction.RECALL, user_id=user_id, source=MemorySource.TOOL,
            conversation_id=conversation_id,
            detail=f"query={query[:200]} memory_ids={[memory['id'] for memory in memories]}",
        )

    lines = [
        f"{index + 1}. [{memory['memory_type']}] {memory['content']}"
        f"（记录于 {(memory.get('created_at') or '')[:10]}）"
        for index, memory in enumerate(memories)
    ]
    return "相关记忆：\n" + "\n".join(lines)
