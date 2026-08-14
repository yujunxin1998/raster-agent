"""记忆机制暴露给 Agent 的两个工具：save_memory / recall_memory。

Memory v2：这两个工具是"模型已经做出的显式决策"，跟后台
`MemoryCaptureMiddleware`/`MemoryUpdateWorker` 那条被动推断链路是两个独立
入口——不经过 `MemoryDelta`/LLM 二次判断，直接写 `UserMemoryFactStore`
（PostgreSQL 规范化主存），但敏感信息前置校验（`MemoryManager.
is_sensitive_content`）仍然保留，跟 `MemoryApplyEngine`/管理 API 走同一套
规则，不因为写入通道不同而降低安全标准。

这两个工具属于"内部工具"，调用方应当在装配 Agent 工具集时同时调用
`tool_filter_registry.register("save_memory", "recall_memory")`
（对应 main.py 里原来的 `tool_filter.register(...)` 调用点），使其不出现在
前端展示事件和持久化记录中。
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agent_core.memory import get_memory_manager
from src.agent_core.memory.elasticsearch_memory_store import get_elasticsearch_memory_store
from src.common.constants import MemoryAuditAction, MemorySource
from src.config.settings import get_settings
from src.storage.memory_audit_store import get_memory_audit_store
from src.storage.user_memory_fact_store import get_user_memory_fact_store
from src.utils.user_utils import resolve_user_id

_MEMORY_FEATURE_DISABLED_MESSAGE = "记忆功能当前已关闭。"


@tool
async def save_memory(
    content: str,
    config: RunnableConfig,
    memory_type: str = "context",
    importance: int = 5,
) -> str:
    """将对话中值得长期保留的信息存入记忆库，供未来对话使用。

    memory_type 可选：preference（偏好习惯）、knowledge（专业知识/技能）、
    context（客观背景事实）、behavior（行为模式）、goal（目标意图/长期规则）。
    importance 范围 1-10，越高越重要。
    """
    settings = get_settings()
    if not settings.MEMORY_ENABLED or not settings.MEMORY_TOOL_ENABLED:
        return _MEMORY_FEATURE_DISABLED_MESSAGE

    configurable = config.get("configurable", {})
    user_id = resolve_user_id(configurable.get("user_id"))
    conversation_id = configurable.get("thread_id")

    manager = get_memory_manager()
    if manager.is_sensitive_content(content):
        await _record_audit(
            MemoryAuditAction.SAVE_REJECTED, user_id, conversation_id, detail="命中敏感信息过滤规则",
        )
        return "记忆保存失败：内容包含敏感信息。"

    try:
        result = await get_user_memory_fact_store().add_or_reinforce(
            user_id=user_id, content=content, category=memory_type, importance=importance,
            confidence=1.0, status="active", source_conversation_id=conversation_id,
        )
    except Exception:
        return "记忆保存失败：记忆服务暂不可用。"

    action = MemoryAuditAction.FACT_REINFORCED if result["action"] == "reinforced" else MemoryAuditAction.CREATE
    await _record_audit(action, user_id, conversation_id, memory_id=result["fact_id"], detail=content)
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

    memories = await get_elasticsearch_memory_store().search(query=query, user_id=user_id)
    memories = [memory for memory in memories if memory.get("score", 1.0) >= settings.MEMORY_MIN_RECALL_SCORE]
    if not memories:
        return "记忆库中未找到相关信息。"

    await _record_audit(
        MemoryAuditAction.RECALL, user_id, conversation_id,
        detail=f"query={query[:200]} memory_ids={[memory['id'] for memory in memories]}",
    )

    lines = [
        f"{index + 1}. [{memory['memory_type']}] {memory['content']}"
        f"（记录于 {(memory.get('created_at') or '')[:10]}）"
        for index, memory in enumerate(memories)
    ]
    return "相关记忆：\n" + "\n".join(lines)


async def _record_audit(
    action: MemoryAuditAction, user_id: str, conversation_id: str | None, *, memory_id: str | None = None,
    detail: str | None = None,
) -> None:
    audit_store = get_memory_audit_store()
    if audit_store is None:
        return
    await audit_store.record(
        action=action, user_id=user_id, source=MemorySource.TOOL,
        memory_id=memory_id, conversation_id=conversation_id, detail=detail,
    )
