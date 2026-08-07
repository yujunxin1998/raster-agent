"""Memory 机制模块：三层记忆体系（短期 checkpointer / 长期向量存储 / 压缩）。

本工程只迁移长期记忆 + 压缩两层的基础设施；短期记忆依赖 LangGraph
checkpointer，checkpointer 的创建属于编排层范畴，不在本工程范围内
（详见 README「本工程范围」一节），因此本模块的 `compress_after_chat`
需要调用方传入一个已编译的 `graph` 实例，本模块自身不持有/不创建 graph。

使用方式::

    from src.agent_core.memory import get_memory_manager

    await get_memory_manager().get_relevant_context(query, user_id)
"""
from src.agent_core.memory.base_memory_store import BaseMemoryStore
from src.agent_core.memory.elasticsearch_memory_store import ElasticsearchMemoryStore
from src.agent_core.memory.memory_compressor import MemoryCompressor
from src.agent_core.memory.memory_extractor import MemoryExtractor
from src.agent_core.memory.memory_manager import (
    MemoryManager,
    get_memory_manager,
    get_memory_with_retry,
    init_memory_manager,
)
from src.agent_core.memory.memory_sensitive_filter import contains_sensitive_info
from src.agent_core.memory.memory_staleness_reviewer import MemoryStalenessReviewer
from src.agent_core.memory.user_profile_updater import UserProfileUpdater

__all__ = [
    "BaseMemoryStore",
    "ElasticsearchMemoryStore",
    "MemoryExtractor",
    "MemoryCompressor",
    "MemoryManager",
    "init_memory_manager",
    "get_memory_manager",
    "get_memory_with_retry",
    "contains_sensitive_info",
    "MemoryStalenessReviewer",
    "UserProfileUpdater",
]
