"""Memory 机制模块（Memory v2：三层记忆体系 + 持久化更新流水线 + 压缩）。

长期记忆更新不再是请求路径里的 fire-and-forget 调用，而是
`MemoryCaptureMiddleware` 落库事件/任务、`MemoryUpdateWorker` +
`MemoryApplyEngine` 异步消费、`FactProjector` 把结果同步进 ES 检索投影——
详见 `docs/raster-agent长期记忆重构设计.md`。`MemoryManager` 只保留画像只读
查询、压缩摘要写入、敏感信息前置校验这三件没有更自然归属的事情。

使用方式::

    from src.agent_core.memory import get_memory_manager

    await get_memory_manager().get_profile(user_id)
"""
from src.agent_core.memory.base_memory_store import BaseMemoryStore
from src.agent_core.memory.elasticsearch_memory_store import ElasticsearchMemoryStore
from src.agent_core.memory.fact_projector import FactProjector
from src.agent_core.memory.memory_apply_engine import MemoryApplyEngine, ProfilePatchConflictError
from src.agent_core.memory.memory_compressor import MemoryCompressor
from src.agent_core.memory.memory_context_builder import MemoryContextBuilder
from src.agent_core.memory.memory_delta import FactOperation, MemoryDelta, ProfilePatch
from src.agent_core.memory.memory_delta_validator import DeltaRejection, validate_delta
from src.agent_core.memory.memory_manager import (
    MemoryManager,
    get_memory_manager,
    init_memory_manager,
)
from src.agent_core.memory.memory_sensitive_filter import contains_sensitive_info
from src.agent_core.memory.memory_staleness_reviewer import MemoryStalenessReviewer
from src.agent_core.memory.memory_update_worker import MemoryUpdateWorker

__all__ = [
    "BaseMemoryStore",
    "ElasticsearchMemoryStore",
    "FactProjector",
    "MemoryApplyEngine",
    "ProfilePatchConflictError",
    "MemoryCompressor",
    "MemoryContextBuilder",
    "MemoryDelta",
    "ProfilePatch",
    "FactOperation",
    "DeltaRejection",
    "validate_delta",
    "MemoryManager",
    "init_memory_manager",
    "get_memory_manager",
    "contains_sensitive_info",
    "MemoryStalenessReviewer",
    "MemoryUpdateWorker",
]
