"""Prompt 机制模块：提示词模板管理，封装为 PromptFactory。

原样迁移自 diit-agent-server 的 `src/core/prompts/`（15 个提示词模板全部
拷贝到本模块的 `templates/` 目录），在原有"扫描 + 注册表"的基础上新增了
`PromptFactory` 这一层，统一处理"取原文"与"按变量渲染"两种使用方式
（原项目里后者由各调用方分别手写 `.format(...)`，现在收口成一个方法）。

模板本身不依赖数据库/沙箱/权限等基础设施，加载不需要等待其它模块初始化，
因此沿用原项目"模块导入时即加载完毕"的做法，在此直接构造进程内唯一的
`prompt_factory` 单例，不需要像 `SkillManager`/`MemoryManager` 那样在
`main.py` 的 lifespan 里显式调用 `init_xxx()`。

使用方式::

    from src.agent_core.prompts import prompt_factory

    prompt_factory.get("SUPERVISOR")
    prompt_factory.render("RAG_QUERY_REWRITE", history_text=..., tool_query=..., user_query=...)
    prompt_factory.names
"""
from loguru import logger

from src.agent_core.prompts.prompt_factory import PromptFactory
from src.agent_core.prompts.prompt_loader import PromptLoader
from src.agent_core.prompts.prompt_registry import PromptRegistry
from src.agent_core.prompts.prompt_template import PromptTemplate

_registry = PromptLoader().load()
prompt_factory: PromptFactory = PromptFactory(_registry)

logger.info(f"[PromptFactory] 就绪 | 共 {len(_registry)} 条模板: {_registry.names}")

__all__ = [
    "prompt_factory",
    "PromptFactory",
    "PromptRegistry",
    "PromptTemplate",
    "PromptLoader",
]
