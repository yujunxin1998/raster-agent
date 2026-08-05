"""Prompt 机制模块，分两类管理，机制不通用：

1. **提示词模板**（`templates/*.md`）—— 单次、单一用途 LLM 调用用的模板/系统
   消息（如 `DB_ANALYSIS`、`RAG_QUERY_REWRITE`、`TITLE_GENERATION`），文件名转
   大写即模板名，走本模块的 `PromptFactory`：`get(name)` 取原文，
   `render(name, **kv)` 做 `.format()` 占位符替换。新增这类提示词直接在
   `templates/` 下加一个 `.md` 文件即可，不需要额外注册代码。
2. **系统提示词模块**（`system/<agent_name>/*.md`）—— 驱动一个带工具循环的
   Agent 的 `system_prompt`（目前只有 `system/lead_agent/`），按
   `role`/`thinking_style`/`clarification_system`/`skill_system`/
   `subagent_system`/`response_style` 六个模块拆开，每个模块文件只写标签内部
   的正文，走 `agent_core.prompts.system_prompt_builder.system_prompt_builder`
   按开关条件拼装（`role` 永远注入，其余 5 个各有一个同名布尔开关，关闭时
   连标签一起整段跳过）。**新增 Agent 的系统提示词不要往 `templates/` 里加
   扁平 `.md`**，应在 `system/` 下新建同名子目录、按六个模块拆分。

`PromptFactory` 管的模板本身不依赖数据库/沙箱/权限等基础设施，加载不需要等待
其它模块初始化，因此沿用"模块导入时即加载完毕"的做法，在此直接构造进程内唯一的
`prompt_factory` 单例，不需要像 `SkillManager`/`MemoryManager` 那样在
`main.py` 的 lifespan 里显式调用 `init_xxx()`。

使用方式::

    from src.agent_core.prompts import prompt_factory
    from src.agent_core.prompts.system_prompt_builder import system_prompt_builder

    prompt_factory.get("TITLE_GENERATION")
    prompt_factory.render("RAG_QUERY_REWRITE", history_text=..., tool_query=..., user_query=...)
    prompt_factory.names

    system_prompt_builder.build("lead_agent", thinking_enabled=True, subagent_enabled=True)
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
