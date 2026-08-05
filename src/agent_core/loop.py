"""Agent Loop 中间件流水线组装（设计文档 4.2 节）。

本工程本轮不包含编排层（见仓库根目录 README「本工程范围」）：这里只提供
`build_middlewares()`，按文档顺序组装 11 个中间件并返回列表，供二期落地
Lead Agent 时传给 `create_agent(model, tools, middleware=build_middlewares(...))`
使用。`main.py` 目前没有编排层入口，不需要调用本模块。

顺序约束（对应文档 4.2 节"这条流水线的关键约束"）：LangChain 中间件列表
"靠前的在最外层"，Guardrail 必须排在 ToolErrorHandling 之前——权限拒绝是
短路（不执行），而 ToolErrorHandling 包裹的是"执行过程中抛出的异常"，
两者是不同职责的两层包装，顺序不能颠倒。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from langchain.agents.middleware import AgentMiddleware

from src.agent_core.guardrail.guardrail_provider import GuardrailProvider
from src.agent_core.memory.memory_manager import MemoryManager
from src.agent_core.middlewares.guardrail import GuardrailMiddleware
from src.agent_core.middlewares.input_sanitization import InputSanitizationMiddleware
from src.agent_core.middlewares.loop_detection import LoopDetectionMiddleware
from src.agent_core.middlewares.memory_extraction import MemoryExtractionMiddleware
from src.agent_core.middlewares.memory_injection import MemoryInjectionMiddleware
from src.agent_core.middlewares.sandbox_middleware import SandboxMiddleware
from src.agent_core.middlewares.summarization import SummarizationMiddleware
from src.agent_core.middlewares.thread_data import ThreadDataMiddleware
from src.agent_core.middlewares.title import TitleMiddleware
from src.agent_core.middlewares.tool_audit import ToolAuditMiddleware
from src.agent_core.middlewares.tool_error_handling import ToolErrorHandlingMiddleware
from src.agent_core.prompts.prompt_factory import PromptFactory
from src.agent_core.sandbox.sandbox_provider import SandboxProvider
from src.agent_core.workspace.thread_workspace_manager import ThreadWorkspaceManager

_DEFAULT_LOOP_DETECTION_THRESHOLD = 3


@dataclass(frozen=True)
class TitleModelSettings:
    """`TitleMiddleware` 需要的 LLM 调用参数，通常直接取自 `SystemConfiguration`。"""

    model_name: str
    provider: str
    api_key: str
    base_url: str


def build_middlewares(
    *,
    guardrail_provider: GuardrailProvider,
    sandbox_provider: SandboxProvider,
    workspace_manager: ThreadWorkspaceManager,
    memory_manager: MemoryManager,
    prompt_factory: PromptFactory,
    title_model_settings: TitleModelSettings,
    loop_detection_threshold: int = _DEFAULT_LOOP_DETECTION_THRESHOLD,
    on_title_generated: Optional[Callable[[str, str], Awaitable[None]]] = None,
) -> list[AgentMiddleware]:
    """按设计文档 4.2 节的顺序组装 11 个中间件。

    Args:
        guardrail_provider: 权限校验器，通常传入 `get_guardrail_provider()`。
        sandbox_provider: 沙箱提供者，通常传入 `get_sandbox_provider()`。
        workspace_manager: 会话隔离工作区管理器，通常传入
            `get_thread_workspace_manager()`。
        memory_manager: 记忆机制门面，通常传入 `get_memory_manager()`。
        prompt_factory: 提示词工厂，供 `TitleMiddleware` 渲染标题生成模板。
        title_model_settings: 标题生成用的 LLM 调用参数。
        loop_detection_threshold: 死循环检测阈值，默认 3（对应旧 supervisor
            "连续 3 次"的判断）。
        on_title_generated: 标题生成完成后的回调，透传给 `TitleMiddleware`。

    Returns:
        按顺序排列的中间件实例列表，可直接传给
        `create_agent(middleware=build_middlewares(...))`。
    """
    return [
        InputSanitizationMiddleware(),
        ThreadDataMiddleware(workspace_manager),
        MemoryInjectionMiddleware(memory_manager),
        GuardrailMiddleware(guardrail_provider),
        SandboxMiddleware(sandbox_provider),
        ToolAuditMiddleware(),
        ToolErrorHandlingMiddleware(),
        LoopDetectionMiddleware(threshold=loop_detection_threshold),
        SummarizationMiddleware(memory_manager),
        TitleMiddleware(
            prompt_factory,
            model_name=title_model_settings.model_name,
            provider=title_model_settings.provider,
            api_key=title_model_settings.api_key,
            base_url=title_model_settings.base_url,
            on_title_generated=on_title_generated,
        ),
        MemoryExtractionMiddleware(memory_manager),
    ]
