"""提示词工厂：对外唯一入口，负责"取模板 + 按需渲染"。

这是相对原项目 `src/core/prompts/` 新增的一层封装。原项目的 `PromptRegistry`
只做"存 + 取原文"，凡是需要拼变量的场景（如 `rag_query_rewrite.md` 里的
`{history_text}`/`{tool_query}`/`{user_query}`）都由调用方各自手写
`.format(...)`，散落在 `rag_query_rewrite_service.py` 等多处，拼错变量名、
漏传变量导致 `KeyError` 只能在运行时才暴露。

`PromptFactory` 把"渲染"这个动作收口成一个方法：`render(name, **variables)`，
渲染失败时统一转换成 `PromptRenderError`（带清晰的模板名 + 缺失变量名），
比裸 `KeyError` 更容易定位问题；不需要变量的静态模板（如 `SUPERVISOR`、
`GENERAL_AGENT`）直接调用 `get(name)` 取原文即可，无需关心是否要渲染。

用法::

    from src.agent_core.prompts import prompt_factory

    prompt_factory.get("SUPERVISOR")                     # 静态模板，原样返回
    prompt_factory.render(
        "RAG_QUERY_REWRITE",
        history_text=history, tool_query=tool_query, user_query=user_query,
    )                                                     # 按占位符渲染
"""
from __future__ import annotations

from src.agent_core.prompts.prompt_registry import PromptRegistry
from src.common.exceptions import PromptRenderError


class PromptFactory:
    """提示词机制对外的唯一入口。"""

    def __init__(self, registry: PromptRegistry) -> None:
        """初始化提示词工厂。

        Args:
            registry: 已完成加载的提示词注册表（通常由 `PromptLoader.load()` 产出）。
        """
        self._registry = registry

    @property
    def registry(self) -> PromptRegistry:
        """暴露底层注册表，供需要访问模板元信息（如 source）的场景使用。"""
        return self._registry

    def get(self, name: str) -> str:
        """取一个模板的原始正文，不做任何渲染。

        适用于没有 `{variable}` 占位符的静态提示词（如 `SUPERVISOR`、
        `GENERAL_AGENT`），也适用于调用方希望自己控制拼接逻辑的场景。

        Args:
            name: 模板名称，大小写不敏感。

        Returns:
            模板正文原文。

        Raises:
            PromptNotFoundError: 模板不存在。
        """
        return self._registry.get_content(name)

    def render(self, name: str, **variables: object) -> str:
        """取一个模板并用 `str.format` 渲染占位符变量。

        Args:
            name: 模板名称，大小写不敏感。
            **variables: 模板里 `{variable}` 占位符对应的值。

        Returns:
            渲染后的最终提示词文本。

        Raises:
            PromptNotFoundError: 模板不存在。
            PromptRenderError: 模板包含调用方未提供的占位符变量。
        """
        content = self._registry.get_content(name)
        try:
            return content.format(**variables)
        except KeyError as exc:
            raise PromptRenderError(
                f"渲染模板 '{name}' 缺少变量: {exc}；已提供变量: {sorted(variables.keys())}"
            ) from exc

    def exists(self, name: str) -> bool:
        """判断某个名称的模板是否存在。"""
        return self._registry.exists(name)

    @property
    def names(self) -> list[str]:
        """全部已注册模板名称。"""
        return self._registry.names
