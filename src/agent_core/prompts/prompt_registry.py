"""提示词注册表（内存索引）。

与原项目 `src/core/prompts/registry.py::PromptRegistry` 的差异：去掉了原实现
里 `__getattr__` 的属性式动态访问（`prompts.TITLE_GENERATION`）。这种"魔法属性"
访问方式对 IDE 补全、静态类型检查都不友好，也和"方法调用应当显式"的
代码规范相悖；改为统一走 `get()`/`get_content()` 显式方法，语义更清晰，
配合 `PromptFactory` 使用体验上并不会更啰嗦。
"""
from __future__ import annotations

from src.agent_core.prompts.prompt_template import PromptTemplate
from src.common.exceptions import PromptNotFoundError


class PromptRegistry:
    """提示词的内存索引，支持按 name 精确查询、列出全量模板。"""

    def __init__(self) -> None:
        self._store: dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        """注册一条提示词模板，同名 name 会被覆盖（后加载的生效）。

        Args:
            template: 待注册的提示词模板。
        """
        self._store[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        """按名称精确查询模板（返回完整对象，含 source 等元信息）。

        Args:
            name: 模板名称，大小写不敏感（内部统一转大写匹配）。

        Returns:
            对应的 PromptTemplate。

        Raises:
            PromptNotFoundError: 未找到该名称对应的模板。
        """
        normalized_name = name.upper()
        if normalized_name not in self._store:
            raise PromptNotFoundError(f"提示词 '{name}' 未找到。已注册：{self.names}")
        return self._store[normalized_name]

    def get_content(self, name: str) -> str:
        """按名称查询模板正文（未渲染的原始文本）。

        Args:
            name: 模板名称。

        Returns:
            模板正文字符串。
        """
        return self.get(name).content

    def exists(self, name: str) -> bool:
        """判断某个名称的模板是否存在。"""
        return name.upper() in self._store

    def __getitem__(self, name: str) -> str:  # noqa: D105
        return self.get_content(name)

    @property
    def all(self) -> list[PromptTemplate]:
        """全部已注册模板。"""
        return list(self._store.values())

    @property
    def names(self) -> list[str]:
        """全部已注册模板名称（升序排列）。"""
        return sorted(self._store.keys())

    def __len__(self) -> int:  # noqa: D105
        return len(self._store)

    def __repr__(self) -> str:  # noqa: D105
        return f"PromptRegistry({self.names})"
