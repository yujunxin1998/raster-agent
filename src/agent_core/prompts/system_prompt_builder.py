"""多轮 Agent 循环用的系统提示词拼装器。

跟 `templates/`（`PromptFactory`/`PromptRegistry` 管的"提示词模板"，单次调用用，
按名称整篇 `.get()`/`.render()`）是两种不同的东西：这里管的是 `system/<agent_name>/`
下按模块拆开的系统提示词片段（`role`/`thinking_style`/`clarification_system`/
`skill_system`/`subagent_system`/`response_style`），每个模块文件只写标签内部的
正文，外层 `<module>...</module>` 标签由 `build()` 加，最终拼成一个 Agent 实际
使用的 `system_prompt` 字符串。

每个非 `role` 模块都有一个同名布尔开关，关闭时该模块连标签一起整段跳过（不是
空标签）——这样模型看到的上下文里不会出现"这个能力不存在"的空壳提示。`role`
永远注入，一个 Agent 没有身份定义没有意义。

用法::

    from src.agent_core.prompts.system_prompt_builder import system_prompt_builder

    system_prompt_builder.build(
        "lead_agent", thinking_enabled=True, subagent_enabled=True, skill_enabled=True,
    )
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

_DEFAULT_SYSTEM_DIR = Path(__file__).parent / "system"

# (模块名, 对应的开关关键字参数名或 None 表示永远注入)
_MODULE_ORDER: tuple[tuple[str, str | None], ...] = (
    ("role", None),
    ("thinking_style", "thinking_enabled"),
    ("clarification_system", "clarification_enabled"),
    ("skill_system", "skill_enabled"),
    ("subagent_system", "subagent_enabled"),
    ("response_style", "response_style_enabled"),
)


class SystemPromptBuilder:
    """按模块开关拼装多轮 Agent 循环用的系统提示词。"""

    def __init__(self, system_dir: Path | None = None) -> None:
        """初始化拼装器。

        Args:
            system_dir: 系统提示词模块的根目录，默认使用本模块内置的
                `system/` 目录。
        """
        self._system_dir = system_dir or _DEFAULT_SYSTEM_DIR

    def build(
        self,
        agent_name: str,
        *,
        thinking_enabled: bool = True,
        clarification_enabled: bool = True,
        skill_enabled: bool = True,
        subagent_enabled: bool = True,
        response_style_enabled: bool = True,
    ) -> str:
        """拼装一个 Agent 的系统提示词。

        Args:
            agent_name: 子目录名（如 `lead_agent`），对应 `system/<agent_name>/`。
            thinking_enabled: 是否注入 `<thinking_style>` 模块。
            clarification_enabled: 是否注入 `<clarification_system>` 模块。
            skill_enabled: 是否注入 `<skill_system>` 模块。
            subagent_enabled: 是否注入 `<subagent_system>` 模块。
            response_style_enabled: 是否注入 `<response_style>` 模块。

        Returns:
            按固定顺序拼接好的系统提示词。模块文件缺失或对应开关为 False 时，
            该模块（含标签）整段跳过。
        """
        switches = {
            "thinking_enabled": thinking_enabled,
            "clarification_enabled": clarification_enabled,
            "skill_enabled": skill_enabled,
            "subagent_enabled": subagent_enabled,
            "response_style_enabled": response_style_enabled,
        }

        agent_dir = self._system_dir / agent_name
        sections: list[str] = []
        for module_name, switch_key in _MODULE_ORDER:
            if switch_key is not None and not switches[switch_key]:
                continue
            content = self._read_module(agent_dir, module_name)
            if content is None:
                continue
            sections.append(f"<{module_name}>\n{content}\n</{module_name}>")

        return "\n\n".join(sections)

    @staticmethod
    def _read_module(agent_dir: Path, module_name: str) -> str | None:
        """读取一个模块文件的正文，缺失时记 warning 日志并返回 None。"""
        module_path = agent_dir / f"{module_name}.md"
        if not module_path.exists():
            logger.warning(f"[SystemPromptBuilder] 模块文件不存在，已跳过: {module_path}")
            return None
        return module_path.read_text(encoding="utf-8").strip()


system_prompt_builder = SystemPromptBuilder()
