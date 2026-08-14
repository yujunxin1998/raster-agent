"""把一个 SkillDefinition 包装成 LangChain 原生 StructuredTool。

与原项目 `src/core/skills/factory.py` 的差异：原来的 `SkillAccessGuard`
（只做"用户是否关闭了这个技能"这一层判断）被替换为统一的
`GuardrailProvider`（对应设计文档 5.1 节），脚本执行现在需要先从
`SandboxProvider` 取一个 Sandbox 实例再传给 `SkillContentReader`
（对应设计文档 5.3 节）。参数 schema 构建逻辑（`SkillParameterSchemaBuilder`）
未改动。
"""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any, Optional

from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import StructuredTool
from loguru import logger
from pydantic import BaseModel, create_model

from src.agent_core.guardrail.guardrail_provider import GuardrailContext, GuardrailProvider
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.sandbox.sandbox_provider import SandboxProvider
from src.agent_core.skills.skill_content_reader import SkillContentReader
from src.agent_core.skills.skill_definition import SkillDefinition

_MISSING = object()

# SKILL.md frontmatter parameter type -> Python type 映射
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

_SANDBOX_UNAVAILABLE_REASON_TEMPLATE = (
    "技能 [{tool_name}] 需要沙箱环境执行，但当前调用缺少会话上下文（thread_id），本次未执行。"
)


class SkillParameterSchemaBuilder:
    """把 SKILL.md frontmatter 里的 `parameters` 列表动态转成 Pydantic 模型。

    思路与 `src/agent_core/tools/custom_tool_converter.py::_schema_to_pydantic`
    一致，只是数据源更简单（扁平列表，不是完整 JSON Schema）。
    """

    def build(self, skill: SkillDefinition) -> type[BaseModel]:
        """构建该技能对应的参数 Pydantic 模型。

        Args:
            skill: 目标技能定义。

        Returns:
            动态生成的 Pydantic 模型类，可直接作为 StructuredTool 的 args_schema。
        """
        fields: dict[str, Any] = {}

        for parameter in skill.parameters:
            if not isinstance(parameter, dict):
                continue
            parameter_name = str(parameter.get("name") or "").strip()
            if not parameter_name:
                continue

            python_type = _TYPE_MAP.get(parameter.get("type", "string"), str)

            if parameter.get("required", False):
                fields[parameter_name] = (python_type, ...)
            else:
                fields[parameter_name] = (Optional[python_type], None)

        model_name = f"SkillInput_{skill.tool_name}_{uuid.uuid4().hex[:6]}"
        return create_model(model_name, **fields)


class SkillToolFactory:
    """把一个 SkillDefinition 包装成 LangChain StructuredTool。

    实际执行委托给 SkillContentReader（按需读取正文/参考资料/经沙箱跑脚本），
    本类负责：建参数 schema + 组装协程 + 权限校验 + 沙箱获取/释放。
    """

    def __init__(
        self,
        guardrail_provider: GuardrailProvider,
        sandbox_provider: SandboxProvider,
        skill_script_timeout_seconds: int,
        schema_builder: SkillParameterSchemaBuilder | None = None,
    ) -> None:
        """初始化技能工具工厂。

        Args:
            guardrail_provider: 权限校验器，调用前检查该用户是否被允许执行该技能。
            sandbox_provider: 沙箱提供者，脚本型技能执行前用它获取 Sandbox 实例。
            skill_script_timeout_seconds: 脚本执行超时时间，对应配置项
                `SKILL_SCRIPT_TIMEOUT_SECONDS`。
            schema_builder: 参数 schema 构建器，默认使用 SkillParameterSchemaBuilder。
        """
        self._guardrail_provider = guardrail_provider
        self._sandbox_provider = sandbox_provider
        self._skill_script_timeout_seconds = skill_script_timeout_seconds
        self._schema_builder = schema_builder or SkillParameterSchemaBuilder()

    def create(self, skill: SkillDefinition) -> StructuredTool:
        """把一个技能包装成可被 LangChain Agent 调用的 StructuredTool。

        Args:
            skill: 目标技能定义。

        Returns:
            对应的 StructuredTool 实例。
        """
        args_schema = self._schema_builder.build(skill)
        reader = SkillContentReader(skill)

        async def _invoke(runtime: ToolRuntime[AgentRuntimeContext], **kwargs: Any) -> str:
            context = runtime.context
            user_id = context.user_id
            conversation_id = context.conversation_id

            decision = await self._guardrail_provider.check(
                user_id=user_id,
                tool_name=skill.tool_name,
                tool_args=kwargs,
                context=GuardrailContext(conversation_id=conversation_id),
            )
            if not decision.is_allowed:
                return decision.reason

            params = dict(kwargs)
            for key in skill.runtime_context_keys:
                value = getattr(context, key, _MISSING)
                if value is not _MISSING:
                    params[key] = value

            if not skill.has_script():
                return await reader.assemble(params, sandbox=None, script_timeout_seconds=self._skill_script_timeout_seconds)

            if not conversation_id:
                logger.warning(
                    f"[SkillToolFactory] 技能 {skill.tool_name} 需要沙箱但缺少 thread_id，跳过脚本执行"
                )
                return _SANDBOX_UNAVAILABLE_REASON_TEMPLATE.format(tool_name=skill.tool_name)

            secret_env = self._resolve_secret_env(skill, context.secrets)

            sandbox = await self._sandbox_provider.acquire(conversation_id, user_id)
            try:
                return await reader.assemble(
                    params,
                    sandbox=sandbox,
                    script_timeout_seconds=self._skill_script_timeout_seconds,
                    secret_env=secret_env,
                )
            finally:
                await self._sandbox_provider.release(sandbox)

        return StructuredTool(
            name=skill.tool_name,
            description=skill.description or f"技能: {skill.tool_name}",
            args_schema=args_schema,
            coroutine=_invoke,
        )

    def _resolve_secret_env(self, skill: SkillDefinition, secrets: Mapping[str, str]) -> dict[str, str]:
        """按"三重交集"规则算出本次调用允许注入的密钥环境变量（设计文档 5.4 节）。

        三个条件缺一不可：技能已被 Guardrail 放行执行（调用到这里之前已校验）×
        调用方本次请求通过 `AgentRuntimeContext.secrets` 提供了值 × SKILL.md
        frontmatter 用 `required_secrets` 声明了这个名字。任何一环缺失都只是
        "这次拿不到该密钥"，不报错、不影响技能其余部分执行。

        Args:
            skill: 目标技能定义。
            secrets: 本次调用的 `AgentRuntimeContext.secrets`，绝不进入对话
                消息/日志/checkpoint。

        Returns:
            允许注入子进程的密钥环境变量字典，可能为空。
        """
        if not skill.required_secrets:
            return {}

        if not isinstance(secrets, Mapping):
            return {}

        declared_names = {secret.name for secret in skill.required_secrets}
        return {
            name: value
            for name, value in secrets.items()
            if name in declared_names
        }
