"""项目统一异常体系。

设计原则：
    - 所有自定义异常必须继承自 AgentCoreError，禁止在业务代码中直接
      raise 裸的 Exception/ValueError（校验类场景可以用 ValueError，
      但跨越模块边界抛出、需要被上层统一捕获处理的异常必须走这里）。
    - 每个子系统（skill / memory / guardrail / sandbox / workspace）拥有
      自己的异常子类，携带 error_code 供日志/监控按类型聚合，不要求调用方
      解析异常消息文本来判断错误类型。
    - 禁止裸 except Exception: pass —— 需要吞掉异常的地方，必须至少记录
      warning 日志并说明为什么可以安全忽略（这一点延续自原项目
      memory/skills 模块"失败不中断主流程"的容错哲学）。
"""
from __future__ import annotations


class AgentCoreError(Exception):
    """项目自定义异常的公共基类。

    Attributes:
        message: 面向开发者/日志的错误描述。
        error_code: 机器可读的错误码，用于日志聚合和前端按错误类型做不同展示。
    """

    error_code: str = "AGENT_CORE_ERROR"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if error_code:
            self.error_code = error_code

    def __str__(self) -> str:  # noqa: D105
        return f"[{self.error_code}] {self.message}"


class ConfigurationError(AgentCoreError):
    """配置缺失或非法（如必需的环境变量未设置）。"""

    error_code = "CONFIGURATION_ERROR"


# ── Skill 机制 ──────────────────────────────────────────────────
class SkillError(AgentCoreError):
    """技能机制相关异常的基类。"""

    error_code = "SKILL_ERROR"


class SkillNotFoundError(SkillError):
    """按 tool_name 查询技能未命中。"""

    error_code = "SKILL_NOT_FOUND"


class SkillDefinitionInvalidError(SkillError):
    """SKILL.md frontmatter 缺失或格式不合法。"""

    error_code = "SKILL_DEFINITION_INVALID"


class DuplicateSkillError(SkillError):
    """两个技能声明了相同的 tool_name，注册时显式拒绝而不是静默覆盖。"""

    error_code = "SKILL_DUPLICATE_NAME"


# ── Memory 机制 ──────────────────────────────────────────────────
class MemoryStoreError(AgentCoreError):
    """长期记忆存储相关异常的基类。"""

    error_code = "MEMORY_STORE_ERROR"


class MemoryNotInitializedError(MemoryStoreError):
    """MemoryManager 尚未完成初始化即被调用。"""

    error_code = "MEMORY_NOT_INITIALIZED"


class SensitiveContentRejectedError(MemoryStoreError):
    """写入内容命中敏感信息规则，被拒绝保存。"""

    error_code = "MEMORY_SENSITIVE_CONTENT_REJECTED"


# ── Guardrail 权限控制 ─────────────────────────────────────────────
class GuardrailError(AgentCoreError):
    """权限控制相关异常的基类。"""

    error_code = "GUARDRAIL_ERROR"


class GuardrailDeniedError(GuardrailError):
    """权限校验被拒绝。

    注意：Guardrail 的常规拒绝路径（工具调用被拦截）不通过抛异常传递，
    而是返回 GuardrailDecision 供调用方转成一段说明文本继续对话
    （详见 src/agent_core/guardrail/guardrail_provider.py 的设计说明）。
    这个异常类仅用于"校验过程本身出错"（如权限存储不可用）等不属于
    正常业务拒绝的场景。
    """

    error_code = "GUARDRAIL_DENIED"


# ── Sandbox 沙箱 ────────────────────────────────────────────────
class SandboxError(AgentCoreError):
    """沙箱执行相关异常的基类。"""

    error_code = "SANDBOX_ERROR"


class SandboxCommandTimeoutError(SandboxError):
    """沙箱内命令执行超时。"""

    error_code = "SANDBOX_COMMAND_TIMEOUT"


class SandboxNotAcquiredError(SandboxError):
    """在未获取 Sandbox 实例的情况下尝试执行沙箱操作。"""

    error_code = "SANDBOX_NOT_ACQUIRED"


# ── Workspace 虚拟文件系统 ───────────────────────────────────────
class WorkspaceError(AgentCoreError):
    """虚拟工作区相关异常的基类。"""

    error_code = "WORKSPACE_ERROR"


class PathTraversalError(WorkspaceError):
    """检测到路径穿越（访问路径越出了允许的根目录范围）。

    统一定义在这里，供 workspace（虚拟工作区）与 skill 文件浏览
    （SkillFileTreeReader）两处共用同一套异常语义。
    """

    error_code = "PATH_TRAVERSAL_DETECTED"


# ── 工具机制 ────────────────────────────────────────────────────
class ToolConversionError(AgentCoreError):
    """前端工具定义（JSON Schema）转换为 LangChain BaseTool 失败。"""

    error_code = "TOOL_CONVERSION_ERROR"


# ── Prompt 机制 ─────────────────────────────────────────────────
class PromptError(AgentCoreError):
    """提示词机制相关异常的基类。"""

    error_code = "PROMPT_ERROR"


class PromptNotFoundError(PromptError):
    """按名称查询提示词模板未命中。"""

    error_code = "PROMPT_NOT_FOUND"


class PromptRenderError(PromptError):
    """提示词模板渲染失败（通常是调用方缺少模板所需的变量）。"""

    error_code = "PROMPT_RENDER_ERROR"
